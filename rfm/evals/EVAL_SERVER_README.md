# RFM Evaluation Server

This FastAPI app (`evals/eval_server.py`) exposes a lightweight multi-GPU evaluation service for RFM/ReWiND checkpoints. It spins up a `MultiGPUEvalServer` that loads identical model replicas across the available CUDA devices and services requests by farming batches to whichever GPU is free.

## Prerequisites
- Python environment with project dependencies installed (`uv pip install -r requirements.txt` or `uv sync`).
- CUDA-visible GPUs (the server will error out if `torch.cuda.device_count() == 0`).
- A valid evaluation config YAML, e.g. `rfm/configs/eval_config_server.yaml`, pointing at the model to load (`model_path`) and any data pre-processing settings.

## Launching the server
```bash
uv run python evals/eval_server.py \
  --config_path rfm/configs/eval_config_server.yaml \
  --host 0.0.0.0 \
  --port 8000 \
  --num_gpus 2 \
  --max_workers 4
```
Key runtime options:
- `--config_path`: path to an `EvalServerConfig` YAML.
- `--num_gpus`: limit how many devices to load (defaults to every visible GPU).
- `--max_workers`: size of the thread pool that handles GPU checkout + inference.
  The YAML must include a `model_path` pointing at a pretrained checkpoint.

The server exposes:
- `POST /evaluate_batch`: JSON payload; accepts either `{"samples": [...]}` or a bare list of samples. Each sample is parsed into one of the `PreferenceSample`, `ProgressSample`, or `SimilaritySample` schemas in `rfm/data/dataset_types.py`.
- `POST /evaluate_batch_npy`: multipart form-data (for large numpy blobs, e.g., precomputed embeddings). Scalar metadata stays in normal form fields, while `.npy` uploads are loaded and merged back into each sample.
- `GET /gpu_status`: insight into pool utilization (`total_requests`, `last_used`, etc.).
- `GET /health`: simple readiness probe.

## Sample JSON payload
Below is the minimum structure for a single preference sample. All tensors/arrays must be JSON-serializable (convert to Python lists first). Optional fields such as `frames` or `metadata` can be omitted if not needed.

```json
[
  {
    "sample_type": "preference",
    "chosen_trajectory": {
      "task": "pick_place",
      "video_embeddings": [[0.1, 0.2], [0.3, 0.4]],
      "text_embedding": [0.5, 0.6],
      "target_progress": [0.2, 0.6, 1.0],
      "frames_shape": [2, 2]
    },
    "rejected_trajectory": {
      "task": "pick_place",
      "video_embeddings": [[-0.1, -0.2], [-0.3, -0.4]],
      "text_embedding": [0.0, 0.1],
      "target_progress": [0.1, 0.3, 0.5],
      "frames_shape": [2, 2]
    }
  }
]
```
Set `frames_shape` to match the `[num_frames, feature_dim]` (or equivalent) of each trajectory so the collator can build padding masks.

Send it with `curl`:
```bash
curl -X POST http://localhost:8000/evaluate_batch \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Health and status checks
Make sure the service is up before sending large batches:
```bash
# basic readiness probe
curl http://localhost:8000/health

# detailed GPU utilization snapshot
curl http://localhost:8000/gpu_status | jq .
```

## Multipart (`/evaluate_batch_npy`) payloads
When individual trajectories contain large numpy tensors, encode each array as a `.npy` upload. Inside the JSON sample, replace the array with a placeholder:
```json
"video_embeddings": {"__numpy_file__": "sample_0_chosen_video_embeddings"}
```
Then attach the matching `.npy` file in the multipart body (FastAPI will pair `sample_0_chosen_video_embeddings` with the uploaded buffer and convert it back to a tensor).

## Client helper script
`evals/sample_eval_request.py` builds a toy preference batch (with randomly generated embeddings) and submits it to `/evaluate_batch`. It is meant as smoke test / reference implementation:
```bash
uv run python evals/sample_eval_request.py \
  --base-url http://localhost:8000 \
  --num-samples 2
```


