# RL-VLM-F Baseline

Direct VLM preference evaluation using Gemini/GPT-4V. No reward model training required.

## Setup

```bash
export GEMINI_API_KEY="your-key"
cd evals/baselines/rlvlmf_base
python test/test_vlm.py  # Verify setup
```

## Usage

### Start Server

```bash
# Production (lightweight logging)
uv run python vlm_server.py --task "robot manipulation" --port 8002

# Debug (saves frames + full responses)
uv run python vlm_server.py --task "robot manipulation" --port 8002 --debug
```

### Run Evaluation

```bash
# Standard evaluation
python evals/run_model_eval.py \
  --config_path rfm/configs/config.yaml \
  --server_url http://localhost:8002 \
  --num_batches 10 --batch_size 10

# Quick test
python evals/run_model_eval.py \
  --config_path rfm/configs/config.yaml \
  --server_url http://localhost:8002 \
  --num_batches 2 --batch_size 5
```

## How It Works

### Smart Frame Selection
- **Input**: Any number of frames per trajectory (1, 8, 32, etc.)
- **Selection**: Always uses last frame (most representative of outcome)
- **Output**: 2 frames total for VLM comparison

### RL-VLM-F Prompting
Exact reproduction of the original paper's format:
```
1. What is shown in Image 1?
2. What is shown in Image 2?  
3. The goal is {task}. Is there any difference...?

Reply 0 if Image 1, 1 if Image 2, -1 if unclear.
```

### Architecture
```
Client → Server → Select Last Frames → Query Gemini → Return Metrics
         ↑
    (Any frame count)
```

## Files

```
rlvlmf_base/
├── vlm_server.py      # FastAPI server
├── vlm_baseline.py    # VLM query logic
├── pyproject.toml     # Dependencies
├── test/              # Test scripts
└── vlm_eval_logs/     # Auto-generated logs
```

## Debug vs Production

| Mode | Frames Saved | VLM Response | Use Case |
|------|--------------|--------------|----------|
| Production | ❌ | Truncated | Regular evaluation |
| Debug | ✅ | Full | Analysis & debugging |

## API

Same as main `server.py`:
- **Endpoint**: `POST /evaluate_batch`
- **Input**: Base64 encoded image pairs
- **Output**: Evaluation metrics (accuracy is the only meaningful metric) 