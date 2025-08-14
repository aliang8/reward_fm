# VLM Baseline

Direct VLM preference evaluation following RL-VLM-F approach. Queries Gemini for trajectory comparisons without training a reward model.

## Folder Structure
```
evals/baselines/rlvlmf_base/
├── eval_config.yaml         # RL-VLM-F evaluation config (1 frame per trajectory)
├── vlm_server.py             # FastAPI server (drop-in replacement for server.py)
├── vlm_baseline.py           # VLM preference logic (Gemini/GPT-4V) 
├── pyproject.toml           # Minimal dependencies (no GPU required)
├── README.md                # This file
└── vlm_eval_logs/           # Detailed evaluation logs (auto-created)
    ├── vlm_eval_YYYYMMDD_HHMMSS.json    # JSON logs with VLM responses
    └── samples_YYYYMMDD_HHMMSS/         # Sample frames for debugging
```

## Setup

Set API key:
```bash
export GEMINI_API_KEY="your-key"
```

Test setup:
```bash
cd evals/baselines/rlvlmf_base
python test_vlm.py
```

## Usage

### Start VLM Server (standalone)
```bash
cd evals/baselines/rlvlmf_base
uv run python vlm_server.py --task "robot manipulation" --port 8002
```

### Run Evaluation

#### Option 1: Use dedicated RL-VLM-F config (recommended for baseline)
```bash
# RL-VLM-F baseline evaluation (1 frame per trajectory)
python evals/run_model_eval.py --config_path evals/baselines/rlvlmf_base/eval_config.yaml --server_url http://localhost:8002 --num_batches 10 --batch_size 10

# Quick test
python evals/run_model_eval.py --config_path evals/baselines/rlvlmf_base/eval_config.yaml --server_url http://localhost:8002 --num_batches 2 --batch_size 5
```

#### Option 2: Override main config  
```bash
# Override frame count in main config
python evals/run_model_eval.py --config_path rfm/configs/config.yaml --server_url http://localhost:8002 --num_batches 10 --batch_size 10 --set data.max_frames=1

# Multi-frame comparison with main config
python evals/run_model_eval.py --config_path rfm/configs/config.yaml --server_url http://localhost:8002 --num_batches 10 --batch_size 10 --set data.max_frames=3
```

### Temporal Experiments (EXPERIMENTAL)
```bash
cd evals/baselines/rlvlmf_base
uv run python vlm_server.py --task "robot manipulation" --temporal --port 8002
```

## Prompting Strategies

### RL-VLM-F Baseline (DEFAULT) 🎯
- **Purpose**: Exact reproduction of RL-VLM-F paper results
- **Format**: "Consider the following two images: Image 1: ... Image 2: ..."
- **Prompts**: "1. What is shown in Image 1? 2. What is shown in Image 2? 3. The goal is {task}..."
- **Output**: "0" (Image 1 better), "1" (Image 2 better), "-1" (tie/unclear)
- **Use for**: Research comparison, paper reproduction, baseline results

### Temporal-Aware Prompting (EXPERIMENTAL) 🧪
- **Purpose**: Better understanding of multi-frame trajectories  
- **Activation**: Automatically when >4 total frames AND `--temporal` flag
- **Format**: "TRAJECTORY A: [sequence] TRAJECTORY B: [sequence]"
- **Prompts**: Analyzes "INITIAL STATE", "PROGRESS", "FINAL OUTCOME", "EFFICIENCY", "TEMPORAL CONSISTENCY"
- **Output**: Same "0"/"1"/"-1" format for compatibility
- **Use for**: Exploring temporal understanding capabilities

## Frame Handling

**Client-Driven**: VLM uses exactly what the client sends.

| Client Sends | Baseline Behavior | Temporal Behavior |
|--------------|-------------------|-------------------|
| 2 frames (1 per traj) | RL-VLM-F format | RL-VLM-F format (no temporal) |
| 6 frames (3 per traj) | RL-VLM-F format + warning | Temporal prompting if `--temporal` |
| 20 frames (10 per traj) | RL-VLM-F format + warning | Temporal prompting if `--temporal` |

**Automatic Warnings**:
- `⚠️ WARNING` if >2 total frames (potential cost issue)
- `📺 Using temporal prompting` when temporal mode activates
- `🔍 Using RL-VLM-F baseline prompting` when using baseline

## API

Same input/output as main `server.py`:
- Input: base64 encoded image pairs + task description  
- Output: preference metrics (loss, accuracy, reward diff)
- Endpoint: `POST /evaluate_batch`

## Architecture

```
                    ┌─ >4 frames + --temporal ──► Temporal Prompts (EXPERIMENTAL)
Client ──► VLM Server ┤  
                    └─ Default ──────────────────► RL-VLM-F Baseline (DEFAULT)
                            │
                            ▼
                      Gemini API ──► Same Metrics
```

**chosen_frames**: Optimal/successful trajectory (should be preferred)  
**rejected_frames**: Suboptimal/failed trajectory (should be rejected)

The VLM server is a drop-in replacement - existing evaluation clients work without modification. 