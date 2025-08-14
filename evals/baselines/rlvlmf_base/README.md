# VLM Baseline

Direct VLM preference evaluation following RL-VLM-F approach. Queries Gemini for trajectory comparisons without training a reward model.

## Structure
```
evals/baselines/rlvlmf_base/
├── vlm_baseline.py      # Core VLM logic (RL-VLM-F baseline + temporal extension)
├── vlm_server.py        # FastAPI server (drop-in replacement for main server.py)
├── test_vlm.py          # Setup verification
├── pyproject.toml       # Minimal deps (no GPU required)
└── README.md
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

### Start VLM Server
```bash
cd evals/baselines/rlvlmf_base

# RL-VLM-F baseline (DEFAULT) - exact paper reproduction
uv run python vlm_server.py --task "pick up red block"

# Temporal-aware prompting (EXPERIMENTAL) - for multi-frame trajectories  
uv run python vlm_server.py --task "pick up red block" --temporal
```

### Run Evaluation (from repo root)
```bash
python evals/run_model_eval.py --server_url http://localhost:8000 --num_batches 10
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