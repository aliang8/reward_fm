# GVL Baseline

Task completion percentage-based preference evaluation using Gemini Vision. Compares trajectories by analyzing task completion percentages throughout the sequence.

## Setup

```bash
export GEMINI_API_KEY="your-key"
cd evals/baselines/gvl_base
python test/test_gvl.py  # Verify setup (optional)
```

## Usage

### Start Server

```bash
# Production (lightweight logging)
uv run python gvl_server.py --task "robot manipulation" --port 8003

# Debug (saves frames + full responses + completion percentages)
uv run python gvl_server.py --task "robot manipulation" --port 8003 --debug
```

### Run Evaluation

```bash
# Standard evaluation
python evals/run_model_eval.py \
  --config_path rfm/configs/config.yaml \
  --server_url http://localhost:8003 \
  --num_batches 10 --batch_size 10

# Quick test
python evals/run_model_eval.py \
  --config_path rfm/configs/config.yaml \
  --server_url http://localhost:8003 \
  --num_batches 2 --batch_size 5
```

## How It Works

### Task Completion Analysis
- **Input**: Any number of frames per trajectory (1, 8, 32, etc.)
- **Processing**: All frames analyzed for task completion percentage (0-100%)
- **Comparison**: Final completion percentages compared between trajectories
- **Output**: Preference based on which trajectory achieved higher completion

### GVL Prompting
Uses the original GVL format for task completion estimation:
```
You are an expert roboticist tasked to predict task completion percentages 
for frames of a robot for the task of {task_description}. 
The task completion percentages are between 0 and 100, where 100 corresponds to full task completion.

Initial robot scene:
[Initial frame shows 0% completion]

Now, for the task of *{task_description}*, output the task completion percentage 
for the following frames that are presented in random order.

Format your response in JSON:
[
  {"frame_number": i, "frame_description": "...", "task_completion_percentage": 0-100}
]
```

### Architecture
```
Client → Server → Extract All Frames → Query Gemini for Completion % → Compare Final % → Return Preference
         ↑
    (Any frame count)
```

## Key Features

### Intelligent Frame Sampling
- **≤ max_frames**: Uses all available frames
- **> max_frames**: 
  - Always includes first and last frames
  - Uniformly samples middle frames
  - Ensures comprehensive trajectory coverage

### Robust Comparison
- **Completion Threshold**: 5% difference threshold to handle noise
- **Error Handling**: Graceful handling of failed completion estimates
- **Randomization**: Avoids position bias in comparisons

### Debug Capabilities
- **Frame Saving**: Saves all trajectory frames for analysis
- **Completion Tracking**: Logs all completion percentages
- **Failure Analysis**: Detailed debug info for incorrect predictions

## Files

```
gvl_base/
├── gvl_server.py       # FastAPI server
├── gvl_baseline.py     # GVL query logic with completion analysis
├── README.md           # This file
└── gvl_eval_logs/      # Auto-generated logs
```

## Debug vs Production

| Mode | Frames Saved | Completion Data | Use Case |
|------|--------------|-----------------|----------|
| Production | ❌ | Summary only | Regular evaluation |
| Debug | ✅ | Full trajectories + percentages | Analysis & debugging |

## API

Same as main `server.py`:
- **Endpoint**: `POST /evaluate_batch`
- **Input**: Base64 encoded image sequences
- **Output**: 
  - Preference evaluation based on task completion
  - Progress predictions (completion percentages 0-1 for each frame)
  - Reward alignment metrics

## Comparison with RL-VLM-F

| Aspect | RL-VLM-F | GVL |
|--------|----------|-----|
| **Approach** | Direct visual preference | Task completion percentage |
| **Analysis** | Last frame comparison | Full trajectory analysis |
| **Output** | A/B/tie choice | Completion percentage comparison |
| **Granularity** | Binary preference | Quantitative progress |
| **Robustness** | Position randomization | Completion threshold |

## Configuration

- **max_frames**: Maximum frames to analyze (default: 15)
- **offset**: Sampling offset for frame selection (default: 0.5)
- **completion_threshold**: Difference threshold for tie detection (default: 5%)
- **port**: Server port (default: 8003)

## Expected Performance

GVL provides more granular analysis than binary preference methods by:
- Quantifying task progress at each frame
- Comparing overall trajectory completion
- Handling partial task completion scenarios
- Providing detailed completion reasoning
