# Additional LIBERO Dataset Evaluation

Complete workflow to download, process, and evaluate additional LIBERO datasets (goal, spatial, object) with both RL-VLM-F and GVL baselines.

## Quick Start

### 1. Download and Process Datasets
```bash
# Process the additional LIBERO datasets using standard preprocessing
./download_additional_libero_simple.sh
```

### 2. Run Evaluations

**Start both servers in separate terminals:**

```bash
# Terminal 1: RL-VLM-F Server (port 8002)
cd evals/baselines/rlvlmf_base
uv run python vlm_server.py --port 8002 --debug

# Terminal 2: GVL Server (port 8003)  
cd evals/baselines/gvl_base
uv run python gvl_server.py --port 8003 --debug
```

**Run evaluations:**
```bash
# Main terminal: Run all evaluations
./eval_additional_libero_simple.sh
```

### 3. Extract Results
```bash
# Get summary of all results
python extract_results.py
```

## What Gets Evaluated

### Datasets:
- **libero_goal**: Goal-conditioned manipulation tasks
- **libero_spatial**: Spatial reasoning tasks  
- **libero_object**: Object manipulation tasks

### Methods:
- **RL-VLM-F**: Direct visual preference (A/B/tie choice)
- **GVL**: Task completion percentage comparison

### Metrics:
- **Preference Accuracy**: How often method chooses better trajectory
- **Reward Difference**: Difference between chosen/rejected scores
- **Progress Accuracy** (GVL only): How well completion percentages align with ground truth

## File Structure

```
├── download_additional_libero_simple.sh    # Download and process datasets  
├── eval_additional_libero_simple.sh        # Run evaluations on both methods
├── extract_results.py                      # Summarize all results
└── evals/logs/                             # Evaluation logs
    ├── rlvlmf_goal_TIMESTAMP.log    # RL-VLM-F results for goal
    ├── rlvlmf_spatial_TIMESTAMP.log # RL-VLM-F results for spatial
    ├── rlvlmf_object_TIMESTAMP.log  # RL-VLM-F results for object
    ├── gvl_goal_TIMESTAMP.log       # GVL results for goal
    ├── gvl_spatial_TIMESTAMP.log    # GVL results for spatial
    └── gvl_object_TIMESTAMP.log     # GVL results for object
```

## Detailed Workflow

### Step 1: Dataset Processing
The `download_additional_libero_simple.sh` script will:
1. Set the dataset path environment variable
2. Add `libero_goal`, `libero_spatial`, `libero_object` to `config.yaml` eval_subsets
3. Use standard preprocessing command: `preprocess_datasets.py --dataset_type=evaluation`
4. Process all additional datasets in one go (same as existing workflow)
5. Generate datasets for evaluation

### Step 2: Evaluation Execution
The `eval_additional_libero_simple.sh` script will:
1. Run RL-VLM-F evaluation on all 3 datasets (port 8002)
2. Run GVL evaluation on all 3 datasets (port 8003)
3. Generate separate log files for each method × dataset combination
4. Use appropriate batch sizes (12 for RL-VLM-F, 8 for GVL)

### Step 3: Results Analysis
The `extract_results.py` script will:
1. Parse all log files for final accuracy metrics
2. Extract additional metrics (reward difference, progress accuracy)
3. Generate comparison tables between methods
4. Show file locations for detailed analysis

## Expected Output

### Results Summary Example:
```
📈 EVALUATION RESULTS SUMMARY
================================================================================

🔍 RL-VLM-F Results:
----------------------------------------
  goal           : 45.2%
  spatial        : 38.7%
  object         : 52.1%

🎯 GVL Results:
----------------------------------------
  goal           : 47.8%
                   Progress acc: 0.762
  spatial        : 41.3%
                   Progress acc: 0.695
  object         : 49.5%
                   Progress acc: 0.831

🆚 METHOD COMPARISON:
------------------------------------------------------------
Dataset         RL-VLM-F     GVL          Difference  
------------------------------------------------------------
goal            45.2%        47.8%        +2.6%
object          52.1%        49.5%        -2.6%
spatial         38.7%        41.3%        +2.6%
```

## Configuration

### Batch Sizes:
- **RL-VLM-F**: 12 samples per batch (faster processing)
- **GVL**: 8 samples per batch (more intensive analysis)

### Server Ports:
- **RL-VLM-F**: 8002
- **GVL**: 8003

### Dataset Source:
- All datasets from `abraranwar/libero_rfm` HuggingFace repository
- Uses existing config: `rfm/configs/config.yaml`

## Troubleshooting

### Missing API Key:
```bash
export GEMINI_API_KEY="your-api-key"
```

### Server Not Running:
Check that both servers are running on correct ports:
```bash
curl http://localhost:8002/health  # RL-VLM-F
curl http://localhost:8003/health  # GVL
```

### Dataset Not Found:
Ensure datasets were processed correctly:
```bash
ls rfm_dataset/  # Should contain libero_goal, libero_spatial, libero_object
```

### Memory Issues:
Reduce batch sizes in the evaluation script if needed.

## Integration with Existing Results

This workflow is designed to complement your existing evaluations:
- Uses same evaluation infrastructure as current setup
- Compatible with existing `libero256_10` and `libero_10_failure` results
- Results can be combined for comprehensive analysis

The `extract_results.py` script will automatically find and include any existing log files in the results summary.

## Next Steps

After running evaluations:
1. Use `extract_results.py` to get quick summary
2. Check individual log files for detailed analysis
3. Compare with existing `libero256_10` and `libero_10_failure` results
4. Use debug logs to investigate any surprising results

This gives you a complete evaluation across **5 LIBERO datasets** with **2 different methods** for comprehensive baseline comparison!
