```
# set model_path under rfm/configs/eval_configs.yaml

# Step 1: Start Qwen server in separate terminal 

uv run python3 evals/eval_server.py --num_gpus=2


# Step 2: Run all evals [reward_alignment, success_failure, confusion_matrix]
./evals/run_all_evals.sh

# OR run them separately

# Step 3: Compile the results 
uv run python3 evals/compile_results.py

# the outputs should be stored under eval_logs/final_metrics.json
```