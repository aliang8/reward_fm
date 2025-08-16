#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p evals/logs

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# LIBERO regular datasets
echo "Running LIBERO regular dataset evaluations at $(date)" | tee evals/logs/libero_regular_${TIMESTAMP}.log

for subset in "libero_10" "libero_goal" "libero_object" "libero_spatial"; do
    echo "=== Evaluating subset: $subset ===" | tee -a evals/logs/libero_regular_${TIMESTAMP}.log
    echo "Start time: $(date)" | tee -a evals/logs/libero_regular_${TIMESTAMP}.log
    
    uv run python evals/run_model_eval.py \
      --config_path=rfm/configs/config.yaml \
      --server_url=http://localhost:8002 \
      --batch_size=12 \
      --set evaluation.eval_dataset_path="abraranwar/libero_rfm" \
      --set evaluation.eval_dataset_subsets=["$subset"] \
      --iterate_all_preferences 2>&1 | tee -a evals/logs/libero_regular_${TIMESTAMP}.log
    
    echo "End time: $(date)" | tee -a evals/logs/libero_regular_${TIMESTAMP}.log
    echo "=== Completed subset: $subset ===" | tee -a evals/logs/libero_regular_${TIMESTAMP}.log
    echo "" | tee -a evals/logs/libero_regular_${TIMESTAMP}.log
done

echo "Completed all LIBERO regular dataset evaluations at $(date)" | tee -a evals/logs/libero_regular_${TIMESTAMP}.log

# LIBERO failure datasets
#echo "Running LIBERO failure dataset evaluations at $(date)" | tee evals/logs/libero_failure_${TIMESTAMP}.log
#
#for subset in "libero_10_failure"; do
#    echo "=== Evaluating subset: $subset ===" | tee -a evals/logs/libero_failure_${TIMESTAMP}.log
#    echo "Start time: $(date)" | tee -a evals/logs/libero_failure_${TIMESTAMP}.log
#    
#    uv run python evals/run_model_eval.py \
#      --config_path=rfm/configs/config.yaml \
#      --server_url=http://localhost:8000 \
#      --batch_size=12 \
#      --set evaluation.eval_dataset_path="ykorkmaz/libero_failure_rfm" \
#      --set evaluation.eval_dataset_subsets=["$subset"] \
#      --iterate_all_preferences 2>&1 | tee -a evals/logs/libero_failure_${TIMESTAMP}.log
#    
#    echo "End time: $(date)" | tee -a evals/logs/libero_failure_${TIMESTAMP}.log
#    echo "=== Completed subset: $subset ===" | tee -a evals/logs/libero_failure_${TIMESTAMP}.log
#    echo "" | tee -a evals/logs/libero_failure_${TIMESTAMP}.log
#done
#
#echo "Completed all LIBERO failure dataset evaluations at $(date)" | tee -a evals/logs/libero_failure_${TIMESTAMP}.log

echo "All evaluations completed! Check logs in evals/logs/"
echo "Regular dataset log: evals/logs/libero_regular_${TIMESTAMP}.log"
#echo "Failure dataset log: evals/logs/libero_failure_${TIMESTAMP}.log"