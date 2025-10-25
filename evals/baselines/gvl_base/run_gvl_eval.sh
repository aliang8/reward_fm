#!/bin/bash

# Script to run GVL evaluation on both datasets
# This can be used standalone or integrated into run_all_evals.sh

# Create logs directory if it doesn't exist
mkdir -p evals/logs

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "🌟 Starting GVL evaluation at $(date)"
echo "📊 Will evaluate both libero256_10 and libero_10_failure datasets"
echo ""

# GVL evaluation on libero256_10
echo "Running GVL evaluation on libero256_10 dataset at $(date)" | tee evals/logs/gvl_libero256_${TIMESTAMP}.log

echo "=== Evaluating GVL on libero256_10 ===" | tee -a evals/logs/gvl_libero256_${TIMESTAMP}.log
echo "Start time: $(date)" | tee -a evals/logs/gvl_libero256_${TIMESTAMP}.log

uv run python evals/run_model_eval.py \
  --config_path=rfm/configs/config.yaml \
  --server_url=http://localhost:8003 \
  --batch_size=8 \
  --set evaluation.eval_dataset_path="abraranwar/libero_rfm" \
  --set evaluation.eval_dataset_subsets=["libero256_10"] \
  --iterate_all_preferences 2>&1 | tee -a evals/logs/gvl_libero256_${TIMESTAMP}.log

echo "End time: $(date)" | tee -a evals/logs/gvl_libero256_${TIMESTAMP}.log
echo "=== Completed GVL on libero256_10 ===" | tee -a evals/logs/gvl_libero256_${TIMESTAMP}.log
echo "" | tee -a evals/logs/gvl_libero256_${TIMESTAMP}.log

# GVL evaluation on libero_10_failure
echo "Running GVL evaluation on libero_10_failure dataset at $(date)" | tee evals/logs/gvl_failure_${TIMESTAMP}.log

echo "=== Evaluating GVL on libero_10_failure ===" | tee -a evals/logs/gvl_failure_${TIMESTAMP}.log
echo "Start time: $(date)" | tee -a evals/logs/gvl_failure_${TIMESTAMP}.log

uv run python evals/run_model_eval.py \
  --config_path=rfm/configs/config.yaml \
  --server_url=http://localhost:8003 \
  --batch_size=8 \
  --set evaluation.eval_dataset_path="ykorkmaz/libero_failure_rfm" \
  --set evaluation.eval_dataset_subsets=["libero_10_failure"] \
  --iterate_all_preferences 2>&1 | tee -a evals/logs/gvl_failure_${TIMESTAMP}.log

echo "End time: $(date)" | tee -a evals/logs/gvl_failure_${TIMESTAMP}.log
echo "=== Completed GVL on libero_10_failure ===" | tee -a evals/logs/gvl_failure_${TIMESTAMP}.log
echo "" | tee -a evals/logs/gvl_failure_${TIMESTAMP}.log

echo "🎉 All GVL evaluations completed! Check logs in evals/logs/"
echo "📊 Results:"
echo "  - libero256_10 log: evals/logs/gvl_libero256_${TIMESTAMP}.log"
echo "  - libero_10_failure log: evals/logs/gvl_failure_${TIMESTAMP}.log"
echo ""
echo "📈 GVL evaluation logs also available in: evals/baselines/gvl_base/gvl_eval_logs/"