#!/bin/bash

# Script to evaluate local datasets by overriding the config properly
# This avoids the eval_datasets/eval_subsets mismatch issue

# Create logs directory if it doesn't exist
mkdir -p evals/logs

# Set FFmpeg library paths for TorchCodec
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg/lib:$DYLD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg/lib:$LD_LIBRARY_PATH"

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "🚀 Starting local dataset evaluation at $(date)"
echo "📊 Datasets: libero_goal, libero_spatial, libero_object"
echo "🔬 Methods: RL-VLM-F (port 8002) and GVL (port 8003)"
echo ""

# List of datasets to evaluate
DATASETS=("libero_goal" "libero_spatial" "libero_object")

# ========================================
# RL-VLM-F EVALUATIONS (PORT 8002)
# ========================================

echo "🔍 Starting RL-VLM-F evaluations..."
echo "⚠️  Make sure RL-VLM-F server is running on port 8002!"
echo ""

for dataset in "${DATASETS[@]}"; do
    echo "=== RL-VLM-F: Evaluating $dataset ===" | tee evals/logs/rlvlmf_${dataset}_${TIMESTAMP}.log
    echo "Start time: $(date)" | tee -a evals/logs/rlvlmf_${dataset}_${TIMESTAMP}.log
    
    # Use the config but override both datasets and subsets to match
    uv run python evals/run_model_eval.py \
      --config_path=rfm/configs/config.yaml \
      --server_url=http://localhost:8002 \
      --batch_size=12 \
      --set data.eval_datasets=[\"datasets/libero_rfm\"] \
      --set data.eval_subsets=[\"$dataset\"] \
      --iterate_all_preferences 2>&1 | tee -a evals/logs/rlvlmf_${dataset}_${TIMESTAMP}.log
    
    echo "End time: $(date)" | tee -a evals/logs/rlvlmf_${dataset}_${TIMESTAMP}.log
    echo "=== Completed RL-VLM-F: $dataset ===" | tee -a evals/logs/rlvlmf_${dataset}_${TIMESTAMP}.log
    echo "" | tee -a evals/logs/rlvlmf_${dataset}_${TIMESTAMP}.log
done

echo "✅ Completed all RL-VLM-F evaluations at $(date)"
echo ""

# ========================================
# GVL EVALUATIONS (PORT 8003)
# ========================================

echo "🔍 Starting GVL evaluations..."
echo "⚠️  Make sure GVL server is running on port 8003!"
echo ""

for dataset in "${DATASETS[@]}"; do
    echo "=== GVL: Evaluating $dataset ===" | tee evals/logs/gvl_${dataset}_${TIMESTAMP}.log
    echo "Start time: $(date)" | tee -a evals/logs/gvl_${dataset}_${TIMESTAMP}.log
    
    # Use the config but override both datasets and subsets to match
    uv run python evals/run_model_eval.py \
      --config_path=rfm/configs/config.yaml \
      --server_url=http://localhost:8003 \
      --batch_size=8 \
      --set data.eval_datasets=[\"datasets/libero_rfm\"] \
      --set data.eval_subsets=[\"$dataset\"] \
      --iterate_all_preferences 2>&1 | tee -a evals/logs/gvl_${dataset}_${TIMESTAMP}.log
    
    echo "End time: $(date)" | tee -a evals/logs/gvl_${dataset}_${TIMESTAMP}.log
    echo "=== Completed GVL: $dataset ===" | tee -a evals/logs/gvl_${dataset}_${TIMESTAMP}.log
    echo "" | tee -a evals/logs/gvl_${dataset}_${TIMESTAMP}.log
done

echo "✅ Completed all GVL evaluations at $(date)"
echo ""

# ========================================
# SUMMARY
# ========================================

echo "🎉 All evaluations completed! Check logs in evals/logs/"
echo ""
echo "📊 RL-VLM-F Results:"
for dataset in "${DATASETS[@]}"; do
    echo "  - $dataset: evals/logs/rlvlmf_${dataset}_${TIMESTAMP}.log"
done

echo ""
echo "📊 GVL Results:"
for dataset in "${DATASETS[@]}"; do
    echo "  - $dataset: evals/logs/gvl_${dataset}_${TIMESTAMP}.log"
done

echo ""
echo "🔍 To check final accuracies, run: python extract_results.py"
echo ""
echo "🏁 Local dataset evaluation complete at $(date)"
