#!/bin/bash
# Simple RL-VLM-F evaluation on LIBERO data

echo "======================================"
echo "RL-VLM-F Baseline Evaluation"
echo "======================================"

# Check for API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not set!"
    echo ""
    echo "Get a key at: https://makersuite.google.com/app/apikey"
    echo "Then: export GEMINI_API_KEY=your-key"
    exit 1
fi

# Configuration
DATASET="${1:-libero_goal}"
BATCHES="${2:-3}"
BATCH_SIZE="${3:-4}"

echo ""
echo "Configuration:"
echo "  Baseline: RL-VLM-F (Gemini 2.5 Flash)"
echo "  Dataset: $DATASET"
echo "  Batches: $BATCHES"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Set dataset path for the eval system
export RFM_PROCESSED_DATASETS_PATH="$(pwd)/datasets"

# Start RL-VLM-F server
echo "Starting RL-VLM-F server on port 8002..."
cd evals/baselines/rlvlmf_base
uv run python vlm_server.py --port 8002 --debug &
SERVER_PID=$!
cd ../../..

# Wait for server
echo "Waiting for server to start..."
sleep 5

# Run evaluation  
echo "Running evaluation on LIBERO $DATASET..."
uv run python evals/run_model_eval.py \
    --config rfm/configs/eval_config.yaml \
    --set server_url="http://localhost:8002" \
    --set batch_size=$BATCH_SIZE \
    --set num_batches=$BATCHES \
    --set data.eval_datasets='["datasets/libero_rfm"]' \
    --set data.eval_subsets='[["'$DATASET'"]]' \
    --set custom_eval.eval_types='["policy_ranking"]' \
    --set custom_eval.policy_ranking='[["datasets/libero_rfm","'$DATASET'"]]'

# Cleanup
echo ""
echo "Stopping server..."
kill $SERVER_PID

echo ""
echo "======================================"
echo "RL-VLM-F Evaluation Complete!"
echo "Check logs in: evals/baselines/rlvlmf_base/vlm_eval_logs/"
echo "======================================" 