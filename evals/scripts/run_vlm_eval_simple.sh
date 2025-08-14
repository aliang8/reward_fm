#!/bin/bash
# Simple VLM evaluation using existing client infrastructure

echo "======================================"
echo "VLM Baseline Evaluation"
echo "======================================"

# Check for API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not set!"
    echo ""
    echo "Get a key at: https://makersuite.google.com/app/apikey"
    echo "Then: export GEMINI_API_KEY=your-key"
    exit 1
fi

# Install dependency
echo "Installing VLM dependency..."
pip install -q google-generativeai

# Task description
TASK="${1:-robot manipulation task}"
BATCHES="${2:-10}"
BATCH_SIZE="${3:-4}"

echo ""
echo "Configuration:"
echo "  Model: VLM (Gemini)"
echo "  Task: $TASK"
echo "  Batches: $BATCHES"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Start VLM server
echo "Starting VLM server..."
python evals/baselines/rlvlmf_base/vlm_server.py --task "$TASK" &
SERVER_PID=$!

# Wait for server
echo "Waiting for server to start..."
sleep 8

# Run evaluation with EXISTING client (no changes!)
echo "Running evaluation..."
python evals/run_model_eval.py \
    --server_url http://localhost:8000 \
    --num_batches $BATCHES \
    --batch_size $BATCH_SIZE

# Cleanup
echo ""
echo "Stopping server..."
kill $SERVER_PID

echo ""
echo "======================================"
echo "VLM Evaluation Complete!"
echo "======================================" 