#!/bin/bash

# Quick test script to evaluate a single dataset
# This will help us debug the config issues

# Set FFmpeg library paths for TorchCodec
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg/lib:$DYLD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg/lib:$LD_LIBRARY_PATH"

echo "🧪 Testing single dataset evaluation: libero_goal"
echo "📊 Using local config with matching dataset/subset lengths"
echo ""

# Test with RL-VLM-F first (simpler, no progress predictions)
echo "=== Testing RL-VLM-F on libero_goal ==="
echo "⚠️  Make sure RL-VLM-F server is running on port 8002!"

uv run python evals/run_model_eval.py \
  --config_path=local_eval_config.yaml \
  --server_url=http://localhost:8002 \
  --batch_size=4 \
  --num_batches=3 \
  --set data.eval_subsets=[\"libero_goal\"] \
  --iterate_all_preferences

echo ""
echo "✅ Test completed!"
