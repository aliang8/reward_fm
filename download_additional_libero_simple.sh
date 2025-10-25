#!/bin/bash

echo "🚀 Processing additional LIBERO datasets using standard preprocessing..."
echo "📦 Will process: libero_goal, libero_spatial, libero_object"
echo ""

# Set dataset path environment variable
export RFM_DATASET_PATH=/Users/kaushiksid/projs/rlplus/reward_fm/rfm_dataset

# Set FFmpeg library paths for TorchCodec
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg/lib:$DYLD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg/lib:$LD_LIBRARY_PATH"

# Use the standard preprocessing command that worked before
echo "=== Running preprocessing for additional LIBERO datasets ==="
echo "Start time: $(date)"

uv run python preprocess_datasets.py \
    --config_path=rfm/configs/config.yaml \
    --dataset_type=evaluation \
    --force_reprocess=true

echo "Completed preprocessing at $(date)"
echo ""
echo "🎉 Additional LIBERO datasets processed!"
echo "📊 Ready for evaluation with RL-VLM-F and GVL"

# Verify the datasets were created
echo ""
echo "🔍 Checking created datasets:"
if [ -d "$RFM_DATASET_PATH" ]; then
    echo "Dataset directory contents:"
    ls -la "$RFM_DATASET_PATH/"
else
    echo "⚠️  Dataset directory not found at $RFM_DATASET_PATH"
fi
