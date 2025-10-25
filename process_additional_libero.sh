#!/bin/bash

echo "🚀 Processing additional LIBERO datasets to HuggingFace format..."
echo "📦 Will process: libero_goal, libero_spatial, libero_object"
echo ""

# Set dataset path environment variable
export RFM_DATASET_PATH=/Users/kaushiksid/projs/rlplus/reward_fm/rfm_dataset

# Set FFmpeg library paths for TorchCodec
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg/lib:$DYLD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg/lib:$LD_LIBRARY_PATH"

# Process libero_goal
echo "=== Processing libero_goal dataset ==="
echo "Start time: $(date)"

uv run python rfm/data/generate_hf_dataset.py \
    --config_path=rfm/configs/data_gen_configs/libero.yaml \
    --dataset.dataset_path=deps/libero/LIBERO/libero/datasets/libero_goal \
    --dataset.dataset_name=libero_goal \
    --hub.push_to_hub=false

echo "Completed libero_goal at $(date)"
echo ""

# Process libero_spatial  
echo "=== Processing libero_spatial dataset ==="
echo "Start time: $(date)"

uv run python rfm/data/generate_hf_dataset.py \
    --config_path=rfm/configs/data_gen_configs/libero.yaml \
    --dataset.dataset_path=deps/libero/LIBERO/libero/datasets/libero_spatial \
    --dataset.dataset_name=libero_spatial \
    --hub.push_to_hub=false

echo "Completed libero_spatial at $(date)"
echo ""

# Process libero_object
echo "=== Processing libero_object dataset ==="
echo "Start time: $(date)"

uv run python rfm/data/generate_hf_dataset.py \
    --config_path=rfm/configs/data_gen_configs/libero.yaml \
    --dataset.dataset_path=deps/libero/LIBERO/libero/datasets/libero_object \
    --dataset.dataset_name=libero_object \
    --hub.push_to_hub=false

echo "Completed libero_object at $(date)"
echo ""

echo "🎉 All additional LIBERO datasets processed!"
echo "📊 Ready for evaluation with RL-VLM-F and GVL"

# Verify the datasets were created
echo ""
echo "🔍 Checking processed datasets:"
echo "Generated datasets should be in: datasets/libero_rfm/"
if [ -d "datasets/libero_rfm" ]; then
    echo "Dataset directory contents:"
    ls -la datasets/libero_rfm/
else
    echo "⚠️  Processed dataset directory not found at datasets/libero_rfm"
fi
