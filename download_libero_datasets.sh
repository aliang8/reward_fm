#!/bin/bash

echo "🚀 Downloading additional LIBERO datasets..."
echo "📦 Will download: libero_goal, libero_spatial, libero_object"
echo ""

cd deps/libero/LIBERO

# Download libero_goal
echo "=== Downloading libero_goal dataset ==="
echo "Start time: $(date)"
uv run python benchmark_scripts/download_libero_datasets.py --datasets libero_goal --use-huggingface
echo "Completed libero_goal at $(date)"
echo ""

# Download libero_spatial  
echo "=== Downloading libero_spatial dataset ==="
echo "Start time: $(date)"
uv run python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial --use-huggingface
echo "Completed libero_spatial at $(date)"
echo ""

# Download libero_object
echo "=== Downloading libero_object dataset ==="
echo "Start time: $(date)"
uv run python benchmark_scripts/download_libero_datasets.py --datasets libero_object --use-huggingface
echo "Completed libero_object at $(date)"
echo ""

# Go back to main directory
cd ../../..

echo "🎉 All additional LIBERO datasets downloaded!"
echo ""
echo "🔍 Checking downloaded datasets:"
if [ -d "deps/libero/LIBERO/libero/datasets" ]; then
    echo "Dataset directory contents:"
    ls -la deps/libero/LIBERO/libero/datasets/
else
    echo "⚠️  Dataset directory not found"
fi
