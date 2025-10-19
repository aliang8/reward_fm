#!/bin/bash
# Script to generate HuggingFace dataset from Humanoid Everyday data

set -euo pipefail

# Configuration
CONFIG_FILE="configs/data_gen_configs/humanoid_everyday.yaml"
DATASET_PATH="${1:-./datasets/humanoid_everyday}"

echo "🤖 Generating Humanoid Everyday RFM dataset..."
echo "📁 Dataset path: $DATASET_PATH"

# Check if dataset path exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Dataset path does not exist: $DATASET_PATH"
    echo "Please download the dataset first using:"
    echo "  bash dataset_upload/data_scripts/humanoid_everyday/download_humanoid_everyday.sh"
    exit 1
fi

# Check if humanoid_everyday package is installed
python -c "import humanoid_everyday" 2>/dev/null || {
    echo "❌ humanoid_everyday package not found"
    echo "Please install it first:"
    echo "  git clone https://github.com/ausbxuse/Humanoid-Everyday"
    echo "  cd Humanoid-Everyday"
    echo "  pip install -e ."
    exit 1
}

# Check if zip files exist in the dataset path
ZIP_COUNT=$(find "$DATASET_PATH" -name "*.zip" | wc -l)
if [ "$ZIP_COUNT" -eq 0 ]; then
    echo "❌ No zip files found in $DATASET_PATH"
    echo "Please download the dataset first using:"
    echo "  bash dataset_upload/data_scripts/humanoid_everyday/download_humanoid_everyday.sh"
    exit 1
fi

echo "✅ Found $ZIP_COUNT zip files in dataset path"

# Update the config file with the correct dataset path
TEMP_CONFIG=$(mktemp)
sed "s|dataset_path: .*|dataset_path: \"$DATASET_PATH\"|" "$CONFIG_FILE" > "$TEMP_CONFIG"

echo "🚀 Starting dataset conversion..."
python dataset_upload/generate_hf_dataset.py --config "$TEMP_CONFIG"

# Clean up
rm "$TEMP_CONFIG"

echo "✅ Humanoid Everyday RFM dataset generation complete!"
