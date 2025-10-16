#!/bin/bash
# Run VLM server from baseline folder

# Check API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not set!"
    echo "Get key: https://makersuite.google.com/app/apikey"
    exit 1
fi

# Install dependency if needed
pip install -q google-generativeai

# Run server
echo "🚀 Starting VLM server..."
python vlm_server.py --task "${1:-robot manipulation task}" --port "${2:-8000}" 