# Makefile for Reward Foundation Model (RFM)
# Simple automation for ML research workflows

.PHONY: help install train eval clean status dataset-libero dataset-agibotworld dataset-all

# Default target
help:
	@echo "RFM - Reward Foundation Model"
	@echo ""
	@echo "Setup:"
	@echo "  make install         Install dependencies with uv"
	@echo "  make dataset-libero  Download LIBERO dataset only"
	@echo "  make dataset-all     Download all datasets"
	@echo ""
	@echo "Training:"
	@echo "  make train           Run training"
	@echo "  make eval            Run evaluation"
	@echo ""
	@echo "Utils:"
	@echo "  make clean           Clean cache files"
	@echo "  make status          Show environment status"

# Install with uv - auto-install uv if needed
install:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "Installing RFM dependencies..."
	uv sync

# Dataset downloads
dataset-libero:
	@echo "Downloading LIBERO dataset..."
	@mkdir -p ${RFM_DATASET_PATH:-./rfm_dataset}
	uv run huggingface-cli download abraranwar/libero_rfm \
		--repo-type dataset \
		--local-dir ${RFM_DATASET_PATH:-./rfm_dataset}/libero_rfm \
		--local-dir-use-symlinks True

dataset-agibotworld:
	@echo "Downloading AgiBotWorld dataset..."
	@mkdir -p ${RFM_DATASET_PATH:-./rfm_dataset}
	uv run huggingface-cli download abraranwar/agibotworld_rfm \
		--repo-type dataset \
		--local-dir ${RFM_DATASET_PATH:-./rfm_dataset}/agibotworld_rfm \
		--local-dir-use-symlinks True

dataset-all:
	@echo "Downloading all datasets..."
	./setup.sh

# Training and evaluation
train:
	uv run accelerate launch --config_file rfm/configs/fsdp.yaml train.py --config_path=rfm/configs/config.yaml

eval:
	uv run accelerate launch --config_file rfm/configs/fsdp.yaml train.py --mode=evaluate

# Utilities
clean:
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

status:
	@echo "Python: $(shell python --version 2>/dev/null || echo 'Not found')"
	@echo "UV: $(shell uv --version 2>/dev/null || echo 'Not installed')"
	@echo "Virtual env: $(shell echo $$VIRTUAL_ENV || echo 'Not activated')" 