#!/usr/bin/env python3
"""
Simple script to upload an already consolidated model to HuggingFace Hub.

This script is for models that are already in the standard format with safetensors files.
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_model_directory(model_dir: Path) -> bool:
    """Validate that the directory contains a valid model."""
    required_files = ["config.json"]

    # Check for required files
    for file in required_files:
        if not (model_dir / file).exists():
            logger.error(f"Required file {file} not found in {model_dir}")
            return False

    # Check for model files (either single or sharded)
    has_model_files = False

    # Check for safetensors files
    safetensors_files = list(model_dir.glob("*.safetensors"))
    if safetensors_files:
        has_model_files = True
        logger.info(f"Found {len(safetensors_files)} safetensors files")

    # Check for pytorch files
    pytorch_files = list(model_dir.glob("pytorch_model*.bin"))
    if pytorch_files:
        has_model_files = True
        logger.info(f"Found {len(pytorch_files)} pytorch files")

    if not has_model_files:
        logger.error("No model files found (*.safetensors or pytorch_model*.bin)")
        return False

    logger.info("✅ Model directory validation passed")
    return True


def create_model_card(model_dir: Path, base_model: str, model_name: str):
    """Create or update the model card."""
    readme_path = model_dir / "README.md"

    # Try to read existing config to get more info
    config_path = model_dir / "config.json"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        model_type = config.get("model_type", "unknown")
        architectures = config.get("architectures", ["unknown"])
    except:
        model_type = "unknown"
        architectures = ["unknown"]

    # Try to read wandb info if available
    wandb_info_path = model_dir / "wandb_info.json"
    wandb_section = ""
    if wandb_info_path.exists():
        try:
            with open(wandb_info_path, "r") as f:
                wandb_info = json.load(f)
            wandb_section = f"""
## Training Run

- **Wandb Run**: [{wandb_info.get('wandb_name', 'N/A')}]({wandb_info.get('wandb_url', '#')})
- **Wandb ID**: `{wandb_info.get('wandb_id', 'N/A')}`
- **Project**: {wandb_info.get('wandb_project', 'N/A')}
"""
        except Exception as e:
            logger.warning(f"Could not read wandb info: {e}")
            wandb_section = ""

    model_card_content = f"""---
license: apache-2.0
base_model: {base_model}
tags:
- reward_model
- rfm
- preference_comparisons
library_name: transformers
---

# {model_name}

## Model Details

- **Base Model**: {base_model}
- **Model Type**: {model_type}
{wandb_section}
## Citation

If you use this model, please cite:
"""

    with open(readme_path, "w") as f:
        f.write(model_card_content)

    logger.info("Created/updated model card (README.md)")


def upload_model_to_hub(
    model_dir: str,
    hub_model_id: str,
    private: bool = False,
    token: str = None,
    commit_message: str = "Upload RFM model",
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
):
    """
    Upload model directory to HuggingFace Hub.

    Args:
        model_dir: Path to the model directory
        hub_model_id: HuggingFace model ID (username/model-name)
        private: Whether to make the model private
        token: HuggingFace token
        commit_message: Commit message for the upload
        base_model: Base model name for the model card
    """

    model_path = Path(model_dir)

    # Validate model directory
    if not model_path.exists():
        raise ValueError(f"Model directory does not exist: {model_path}")

    if not validate_model_directory(model_path):
        raise ValueError("Model directory validation failed")

    # Create/update model card
    create_model_card(model_path, base_model, hub_model_id)

    # Login to HuggingFace
    if token:
        login(token=token)
        logger.info("Logged in to HuggingFace Hub")
    elif os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
        logger.info("Logged in using HF_TOKEN environment variable")
    else:
        logger.warning("No HuggingFace token provided. You may need to login manually.")

    # Upload to Hub
    logger.info(f"Uploading model to: {hub_model_id}")
    logger.info(f"Private: {private}")
    logger.info(f"Model directory: {model_path}")

    api = HfApi()

    # Create the repository if it doesn't exist
    if not api.repo_exists(repo_id=hub_model_id):
        try:
            api.create_repo(repo_id=hub_model_id, repo_type="model", private=private, exist_ok=True)
            logger.info(f"Repository {hub_model_id} created/verified")
        except Exception as e:
            logger.warning(f"Could not create repository (may already exist): {e}")

        # Upload the entire directory
        api.upload_folder(
            folder_path=str(model_path), repo_id=hub_model_id, commit_message=commit_message, repo_type="model"
        )

        logger.info(f"✅ Successfully uploaded model to: https://huggingface.co/{hub_model_id}")

    # Also upload the config.yaml which is in the directory above
    logger.info(f"Uploading config.yaml to: {hub_model_id}")
    logger.info(f"Model directory: {model_path.parent}")
    api.upload_file(
        path_or_fileobj=str(model_path.parent / "config.yaml"),
        path_in_repo="config.yaml",
        repo_id=hub_model_id,
        commit_message=commit_message,
        repo_type="model",
    )

    return f"https://huggingface.co/{hub_model_id}"


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--hub_model_id", type=str, required=True, help="HuggingFace model ID (username/model-name)")
    parser.add_argument("--private", action="store_true", help="Make the model private")
    parser.add_argument("--token", type=str, help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--commit_message", type=str, default="Upload RFM model", help="Commit message for the upload")
    parser.add_argument(
        "--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Base model name for the model card"
    )

    args = parser.parse_args()

    try:
        url = upload_model_to_hub(
            model_dir=args.model_dir,
            hub_model_id=args.hub_model_id,
            private=args.private,
            token=args.token,
            commit_message=args.commit_message,
            base_model=args.base_model,
        )

        logger.info(f"\n🎉 Upload completed successfully!")
        logger.info(f"Model URL: {url}")

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise


if __name__ == "__main__":
    main()
