#!/usr/bin/env python3
from __future__ import annotations

import torch
import io
import json
import re
from dataclasses import fields
from typing import Any, Optional, Tuple

import aiohttp
import numpy as np
import requests
import yaml

from huggingface_hub import hf_hub_download

from rfm.configs.experiment_configs import ExperimentConfig
from rfm.utils.setup_utils import setup_model_and_processor
from rfm.data.dataset_types import PreferenceSample, SimilaritySample, ProgressSample


def extract_answer_from_text(text):
    m = re.search(r"<ans>(.*?)</ans>", text, re.DOTALL)
    ans = m.group(1).strip() if m else ""
    return ans


def load_model_from_hf(
    model_path: str,
    device: torch.device,
) -> Tuple[Optional[ExperimentConfig], Optional[Any], Optional[Any], Optional[Any]]:
    """
    Load reward model config and model from HuggingFace.

    This function matches the config loading logic used in train.py:
    - Downloads config.yaml from HuggingFace
    - Uses custom YAML loader for ReWIND models
    - Filters config keys to match ExperimentConfig
    - Clears training and logging sections
    - Loads model using setup_model_and_processor

    Args:
        model_path: HuggingFace model repository ID or local path
        device: Device to load model on

    Returns:
        Tuple of (exp_config, tokenizer, processor, reward_model)
    """
    if hf_hub_download is None:
        raise ImportError("huggingface_hub not available. Install with: pip install huggingface_hub")

    # Download model config from Hugging Face
    model_config_path = hf_hub_download(repo_id=model_path, filename="config.yaml")

    # Load YAML with custom loader for ReWIND models
    with open(model_config_path) as f:
        yaml_text = f.read()

    class _HFConfigSafeLoader(yaml.SafeLoader):
        pass

    _HFConfigSafeLoader.add_constructor(
        "tag:yaml.org,2002:python/object:rfm.models.rewind_transformer.ReWINDTransformerConfig",
        lambda loader, node: loader.construct_mapping(node),
    )

    model_config_dict = yaml.load(yaml_text, Loader=_HFConfigSafeLoader)

    # Filter out keys that ExperimentConfig doesn't accept (for backwards compatibility)
    valid_keys = {f.name for f in fields(ExperimentConfig)}
    filtered_config = {k: v for k, v in model_config_dict.items() if k in valid_keys}

    # Remove training and logging sections (same as train.py)
    filtered_config["training"] = {}
    filtered_config["logging"] = {}

    exp_config = ExperimentConfig(**filtered_config)
    tokenizer, processor, reward_model = setup_model_and_processor(exp_config.model, model_path)
    reward_model = reward_model.to(device)
    reward_model.eval()

    return exp_config, tokenizer, processor, reward_model


def build_payload(
    samples: list[PreferenceSample | SimilaritySample | ProgressSample],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a payload with numpy array handling.

    Args:
        samples: List of samples to convert

    Returns:
        Tuple of (files, sample_data) where:
        - files: Dict of numpy arrays converted to .npy format
        - sample_data: List of sample dictionaries with numpy arrays replaced by file references
    """
    files = {}
    sample_data = []

    for sample_idx, sample in enumerate(samples):
        # Copy the original sample and handle numpy arrays
        processed_sample = sample.model_dump().copy()

        # Handle trajectory objects with numpy arrays
        for key in [
            "chosen_trajectory",
            "rejected_trajectory",
            "reference_trajectory",
            "traj_sim_trajectory",
            "traj_diff_trajectory",
            "trajectory",
        ]:
            if key in processed_sample and isinstance(processed_sample[key], dict):
                trajectory = processed_sample[key]

                # Convert numpy arrays to .npy files
                numpy_fields = ["frames", "lang_vector", "video_embeddings", "text_embedding"]
                for field_name in numpy_fields:
                    # if it is a tensor, first convert it to a numpy array
                    if field_name in trajectory and isinstance(trajectory[field_name], torch.Tensor):
                        trajectory[field_name] = trajectory[field_name].numpy()

                    if field_name in trajectory and isinstance(trajectory[field_name], np.ndarray):
                        # Convert numpy array to .npy file
                        buf = io.BytesIO()
                        np.save(buf, trajectory[field_name])
                        buf.seek(0)
                        file_key = f"sample_{sample_idx}_{key}_{field_name}"
                        files[file_key] = (
                            f"sample_{sample_idx}_{key}_{field_name}.npy",
                            buf,
                            "application/octet-stream",
                        )
                        trajectory[field_name] = {"__numpy_file__": file_key}

        sample_data.append(processed_sample)

    return files, sample_data


def post_batch(url: str, payload: dict[str, Any], timeout_s: float = 120.0) -> dict[str, Any]:
    """POST a batch payload to the evaluation server and return parsed JSON."""
    resp = requests.post(url.rstrip("/") + "/evaluate_batch", json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def post_batch_npy(
    url: str, files: dict[str, Any], sample_data: list[dict[str, Any]], timeout_s: float = 120.0
) -> dict[str, Any]:
    """POST batch using .npy format for numpy arrays."""
    # Convert sample_data to form data
    data = {f"sample_{i}": json.dumps(sample) for i, sample in enumerate(sample_data)}

    # Send as multipart form data
    resp = requests.post(url.rstrip("/") + "/evaluate_batch_npy", files=files, data=data, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


async def post_batch_npy_async(
    session: aiohttp.ClientSession,
    url: str,
    files: dict[str, Any],
    sample_data: list[dict[str, Any]],
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Async version of post_batch_npy using aiohttp."""
    # Create FormData for aiohttp
    form_data = aiohttp.FormData()

    # Add files
    for key, (filename, file_obj, content_type) in files.items():
        form_data.add_field(key, file_obj, filename=filename, content_type=content_type)

    # Add sample data
    for i, sample in enumerate(sample_data):
        form_data.add_field(f"sample_{i}", json.dumps(sample))

    headers = {"Connection": "close"}
    # Send as multipart form data using aiohttp
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with session.post(
        url.rstrip("/") + "/evaluate_batch_npy", data=form_data, timeout=timeout, headers=headers
    ) as resp:
        resp.raise_for_status()
        return await resp.json()
