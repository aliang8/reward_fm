#!/usr/bin/env python3
from __future__ import annotations

import torch
import io
import json
import os
import re
from pathlib import Path
from dataclasses import fields
from typing import Any, Optional, Tuple

import aiohttp
import numpy as np
import requests
import yaml

from huggingface_hub import hf_hub_download

from rfm.configs.experiment_configs import ExperimentConfig
from rfm.utils.setup_utils import setup_model_and_processor
from rfm.utils.save import resolve_checkpoint_path
from rfm.data.dataset_types import PreferenceSample, SimilaritySample, ProgressSample


def extract_answer_from_text(text):
    m = re.search(r"<ans>(.*?)</ans>", text, re.DOTALL)
    ans = m.group(1).strip() if m else ""
    return ans


def load_model_from_hf(
    model_path: str,
    device: torch.device,
    hub_token: Optional[str] = None,
) -> Tuple[Optional[ExperimentConfig], Optional[Any], Optional[Any], Optional[Any]]:
    """
    Load reward model config and model from HuggingFace or local checkpoint.

    This mirrors the logic used by the training/eval scripts:
    - Resolve checkpoint path (supports HF Hub with @ notation)
    - Locate config.yaml locally (if model_path is a directory) or download from HF
    - Use custom YAML loader for ReWiND configs
    - Filter config keys to ExperimentConfig
    - Clear training/logging sections
    - Load model artifacts via setup_model_and_processor

    Args:
        model_path: HuggingFace model repository ID or local checkpoint path.
                   Supports @ notation for tags: username/model@tag-name
        device: Device to load model on
        hub_token: Optional HuggingFace token for private repos

    Returns:
        Tuple of (exp_config, tokenizer, processor, reward_model)
    """
    # Resolve checkpoint path (handles HF Hub downloads with @ notation)
    resolved_path = resolve_checkpoint_path(model_path, hub_token=hub_token)
    if resolved_path is None:
        raise ValueError(f"Could not resolve checkpoint path: {model_path}")
    
    config_path: Optional[str] = None

    # Parse repo_id and revision (tag) from model_path if using @tag format
    # This is used for downloading config.yaml if needed
    if "@" in model_path:
        repo_id, revision = model_path.split("@", 1)
    else:
        repo_id, revision = model_path, None

    if os.path.exists(resolved_path):
        # Local checkpoint: look for config.yaml
        candidate_paths = [
            resolved_path if resolved_path.endswith(".yaml") else os.path.join(resolved_path, "config.yaml"),
            os.path.join(os.path.dirname(resolved_path), "config.yaml"),
        ]
        for candidate in candidate_paths:
            if os.path.isfile(candidate):
                config_path = candidate
                break
        if config_path is None:
            raise FileNotFoundError(
                f"config.yaml not found near local checkpoint {resolved_path}. Provide a path that contains config.yaml."
            )
    else:
        if hf_hub_download is None:
            raise ImportError("huggingface_hub not available. Install with: pip install huggingface_hub")
        # Download config with revision if specified
        config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml", revision=revision, token=hub_token)

    with open(config_path) as f:
        yaml_text = f.read()

    class _ConfigSafeLoader(yaml.SafeLoader):
        pass

    _ConfigSafeLoader.add_constructor(
        "tag:yaml.org,2002:python/object:rfm.models.rewind_transformer.ReWINDTransformerConfig",
        lambda loader, node: loader.construct_mapping(node),
    )

    model_config_dict = yaml.load(yaml_text, Loader=_ConfigSafeLoader)

    valid_keys = {f.name for f in fields(ExperimentConfig)}
    filtered_config = {k: v for k, v in model_config_dict.items() if k in valid_keys}
    filtered_config["training"] = {}
    filtered_config["logging"] = {}

    exp_config = ExperimentConfig(**filtered_config)
    # Use resolved_path for loading the actual model
    tokenizer, processor, reward_model = setup_model_and_processor(exp_config.model, resolved_path)
    reward_model = reward_model.to(device)
    reward_model.eval()

    return exp_config, tokenizer, processor, reward_model


def load_wandb_run_info(model_path: str, hub_token: Optional[str] = None) -> Optional[dict[str, Any]]:
    """
    Retrieve saved wandb metadata for a checkpoint.

    Checks for a local `wandb_info.json` (written during training) and, if the
    checkpoint lives on HuggingFace, falls back to parsing the README that
    `upload_to_hub.py` generates (which embeds wandb fields).
    
    Args:
        model_path: HuggingFace model repository ID or local checkpoint path.
                   Supports @ notation for tags: username/model@tag-name
        hub_token: Optional HuggingFace token for private repos
    """

    def _load_json(path: Path) -> Optional[dict[str, Any]]:
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    # Resolve checkpoint path first
    resolved_path = resolve_checkpoint_path(model_path, hub_token=hub_token)
    if resolved_path:
        path = Path(resolved_path)
        if path.exists():
            candidates = []
            if path.is_file():
                candidates.append(path.parent / "wandb_info.json")
            else:
                candidates.append(path / "wandb_info.json")
                candidates.append(path.parent / "wandb_info.json")
            for candidate in candidates:
                info = _load_json(candidate)
                if info:
                    return info

    if hf_hub_download is None:
        return None

    # Parse repo_id and revision (tag) from model_path if using @tag format
    if "@" in model_path:
        repo_id, revision = model_path.split("@", 1)
    else:
        repo_id, revision = model_path, None

    try:
        readme_path = hf_hub_download(
            repo_id=repo_id, filename="README.md", revision=revision, token=hub_token, local_files_only=False
        )
    except Exception:
        return None

    try:
        readme_text = Path(readme_path).read_text()
    except OSError:
        return None

    wandb_info: dict[str, Any] = {}

    run_match = re.search(r"\*\*Wandb Run\*\*:\s*\[(?P<name>.+?)\]\((?P<url>.+?)\)", readme_text)
    if run_match:
        wandb_info["wandb_name"] = run_match.group("name")
        wandb_info["wandb_url"] = run_match.group("url")

    id_match = re.search(r"\*\*Wandb ID\*\*:\s*`(?P<id>[^`]+)`", readme_text)
    if id_match:
        wandb_info["wandb_id"] = id_match.group("id")

    project_match = re.search(r"\*\*Project\*\*:\s*(?P<project>[^\n]+)", readme_text)
    if project_match:
        wandb_info["wandb_project"] = project_match.group("project").strip()

    entity_match = re.search(r"\*\*Entity\*\*:\s*(?P<entity>[^\n]+)", readme_text)
    if entity_match:
        wandb_info["wandb_entity"] = entity_match.group("entity").strip()

    return wandb_info or None


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
