#!/usr/bin/env python3
"""
Evaluation utilities for building and sending batches to the evaluation server.

This module provides helpers to:
- Load the experiment config from YAML into dataclasses
- Convert dataset samples to a JSON-serializable payload (base64-encoded frames)
- Post batches to a server and parse responses
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

from rfm.configs.experiment_configs import (
    ExperimentConfig,
    ModelConfig,
    PEFTConfig,
    DataConfig,
    TrainingConfig,
    LoggingConfig,
)


def _dict_to_dataclass(cfg_dict: Dict[str, Any]) -> ExperimentConfig:
    """Construct `ExperimentConfig` and nested dataclasses from a dict.

    Only keys present in the dataclasses are used; extra keys are ignored.
    """

    def subset(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        return {k: d[k] for k in keys if k in d}

    model = ModelConfig(
        **subset(
            cfg_dict.get("model", {}),
            [
                "base_model_id",
                "torch_dtype",
                "trust_remote_code",
                "train_vision_encoder",
                "train_language_model",
                "train_value_head",
                "train_progress_head",
                "train_preference_head",
                "train_similarity_head",
            ],
        )
    )

    peft = PEFTConfig(
        **subset(cfg_dict.get("peft", {}), ["use_peft", "r", "lora_alpha", "lora_dropout", "bias", "target_modules"])
    )

    data = DataConfig(
        **subset(
            cfg_dict.get("data", {}),
            [
                "train_datasets",
                "train_subsets",
                "eval_datasets",
                "eval_subsets",
                "eval_subset_size",
                "max_frames",
                "video_frame_sampling",
                "resized_height",
                "resized_width",
                "preference_ratio",
                "dataset_preference_ratio",
                "shuffle",
                "seed",
                "num_proc",
                "force_reprocess",
                "dataloader_pin_memory",
                "dataloader_num_workers",
            ],
        )
    )

    training = TrainingConfig(
        **subset(
            cfg_dict.get("training", {}),
            [
                "num_gpus",
                "output_dir",
                "max_seq_length",
                "beta",
                "resume_from_checkpoint",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "learning_rate",
                "num_train_epochs",
                "save_strategy",
                "logging_steps",
                "bf16",
                "fp16",
                "remove_unused_columns",
                "gradient_checkpointing",
                "ddp_find_unused_parameters",
                "ddp_bucket_cap_mb",
                "max_steps",
                "save_steps",
                "evaluation_strategy",
                "eval_steps",
                "per_device_eval_batch_size",
                "do_eval",
                "prediction_loss_only",
            ],
        )
    )

    logging = LoggingConfig(
        **subset(
            cfg_dict.get("logging", {}),
            [
                "print_trainable_parameters",
                "save_model",
                "save_processor",
                "use_wandb",
                "wandb_project",
                "wandb_entity",
                "wandb_run_name",
            ],
        )
    )

    return ExperimentConfig(
        mode=cfg_dict.get("mode", "train"),
        debug=cfg_dict.get("debug", False),
        model=model,
        peft=peft,
        data=data,
        training=training,
        logging=logging,
    )


def load_experiment_config_from_yaml(yaml_path: str) -> ExperimentConfig:
    import yaml

    with open(yaml_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return _dict_to_dataclass(cfg_dict)


def _ensure_numpy_frames(frames: Any, frames_shape: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Ensure frames are a numpy array of shape (T, H, W, C).

    Accepts bytes (with shape), numpy array, list of numpy frames, or single frame.
    """
    if frames is None:
        return np.empty((0,))

    # Bytes -> numpy using provided shape
    if isinstance(frames, (bytes, bytearray)):
        if frames_shape is None:
            # Fallback: interpret as uint8 flat array (cannot reshape reliably)
            arr = np.frombuffer(frames, dtype=np.uint8)
            return arr
        if isinstance(frames_shape, list):
            frames_shape = tuple(frames_shape)
        try:
            return np.frombuffer(frames, dtype=np.uint8).reshape(frames_shape)
        except Exception:
            return np.frombuffer(frames, dtype=np.uint8)

    # Already a numpy array
    if isinstance(frames, np.ndarray):
        return frames

    # List of numpy arrays
    if isinstance(frames, list) and all(isinstance(f, np.ndarray) for f in frames):
        return np.stack(frames, axis=0)

    # Unsupported (e.g., file paths) – return as empty; upstream should handle
    return np.empty((0,))


def decode_frames_b64(frames_b64: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for s in frames_b64:
        try:
            buf = io.BytesIO(base64.b64decode(s))
            img = Image.open(buf).convert("RGB")
            images.append(img)
        except Exception:
            continue
    return images


def frames_to_base64_images(frames: Any, frames_shape: Optional[Tuple[int, int, int, int]] = None) -> List[str]:
    """Convert frames to a list of base64-encoded JPEG strings.

    Frames can be ndarray (T,H,W,C), bytes + shape, list of ndarray, or a single frame.
    """
    arr = _ensure_numpy_frames(frames, frames_shape)
    if arr.size == 0:
        return []

    # Normalize to (T, H, W, C)
    if arr.ndim == 3:  # single frame (H,W,C)
        arr = arr[None, ...]
    elif arr.ndim != 4:
        # Unknown shape: cannot encode reliably
        return []

    encoded: List[str] = []
    for i in range(arr.shape[0]):
        frame = arr[i]
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return encoded


def similarity_to_preference_payload(sim_sample: Any) -> Dict[str, Any]:
    """Convert a SimilaritySample to a preference-style payload using traj_sim as chosen."""
    return {
        "task": sim_sample.task if hasattr(sim_sample, "task") else getattr(sim_sample, "task_ref", ""),
        "sample_type": "preference",
        "chosen_frames_b64": frames_to_base64_images(
            sim_sample.traj_sim_frames, getattr(sim_sample, "traj_sim_frames_shape", None)
        ),
        "rejected_frames_b64": frames_to_base64_images(
            sim_sample.traj_diff_frames, getattr(sim_sample, "traj_diff_frames_shape", None)
        ),
        "target_progress_A": getattr(sim_sample, "target_progress_A", None),
        "target_progress_B": getattr(sim_sample, "target_progress_B", None),
    }


def preference_to_payload(pref_sample: Any) -> Dict[str, Any]:
    """Convert a PreferenceSample to a JSON-serializable payload for the server."""
    return {
        "task": getattr(pref_sample, "task", ""),
        "sample_type": "preference",
        "chosen_frames_b64": frames_to_base64_images(
            pref_sample.chosen_frames, getattr(pref_sample, "chosen_frames_shape", None)
        ),
        "rejected_frames_b64": frames_to_base64_images(
            pref_sample.rejected_frames, getattr(pref_sample, "rejected_frames_shape", None)
        ),
        "target_progress_A": getattr(pref_sample, "target_progress_A", None),
        "target_progress_B": getattr(pref_sample, "target_progress_B", None),
    }


def build_batch_payload(samples: List[Any]) -> Dict[str, Any]:
    """Build a batch payload from samples of either PreferenceSample or SimilaritySample."""
    payload_samples: List[Dict[str, Any]] = []
    for s in samples:
        sample_type = getattr(s, "sample_type", None)
        if sample_type == "preference":
            payload_samples.append(preference_to_payload(s))
        elif sample_type == "similarity":
            payload_samples.append(similarity_to_preference_payload(s))
        else:
            # Skip unknown types
            continue
    return {"samples": payload_samples}


def post_batch(url: str, payload: Dict[str, Any], timeout_s: float = 120.0) -> Dict[str, Any]:
    """POST a batch payload to the evaluation server and return parsed JSON."""
    resp = requests.post(url.rstrip("/") + "/evaluate_batch", json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()
