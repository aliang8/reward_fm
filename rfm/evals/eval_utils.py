#!/usr/bin/env python3
from __future__ import annotations

import re
import torch
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Tuple
from datetime import datetime

import aiohttp
import numpy as np
import requests
import torch

from rfm.data.dataset_types import PreferenceSample, SimilaritySample, ProgressSample, Trajectory
from rfm.data.datasets.helpers import linspace_subsample_frames, pad_trajectory_to_max_frames_np


def extract_answer_from_text(text: str) -> str:
    """Extract answer from text using <ans> tags."""
    m = re.search(r"<ans>(.*?)</ans>", text, re.DOTALL)
    ans = m.group(1).strip() if m else ""
    return ans


def raw_dict_to_sample(
    raw_data: Union[Tuple[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    max_frames: int = 16,
    sample_type: str = "progress",
) -> Union[ProgressSample, PreferenceSample]:
    """
    Convert raw data dictionary to a ProgressSample or PreferenceSample.

    Args:
        raw_data: Dict with 'frames', 'task', 'id', 'metadata', 'video_embeddings', 'text_embedding' or Tuple of (Dict[str, Any], Dict[str, Any])
        max_frames: Maximum number of frames to use (default: 16)
        sample_type: Either "progress" or "preference" (default: "progress")

    Returns:
        ProgressSample or PreferenceSample
    """
    def _build_trajectory(raw_data: Dict[str, Any], num_frames: int) -> Trajectory:
        processed_item: Dict[str, Any] = {}

        # Process frames
        frames_array = raw_data["frames"]

        # Ensure we have the correct shape: (T, H, W, C)
        if len(frames_array.shape) != 4:
            raise ValueError(f"Expected 4D array (T, H, W, C), got shape {frames_array.shape}")

        # Convert from CxHxW to HxWxC if needed
        if frames_array.shape[1] == 3:
            frames_array = np.transpose(frames_array, (0, 2, 3, 1))

        frames_array, _ = linspace_subsample_frames(frames_array, num_frames)
        dummy_progress = [0.0] * len(frames_array)
        frames_array, _ = pad_trajectory_to_max_frames_np(frames_array, dummy_progress, num_frames, pad_from="right")

        if frames_array.size == 0:
            raise ValueError("No frames processed for example")

        processed_item["frames"] = frames_array
        processed_item["frames_shape"] = frames_array.shape
        processed_item["task"] = raw_data["task"]
        processed_item["lang_vector"] = None
        processed_item["metadata"] = raw_data.get("metadata", None)

        # Process video embeddings using same helper functions
        video_embeddings = raw_data.get("video_embeddings")
        if video_embeddings is not None:
            video_embeddings, _ = linspace_subsample_frames(video_embeddings, num_frames)
            dummy_progress_emb = [0.0] * len(video_embeddings)
            video_embeddings, _ = pad_trajectory_to_max_frames_np(
                video_embeddings, dummy_progress_emb, num_frames, pad_from="right"
            )

        text_embedding = raw_data.get("text_embedding")

        # Convert to tensors if they are numpy arrays
        if video_embeddings is not None and isinstance(video_embeddings, np.ndarray):
            video_embeddings = torch.tensor(video_embeddings)
        if text_embedding is not None and isinstance(text_embedding, np.ndarray):
            text_embedding = torch.tensor(text_embedding)

        processed_item["video_embeddings"] = video_embeddings
        processed_item["text_embedding"] = text_embedding
        processed_item["video_shape"] = video_embeddings.shape if video_embeddings is not None else None
        processed_item["text_shape"] = text_embedding.shape if text_embedding is not None else None

        trajectory = Trajectory(**processed_item)
        return trajectory

    if sample_type == "progress":
        assert isinstance(raw_data, dict), "raw_data must be a dictionary"
        trajectory = _build_trajectory(raw_data=raw_data, num_frames=max_frames)
        return ProgressSample(trajectory=trajectory)
    elif sample_type == "preference":
        assert isinstance(raw_data, tuple), "raw_data must be a tuple"
        assert len(raw_data) == 2, "raw_data must be a tuple of two dictionaries"
        trajectories: List[Trajectory] = []
        for trajectory_data in raw_data:
            trajectory = _build_trajectory(raw_data=trajectory_data, num_frames=max_frames)
            trajectories.append(trajectory)
        return PreferenceSample(chosen_trajectory=trajectories[0], rejected_trajectory=trajectories[1])
    else:
        raise ValueError(f"Unsupported sample_type: {sample_type}")


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


def find_video_files(directory: str) -> list[str]:
    """Find all video files in a directory.

    Args:
        directory: Path to directory containing video files

    Returns:
        List of paths to video files
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
    video_files = []

    directory_path = Path(directory)
    if not directory_path.is_dir():
        return []

    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(str(file_path))

    video_files.sort()
    return video_files


def infer_task_from_video_name(video_path: str) -> str:
    """Infer task name from video filename.

    Task is everything before the comma (if comma exists), or everything before success/fail/failure.

    Args:
        video_path: Path to video file

    Returns:
        Inferred task name
    """
    video_name = Path(video_path).stem  # Get filename without extension

    # If there's a comma, task is everything before the comma
    if "," in video_name:
        task_part = video_name.split(",")[0]
    else:
        # Otherwise, split by underscore and remove success/fail/failure suffixes
        parts = video_name.split("_")
        filtered_parts = []
        for part in parts:
            part_lower = part.lower()
            if part_lower not in ["success", "fail", "failure"]:
                filtered_parts.append(part)

        if not filtered_parts:
            return "Complete the task"

        task_part = "_".join(filtered_parts)

    # Split by underscore and join with spaces
    task_words = task_part.split("_")
    task = " ".join(task_words)

    if task:
        # Capitalize first letter of first word, keep rest as is
        task = task[0].upper() + task[1:] if len(task) > 1 else task.upper()
    else:
        task = "Complete the task"

    return task


def setup_output_directory(output_dir: Optional[str], video_path: Optional[str] = None) -> str:
    """Create output directory and return path."""
    if output_dir:
        save_dir = output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(".", f"eval_outputs/{timestamp}")

    os.makedirs(save_dir, exist_ok=True)
    return save_dir
