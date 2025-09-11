#!/usr/bin/env python3
from __future__ import annotations

import re
import io
import json
from typing import Any, Dict, List, Union, Tuple

import numpy as np
import requests
import aiohttp
from rfm.data.dataset_types import PreferenceSample, SimilaritySample


def extract_answer_from_text(text):
    m = re.search(r"<ans>(.*?)</ans>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def post_batch(url: str, payload: Dict[str, Any], timeout_s: float = 120.0) -> Dict[str, Any]:
    """POST a batch payload to the evaluation server and return parsed JSON."""
    resp = requests.post(url.rstrip("/") + "/evaluate_batch", json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def build_payload(
    samples: List[Union[PreferenceSample, SimilaritySample]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
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
                if "frames" in trajectory and isinstance(trajectory["frames"], np.ndarray):
                    # Convert frames to .npy file
                    buf = io.BytesIO()
                    np.save(buf, trajectory["frames"])
                    buf.seek(0)
                    file_key = f"sample_{sample_idx}_{key}_frames"
                    files[file_key] = (f"sample_{sample_idx}_{key}_frames.npy", buf, "application/octet-stream")
                    trajectory["frames"] = {"__numpy_file__": file_key}

                if "lang_vector" in trajectory and isinstance(trajectory["lang_vector"], np.ndarray):
                    # Convert lang_vector to .npy file
                    buf = io.BytesIO()
                    np.save(buf, trajectory["lang_vector"])
                    buf.seek(0)
                    file_key = f"sample_{sample_idx}_{key}_lang_vector"
                    files[file_key] = (f"sample_{sample_idx}_{key}_lang_vector.npy", buf, "application/octet-stream")
                    trajectory["lang_vector"] = {"__numpy_file__": file_key}

        sample_data.append(processed_sample)

    return files, sample_data


def post_batch_npy(
    url: str, files: Dict[str, Any], sample_data: List[Dict[str, Any]], timeout_s: float = 120.0
) -> Dict[str, Any]:
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
    files: Dict[str, Any],
    sample_data: List[Dict[str, Any]],
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
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
