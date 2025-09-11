#!/usr/bin/env python3
"""
Custom dataset classes for RFM data generation.
"""

import numpy as np
from typing import List, Dict


def create_binned_subsequences(frames: np.ndarray, num_bins: int = 10) -> List[Dict]:
    """
    Create binned subsequences from frames.

    Args:
        frames: numpy array of shape (num_frames, H, W, C)
        num_bins: number of bins to create

    Returns:
        List of dictionaries with 'start_frame', 'end_frame', 'frames', 'bin_idx', 'progress'
    """
    num_frames = frames.shape[0]

    if num_frames < num_bins:
        # If fewer frames than bins, create one frame per bin
        bin_size = 1
        num_bins = num_frames
    else:
        bin_size = num_frames // num_bins

    binned_subsequences = []

    for bin_idx in range(num_bins):
        start_frame = bin_idx * bin_size
        end_frame = min(start_frame + bin_size, num_frames)

        # Handle last bin to include remaining frames
        if bin_idx == num_bins - 1:
            end_frame = num_frames

        if start_frame >= end_frame:
            continue

        bin_frames = frames[start_frame:end_frame]
        progress = start_frame / (num_frames - 1) if num_frames > 1 else 0.0

        binned_subsequences.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frames": bin_frames,
                "bin_idx": bin_idx,
                "progress": progress,
            }
        )

    return binned_subsequences
