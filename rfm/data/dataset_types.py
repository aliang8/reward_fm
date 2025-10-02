#!/usr/bin/env python3
"""
Dataclasses for RFM model dataset trajectory structures.
Defines the standard format for HuggingFace dataset trajectories.
"""

from typing import Any, Union

import numpy as np
from pydantic import BaseModel
import torch


class Trajectory(BaseModel):
    """Trajectory structure containing frames, metadata, and progress information."""

    # Core trajectory fields
    frames: list[str] | np.ndarray | None = None
    frames_shape: tuple | None = None

    # If embeddings are precomputed
    embeddings_path: str | None = None
    video_embeddings: torch.Tensor | None = None
    text_embedding: torch.Tensor | None = None

    id: str | None = None
    task: str | None = None
    lang_vector: np.ndarray | list[float] | None = None
    data_source: str | None = None
    quality_label: str | None = None
    is_robot: bool | None = None

    data_gen_strategy: str | None = None

    # Progress and metadata
    target_progress: list[float] | None = None
    metadata: dict[str, Any] | None = None

    class Config:
        arbitrary_types_allowed = True


class ProgressSample(BaseModel):
    """Sample structure for progress evaluation."""

    trajectory: Trajectory
    sample_type: str = "progress"


class PreferenceSample(BaseModel):
    """Sample structure for preference prediction: chosen vs rejected where chosen is preferred."""

    # Trajectories
    chosen_trajectory: Trajectory
    rejected_trajectory: Trajectory

    sample_type: str = "preference"


class SimilaritySample(BaseModel):
    """Sample structure for similarity scoring: traj_sim and traj_diff ranked against o^ref."""

    # Trajectories
    reference_trajectory: Trajectory  # o^ref
    traj_sim_trajectory: Trajectory  # Similar trajectory
    traj_diff_trajectory: Trajectory  # Different trajectory

    sample_type: str = "similarity"


SampleType = Union[PreferenceSample, SimilaritySample, ProgressSample]
