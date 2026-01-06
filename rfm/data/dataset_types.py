#!/usr/bin/env python3
"""
Dataclasses for RFM model dataset trajectory structures.
Defines the standard format for HuggingFace dataset trajectories.
"""

from typing import Any, Union, List, Dict, Tuple, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch


class Trajectory(BaseModel):
    """Trajectory structure containing frames, metadata, and progress information."""

    # Core trajectory fields
    frames: Union[List[str], np.ndarray, None] = None
    frames_shape: Optional[Tuple] = None

    # If embeddings are precomputed
    embeddings_path: Optional[str] = None
    video_embeddings: Union[torch.Tensor, np.ndarray, None] = None
    text_embedding: Union[torch.Tensor, np.ndarray, None] = None

    id: Optional[str] = None
    task: Optional[str] = None
    lang_vector: Union[np.ndarray, List[float], None] = None
    data_source: Optional[str] = None
    quality_label: Optional[str] = None
    is_robot: Optional[bool] = None

    # Progress and metadata
    target_progress: Optional[List[float]] = None
    partial_success: Optional[float] = None
    success_label: Optional[List[float]] = None  # Success labels for each frame (1.0 for success, 0.0 for failure)
    metadata: Optional[Dict[str, Any]] = None
    data_gen_strategy: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ProgressSample(BaseModel):
    """Sample structure for progress evaluation."""

    trajectory: Trajectory
    sample_type: str = "progress"
    data_gen_strategy: Optional[str] = None
    resample_attempts: int = 1


class PreferenceSample(BaseModel):
    """Sample structure for preference prediction: chosen vs rejected where chosen is preferred."""

    # Trajectories
    chosen_trajectory: Trajectory
    rejected_trajectory: Trajectory

    sample_type: str = "preference"
    data_gen_strategy: Optional[str] = None
    resample_attempts: int = 1


class SimilaritySample(BaseModel):
    """Sample structure for similarity scoring: traj_sim and traj_diff ranked against o^ref."""

    # Trajectories
    ref_trajectory: Trajectory  # o^ref
    sim_trajectory: Trajectory  # Similar trajectory
    diff_trajectory: Optional[Trajectory] = None  # Different trajectory (optional in inference mode)

    sample_type: str = "similarity"
    data_gen_strategy: Optional[str] = None
    resample_attempts: int = 1


SampleType = Union[PreferenceSample, SimilaritySample, ProgressSample]
