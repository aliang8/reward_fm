#!/usr/bin/env python3
"""
Dataclasses for RFM model dataset trajectory structures.
Defines the standard format for HuggingFace dataset trajectories.
"""

from pydantic import BaseModel
from typing import Optional, Union, List, Dict, Any
import numpy as np


class Trajectory(BaseModel):
    """Trajectory structure containing frames, metadata, and progress information."""

    # Core trajectory fields
    frames: Optional[Union[List[str], np.ndarray]] = None
    frames_shape: Optional[tuple] = None
    id: Optional[str] = None
    task: Optional[str] = None
    lang_vector: Optional[Union[np.ndarray, List[float]]] = None
    data_source: Optional[str] = None
    quality_label: Optional[str] = None
    is_robot: Optional[bool] = None

    data_gen_strategy: Optional[str] = None

    # Progress and metadata
    target_progress: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

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
