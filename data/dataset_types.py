#!/usr/bin/env python3
"""
Dataclasses for RFM model dataset trajectory structures.
Defines the standard format for HuggingFace dataset trajectories.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np


@dataclass
class Trajectory:
    """Standard trajectory structure for HuggingFace format."""
    
    id: str
    task: str
    lang_vector: np.ndarray
    data_source: str
    frames: List[str]
    optimal: bool
    is_robot: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for HuggingFace dataset trajectory."""
        return {
            "id": self.id,
            "task": self.task,
            "lang_vector": self.lang_vector,
            "data_source": self.data_source,
            "frames": self.frames,
            "optimal": self.optimal,
            "is_robot": self.is_robot,
            "metadata": self.metadata,
        }


@dataclass
class DatasetMetadata:
    """Metadata for the entire dataset."""
    
    dataset_name: str
    num_entries: int
    max_trajectories: Optional[int]
    max_frames: int
    default_ranking: int
    data_source: str
    created_at: str
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "dataset_name": self.dataset_name,
            "num_entries": self.num_entries,
            "max_trajectories": self.max_trajectories,
            "max_frames": self.max_frames,
            "default_ranking": self.default_ranking,
            "data_source": self.data_source,
            "created_at": self.created_at,
            "additional_info": self.additional_info,
        }


@dataclass
class Preference:
    """Preference data structure for trajectory comparisons."""
    
    traj_id: str
    chosen_id: str
    rejected_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "traj_id": self.traj_id,
            "chosen_id": self.chosen_id,
            "rejected_id": self.rejected_id,
        } 