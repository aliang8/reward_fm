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
    frames: str
    is_robot: bool
    quality_label: str
    preference_group_id: Optional[str] = None
    preference_rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for HuggingFace dataset trajectory."""
        return {
            "id": self.id,
            "task": self.task,
            "lang_vector": self.lang_vector,
            "data_source": self.data_source,
            "frames": self.frames,
            "is_robot": self.is_robot,
            "quality_label": self.quality_label,
            "preference_group_id": self.preference_group_id,
            "preference_rank": self.preference_rank,
            "metadata": self.metadata,
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
