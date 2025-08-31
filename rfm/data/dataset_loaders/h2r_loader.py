#!/usr/bin/env python3
"""
LIBERO dataset loader for the generic dataset converter for RFM model training.
This module contains LIBERO-specific logic for loading and processing HDF5 files.
"""

import h5py
import os
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
from rfm.data.helpers import generate_unique_id
import cv2


class H2RFrameLoader:
    """Pickle-able loader that reads LIBERO frames from an HDF5 dataset on demand.

    Stores only simple fields so it can be safely passed across processes.
    """

    def __init__(self, hdf5_path: str, convert_to_rgb: bool = True):
        self.hdf5_path = hdf5_path
        self.convert_to_rgb = convert_to_rgb

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load frames from HDF5 when called.

        Returns:
            np.ndarray of shape (T, H, W, 3), dtype uint8
        """
        with h5py.File(self.hdf5_path, "r") as f:
            human_frames = f["/cam_data/human_camera"][:]
            robot_frames = f["/cam_data/robot_camera"][:]

        if self.convert_to_rgb:
            human_frames = human_frames[..., [2, 1, 0]]
            robot_frames = robot_frames[..., [2, 1, 0]]

        # Ensure shape and dtype sanity
        if not isinstance(human_frames, np.ndarray) or human_frames.ndim != 4 or human_frames.shape[-1] != 3:
            raise ValueError(f"Unexpected frames shape for {self.hdf5_path}: {getattr(human_frames, 'shape', None)}")

        if not isinstance(robot_frames, np.ndarray) or robot_frames.ndim != 4 or robot_frames.shape[-1] != 3:
            raise ValueError(f"Unexpected frames shape for {self.hdf5_path}: {getattr(robot_frames, 'shape', None)}")

        # Ensure uint8
        if human_frames.dtype != np.uint8:
            human_frames = human_frames.astype(np.uint8, copy=False)
        if robot_frames.dtype != np.uint8:
            robot_frames = robot_frames.astype(np.uint8, copy=False)

        return human_frames, robot_frames


# Task mapping from folder names to task descriptions
FOLDER_TO_TASK_NAME = {
    "grab_both_cubes_v1": "pick up each cube individually and place them onto the plate.",
    "grab_cup_v1": "pick up the cup and place it in another location",
    "pull_plate_v1": "pull the plate from bottom to top.",
}


def _get_task_name_from_folder(folder_name: str) -> str:
    """Convert folder name to task name using the mapping."""
    # First try to find exact match
    if folder_name in FOLDER_TO_TASK_NAME:
        return FOLDER_TO_TASK_NAME[folder_name]

    # If no exact match, try partial matching
    for folder_key, task_name in FOLDER_TO_TASK_NAME.items():
        if folder_key in folder_name or folder_name in folder_key:
            return task_name

    # If no mapping found, convert folder name to readable task
    task = folder_name.replace("_", " ").replace("-", " ")
    return task.strip()


def _discover_h2r_files(dataset_path: Path) -> List[Tuple[Path, str]]:
    """Discover all video files in the H2R dataset structure.

    Expected structure:
    dataset_path/
        folder_name_1/
            hdf5_file_1.hdf5
            hdf5_file_2.hdf5
            hdf5_file_3.hdf5
            ...
        folder_name_2/
            hdf5_file_1.hdf5
            hdf5_file_2.hdf5
            hdf5_file_3.hdf5
            ...
        ...

    Returns:
        List of tuples: (hdf5_file_path, task_name)
    """
    trajectory_files: List[Tuple[Path, str]] = []
    for folder in dataset_path.iterdir():
        if folder.is_dir():
            for file in folder.glob("*.hdf5"):
                trajectory_files.append((file, folder.name))

    return trajectory_files


def load_h2r_dataset(base_path: str) -> Dict[str, List[Dict]]:
    """Load H2R dataset from HDF5 files and organize by task.

    Args:
        base_path: Path to the H2R dataset directory containing HDF5 files

    Returns:
        Dictionary mapping task names to lists of trajectory dictionaries
    """

    print(f"Loading H2R dataset from: {base_path}")

    task_data = {}

    # Find all HDF5 files in the base path
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"H2R dataset path not found: {base_path}")

    hdf5_files = _discover_h2r_files(base_path)
    print("=" * 100)
    print("LOADING H2R DATASET")
    print("=" * 100)

    print(f"Found {len(hdf5_files)} HDF5 files")

    for file_path, folder_name in tqdm(hdf5_files, desc=f"Processing H2R dataset, {len(hdf5_files)} files"):
        trajectory_info_human = {"frames": [], "actions": []}
        trajectory_info_robot = {"frames": [], "actions": []}
        human_frames, robot_frames = H2RFrameLoader(file_path)()

        trajectory_info_human["frames"] = human_frames
        trajectory_info_robot["frames"] = robot_frames

        # TODO: add actions

        trajectory_info_human["is_robot"] = False
        trajectory_info_robot["is_robot"] = True

        trajectory_info_human["quality_label"] = "successful"
        trajectory_info_robot["quality_label"] = "successful"

        trajectory_info_human["preference_group_id"] = None
        trajectory_info_robot["preference_group_id"] = None

        trajectory_info_human["preference_rank"] = None
        trajectory_info_robot["preference_rank"] = None

        task_name = _get_task_name_from_folder(folder_name)
        trajectory_info_human["task"] = task_name
        trajectory_info_robot["task"] = task_name

        trajectory_info_human["id"] = generate_unique_id()
        trajectory_info_robot["id"] = generate_unique_id()

        if folder_name not in task_data:
            task_data[folder_name] = []

        task_data[folder_name].append(trajectory_info_human)
        task_data[folder_name].append(trajectory_info_robot)

    print(
        f"Loaded {sum(len(trajectories) for trajectories in task_data.values())} trajectories from {len(task_data)} tasks"
    )
    return task_data
