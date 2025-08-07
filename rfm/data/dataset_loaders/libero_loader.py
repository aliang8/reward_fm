#!/usr/bin/env python3
"""
LIBERO dataset loader for the generic dataset converter for RFM model training.
This module contains LIBERO-specific logic for loading and processing HDF5 files.
"""

import h5py
import os
import numpy as np
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

def load_libero_dataset(base_path: str) -> Dict[str, List[Dict]]:
    """Load LIBERO dataset from HDF5 files and organize by task.
    
    Args:
        base_path: Path to the LIBERO dataset directory containing HDF5 files
        
    Returns:
        Dictionary mapping task names to lists of trajectory dictionaries
    """
    
    print(f"Loading LIBERO dataset from: {base_path}")
    
    task_data = {}
    
    # Find all HDF5 files in the base path
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"LIBERO dataset path not found: {base_path}")
    
    hdf5_files = list(base_path.glob("*.hdf5"))
    print("="*100)
    print("LOADING LIBERO DATASET")
    print("="*100)

    print(f"Found {len(hdf5_files)} HDF5 files")
    
    for file_path in tqdm(hdf5_files, desc=f"Processing LIBERO dataset, {len(hdf5_files)} files"):
        task_name = file_path.stem  # Remove .hdf5 extension
        # print(f"Loading task: {task_name}")
        
        with h5py.File(file_path, 'r') as f:
            if 'data' not in f:
                print(f"No 'data' group in {task_name}")
                continue
            
            data_group = f['data']
            trajectories = []
            
            for trajectory_key in data_group.keys():
                trajectory = data_group[trajectory_key]
                if isinstance(trajectory, h5py.Group):
                    # Extract trajectory data
                    trajectory_info = {
                        'frames': [],
                        'actions': []
                    }
                    
                    # Get trajectory length from observations
                    if 'obs' in trajectory and 'agentview_rgb' in trajectory['obs']:
                        frames = trajectory['obs']['agentview_rgb'][:]  # (T, H, W, 3)
                        # Rotate frames by 180 degrees to correct LIBERO camera orientation
                        # Equivalent to flipping vertically and horizontally
                        if isinstance(frames, np.ndarray) and frames.ndim == 4 and frames.shape[-1] == 3:
                            frames = frames[:, ::-1, ::-1, :].copy()
                        trajectory_info['frames'] = frames
                    
                    # Get actions if available
                    if 'actions' in trajectory:
                        trajectory_info['actions'] = trajectory['actions'][:]

                    # Assume all LIBERO trajectories are successful
                    trajectory_info['optimal'] = "optimal"
                    trajectory_info['is_robot'] = True
                    
                    # Parse the original file path to extract scene and task info
                    file_name = os.path.basename(file_path).replace('.hdf5', '')
                    
                    # Extract scene and task from the file name
                    # Example: LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray
                    parts = file_name.split('_')
                    
                    # Find the scene part (contains "SCENE")
                    scene_part = None
                    task_parts = []
                    
                    for i, part in enumerate(parts):
                        if 'SCENE' in part:
                            scene_part = part
                            # Everything after the scene is the task
                            task_parts = parts[i+1:]
                            break
                    
                    # If no scene found, use the first part as scene
                    if scene_part is None:
                        scene_part = parts[0] if parts else "UNKNOWN_SCENE"
                        task_parts = parts[1:] if len(parts) > 1 else []
                    
                    # Convert task parts to readable string
                    task_string = " ".join(task_parts).replace('_', ' ')
                    task_string = task_string.replace("demo", "")
                    
                    
                    # Add parsed information to trajectory
                    trajectory_info['task'] = task_string.strip()                    
                    trajectories.append(trajectory_info)
            
            task_data[task_name] = trajectories
            # print(f"  Loaded {len(trajectories)} trajectories for {task_name}")
    
    print(f"Loaded {sum(len(trajectories) for trajectories in task_data.values())} trajectories from {len(task_data)} tasks")
    return task_data


 