#!/usr/bin/env python3
"""
Metaworld dataset loader for the generic dataset converter for RFM model training.
This module contains logic for loading metaworld data organized by task and quality.
"""

import os
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
from rfm.data.video_helpers import load_video_frames


def map_quality_label(original_label: str) -> str:
    """Map original quality labels to standardized RFM labels."""
    label_mapping = {
        'GT': 'successful',
        'success': 'successful', 
        'all_fail': 'failure',
        'close_succ': 'suboptimal'
    }
    return label_mapping.get(original_label, original_label)


def map_task_to_natural_language(task_name: str) -> str:
    """Convert task names to natural language descriptions."""
    task_mapping = {
        'window_open': 'Open the window',
        'window_close': 'Close the window',
        'button_press': 'Press the button',
        'button_press_topdown_wall': 'Press the button on the top of the wall',
        'coffee_pull': 'Pull the coffee lever',
        'door_open': 'Open the door',
        'drawer_close': 'Close the drawer',
        'faucet_open': 'Open the faucet',
        'hand_insert': 'Insert your hand into the opening',
        'handle_press_side': 'Press the side of the handle',
        'handle_pull_side': 'Pull the side of the handle',
        'lever_pull': 'Pull the lever',
        'plate_slide_back_side': 'Slide the plate back from the side',
        'push_wall': 'Push against the wall',
        'eval_tasks': 'Evaluation tasks',
        'train_tasks': 'Training tasks'
    }
    return task_mapping.get(task_name, task_name.replace('_', ' ').title())


def load_metaworld_dataset(base_path: str) -> Dict[str, List[Dict]]:
    """Load metaworld dataset and organize by task.
    
    Args:
        base_path: Path to the metaworld dataset directory
        
    Returns:
        Dictionary mapping task names to lists of trajectory dictionaries
    """
    
    print(f"Loading metaworld dataset from: {base_path}")
    
    task_data = {}
    base_path = Path(base_path)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Metaworld dataset path not found: {base_path}")
    
    # Find all task directories
    task_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for task_dir in tqdm(task_dirs, desc="Processing tasks"):
        task_name = task_dir.name
        
        if task_name in ['.DS_Store']:
            continue
            
        task_trajectories = []
        
        # Find all quality label directories within this task
        quality_dirs = [d for d in task_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        for quality_dir in quality_dirs:
            original_quality_label = quality_dir.name
            
            # Map quality label to standardized format
            quality_label = map_quality_label(original_quality_label)
            
            # Find all video files in this quality directory
            video_files = list(quality_dir.glob("*.mp4")) + list(quality_dir.glob("*.gif"))
            
            for video_file in video_files:
                # Extract index from filename (e.g., "1.mp4" -> 1)
                try:
                    idx = int(video_file.stem)
                except ValueError:
                    print(f"Warning: Could not parse index from filename: {video_file.name}")
                    idx = 0
                            
                # Create trajectory entry
                trajectory = {
                    'frames': load_video_frames(video_file), 
                    'task': map_task_to_natural_language(task_name),  # Natural language task
                    'quality_label': quality_label,  # Mapped quality label
                    'is_robot': True,
                    'original_quality_label': original_quality_label,  # Keep original for reference
                    'original_task_name': task_name  # Keep original task name for reference
                }
                
                task_trajectories.append(trajectory)
        
        if task_trajectories:
            task_data[task_name] = task_trajectories
            print(f"  Task '{task_name}' -> '{map_task_to_natural_language(task_name)}': {len(task_trajectories)} trajectories")
    
    total_trajectories = sum(len(trajectories) for trajectories in task_data.values())
    print(f"\nLoaded {total_trajectories} trajectories from {len(task_data)} tasks")
    
    return task_data
    """Get statistics about the dataset structure."""
    base_path = Path(base_path)
    
    if not base_path.exists():
        return {}
    
    stats = {
        'total_tasks': 0,
        'total_trajectories': 0,
        'tasks': {},
        'quality_labels': set(),
        'file_extensions': set(),
        'label_mapping': {}
    }
    
    task_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for task_dir in task_dirs:
        task_name = task_dir.name
        if task_name in ['.DS_Store']:
            continue
            
        task_stats = {
            'trajectories': 0,
            'quality_labels': set(),
            'file_extensions': set(),
            'natural_language_task': map_task_to_natural_language(task_name)
        }
        
        quality_dirs = [d for d in task_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        for quality_dir in quality_dirs:
            original_quality_label = quality_dir.name
            mapped_quality_label = map_quality_label(original_quality_label)
            
            stats['quality_labels'].add(mapped_quality_label)
            task_stats['quality_labels'].add(mapped_quality_label)
            
            # Track label mapping
            if original_quality_label not in stats['label_mapping']:
                stats['label_mapping'][original_quality_label] = mapped_quality_label
            
            video_files = list(quality_dir.glob("*.mp4")) + list(quality_dir.glob("*.gif"))
            
            for video_file in video_files:
                stats['file_extensions'].add(video_file.suffix)
                task_stats['file_extensions'].add(video_file.suffix)
                task_stats['trajectories'] += 1
                stats['total_trajectories'] += 1
        
        if task_stats['trajectories'] > 0:
            stats['tasks'][task_name] = task_stats
            stats['total_tasks'] += 1
    
    return stats