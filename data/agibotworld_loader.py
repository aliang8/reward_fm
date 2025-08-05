"""
AgiBotWorld dataset loader for the generic dataset converter for RFM model training.
This module contains AgiBotWorld-specific logic for loading and processing data using
HuggingFace streaming to efficiently handle large datasets.
"""

import os
import json
import h5py
import numpy as np
import cv2
import tempfile
from typing import List, Dict, Tuple, Optional, Callable, Any
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from helpers import downsample_frames

def load_agibotworld_dataset(dataset_name_or_path: str, max_trajectories: int = 100) -> Dict[str, List[Dict]]:
    """Load AgiBotWorld dataset using HuggingFace streaming and extract head_color.mp4 files.
    
    Args:
        dataset_name_or_path: HuggingFace dataset name (e.g. "agibot-world/AgiBotWorld-Alpha") 
                             or local path to the dataset
        
    Returns:
        Dictionary mapping task names to lists of trajectory dictionaries
    """
    
    print(f"Loading AgiBotWorld dataset from: {dataset_name_or_path}")
    print("=" * 100)
    print("LOADING AGIBOTWORLD DATASET")
    print("=" * 100)
    
    task_data = {}
    
    # Check if it's a local path or HuggingFace dataset name
    if os.path.exists(dataset_name_or_path):
        # Local dataset
        task_data = _load_local_agibotworld(dataset_name_or_path, max_trajectories)
    else:
        # HuggingFace dataset - use streaming
        task_data = _load_streaming_agibotworld(dataset_name_or_path, max_trajectories)
    
    print(f"Loaded {sum(len(trajectories) for trajectories in task_data.values())} trajectories from {len(task_data)} tasks")
    return task_data


def _load_local_agibotworld(base_path: str, max_trajectories: int = 100, max_frames: int = 32) -> Dict[str, List[Dict]]:
    """Load AgiBotWorld dataset from local files, starting with task_info JSON files."""
    base_path = Path(base_path)
    task_data = {}
    
    # Define required directories
    observations_dir = base_path / "observations"
    task_info_dir = base_path / "task_info"
    proprio_stats_dir = base_path / "proprio_stats"
    
    if not observations_dir.exists():
        raise FileNotFoundError(f"Observations directory not found: {observations_dir}")
    if not task_info_dir.exists():
        raise FileNotFoundError(f"Task info directory not found: {task_info_dir}")
    
    # Start by iterating over task_info JSON files to get proper task names
    task_info_files = list(task_info_dir.glob("*.json"))
    
    if not task_info_files:
        raise FileNotFoundError(f"No task info JSON files found in: {task_info_dir}")
    
    print(f"Found {len(task_info_files)} task info files")
    
    total_trajectories = 0
    
    for task_info_file in tqdm(task_info_files, desc="Processing tasks"):
        if total_trajectories >= max_trajectories:
            print(f"Reached max_trajectories limit ({max_trajectories}), stopping...")
            break
            
        # Extract task ID from filename (e.g., "task_392.json" -> "392")
        task_id = task_info_file.stem.replace("task_", "")
        
        # Load task information from JSON
        task_info = _load_task_info(task_info_file)
        
        if not task_info:
            print(f"Skipping task {task_id} - no valid task info")
            continue
        
        # Extract proper task name from first episode (they should all have the same task)
        if task_info and len(task_info) > 0:
            first_episode = task_info[0]
            task_name = first_episode.get('task_name', f"Task {task_id}")
            task_description = first_episode.get('task_description', f"AgiBotWorld Task {task_id}")
        else:
            task_name = f"Task {task_id}"
            task_description = f"AgiBotWorld Task {task_id}"
        
        print(f"Processing task {task_id}: '{task_name}'")
        
        # Get the corresponding task directory
        task_dir = observations_dir / task_id
        if not task_dir.exists():
            print(f"Task directory not found: {task_dir}, skipping...")
            continue
        
        trajectories = []
        
        # Process episodes based on the information in task_info JSON
        for episode_info in task_info:
            if total_trajectories >= max_trajectories:
                break
                
            episode_id = str(episode_info.get('episode_id', ''))
            if not episode_id:
                continue
                
            # Check if episode directory exists
            episode_dir = task_dir / episode_id
            if not episode_dir.exists():
                print(f"Episode directory not found: {episode_dir}, skipping episode {episode_id}")
                continue
            
            # Look for head_color.mp4 file
            videos_dir = episode_dir / "videos"
            head_color_video = videos_dir / "head_color.mp4"
            
            if head_color_video.exists():
                # Load proprioceptive data
                proprio_file = proprio_stats_dir / task_id / episode_id / "proprio_stats.h5"
                actions = _load_actions_from_h5(proprio_file)
                
                # Process video: resize to 256x256 and downsample frames
                try:
                    processed_frames = _process_video_for_dataset(head_color_video)
                    
                    trajectory = {
                        'frames': processed_frames,  # Processed video frames
                        'actions': actions,
                        'is_robot': True,  # AgiBotWorld is robot data
                        'task': task_name,  # Use the descriptive task name from JSON
                        'optimal': 'optimal'  # Assume all AgiBotWorld trajectories are optimal
                    }
                except Exception as e:
                    print(f"  ❌ Failed to process video {head_color_video}: {e}")
                    continue
                
                trajectories.append(trajectory)
                total_trajectories += 1
                
                print(f"  ✅ Loaded episode {episode_id} ({total_trajectories}/{max_trajectories})")
            else:
                print(f"  ❌ head_color.mp4 not found for episode {episode_id}")
        
        if trajectories:
            # Use proper task name from JSON instead of generic "task_{id}"
            task_data[task_name] = trajectories
            print(f"Added {len(trajectories)} trajectories for task '{task_name}'")
    
    print(f"Loaded {total_trajectories} total trajectories from {len(task_data)} tasks")
    return task_data


def _load_streaming_agibotworld(dataset_name: str, max_trajectories: int = 100) -> Dict[str, List[Dict]]:
    """Load AgiBotWorld dataset using HuggingFace streaming with webdataset format."""
    print(f"Streaming from HuggingFace dataset: {dataset_name}")
    print("Processing webdataset format...")
    
    # Try to load as streaming dataset without enforcing schema
    try:
        # Load dataset with streaming
        dataset = load_dataset(dataset_name, streaming=True, split='train')
        print(f"Successfully loaded streaming dataset: {dataset_name}")
        
        # Inspect first few samples to understand structure
        task_data = {}
        sample_count = 0
        processed_count = 0
        
        print("Inspecting dataset structure and processing valid samples...")
        
        # Use itertools to handle the casting errors gracefully
        import itertools
        dataset_iter = iter(dataset)
        
        # Use max_trajectories parameter with reasonable sample examination limit
        max_samples_to_examine = max(max_trajectories * 10, 1000)  # Examine more samples to find valid ones
        print(f"Looking for up to {max_trajectories} trajectories by examining up to {max_samples_to_examine} samples...")
        
        while processed_count < max_trajectories and sample_count < max_samples_to_examine:
            try:
                # Get next example, handling casting errors gracefully
                try:
                    example = next(dataset_iter)
                except StopIteration:
                    print(f"Reached end of dataset stream after {sample_count} samples")
                    break
                except Exception as cast_error:
                    # Only print first few casting errors to avoid spam
                    if sample_count <= 10:
                        print(f"Skipping sample due to casting error: {cast_error}")
                    elif sample_count % 100 == 0:
                        print(f"Processed {sample_count} samples, found {processed_count} valid head camera videos...")
                    sample_count += 1
                    continue
                
                sample_count += 1
                
                # Print structure of first valid sample
                if processed_count == 0:
                    print(f"Sample structure: {list(example.keys())}")
                    for key, value in example.items():
                        if value is not None:
                            print(f"  {key}: {type(value)} - {len(value) if hasattr(value, '__len__') else 'no len'}")
                
                # Extract key information from webdataset format
                key = example.get('__key__', f'sample_{sample_count}')
                
                # Print all keys for first 20 samples to understand data structure
                if sample_count <= 20:
                    print(f"Sample {sample_count}: key='{key}'")
                
                # More flexible filtering - accept any video that might be a camera view
                is_video_sample = any(pattern in key.lower() for pattern in [
                    'head_color', 'head_rgb', 'head_cam', 'head', 'color', 'rgb', 'camera', 'cam', 'video'
                ])
                
                if not is_video_sample:
                    # Print skipped keys occasionally for debugging
                    if sample_count <= 10:
                        print(f"  ❌ Skipping non-video sample: {key}")
                    continue
                else:
                    if sample_count <= 20:
                        print(f"  ✅ Potential video sample: {key}")
                
                # Look for video data in various fields
                video_data = None
                if 'mp4' in example and example['mp4'] is not None:
                    video_data = example['mp4']
                elif 'webm' in example and example['webm'] is not None:
                    video_data = example['webm']
                elif 'avi' in example and example['avi'] is not None:
                    video_data = example['avi']
                
                # Skip if no video data found
                if video_data is None or len(video_data) == 0:
                    continue
                
                # Parse task information from key 
                # Key format might be like: observations/327/648642/videos/head_color
                parts = key.split('/')
                task_id = None
                episode_id = None
                
                # Try to extract task and episode IDs from path structure
                for i, part in enumerate(parts):
                    if part.isdigit() and len(part) <= 4:  # Task ID (short number)
                        task_id = part
                        if i + 1 < len(parts) and parts[i + 1].isdigit():
                            episode_id = parts[i + 1]
                        break
                
                if task_id is None:
                    task_id = f"{processed_count // 10}"  # Group into tasks
                    episode_id = f"{processed_count}"
                
                task_name = f"task_{task_id}"
                
                print(f"✅ Found valid head camera video #{processed_count+1}: {key} (task {task_id}, episode {episode_id}, {len(video_data)} bytes)")
                
                # Process video: resize to 256x256 and downsample frames
                try:
                    processed_frames = _process_video_for_dataset(video_data)
                    
                    # Create trajectory entry
                    trajectory = {
                        'frames': processed_frames,  # Processed video frames
                        'actions': np.random.randn(50, 14),  # Placeholder actions (longer sequence)
                        'is_robot': True,
                        'task': f"AgiBotWorld task {task_id} - episode {episode_id}",
                        'optimal': 'optimal'
                    }
                except Exception as e:
                    print(f"  ❌ Failed to process video for {key}: {e}")
                    continue
                
                # Add to task data
                if task_name not in task_data:
                    task_data[task_name] = []
                task_data[task_name].append(trajectory)
                
                processed_count += 1
                
                # Print progress every 10 valid samples
                if processed_count % 10 == 0:
                    print(f"📊 Progress: {processed_count} valid videos found from {sample_count} samples examined")
                
            except Exception as e:
                print(f"Error processing sample {sample_count}: {e}")
                continue
        
        print(f"Processed {processed_count} valid samples from {sample_count} total samples")
        return task_data
        
    except Exception as e:
        print(f"Failed to load as streaming dataset: {e}")
        print("The AgiBotWorld dataset may require authentication or different access method.")
        print("Try:")
        print("1. Logging into HuggingFace: huggingface-cli login")
        print("2. Accepting the dataset license on the HuggingFace page")
        print("3. Using a local download instead")
        return {}


def _load_task_info(task_info_file: Path) -> List[Dict]:
    """Load task information from JSON file."""
    if not task_info_file.exists():
        print(f"Task info file not found: {task_info_file}")
        return []
    
    try:
        with open(task_info_file, 'r') as f:
            task_info = json.load(f)
        return task_info if isinstance(task_info, list) else [task_info]
    except Exception as e:
        print(f"Error loading task info from {task_info_file}: {e}")
        return []

def _process_video_for_dataset(video_input) -> np.ndarray:
    """Load video frames.
    
    Args:
        video_input: Either a file path (str/Path) or video bytes
        max_frames: Maximum number of frames to keep
        fps: Output video frame rate
        
    Returns:
        Processed video frames as numpy array
    """
    # Load video frames
    if isinstance(video_input, (str, Path)):
        # Load from file path
        video_path = str(video_input)
        cap = cv2.VideoCapture(video_path)
    else:
        # Load from bytes - save to temp file first
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(video_input)
            temp_path = temp_file.name
        cap = cv2.VideoCapture(temp_path)
    
    try:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for consistency
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        
        cap.release()
        
        # Clean up temp file if we created one
        if not isinstance(video_input, (str, Path)):
            os.unlink(temp_path)
        
        if len(frames) == 0:
            raise ValueError("No frames could be extracted from video")
        
        # Convert to numpy array
        frames_array = np.array(frames)
        
        return frames_array
        
    except Exception as e:
        cap.release()
        # Clean up temp files in case of error
        if not isinstance(video_input, (str, Path)) and 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        raise e


def _load_actions_from_h5(proprio_file: Path) -> np.ndarray:
    """Load actions from proprioceptive H5 file."""
    if not proprio_file.exists():
        print(f"Proprioceptive file not found: {proprio_file}")
        return np.array([])
    
    try:
        with h5py.File(proprio_file, 'r') as f:
            # According to AgiBotWorld docs, actions are stored under /action
            if 'action' in f:
                action_group = f['action']
                
                # Try to extract joint actions (most common for manipulation)
                if 'joint' in action_group and 'position' in action_group['joint']:
                    actions = action_group['joint']['position'][:]
                elif 'end' in action_group and 'position' in action_group['end']:
                    # Use end-effector positions if joint positions not available
                    end_positions = action_group['end']['position'][:]
                    end_orientations = action_group['end']['orientation'][:] if 'orientation' in action_group['end'] else None
                    
                    if end_orientations is not None:
                        # Concatenate position and orientation for full 6-DOF actions
                        # Reshape orientations from [N, 2, 4] to [N, 8] (both arms)
                        end_orientations_flat = end_orientations.reshape(end_orientations.shape[0], -1)
                        # Reshape positions from [N, 2, 3] to [N, 6] 
                        end_positions_flat = end_positions.reshape(end_positions.shape[0], -1)
                        actions = np.concatenate([end_positions_flat, end_orientations_flat], axis=1)
                    else:
                        actions = end_positions.reshape(end_positions.shape[0], -1)
                else:
                    print(f"No recognizable action data found in {proprio_file}")
                    return np.array([])
                
                return actions
            else:
                print(f"No action group found in {proprio_file}")
                return np.array([])
                
    except Exception as e:
        print(f"Error loading actions from {proprio_file}: {e}")
        return np.array([])