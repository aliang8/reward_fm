#!/usr/bin/env python3
"""
Helper functions for RFM model dataset conversion.
Contains utility functions for processing frames, saving images, and managing data.
"""

import os
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from pathlib import Path
import uuid
from sentence_transformers import SentenceTransformer
import cv2


def save_frame_as_image(frame_data: np.ndarray, output_path: str) -> str:
    """Save a frame as a JPG image."""
    # Convert from HDF5 format to PIL Image
    if frame_data.dtype != np.uint8:
        frame_data = (frame_data * 255).astype(np.uint8)
    
    image = Image.fromarray(frame_data)
    image.save(output_path, "JPEG", quality=95)
    return output_path


def downsample_frames(frames: np.ndarray, max_frames: int = 32) -> np.ndarray:
    """Downsample frames to at most max_frames using linear interpolation."""
    # If max_frames is -1, don't downsample
    if max_frames == -1:
        return frames
    
    if len(frames) <= max_frames:
        return frames
    
    # Use linear interpolation to downsample
    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return frames[indices]


def create_trajectory_video(frames: np.ndarray, output_dir: str, sequence_name: str, max_frames: int = -1, fps: int = 10) -> bytes:
    """Create a trajectory video from frames and return as bytes."""
    
    # Downsample frames
    frames = downsample_frames(frames, max_frames)
    
    # Get video dimensions from first frame
    if len(frames) == 0:
        raise ValueError("No frames provided for video creation")
    
    height, width = frames[0].shape[:2]
    
    # Create temporary video file
    temp_video_path = os.path.join(output_dir, f"temp_{sequence_name}.mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame in frames:
        # Ensure frame is in uint8 format
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        video_writer.write(frame)
    
    # Release video writer
    video_writer.release()
    
    # Read video file as bytes
    with open(temp_video_path, 'rb') as f:
        video_bytes = f.read()
    
    # Clean up temporary file
    os.remove(temp_video_path)
    
    return video_bytes


def create_trajectory_sequence(frames: List[str], output_dir: str, sequence_name: str, max_frames: int = -1) -> List[str]:
    """Create a trajectory sequence from frames and save as images."""
    
    sequence_dir = os.path.join(output_dir, sequence_name)
    os.makedirs(sequence_dir, exist_ok=True)
    
    # Downsample frames
    frames = downsample_frames(frames, max_frames)
    
    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = os.path.join(sequence_dir, f"frame_{i:02d}.jpg")
        saved_path = save_frame_as_image(frame, frame_path)
        frame_paths.append(saved_path)
    
    return frame_paths


def generate_unique_id() -> str:
    """Generate a unique UUID for dataset entries."""
    return str(uuid.uuid4())


def create_hf_trajectory(
    traj_dict: Dict,
    output_dir: str,
    sequence_name: str,
    lang_model: SentenceTransformer,
    max_frames: int = -1,
    dataset_name: str = "",
    use_video: bool = True,
    fps: int = 10
) -> Dict:
    """Create a HuggingFace dataset trajectory."""
    
    if use_video:
        # Create trajectory video
        video_bytes = create_trajectory_video(traj_dict['frames'], output_dir, sequence_name, max_frames, fps)
        frames = [video_bytes]  # Store video bytes as single frame entry
    else:
        # Create trajectory sequence (original behavior)
        frame_paths = create_trajectory_sequence(traj_dict['frames'], output_dir, sequence_name, max_frames)
        frames = frame_paths
    
    # Generate unique ID
    unique_id = generate_unique_id()
    
    # Get task description
    task_description = traj_dict["task"]
    
    # Generate language embedding
    lang_vector = lang_model.encode(task_description)

    # Create dataset trajectory
    trajectory = {
        "id": unique_id,
        "task": task_description,
        "lang_vector": lang_vector,
        "data_source": dataset_name,
        "frames": frames,
        "optimal": traj_dict['optimal'],
        "is_robot": traj_dict['is_robot'],
    }
    
    return trajectory


def load_sentence_transformer_model() -> SentenceTransformer:
    """Load the sentence transformer model for language embeddings."""
    return SentenceTransformer('all-MiniLM-L6-v2')


def create_output_directory(output_dir: str) -> None:
    """Create the output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)


def flatten_task_data(task_data: Dict[str, List[Dict]]) -> List[Dict]:
    """Flatten task data into a list of trajectories."""
    all_trajectories = []
    for task_name, trajectories in task_data.items():
        for trajectory in trajectories:
            trajectory['task_name'] = task_name
            all_trajectories.append(trajectory)
    return all_trajectories 