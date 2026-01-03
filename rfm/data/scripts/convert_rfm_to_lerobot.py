#!/usr/bin/env python3
"""
Convert RFM datasets to LeRobot format for OpenGVL compatibility.

This script:
1. Loads RFM datasets from HuggingFace Hub
2. Converts them to LeRobot format
3. Optionally uploads to HuggingFace Hub

LeRobot format requirements:
- Episodes organized by index
- Camera keys for multi-view support
- Task descriptions
- Video files in MP4 format
- Metadata files (info.json, episodes.jsonl, etc.)

Usage:
    python rfm/data/scripts/convert_rfm_to_lerobot.py \
        --dataset-name "aliangdw/metaworld" \
        --subset "metaworld_eval" \
        --output-dir "./lerobot_datasets/metaworld_eval" \
        --push-to-hub \
        --repo-id "your-username/metaworld-eval-lerobot"
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, create_repo, delete_repo, login, whoami
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from loguru import logger
from PIL import Image
from tqdm import tqdm

from rfm.data.datasets.helpers import load_frames_from_npz

def load_rfm_dataset(dataset_name: str, subset: Optional[str] = None) -> List[Dict]:
    """Load an RFM dataset from processed cache.
    
    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "aliangdw/metaworld")
        subset: Optional subset name
        
    Returns:
        List of trajectory dictionaries
    """
    logger.info(f"Loading RFM dataset: {dataset_name}/{subset}")
    
    # Load from processed cache using RFM_PROCESSED_DATASETS_PATH
    cache_dir = os.environ.get("RFM_PROCESSED_DATASETS_PATH", "")
    if not cache_dir:
        raise ValueError(
            "RFM_PROCESSED_DATASETS_PATH environment variable not set. "
            "Please set it to the directory containing your processed datasets."
        )
    
    # Build cache key (same format as preprocessing script)
    cache_key = f"{dataset_name}/{subset}" if subset else dataset_name
    individual_cache_dir = os.path.join(cache_dir, cache_key.replace("/", "_").replace(":", "_"))
    
    if not os.path.exists(individual_cache_dir):
        raise FileNotFoundError(
            f"Processed dataset cache not found at {individual_cache_dir}. "
            f"Please run preprocess_datasets.py first to create the cache."
        )
    
    # Load the processed dataset
    dataset_cache_dir = os.path.join(individual_cache_dir, "processed_dataset")
    if not os.path.exists(dataset_cache_dir):
        raise FileNotFoundError(f"Processed dataset not found at {dataset_cache_dir}")
    
    logger.info(f"Loading from processed cache: {dataset_cache_dir}")
    dataset = Dataset.load_from_disk(dataset_cache_dir, keep_in_memory=False)
    
    # Convert to list of dicts
    trajectories = []
    for idx, example in enumerate(tqdm(dataset, desc="Loading trajectories")):
        traj = {
            "id": example.get("id", f"trajectory_{idx}"),
            "task": example.get("task", ""),
            "frames": example.get("frames", ""),  # Processed datasets store npz path in frames field
            "is_robot": example.get("is_robot", True),
            "quality_label": example.get("quality_label", "successful"),
            "partial_success": example.get("partial_success", 1.0),
            "data_source": example.get("data_source", dataset_name),
        }
        trajectories.append(traj)
    
    logger.info(f"Loaded {len(trajectories)} trajectories")
    return trajectories




def save_frames_as_video(frames: np.ndarray, output_path: str, fps: int = 1) -> bool:
    """Save frames as MP4 video file.
    
    Args:
        frames: numpy array with shape (T, H, W, C) containing frames
        output_path: Path to save the video file
        fps: Frames per second for the video
        
    Returns:
        True if successful, False otherwise
    """
    if len(frames) == 0:
        logger.warning(f"No frames to save to {output_path}")
        return False
    
    # Ensure frames are uint8
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    
    # Get video properties
    height, width = frames.shape[1], frames.shape[2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames (convert RGB to BGR for OpenCV)
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    return True


def convert_to_lerobot_format(
    trajectories: List[Dict],
    output_dir: str,
    camera_key: str = "rgb",
    max_episodes: Optional[int] = None,
    repo_id: Optional[str] = None,
    robot_type: str = "unknown",
    fps: int = 1,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
    hub_token: Optional[str] = None,
    hub_private: bool = False,
    hub_delete_existing: bool = False,
) -> str:
    """Convert RFM trajectories to LeRobot v3.0 format using LeRobotDataset API.
    
    Args:
        trajectories: List of RFM trajectory dictionaries
        output_dir: Output directory for LeRobot dataset
        camera_key: Camera key name (default: "rgb")
        max_episodes: Maximum number of episodes to convert
        repo_id: Repository ID for the dataset (optional, defaults to output_dir name)
        robot_type: Robot type identifier
        fps: Frames per second for the dataset
        push_to_hub: Whether to push the dataset to HuggingFace Hub after conversion
        hub_repo_id: HuggingFace repository ID for upload (e.g., "username/dataset-name")
        hub_token: HuggingFace token (optional, uses existing login if not provided)
        hub_private: Whether the repository should be private
        hub_delete_existing: Whether to delete existing repository before pushing
        
    Returns:
        Path to the created LeRobot dataset
    """
    logger.info(f"Converting {len(trajectories)} trajectories to LeRobot v3.0 format using LeRobotDataset API...")
    
    output_path = Path(output_dir)
    
    # Use output_dir name as repo_id if not provided
    if repo_id is None:
        repo_id = output_path.name
    
    # Clean up existing directory if it exists (LeRobotDataset.create() requires a fresh directory)
    if output_path.exists():
        logger.info(f"Removing existing directory: {output_path}")
        import shutil
        shutil.rmtree(output_path)
    
    # Limit episodes if specified
    if max_episodes:
        trajectories = trajectories[:max_episodes]
    
    # Get video shape from first trajectory
    video_shape = None
    if trajectories:
        first_npz = trajectories[0].get("frames", "")
        if first_npz:
            sample_frames = load_frames_from_npz(first_npz)
            if len(sample_frames) > 0:
                h, w, c = sample_frames.shape[1], sample_frames.shape[2], sample_frames.shape[3] if len(sample_frames.shape) > 3 else 3
                video_shape = (h, w, c)
    
    if video_shape is None:
        video_shape = (480, 640, 3)  # Default reasonable size
        logger.warning(f"Could not determine video shape, using default: {video_shape}")
    
    # Create LeRobot dataset using the API
    # Note: LeRobotDataset.create() handles v3.0 format automatically
    # Only define observation features - LeRobotDataset automatically handles:
    # - episode_index, frame_index, index, timestamp, next.done
    # - task (special field, must be in each frame but not in features dict)
    # Specify root=output_path to use our custom output directory instead of default LeRobot home
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        root=str(output_path),  # Use our specified output directory
        features={
            f"observation.images.{camera_key}": {
                "dtype": "video",  # Use "video" dtype to ensure videos are created and uploaded
                "shape": video_shape,
                "names": ["height", "width", "channel"],
            },
        },
    )
    
    # Process trajectories and add to dataset
    episode_index = 0
    global_index = 0
    
    for traj_idx, traj in enumerate(tqdm(trajectories, desc="Converting trajectories")):
        # Get npz path (processed datasets store npz path in frames field as string)
        npz_path = traj.get("frames", "")
        if not npz_path or not isinstance(npz_path, str):
            logger.warning(f"Skipping trajectory {traj_idx}: frames path not found or invalid")
            continue
        
        # Load frames from npz
        frames = load_frames_from_npz(npz_path)
        
        if len(frames) == 0:
            logger.warning(f"Skipping trajectory {traj_idx}: no frames loaded from {npz_path}")
            continue
        
        # Get task description
        task = traj.get("task", "")
        
        # Add frames to dataset
        for frame_idx, frame in enumerate(frames):
            # Ensure frame is uint8 and correct shape
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            # Ensure 3D array (H, W, C)
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.ndim == 4:
                frame = frame[0]
            
            # Add frame to dataset
            # Note: LeRobotDataset automatically handles episode_index, frame_index, index, timestamp, next.done
            # 'task' is required in each frame (LeRobotDataset extracts it to episode metadata)
            dataset.add_frame({
                f"observation.images.{camera_key}": frame,
                "task": task,  # Required: LeRobotDataset extracts this to episode metadata
            })
            global_index += 1
        
        # Save episode (task is automatically extracted from episode_buffer populated by add_frame)
        dataset.save_episode()
        episode_index += 1
    
    # Finalize the dataset (required before push_to_hub)
    logger.info("Finalizing dataset...")
    dataset.finalize()
    
    logger.info(f"✅ Converted {episode_index} episodes to LeRobot format")
    logger.info(f"   Output directory: {output_path}")
    
    # Push to hub if requested
    if push_to_hub:
        if not hub_repo_id:
            # Generate repo_id from output_dir name
            hub_repo_id = output_path.name.replace("_", "-")
            logger.info(f"Using generated hub_repo_id: {hub_repo_id}")
        
        # Handle authentication
        if hub_token:
            login(token=hub_token)
        else:
            # Use existing login
            whoami()
            logger.info("Using existing HuggingFace login")
        
        # Check if repository exists and handle accordingly
        api = HfApi(token=hub_token)
        repo_exists = False
        try:
            api.repo_info(repo_id=hub_repo_id, repo_type="dataset")
            repo_exists = True
        except Exception:
            repo_exists = False
        
        if repo_exists:
            if hub_delete_existing:
                logger.info(f"Repository {hub_repo_id} exists. Deleting it first...")
                delete_repo(repo_id=hub_repo_id, repo_type="dataset", token=hub_token)
                logger.info(f"✅ Deleted existing repository {hub_repo_id}")
                repo_exists = False  # Mark as not existing after deletion
            else:
                logger.info(f"Repository {hub_repo_id} exists. Uploading will overwrite existing files...")
        
        # Update dataset's repo_id to match hub_repo_id (push_to_hub uses self.repo_id)
        dataset.repo_id = hub_repo_id
        
        # Create repository if it doesn't exist (push_to_hub will create it, but we do it explicitly for clarity)
        if not repo_exists:
            logger.info(f"Creating repository {hub_repo_id} on HuggingFace Hub...")
            create_repo(
                repo_id=hub_repo_id,
                repo_type="dataset",
                private=hub_private,
                exist_ok=True,
                token=hub_token,
            )
            logger.info(f"✅ Created repository {hub_repo_id}")
        
        # Verify videos and images exist before pushing
        videos_dir = output_path / "videos"
        images_dir = output_path / "images"
        
        if videos_dir.exists():
            video_count = sum(1 for _ in videos_dir.rglob("*.mp4"))
            logger.info(f"Found {video_count} video files in {videos_dir}")
        else:
            logger.warning(f"Videos directory not found: {videos_dir}")
        
        # Check if images folder exists and should be uploaded
        # Note: LeRobotDataset by default ignores "images/" folder, but we can override with allow_patterns
        upload_images = False
        if images_dir.exists():
            image_count = sum(1 for _ in images_dir.rglob("*.png")) + sum(1 for _ in images_dir.rglob("*.jpg"))
            if image_count > 0:
                logger.info(f"Found {image_count} image files in {images_dir}")
                upload_images = True
            else:
                logger.info(f"Images directory exists but is empty: {images_dir}")
        
        # Push to hub using LeRobotDataset API
        # Use upload_large_folder=True for video files which can be large
        logger.info(f"Pushing dataset to HuggingFace Hub: {hub_repo_id}")
        logger.info(f"  - push_videos: True")
        logger.info(f"  - upload_large_folder: True (for large video files)")
        if upload_images:
            logger.info(f"  - upload_images: True (overriding default ignore)")
        
        # Allow images folder to be uploaded if it exists (override default ignore_patterns)
        push_kwargs = {
            "private": hub_private,
            "push_videos": True,  # Explicitly enable video upload
            "upload_large_folder": True,  # Use large folder upload for videos (handles files > 10MB)
        }
        if upload_images:
            push_kwargs["allow_patterns"] = ["images/**"]  # Override default ignore of images/
        
        dataset.push_to_hub(**push_kwargs)
        
        logger.info(f"✅ Successfully pushed dataset to: https://huggingface.co/datasets/{hub_repo_id}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert RFM datasets to LeRobot format for OpenGVL"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="HuggingFace dataset identifier (e.g., 'aliangdw/metaworld')",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional subset name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for LeRobot dataset",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default="rgb",
        help="Camera key name (default: 'rgb')",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to convert (for testing)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub after conversion",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace repository ID for upload (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional, will use existing login if not provided)",
    )
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing repository on HuggingFace Hub before uploading (replaces everything)",
    )
    
    args = parser.parse_args()
    
    # Load RFM dataset
    trajectories = load_rfm_dataset(args.dataset_name, args.subset)
    
    if len(trajectories) == 0:
        logger.error("No trajectories loaded. Exiting.")
        return
    
    # Convert to LeRobot format (with optional push to hub)
    dataset_path = convert_to_lerobot_format(
        trajectories=trajectories,
        output_dir=args.output_dir,
        camera_key=args.camera_key,
        max_episodes=args.max_episodes,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.repo_id,
        hub_token=args.token,
        hub_private=args.private,
        hub_delete_existing=args.delete_existing,
    )
    
    if not args.push_to_hub:
        logger.info(f"Dataset ready at: {dataset_path}")
        logger.info("To upload to HuggingFace Hub, run with --push-to-hub --repo-id <your-repo-id>")


if __name__ == "__main__":
    main()