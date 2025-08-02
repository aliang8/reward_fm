#!/usr/bin/env python3
"""
Script to convert LIBERO dataset to HuggingFace format with the specified schema:
{
    "id": id for the item
    "task": language instruction in text,
    "lang_vector": np.array of sentence-transformers/all-MiniLM-L6-v2 model vector of size (384,),
    "data_source": dataset name (e.g., LIBERO),
    "frames": python list of strings of file paths, each traj is downsampled to at most 32 frames,
    "optimal": True/False python bool (optimal is if the traj is successful),
    "ranking": python int for the ranking of this trajectory among a set of ranked N trajectories,
    "preference_embedding": unique 384-length random vector that all N compared trajs share,
    "is_robot": True/False python bool for if this trajectory is a robot video,
}
"""

import h5py
import json
import os
import numpy as np
from PIL import Image
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import shutil
import imageio.v2 as imageio
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import uuid
from tqdm import tqdm
import pickle
import argparse


def load_libero_dataset(base_path: str = "LIBERO/libero/datasets/libero_90") -> Dict[str, List[Dict]]:
    """Load LIBERO dataset from HDF5 files and organize by task."""
    
    print(f"Loading LIBERO dataset from: {base_path}")
    
    task_data = {}
    
    # Find all HDF5 files in the base path
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"LIBERO dataset path not found: {base_path}")
    
    hdf5_files = list(base_path.glob("*.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    for file_path in hdf5_files:
        task_name = file_path.stem  # Remove .hdf5 extension
        print(f"Loading task: {task_name}")
        
        with h5py.File(file_path, 'r') as f:
            if 'data' not in f:
                print(f"No 'data' group in {task_name}")
                continue
            
            data_group = f['data']
            demos = []
            
            for demo_key in data_group.keys():
                demo = data_group[demo_key]
                if isinstance(demo, h5py.Group):
                    # Extract trajectory data
                    demo_info = {
                        'demo_id': demo_key,
                        'task_name': task_name,
                        'file_path': str(file_path),
                        'trajectory_length': None,
                        'frames': [],
                        'actions': [],
                        'rewards': []
                    }
                    
                    # Get trajectory length from observations
                    if 'obs' in demo and 'agentview_rgb' in demo['obs']:
                        demo_info['trajectory_length'] = demo['obs']['agentview_rgb'].shape[0]
                        demo_info['frames'] = demo['obs']['agentview_rgb'][:]  # RGB frames
                    
                    # Get actions if available
                    if 'actions' in demo:
                        demo_info['actions'] = demo['actions'][:]
                    
                    # Get rewards if available
                    if 'rewards' in demo:
                        demo_info['rewards'] = demo['rewards'][:]
                    
                    # Assume all LIBERO demos are successful
                    demo_info['success'] = True
                    
                    demos.append(demo_info)
            
            task_data[task_name] = demos
            print(f"  Loaded {len(demos)} demos for {task_name}")
    
    print(f"Loaded {sum(len(demos) for demos in task_data.values())} demos from {len(task_data)} tasks")
    return task_data


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
    if len(frames) <= max_frames:
        return frames
    
    # Use linear interpolation to downsample
    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return frames[indices]


def create_trajectory_sequence(demo: Dict, output_dir: str, sequence_name: str, max_frames: int = 32) -> List[str]:
    """Create a trajectory sequence from demo frames and save as images."""
    
    sequence_dir = os.path.join(output_dir, sequence_name)
    os.makedirs(sequence_dir, exist_ok=True)
    
    # Downsample frames
    frames = downsample_frames(demo['frames'], max_frames)
    
    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = os.path.join(sequence_dir, f"frame_{i:02d}.jpg")
        saved_path = save_frame_as_image(frame, frame_path)
        frame_paths.append(saved_path)
    
    return frame_paths


def create_ranking_groups(task_data: Dict[str, List[Dict]], num_groups: int = 100, trajectories_per_group: int = 5) -> List[List[Dict]]:
    """Create ranking groups where each group contains multiple trajectories to be ranked."""
    
    ranking_groups = []
    
    # Collect all demos
    all_demos = []
    for task_name, demos in task_data.items():
        for demo in demos:
            demo['task_name'] = task_name
            all_demos.append(demo)
    
    # Create ranking groups
    for group_idx in range(num_groups):
        if len(all_demos) < trajectories_per_group:
            break
        
        # Randomly sample trajectories for this group
        group_demos = random.sample(all_demos, min(trajectories_per_group, len(all_demos)))
        
        # Remove selected demos to avoid duplicates
        for demo in group_demos:
            all_demos.remove(demo)
        
        ranking_groups.append(group_demos)
    
    print(f"Created {len(ranking_groups)} ranking groups")
    return ranking_groups


def create_hf_dataset_entry(
    demo: Dict, 
    output_dir: str, 
    sequence_name: str, 
    ranking: int, 
    preference_embedding: np.ndarray,
    lang_model,
    max_frames: int = 32
) -> Dict:
    """Create a single HuggingFace dataset entry."""
    
    # Create trajectory sequence
    frame_paths = create_trajectory_sequence(demo, output_dir, sequence_name, max_frames)
    
    # Generate unique ID
    unique_id = str(uuid.uuid4())
    
    # Parse the original file path to extract scene and task info
    file_path = demo['file_path']
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
    
    # Extract demo and trajectory info from sequence_name
    # sequence_name is like "trajectory_00"
    demo_trajectory_info = sequence_name
    
    # Create clean task description (without "Robot trajectory for task:")
    task_description = task_string
    
    # Generate language embedding
    lang_vector = lang_model.encode(task_description)
    
    # Create metadata dictionary
    metadata = {
        "original_file": file_name,
        "scene": scene_part,
        "demo_id": demo['demo_id'],
        "trajectory_info": demo_trajectory_info,
        "trajectory_length": demo.get('trajectory_length'),
        "file_path": file_path
    }
    
    # Create dataset entry
    entry = {
        "id": unique_id,
        "task": task_description,
        "lang_vector": lang_vector,
        "data_source": "LIBERO",
        "frames": frame_paths,
        "optimal": demo.get('success', True),  # Assume LIBERO demos are optimal
        "ranking": ranking,
        "preference_embedding": preference_embedding,
        "is_robot": True,  # LIBERO contains robot trajectories
        "metadata": metadata,
    }
    
    return entry


def create_libero_hf_dataset(
    task_data: Dict[str, List[Dict]], 
    output_dir: str = "libero_hf_dataset",
    num_ranking_groups: int = 100,
    trajectories_per_group: int = 5,
    max_frames: int = 32,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
    hub_token: Optional[str] = None
) -> None:
    """Create the complete LIBERO HuggingFace dataset."""
    
    print(f"Creating LIBERO HuggingFace dataset...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sentence transformer model
    print("Loading sentence transformer model...")
    lang_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create ranking groups
    ranking_groups = create_ranking_groups(task_data, num_ranking_groups, trajectories_per_group)
    
    # Process each ranking group
    all_entries = []
    
    for group_idx, group_demos in enumerate(tqdm(ranking_groups, desc="Processing ranking groups")):
        # Generate unique preference embedding for this group
        preference_embedding = np.random.randn(384).astype(np.float32)
        
        # Create output directory for this group
        group_dir = os.path.join(output_dir, f"group_{group_idx:04d}")
        os.makedirs(group_dir, exist_ok=True)
        
        # Set all rankings to 0 for LIBERO dataset
        # Create entries for each trajectory in the group
        for demo_idx, demo in enumerate(group_demos):
            sequence_name = f"trajectory_{demo_idx:02d}"
            
            entry = create_hf_dataset_entry(
                demo=demo,
                output_dir=group_dir,
                sequence_name=sequence_name,
                ranking=0,  # Set all rankings to 0 for LIBERO
                preference_embedding=preference_embedding,
                lang_model=lang_model,
                max_frames=max_frames
            )
            
            all_entries.append(entry)
    
    # Create HuggingFace dataset
    print(f"Creating HuggingFace dataset with {len(all_entries)} entries...")
    dataset = Dataset.from_list(all_entries)
    
    # Save dataset
    dataset_path = os.path.join(output_dir, "libero_hf_dataset")
    dataset.save_to_disk(dataset_path)
    
    # Save metadata
    metadata = {
        "num_entries": len(all_entries),
        "num_ranking_groups": len(ranking_groups),
        "trajectories_per_group": trajectories_per_group,
        "max_frames": max_frames,
        "data_source": "LIBERO",
        "created_at": str(np.datetime64('now')),
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"LIBERO HuggingFace dataset created successfully!")
    print(f"Dataset saved to: {dataset_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Total entries: {len(all_entries)}")
    print(f"Ranking groups: {len(ranking_groups)}")
    
    # Push to HuggingFace Hub if requested
    if push_to_hub and hub_repo_id:
        print(f"\nPushing dataset to HuggingFace Hub: {hub_repo_id}")
        try:
            # Push the dataset to the hub
            dataset.push_to_hub(
                hub_repo_id,
                token=hub_token,
                private=False,  # Set to True if you want a private repository
                commit_message="Add LIBERO dataset for VLM reward modeling"
            )
            print(f"✅ Successfully pushed dataset to: https://huggingface.co/datasets/{hub_repo_id}")
            
            # Also push the metadata
            from huggingface_hub import HfApi
            api = HfApi(token=hub_token)
            api.upload_file(
                path_or_fileobj=metadata_path,
                path_in_repo="metadata.json",
                repo_id=hub_repo_id,
                repo_type="dataset"
            )
            print("✅ Successfully pushed metadata to hub")
            
        except Exception as e:
            print(f"❌ Error pushing to hub: {e}")
            print("Dataset was created locally but failed to push to hub")
    elif push_to_hub and not hub_repo_id:
        print("❌ push_to_hub=True but no hub_repo_id provided")
    
    return dataset


def main():
    """Main function to create the LIBERO HuggingFace dataset."""
    
    parser = argparse.ArgumentParser(description="Create LIBERO HuggingFace dataset")
    parser.add_argument("--libero_path", type=str, default="LIBERO/libero/datasets/libero_90",
                       help="Path to LIBERO dataset")
    parser.add_argument("--output_dir", type=str, default="libero_hf_dataset",
                       help="Output directory for the dataset")
    parser.add_argument("--num_groups", type=int, default=100,
                       help="Number of ranking groups to create")
    parser.add_argument("--trajectories_per_group", type=int, default=5,
                       help="Number of trajectories per ranking group")
    parser.add_argument("--max_frames", type=int, default=32,
                       help="Maximum number of frames per trajectory")
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push dataset to HuggingFace Hub")
    parser.add_argument("--hub_repo_id", type=str, default=None,
                       help="HuggingFace Hub repository ID (e.g., 'username/libero-vlm-dataset')")
    parser.add_argument("--hub_token", type=str, default=None,
                       help="HuggingFace Hub token (or set HF_TOKEN environment variable)")
    
    args = parser.parse_args()
    
    # Get hub token from environment if not provided
    if args.hub_token is None:
        args.hub_token = os.getenv("HF_TOKEN")
    
    # Load LIBERO dataset
    task_data = load_libero_dataset(args.libero_path)
    
    if not task_data:
        print("No LIBERO data found. Please check the path.")
        return
    
    # Create HuggingFace dataset
    dataset = create_libero_hf_dataset(
        task_data=task_data,
        output_dir=args.output_dir,
        num_ranking_groups=args.num_groups,
        trajectories_per_group=args.trajectories_per_group,
        max_frames=args.max_frames,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
        hub_token=args.hub_token
    )
    
    print("Dataset creation complete!")


if __name__ == "__main__":
    main() 