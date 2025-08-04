#!/usr/bin/env python3
"""
Main dataset converter that can convert any dataset to HuggingFace format for RFM model training.
This is a generic converter that works with any dataset-specific loader.
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm
from dataclasses import dataclass, field
from pyrallis import wrap
from helpers import (
    load_sentence_transformer_model, 
    create_output_directory,
    flatten_task_data,
    create_hf_trajectory
)
from dataset_types import Trajectory
from functools import partial

@dataclass
class DatasetConfig:
    """Config for dataset settings"""
    dataset_path: str = field(default="", metadata={"help": "Path to the dataset"})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Name of the dataset (defaults to dataset_type)"})


@dataclass
class OutputConfig:
    """Config for output settings"""
    output_dir: str = field(default="rfm_dataset", metadata={"help": "Output directory for the dataset"})
    max_trajectories: Optional[int] = field(default=None, metadata={"help": "Maximum number of trajectories to process (None for all)"})
    max_frames: int = field(default=-1, metadata={"help": "Maximum number of frames per trajectory (-1 for no downsampling)"})
    use_video: bool = field(default=True, metadata={"help": "Use MP4 videos instead of individual frame images"})
    fps: int = field(default=10, metadata={"help": "Frames per second for video creation"})


@dataclass
class HubConfig:
    """Config for HuggingFace Hub settings"""
    push_to_hub: bool = field(default=False, metadata={"help": "Push dataset to HuggingFace Hub"})
    hub_repo_id: Optional[str] = field(default=None, metadata={"help": "HuggingFace Hub repository ID"})
    hub_token: Optional[str] = field(default=None, metadata={"help": "HuggingFace Hub token (or set HF_TOKEN environment variable)"})


@dataclass
class GenerateConfig:
    """Main configuration for dataset generation"""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    hub: HubConfig = field(default_factory=HubConfig)

def convert_dataset_to_hf_format(
    trajectories: List[Dict],
    create_hf_trajectory: Callable[[Dict, str, str, int, Any, int, str], Trajectory],
    output_dir: str = "rfm_dataset",
    dataset_name: str = "",
    max_trajectories: int = None,
    max_frames: int = -1,
    use_video: bool = True,
    fps: int = 10,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
    hub_token: Optional[str] = None
) -> Dataset:
    """Convert a list of trajectories to HuggingFace format."""
    
    print(f"Converting {dataset_name} dataset to HuggingFace format...")
    
    # Create output directory
    create_output_directory(output_dir)
    
    # Load sentence transformer model
    print("Loading sentence transformer model...")
    lang_model = load_sentence_transformer_model()
    
    # Validate input
    if not trajectories:
        raise ValueError(f"No trajectories provided for {dataset_name} dataset.")
    
    print(f"Processing {len(trajectories)} trajectories")
    
    # Limit trajectories if specified
    if max_trajectories is not None:
        trajectories = trajectories[:max_trajectories]
    
    # Process each trajectory
    all_entries = []
    
    for trajectory_idx, trajectory in enumerate(tqdm(trajectories, desc="Processing trajectories")):            
        # Create output directory for this trajectory
        trajectory_dir = os.path.join(output_dir, f"trajectory_{trajectory_idx:04d}")
        os.makedirs(trajectory_dir, exist_ok=True)
        sequence_name = f"trajectory_{trajectory_idx:04d}"
        
        trajectory = create_hf_trajectory(
            traj_dict=trajectory,
            output_dir=trajectory_dir,
            sequence_name=sequence_name,
            lang_model=lang_model,
            max_frames=max_frames,
            dataset_name=dataset_name,
            use_video=use_video,
            fps=fps
        )
        
        all_entries.append(trajectory)
    
    # Create HuggingFace dataset
    print(f"Creating HuggingFace dataset with {len(all_entries)} entries...")
    dataset = Dataset.from_list(all_entries)
    
    # Save dataset as a split
    dataset_path = os.path.join(output_dir, dataset_name.lower())
    dataset.save_to_disk(dataset_path)
    
    print(f"{dataset_name} HuggingFace dataset created successfully!")
    print(f"Dataset saved to: {dataset_path}")
    print(f"Total entries: {len(all_entries)}")
    
    # Push to HuggingFace Hub if requested
    if push_to_hub and hub_repo_id:
        print(f"\nPushing dataset to HuggingFace Hub: {hub_repo_id}")
        try:
            # Push the dataset to the hub with dataset name as config name
            dataset.push_to_hub(
                hub_repo_id,
                config_name=dataset_name.lower(),  # Use dataset name as config name
                token=hub_token,
                private=False,
                commit_message=f"Add {dataset_name} dataset for RFM training"
            )
            print(f"✅ Successfully pushed dataset to: https://huggingface.co/datasets/{hub_repo_id}")
            print(f"📁 Dataset available as config: {dataset_name.lower()}")
            
        except Exception as e:
            print(f"❌ Error pushing to hub: {e}")
            print("Dataset was created locally but failed to push to hub")
    elif push_to_hub and not hub_repo_id:
        print("❌ push_to_hub=True but no hub_repo_id provided")
    
    return dataset

@wrap()
def main(cfg: GenerateConfig):
    """Main function to convert any dataset to HuggingFace format."""
    
    # Get hub token from environment if not provided
    if cfg.hub.hub_token is None:
        cfg.hub.hub_token = os.getenv("HF_TOKEN")
    
    # Import the appropriate dataset loader and trajectory creator
    if "libero" in cfg.dataset.dataset_name:
        from libero_loader import load_libero_dataset
        # Load the trajectories using the loader
        task_data = load_libero_dataset(cfg.dataset.dataset_path)
        trajectories = flatten_task_data(task_data)
    else:
        raise ValueError(f"Unknown dataset type: {cfg.dataset.dataset_name}")
    
    # Convert dataset
    convert_dataset_to_hf_format(
        trajectories=trajectories,
        create_hf_trajectory=partial(create_hf_trajectory, dataset_name=cfg.dataset.dataset_name, use_video=cfg.output.use_video, fps=cfg.output.fps),
        output_dir=cfg.output.output_dir,
        dataset_name=cfg.dataset.dataset_name,
        max_trajectories=cfg.output.max_trajectories,
        max_frames=cfg.output.max_frames,
        use_video=cfg.output.use_video,
        fps=cfg.output.fps,
        push_to_hub=cfg.hub.push_to_hub,
        hub_repo_id=cfg.hub.hub_repo_id,
        hub_token=cfg.hub.hub_token
    )
    
    print("Dataset conversion complete!")


if __name__ == "__main__":
    main() 