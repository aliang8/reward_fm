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
    flatten_task_data
)
from dataset_types import DatasetTrajectory, DatasetMetadata


def convert_dataset_to_hf_format(
    trajectories: List[Dict],
    create_hf_trajectory: Callable[[Dict, str, str, int, np.ndarray, Any, int, str], DatasetTrajectory],
    output_dir: str = "rfm_dataset",
    dataset_name: str = "UNKNOWN",
    max_trajectories: int = None,
    max_frames: int = 32,
    default_ranking: int = 0,
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
        # Generate unique preference embedding for each trajectory (all same for now)
        preference_embedding = np.random.randn(384).astype(np.float32)
        
        # Create output directory for this trajectory
        trajectory_dir = os.path.join(output_dir, f"trajectory_{trajectory_idx:04d}")
        os.makedirs(trajectory_dir, exist_ok=True)
        
        # Create trajectory for this trajectory
        sequence_name = f"trajectory_00"
        
        trajectory = create_hf_trajectory(
            demo=trajectory,
            output_dir=trajectory_dir,
            sequence_name=sequence_name,
            ranking=default_ranking,
            preference_embedding=preference_embedding,
            lang_model=lang_model,
            max_frames=max_frames,
            dataset_name=dataset_name
        )
        
        all_entries.append(trajectory)
    
    # Create HuggingFace dataset
    print(f"Creating HuggingFace dataset with {len(all_entries)} entries...")
    dataset = Dataset.from_list(all_entries)
    
    # Save dataset as a split
    dataset_path = os.path.join(output_dir, dataset_name.lower())
    dataset.save_to_disk(dataset_path)
    
    # Save metadata
    metadata = DatasetMetadata(
        dataset_name=dataset_name,
        num_entries=len(all_entries),
        max_trajectories=max_trajectories,
        max_frames=max_frames,
        default_ranking=default_ranking,
        data_source=dataset_name,
        created_at=str(np.datetime64('now'))
    )
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata.to_dict(), f, indent=2)
    

    
    print(f"{dataset_name} HuggingFace dataset created successfully!")
    print(f"Dataset saved to: {dataset_path}")
    print(f"Metadata saved to: {metadata_path}")
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
                commit_message=f"Add {dataset_name} dataset for VLM reward modeling"
            )
            print(f"✅ Successfully pushed dataset to: https://huggingface.co/datasets/{hub_repo_id}")
            print(f"📁 Dataset available as config: {dataset_name.lower()}")
            
            # Also push the metadata
            from huggingface_hub import HfApi
            api = HfApi(token=hub_token)
            api.upload_file(
                path_or_fileobj=metadata_path,
                path_in_repo=f"{dataset_name.lower()}_metadata.json",  # Include dataset name in metadata filename
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


def create_rfm_dataset(
    output_dir: str = "rfm_dataset",
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
    hub_token: Optional[str] = None
) -> None:
    """Create RFM dataset with multiple splits from existing datasets."""
    
    print("Crea  RFM dataset...")
    
    # Create output directory
    create_output_directory(output_dir)
    
    # Find all dataset splits
    splits = {}
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "dataset_info.json")):
            splits[item] = item_path
    
    if not splits:
        print("No dataset splits found. Please run dataset conversion first.")
        return
    
    print(f"Found {len(splits)} dataset splits: {list(splits.keys())}")
    
    # Load all splits
    datasets = {}
    for split_name, split_path in splits.items():
        try:
            dataset = Dataset.load_from_disk(split_path)
            datasets[split_name] = dataset
            print(f"Loaded {split_name}: {len(dataset)} entries")
        except Exception as e:
            print(f"Error loading {split_name}: {e}")
    
    if not datasets:
        print("No valid datasets found.")
        return
    
    # Create unified dataset
    from datasets import DatasetDict
    
    # DatasetDict automatically uses split names as config names
    dataset_dict = DatasetDict(datasets)
    
    # Save unified dataset
    unified_path = os.path.join(output_dir, "rfm_unified")
    dataset_dict.save_to_disk(unified_path)
    
    # Create metadata
    metadata = {
        "dataset_name": "RFM",
        "description": "Robot Foundation Model dataset with multiple data sources",
        "splits": list(datasets.keys()),
        "total_entries": sum(len(dataset) for dataset in datasets.values()),
        "split_entries": {name: len(dataset) for name, dataset in datasets.items()},
        "created_at": str(np.datetime64('now')),
    }
    
    metadata_path = os.path.join(output_dir, "rfm_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Unified RFM dataset created successfully!")
    print(f"Dataset saved to: {unified_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Total entries: {metadata['total_entries']}")
    print(f"Splits: {metadata['splits']}")
    
    # Push to HuggingFace Hub if requested
    if push_to_hub and hub_repo_id:
        print(f"\nPushing unified dataset to HuggingFace Hub: {hub_repo_id}")
        try:
            # Push the dataset to the hub
            dataset_dict.push_to_hub(
                hub_repo_id,
                token=hub_token,
                private=False,
                commit_message="Add unified RFM dataset with multiple data sources"
            )
            print(f"✅ Successfully pushed dataset to: https://huggingface.co/datasets/{hub_repo_id}")
            
            # Also push the metadata
            from huggingface_hub import HfApi
            api = HfApi(token=hub_token)
            api.upload_file(
                path_or_fileobj=metadata_path,
                path_in_repo="rfm_metadata.json",
                repo_id=hub_repo_id,
                repo_type="dataset"
            )
            print("✅ Successfully pushed metadata to hub")
            
        except Exception as e:
            print(f"❌ Error pushing to hub: {e}")
            print("Dataset was created locally but failed to push to hub")
    elif push_to_hub and not hub_repo_id:
        print("❌ push_to_hub=True but no hub_repo_id provided")


@dataclass
class DatasetConfig:
    """Config for dataset settings"""
    dataset_type: str = field(default="libero", metadata={"help": "Type of dataset to convert"})
    dataset_path: str = field(default="", metadata={"help": "Path to the dataset"})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Name of the dataset (defaults to dataset_type)"})


@dataclass
class OutputConfig:
    """Config for output settings"""
    output_dir: str = field(default="rfm_dataset", metadata={"help": "Output directory for the dataset"})
    max_trajectories: Optional[int] = field(default=None, metadata={"help": "Maximum number of trajectories to process (None for all)"})
    max_frames: int = field(default=32, metadata={"help": "Maximum number of frames per trajectory"})
    default_ranking: int = field(default=0, metadata={"help": "Default ranking value for all trajectories"})


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


@wrap()
def main(cfg: GenerateConfig):
    """Main function to convert any dataset to HuggingFace format."""
    
    # Get hub token from environment if not provided
    if cfg.hub.hub_token is None:
        cfg.hub.hub_token = os.getenv("HF_TOKEN")
    
    # Set dataset name if not provided
    if cfg.dataset.dataset_name is None:
        cfg.dataset.dataset_name = cfg.dataset.dataset_type.upper()
    
    # Import the appropriate dataset loader and trajectory creator
    if cfg.dataset.dataset_type == "libero":
        from libero_loader import load_libero_dataset
        from helpers import create_hf_trajectory
        # Load the trajectories using the loader
        task_data = load_libero_dataset(cfg.dataset.dataset_path)
        trajectories = flatten_task_data(task_data)
    elif cfg.dataset.dataset_type == "custom":
        # For custom datasets, you would need to provide a loader function
        raise NotImplementedError("Custom dataset loader not implemented yet. Please create a loader function.")
    else:
        raise ValueError(f"Unknown dataset type: {cfg.dataset.dataset_type}")
    
    # Convert dataset
    dataset = convert_dataset_to_hf_format(
        trajectories=trajectories,
        create_hf_trajectory=create_hf_trajectory,
        output_dir=cfg.output.output_dir,
        dataset_name=cfg.dataset.dataset_name,
        max_trajectories=cfg.output.max_trajectories,
        max_frames=cfg.output.max_frames,
        default_ranking=cfg.output.default_ranking,
        push_to_hub=cfg.hub.push_to_hub,
        hub_repo_id=cfg.hub.hub_repo_id,
        hub_token=cfg.hub.hub_token
    )
    
    print("Dataset conversion complete!")


if __name__ == "__main__":
    main() 