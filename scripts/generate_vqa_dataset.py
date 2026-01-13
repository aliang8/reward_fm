#!/usr/bin/env python3
"""
Generate a static VQA dataset from RFM trajectories using RFMDataset.

This script generates a HuggingFace dataset with VQA-style prompts and answers,
storing references to .npz files and frame indices instead of loading actual frames.
The generated dataset can be used for training with train_vqa_sft.py.

The script iterates through the dataset for a specified number of epochs. Total samples
generated = dataset_size × num_epochs. For example, with a 43k dataset and 2.0 epochs,
you'll get 86k samples.

Usage:
    python scripts/generate_vqa_dataset.py \\
        --num_epochs 1.0 \\
        --output_path /path/to/output/dataset \\
        --seed 42 \\
        --num_workers 4 \\
        --config_overrides data.train_datasets=[jesbu1_roboreward_rfm_roboreward_train]
"""

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union
from rfm.utils.setup_utils import (
    setup_dataset,
)

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from rfm.data.datasets.base import resolve_dataset_keys
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rfm.configs.experiment_configs import (
    ExperimentConfig,
    ModelConfig,
    PEFTConfig,
    DataConfig,
    TrainingConfig,
    LossConfig,
    LoggingConfig,
    SaveBestConfig,
    CustomEvaluationConfig,
)
from rfm.data.dataset_types import PreferenceSample, ProgressSample
from rfm.data.datasets import RFMDataset
from rfm.utils.config_utils import convert_hydra_to_dataclass

# VQA prompt templates (from rfm/data/collators/vqa.py)
RESPONSE_PREFIX = "ANS:"

PREFERENCE_PROMPT_TEMPLATE = """Given these two robot videos, which one makes the most progress towards solving the task, Video 1 or 2? Format your answer as: {response_prefix} 1/2

Task: {task}"""

PROGRESS_PROMPT_TEMPLATE = """Given the task, assign an integer-valued progress score from 0 to 100 for the robot in the video in the format: ANS: score
End of episode progress should be judged only on the final state, without time limits.
Rubric for end-of-episode progress (judge only the final state without time limits):
0 - No Progress: Final state shows no goal-relevant change for the command.
100 - Perfect Completion: Final state satisfies all requirements to solve the task.
Anything in between represents partial progress towards the goal.

Task: {task}"""


def sample_to_vqa_dict(sample: Union[PreferenceSample, ProgressSample]) -> Dict[str, Any]:
    """
    Convert RFM sample (PreferenceSample or ProgressSample) to VQA training format.
    
    Args:
        sample: PreferenceSample or ProgressSample from RFMDataset
        
    Returns:
        Dictionary with VQA training fields
    """
    if sample.sample_type == "preference":
        # Extract trajectory information
        chosen_traj = sample.chosen_trajectory
        rejected_traj = sample.rejected_trajectory
        
        # Randomly decide which trajectory comes first
        chosen_is_first = random.random() < 0.5
        
        if chosen_is_first:
            first_traj = chosen_traj
            second_traj = rejected_traj
            answer = "1"
        else:
            first_traj = rejected_traj
            second_traj = chosen_traj
            answer = "2"
        
        # Build prompt
        prompt = PREFERENCE_PROMPT_TEMPLATE.format(
            response_prefix=RESPONSE_PREFIX,
            task=chosen_traj.task
        )
        
        # Return VQA dict with all fields (some will be None)
        return {
            "sample_type": "preference",
            "first_npz_path": first_traj.npz_path,
            "first_frame_indices": first_traj.frame_indices,
            "first_frames_shape": list(first_traj.frames_shape) if first_traj.frames_shape else None,
            "second_npz_path": second_traj.npz_path,
            "second_frame_indices": second_traj.frame_indices,
            "second_frames_shape": list(second_traj.frames_shape) if second_traj.frames_shape else None,
            "npz_path": None,  # Not used for preference
            "frame_indices": None,  # Not used for preference
            "frames_shape": None,  # Not used for preference
            "prompt": prompt,
            "answer": answer,
            "task": chosen_traj.task,
            "data_source": chosen_traj.data_source,
            "data_gen_strategy": sample.data_gen_strategy,
            "resample_attempts": sample.resample_attempts,
            "chosen_is_first": chosen_is_first,
            "target_progress": None,  # Not used for preference
        }
    
    elif sample.sample_type == "progress":
        # Extract trajectory information
        traj = sample.trajectory
        
        # Get target progress (last frame progress)
        if traj.target_progress and len(traj.target_progress) > 0:
            # Get the last non-padded progress value
            target_progress = traj.target_progress[-1]
            # Convert to integer score (0-100)
            if isinstance(target_progress, (float, np.floating)):
                progress_score = int(round(target_progress * 100))
            else:
                progress_score = int(target_progress)
            # Clamp to 0-100
            progress_score = max(0, min(100, progress_score))
        else:
            progress_score = 0
        
        # Build prompt
        prompt = PROGRESS_PROMPT_TEMPLATE.format(task=traj.task)
        
        # Return VQA dict with all fields (some will be None)
        return {
            "sample_type": "progress",
            "npz_path": traj.npz_path,
            "frame_indices": traj.frame_indices,
            "frames_shape": list(traj.frames_shape) if traj.frames_shape else None,
            "first_npz_path": None,  # Not used for progress
            "first_frame_indices": None,  # Not used for progress
            "first_frames_shape": None,  # Not used for progress
            "second_npz_path": None,  # Not used for progress
            "second_frame_indices": None,  # Not used for progress
            "second_frames_shape": None,  # Not used for progress
            "prompt": prompt,
            "answer": str(progress_score),
            "task": traj.task,
            "data_source": traj.data_source,
            "data_gen_strategy": sample.data_gen_strategy,
            "resample_attempts": sample.resample_attempts,
            "chosen_is_first": None,  # Not used for progress
            "target_progress": progress_score,
        }
    
    else:
        raise ValueError(f"Unknown sample type: {sample.sample_type}")


def vqa_collate_fn(batch: List[Union[PreferenceSample, ProgressSample]]) -> List[Dict[str, Any]]:
    """
    Collate function to convert RFM samples to VQA format.
    
    Args:
        batch: List of PreferenceSample or ProgressSample objects
        
    Returns:
        List of VQA dictionaries
    """
    return [sample_to_vqa_dict(sample) for sample in batch]


def save_batch_to_temp(samples: List[Dict[str, Any]], temp_dir: str, batch_idx: int) -> str:
    """
    Save a batch of samples to a temporary HuggingFace dataset file.
    
    Args:
        samples: List of VQA sample dictionaries
        temp_dir: Temporary directory to save to
        batch_idx: Batch index for filename
        
    Returns:
        Path to saved temporary dataset
    """
    temp_path = os.path.join(temp_dir, f"batch_{batch_idx:06d}")
    dataset = Dataset.from_list(samples)
    dataset.save_to_disk(temp_path)
    return temp_path


def main():
    parser = argparse.ArgumentParser(description="Generate VQA dataset from RFM trajectories")
    
    # Dataset generation arguments
    parser.add_argument(
        "--num_epochs",
        type=float,
        required=True,
        help="Number of epochs to iterate through the dataset (can be fractional, e.g., 0.5 for half epoch)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the generated HuggingFace dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    # Config arguments
    parser.add_argument(
        "--config_name",
        type=str,
        default="config",
        help="Name of the config file (without .yaml extension)",
    )
    parser.add_argument(
        "--config_overrides",
        type=str,
        nargs="*",
        default=[],
        help="Config overrides in Hydra format (e.g., data.max_frames=16)",
    )
    
    # Evaluation mode
    parser.add_argument(
        "--eval_mode",
        action="store_true",
        help="Generate evaluation dataset (no augmentations)",
    )
    
    # DataLoader arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for DataLoader (not training batch size)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers for parallel processing",
    )
    
    # Incremental saving
    parser.add_argument(
        "--save_batch_size",
        type=int,
        default=50000,
        help="Save dataset incrementally every N samples to avoid OOM",
    )
    
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 100)
    print("VQA Dataset Generation Configuration")
    print("=" * 100)
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Output path: {args.output_path}")
    print(f"Seed: {args.seed}")
    print(f"Evaluation mode: {args.eval_mode}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Save batch size: {args.save_batch_size}")
    print("=" * 100)

    # Load Hydra config
    print("Loading Hydra configuration...")
    config_path = str(project_root / "rfm" / "configs")
    
    # Register config schemas
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=ExperimentConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="peft", name="base_peft", node=PEFTConfig)
    cs.store(group="data", name="base_data", node=DataConfig)
    cs.store(group="training", name="base_training", node=TrainingConfig)
    cs.store(group="loss", name="base_loss", node=LossConfig)
    cs.store(group="logging", name="base_logging", node=LoggingConfig)
    cs.store(group="save_best", name="base_save_best", node=SaveBestConfig)
    cs.store(group="custom_evaluation", name="base_custom_evaluation", node=CustomEvaluationConfig)

    with initialize_config_dir(config_dir=config_path, version_base=None):
        hydra_config = compose(
            config_name=args.config_name,
            overrides=args.config_overrides,
        )
    
    # Convert to dataclass
    config = convert_hydra_to_dataclass(hydra_config, ExperimentConfig)
    
    print(f"Loaded config: {args.config_name}")
    print(f"Sample type ratio: {config.data.sample_type_ratio}")
    print(f"Max frames: {config.data.max_frames}")

    print("Resolving dataset keys")
    config.data.train_datasets = resolve_dataset_keys(config.data.train_datasets, split="train")
    print(f"Resolved train datasets: {config.data.train_datasets}")

    if args.eval_mode:
        config.data.eval_datasets = resolve_dataset_keys(config.data.train_datasets, split="eval")
        config.data.train_datasets = config.data.eval_datasets
        print(f"Resolved eval datasets: {config.data.eval_datasets}")
    
    print(f"Train datasets: {config.data.train_datasets}")
    print(f"Eval datasets: {config.data.eval_datasets}")

    # Create RFMDataset with return_npz_paths=True
    print("\nCreating RFMDataset...")
    dataset = setup_dataset(cfg=config.data, is_eval=args.eval_mode, return_npz_paths=True)
    
    # Calculate number of samples based on epochs
    dataset_size = len(dataset)
    num_samples = int(dataset_size * args.num_epochs)
    
    print(f"Dataset size: {dataset_size} trajectories")
    print(f"Epochs: {args.num_epochs}")
    print(f"Total samples to generate: {num_samples} ({args.num_epochs} × {dataset_size})")

    # Create DataLoader
    print(f"Creating DataLoader with {args.num_workers} workers...")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=vqa_collate_fn,
        shuffle=False,
        pin_memory=False,  # Not needed since we're not loading tensors
    )

    # Generate samples
    print("\nGenerating VQA samples...")
    all_samples = []
    temp_dir = tempfile.mkdtemp(prefix="vqa_dataset_")
    temp_files = []
    batch_counter = 0
    
    try:
        progress_bar = tqdm(
            total=num_samples,
            desc="Generating samples",
            unit="samples",
        )
        
        samples_generated = 0
        epoch = 0
        
        # Loop over multiple epochs if needed
        while samples_generated < num_samples:
            for batch in dataloader:
                all_samples.extend(batch)
                samples_generated += len(batch)
                progress_bar.update(len(batch))
                
                # Incremental save
                if len(all_samples) >= args.save_batch_size:
                    print(f"\nSaving batch {batch_counter} ({len(all_samples)} samples)...")
                    temp_path = save_batch_to_temp(all_samples, temp_dir, batch_counter)
                    temp_files.append(temp_path)
                    all_samples = []
                    batch_counter += 1
                
                # Stop if we've reached the target number
                if samples_generated >= num_samples:
                    break
            
            epoch += 1
            # Break outer loop if we've reached target
            if samples_generated >= num_samples:
                break
        
        progress_bar.close()
        
        # Save remaining samples
        if all_samples:
            print(f"\nSaving final batch {batch_counter} ({len(all_samples)} samples)...")
            temp_path = save_batch_to_temp(all_samples, temp_dir, batch_counter)
            temp_files.append(temp_path)
        
        # Concatenate all temporary datasets
        print(f"\nConcatenating {len(temp_files)} batch files...")
        datasets = [Dataset.load_from_disk(temp_path) for temp_path in temp_files]
        final_dataset = concatenate_datasets(datasets)
        
        # Truncate to exact number of samples if needed
        if len(final_dataset) > num_samples:
            final_dataset = final_dataset.select(range(num_samples))
        
        print(f"\nFinal dataset size: {len(final_dataset)} samples")
        
        # Save final dataset
        print(f"Saving final dataset to {args.output_path}...")
        os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
        final_dataset.save_to_disk(args.output_path)
        
        print("=" * 100)
        print("Dataset generation complete!")
        print(f"Saved to: {args.output_path}")
        print(f"Total samples: {len(final_dataset)}")
        
        # Print sample distribution
        sample_types = final_dataset["sample_type"]
        pref_count = sum(1 for t in sample_types if t == "preference")
        prog_count = sum(1 for t in sample_types if t == "progress")
        print(f"Preference samples: {pref_count}")
        print(f"Progress samples: {prog_count}")
        print("=" * 100)
        
    finally:
        # Clean up temporary files
        print("\nCleaning up temporary files...")
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print("Cleanup complete!")


if __name__ == "__main__":
    main()
