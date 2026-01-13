#!/usr/bin/env python3
"""
Generate a static VQA dataset from RFM trajectories.

This script generates a HuggingFace dataset with VQA-style prompts and answers,
storing references to .npz files and frame indices instead of loading actual frames.
The generated dataset can be used for training with train_vqa_sft.py.

Usage:
    python scripts/generate_vqa_dataset.py \\
        --num_samples 10000 \\
        --output_path /path/to/output/dataset \\
        --seed 42
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
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
from rfm.data.dataset_types import PreferenceSample, ProgressSample, Trajectory
from rfm.data.datasets import RFMDataset
from rfm.data.datasets.base import resolve_dataset_keys
from rfm.data.samplers.base import (
    get_segment_indices_with_middle,
    linspace_subsample_frames,
    compute_progress_from_segment,
)
from rfm.data.samplers.pref import PrefSampler
from rfm.data.samplers.progress import ProgressSampler
from rfm.utils.config_utils import convert_hydra_to_dataclass
from rfm.utils.logger import rank_0_info

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


def extract_frame_indices_from_trajectory(
    traj_dict: Dict[str, Any],
    max_frames: int,
    subsample_strategy: Optional[str] = None,
    config: Any = None,
) -> List[int]:
    """
    Simulate the frame index extraction logic from _get_traj_from_data WITHOUT loading frames.
    
    Args:
        traj_dict: Raw trajectory dictionary from HF dataset
        max_frames: Maximum number of frames to extract
        subsample_strategy: Subsampling strategy ("subsample_forward", "subsample_reverse", "subsample_rewind", or None)
        config: Config object with progress_pred_type
        
    Returns:
        List of frame indices that would be used
    """
    # Get total frame count from the npz file's metadata
    # In the preprocessed datasets, this is stored in the dataset
    frames_path = traj_dict.get("frames")
    if frames_path and isinstance(frames_path, str):
        # Load minimal metadata from npz to get frame count
        try:
            with np.load(frames_path) as data:
                num_frames_total = int(data.get("num_frames", len(data["frames"])))
        except Exception as e:
            rank_0_info(f"Warning: Could not load frame count from {frames_path}: {e}")
            # Fallback: assume we have frames
            num_frames_total = max_frames
    else:
        # Frames might be directly stored (shouldn't happen in preprocessed datasets)
        if isinstance(traj_dict.get("frames"), (list, np.ndarray)):
            num_frames_total = len(traj_dict["frames"])
        else:
            num_frames_total = max_frames

    # Determine which indices to use based on subsampling strategy
    if subsample_strategy is not None:
        # Simulate _get_subsample_indices logic
        if subsample_strategy == "subsample_forward":
            start_idx = 0
            end_idx = num_frames_total - 1
            middle_idx = (start_idx + end_idx) // 2
        elif subsample_strategy == "subsample_reverse":
            start_idx = num_frames_total - 1
            end_idx = 0
            middle_idx = (start_idx + end_idx) // 2
        elif subsample_strategy == "subsample_rewind":
            start_idx = 0
            end_idx = random.randint(0, num_frames_total - 1) if num_frames_total > 1 else 0
            middle_idx = (start_idx + end_idx) // 2 if num_frames_total >= 3 else None
        else:  # bidirectional
            start_idx = 0
            end_idx = num_frames_total - 1
            middle_idx = (start_idx + end_idx) // 2

        # Use middle_idx only for rewind strategy
        use_middle = subsample_strategy == "subsample_rewind" and middle_idx is not None and num_frames_total >= 3

        # Construct indices
        indices = get_segment_indices_with_middle(
            num_frames_total=num_frames_total,
            start_idx=start_idx,
            end_idx=end_idx,
            middle_idx=middle_idx if use_middle else None,
            max_frames=max_frames,
        )
    else:
        # No subsampling strategy - use all frames
        indices = list(range(num_frames_total))

    # Subsample uniformly if needed
    current_frame_count = len(indices)
    if current_frame_count > max_frames:
        # Simulate linspace_subsample_frames
        subsample_indices = np.linspace(0, current_frame_count - 1, max_frames, dtype=int).tolist()
        indices = [indices[i] for i in subsample_indices]

    return indices


def compute_target_progress_for_indices(
    traj_dict: Dict[str, Any],
    frame_indices: List[int],
    config: Any,
    dataset_success_cutoff_map: Dict[str, float],
) -> List[float]:
    """Compute target progress values for given frame indices."""
    num_frames_total = max(frame_indices) + 1 if frame_indices else 1
    
    ds_key = traj_dict["data_source"]
    success_cutoff = dataset_success_cutoff_map.get(ds_key, config.max_success)
    partial_success = traj_dict.get("partial_success")
    
    target_progress = compute_progress_from_segment(
        num_frames_total=num_frames_total,
        frame_indices=frame_indices,
        progress_pred_type=config.progress_pred_type,
        success_cutoff=success_cutoff,
        partial_success=partial_success,
    )
    
    return target_progress


def extract_preference_metadata_from_dicts(
    chosen_traj_dict: Dict[str, Any],
    rejected_traj_dict: Dict[str, Any],
    data_gen_strategy: str,
    max_frames: int,
    config: Any,
    dataset_success_cutoff_map: Dict[str, float],
) -> Dict[str, Any]:
    """
    Extract metadata from trajectory dicts WITHOUT loading frames.
    
    Args:
        chosen_traj_dict: Raw trajectory dict from dataset
        rejected_traj_dict: Raw trajectory dict from dataset
        data_gen_strategy: Strategy used for generation
        max_frames: Maximum frames to use
        config: Configuration object
        dataset_success_cutoff_map: Dataset success cutoff mapping
        
    Returns:
        Dictionary with all metadata needed for training
    """
    # Randomly decide which trajectory is "1" or "2"
    if random.random() < 0.5:
        answer = "1"
        first_traj_dict = chosen_traj_dict
        second_traj_dict = rejected_traj_dict
        chosen_is_first = True
    else:
        answer = "2"
        first_traj_dict = rejected_traj_dict
        second_traj_dict = chosen_traj_dict
        chosen_is_first = False

    # Generate prompt
    prompt = PREFERENCE_PROMPT_TEMPLATE.format(
        response_prefix=RESPONSE_PREFIX,
        task=chosen_traj_dict["task"],
    )

    # Extract npz paths and determine frame indices
    # The frames field in trajectory dict is a string path to the npz file
    first_npz_path = first_traj_dict["frames"]
    second_npz_path = second_traj_dict["frames"]
    
    # Compute frame indices using the same logic as samplers
    # For now, use simple forward sampling indices
    first_frame_indices = extract_frame_indices_from_trajectory(
        first_traj_dict, max_frames, subsample_strategy="subsample_forward", config=config
    )
    second_frame_indices = extract_frame_indices_from_trajectory(
        second_traj_dict, max_frames, subsample_strategy="subsample_rewind", config=config  # rewind for rejected
    )
    
    # Get frame shapes from frame indices
    first_frames_shape = [len(first_frame_indices), 224, 224, 3]  # Placeholder shape
    second_frames_shape = [len(second_frame_indices), 224, 224, 3]

    metadata = {
        "sample_type": "preference",
        "prompt": prompt,
        "answer": answer,
        "first_npz_path": first_npz_path,
        "second_npz_path": second_npz_path,
        "first_frame_indices": first_frame_indices,
        "second_frame_indices": second_frame_indices,
        "first_frames_shape": first_frames_shape,
        "second_frames_shape": second_frames_shape,
        "task": chosen_traj_dict["task"],
        "data_source": chosen_traj_dict["data_source"],
        "data_gen_strategy": data_gen_strategy,
        "resample_attempts": 1,
        "chosen_is_first": chosen_is_first,
    }

    return metadata


def extract_progress_metadata_from_dict(
    traj_dict: Dict[str, Any],
    data_gen_strategy: str,
    max_frames: int,
    config: Any,
    dataset_success_cutoff_map: Dict[str, float],
) -> Dict[str, Any]:
    """
    Extract metadata from trajectory dict WITHOUT loading frames.
    
    Args:
        traj_dict: Raw trajectory dict from dataset
        data_gen_strategy: Strategy used for generation
        max_frames: Maximum frames to use
        config: Configuration object
        dataset_success_cutoff_map: Dataset success cutoff mapping
        
    Returns:
        Dictionary with all metadata needed for training
    """
    # Extract npz path
    npz_path = traj_dict["frames"]
    
    # Compute frame indices
    frame_indices = extract_frame_indices_from_trajectory(
        traj_dict, max_frames, subsample_strategy="subsample_forward", config=config
    )
    
    # Compute target progress
    target_progress = compute_target_progress_for_indices(
        traj_dict, frame_indices, config, dataset_success_cutoff_map
    )
    
    # Extract last frame's progress as answer (0-100 integer)
    if target_progress:
        last_progress = target_progress[-1]
        answer = str(int(round(last_progress * 100)))
    else:
        answer = "0"

    # Generate prompt
    prompt = PROGRESS_PROMPT_TEMPLATE.format(task=traj_dict["task"])

    # Get frame shape
    frames_shape = [len(frame_indices), 224, 224, 3]  # Placeholder shape

    metadata = {
        "sample_type": "progress",
        "prompt": prompt,
        "answer": answer,
        "npz_path": npz_path,
        "frame_indices": frame_indices,
        "frames_shape": frames_shape,
        "target_progress": target_progress,
        "task": traj_dict["task"],
        "data_source": traj_dict["data_source"],
        "data_gen_strategy": data_gen_strategy,
        "resample_attempts": 1,
    }

    return metadata


def generate_dataset(
    config: ExperimentConfig,
    num_samples: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate dataset samples using RFM samplers.
    
    Args:
        config: Experiment configuration
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of sample dictionaries
    """
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)

    rank_0_info("=" * 100)
    rank_0_info("Setting up dataset and samplers")
    rank_0_info("=" * 100)

    # Resolve dataset keys
    config.data.train_datasets = resolve_dataset_keys(config.data.train_datasets, split="train")
    rank_0_info(f"Resolved train datasets: {config.data.train_datasets}")

    # Create RFM dataset (this will load the HF datasets)
    dataset = RFMDataset(config=config.data, is_evaluation=False)
    rank_0_info(f"Loaded dataset with {len(dataset.dataset)} trajectories")

    # Initialize samplers
    sampler_kwargs = {
        "config": config.data,
        "dataset": dataset.dataset,
        "combined_indices": dataset._combined_indices,
        "dataset_success_cutoff_map": dataset.dataset_success_cutoff_map,
        "verbose": False,
    }

    pref_sampler = None
    progress_sampler = None

    # Check sample type ratios to determine which samplers to create
    if config.data.sample_type_ratio[0] > 0:
        rank_0_info("Initializing preference sampler")
        pref_sampler = PrefSampler(is_evaluation=False, **sampler_kwargs)

    if config.data.sample_type_ratio[1] > 0:
        rank_0_info("Initializing progress sampler")
        progress_sampler = ProgressSampler(is_evaluation=False, **sampler_kwargs)

    if pref_sampler is None and progress_sampler is None:
        raise ValueError("At least one of preference or progress samplers must be enabled (sample_type_ratio)")

    # Calculate sample type distribution
    pref_ratio = config.data.sample_type_ratio[0]
    prog_ratio = config.data.sample_type_ratio[1]
    total_ratio = pref_ratio + prog_ratio

    if total_ratio == 0:
        raise ValueError("Sample type ratio sum is zero")

    pref_prob = pref_ratio / total_ratio if total_ratio > 0 else 0
    prog_prob = prog_ratio / total_ratio if total_ratio > 0 else 0

    rank_0_info(f"Sample type probabilities: preference={pref_prob:.2f}, progress={prog_prob:.2f}")
    rank_0_info("=" * 100)
    rank_0_info(f"Generating {num_samples} samples")
    rank_0_info("=" * 100)

    # Generate samples
    generated_samples = []
    failed_samples = 0

    for i in tqdm(range(num_samples), desc="Generating samples"):
        # Select sample type
        if random.random() < pref_prob:
            sample_type = "preference"
        else:
            sample_type = "progress"

        # Get a trajectory from the dataset
        dataset_idx = i % len(dataset.dataset)
        item = dataset.dataset[dataset_idx]

        # Generate sample WITHOUT loading frames
        try:
            if sample_type == "preference" and pref_sampler:
                # Generate preference sample by selecting two trajectories
                # Use a simple strategy: rewind for now
                # Get chosen trajectory (current item)
                chosen_traj_dict = item
                
                # Create rejected trajectory (same trajectory, will be rewound)
                rejected_traj_dict = item
                data_gen_strategy = "rewind"
                
                metadata = extract_preference_metadata_from_dicts(
                    chosen_traj_dict=chosen_traj_dict,
                    rejected_traj_dict=rejected_traj_dict,
                    data_gen_strategy=data_gen_strategy,
                    max_frames=config.data.max_frames,
                    config=config.data,
                    dataset_success_cutoff_map=dataset.dataset_success_cutoff_map,
                )
                generated_samples.append(metadata)
                
            elif sample_type == "progress" and progress_sampler:
                # Generate progress sample from single trajectory
                data_gen_strategy = "forward_progress"
                
                metadata = extract_progress_metadata_from_dict(
                    traj_dict=item,
                    data_gen_strategy=data_gen_strategy,
                    max_frames=config.data.max_frames,
                    config=config.data,
                    dataset_success_cutoff_map=dataset.dataset_success_cutoff_map,
                )
                generated_samples.append(metadata)
            else:
                failed_samples += 1
        except Exception as e:
            rank_0_info(f"Error generating sample {i}: {e}")
            import traceback
            traceback.print_exc()
            failed_samples += 1
            continue

    rank_0_info("=" * 100)
    rank_0_info(f"Successfully generated {len(generated_samples)} samples")
    rank_0_info(f"Failed to generate {failed_samples} samples")
    rank_0_info("=" * 100)

    return generated_samples


def main():
    # Register structured configs with Hydra (like in train.py)
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=ExperimentConfig)
    cs.store(group="model", name="model_config", node=ModelConfig)
    cs.store(group="peft", name="peft_config", node=PEFTConfig)
    cs.store(group="data", name="data_config", node=DataConfig)
    cs.store(group="training", name="training_config", node=TrainingConfig)
    cs.store(group="loss", name="loss_config", node=LossConfig)
    cs.store(group="logging", name="logging_config", node=LoggingConfig)
    cs.store(group="logging/save_best", name="save_best_config", node=SaveBestConfig)
    cs.store(group="custom_eval", name="custom_eval_config", node=CustomEvaluationConfig)
    
    parser = argparse.ArgumentParser(description="Generate VQA dataset from RFM trajectories")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to generate (default: 10000)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the generated dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="config",
        help="Name of the config file to use (default: config)",
    )
    parser.add_argument(
        "--config_overrides",
        type=str,
        nargs="*",
        default=[],
        help="Config overrides in key=value format (e.g., data.max_frames=16)",
    )

    args = parser.parse_args()

    # Initialize Hydra config
    config_dir = str(project_root / "rfm" / "configs")
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Compose config with overrides
        hydra_cfg = compose(config_name=args.config_name, overrides=args.config_overrides)
        
        # Convert to dataclass
        config = convert_hydra_to_dataclass(hydra_cfg, ExperimentConfig)

    rank_0_info("=" * 100)
    rank_0_info("VQA Dataset Generation")
    rank_0_info("=" * 100)
    rank_0_info(f"Number of samples: {args.num_samples}")
    rank_0_info(f"Output path: {args.output_path}")
    rank_0_info(f"Random seed: {args.seed}")
    rank_0_info(f"Max frames: {config.data.max_frames}")
    rank_0_info(f"Sample type ratio: {config.data.sample_type_ratio}")
    rank_0_info("=" * 100)

    # Generate samples
    samples = generate_dataset(config, args.num_samples, args.seed)

    # Save as HuggingFace Dataset
    rank_0_info(f"Saving dataset to {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    
    dataset = Dataset.from_list(samples)
    dataset.save_to_disk(args.output_path)

    # Also save config for reference
    config_path = os.path.join(args.output_path, "generation_config.json")
    config_dict = {
        "num_samples": args.num_samples,
        "seed": args.seed,
        "max_frames": config.data.max_frames,
        "sample_type_ratio": config.data.sample_type_ratio,
        "train_datasets": config.data.train_datasets,
        "preference_strategy_ratio": config.data.preference_strategy_ratio,
        "progress_strategy_ratio": config.data.progress_strategy_ratio,
    }
    
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    rank_0_info(f"Dataset saved successfully to {args.output_path}")
    rank_0_info(f"Config saved to {config_path}")
    rank_0_info("=" * 100)
    
    # Print sample statistics
    sample_types = {}
    for sample in samples:
        sample_type = sample["sample_type"]
        sample_types[sample_type] = sample_types.get(sample_type, 0) + 1
    
    rank_0_info("Sample statistics:")
    for sample_type, count in sample_types.items():
        rank_0_info(f"  {sample_type}: {count} ({100 * count / len(samples):.1f}%)")
    rank_0_info("=" * 100)


if __name__ == "__main__":
    main()
