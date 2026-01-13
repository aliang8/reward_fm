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
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    eval_mode: bool = False,
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
    first_frame_indices = extract_frame_indices_from_trajectory(
        first_traj_dict, max_frames, subsample_strategy="subsample_forward", config=config
    )
    
    # For eval mode, use forward for both (no rewind augmentation)
    # For training mode, use rewind for second trajectory
    second_strategy = "subsample_forward" if eval_mode else "subsample_rewind"
    second_frame_indices = extract_frame_indices_from_trajectory(
        second_traj_dict, max_frames, subsample_strategy=second_strategy, config=config
    )
    
    # Get frame shapes from frame indices
    first_frames_shape = [len(first_frame_indices), 224, 224, 3]  # Placeholder shape
    second_frames_shape = [len(second_frame_indices), 224, 224, 3]

    metadata = {
        "sample_type": "preference",
        "prompt": prompt,
        "answer": answer,
        # Preference-specific fields
        "first_npz_path": first_npz_path,
        "second_npz_path": second_npz_path,
        "first_frame_indices": first_frame_indices,
        "second_frame_indices": second_frame_indices,
        "first_frames_shape": first_frames_shape,
        "second_frames_shape": second_frames_shape,
        "chosen_is_first": chosen_is_first,
        # Progress-specific fields (set to None for preference samples)
        "npz_path": None,
        "frame_indices": None,
        "frames_shape": None,
        "target_progress": None,
        # Common fields
        "task": chosen_traj_dict["task"],
        "data_source": chosen_traj_dict["data_source"],
        "data_gen_strategy": data_gen_strategy,
        "resample_attempts": 1,
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
        # Progress-specific fields
        "npz_path": npz_path,
        "frame_indices": frame_indices,
        "frames_shape": frames_shape,
        "target_progress": target_progress,
        # Preference-specific fields (set to None for progress samples)
        "first_npz_path": None,
        "second_npz_path": None,
        "first_frame_indices": None,
        "second_frame_indices": None,
        "first_frames_shape": None,
        "second_frames_shape": None,
        "chosen_is_first": None,
        # Common fields
        "task": traj_dict["task"],
        "data_source": traj_dict["data_source"],
        "data_gen_strategy": data_gen_strategy,
        "resample_attempts": 1,
    }

    return metadata


def generate_single_sample(args_tuple):
    """
    Generate a single sample. This function is designed to be called in parallel.
    
    Args:
        args_tuple: Tuple containing (sample_idx, dataset_item, config_dict, eval_mode_info, seed)
        
    Returns:
        Dict with sample metadata or None if failed
    """
    try:
        sample_idx, dataset_item, config_dict, eval_mode_info, seed_offset = args_tuple
        
        # Set random seed for this worker (deterministic but different per sample)
        random.seed(seed_offset + sample_idx)
        np.random.seed(seed_offset + sample_idx)
        
        # Unpack config dict
        max_frames = config_dict['max_frames']
        pref_prob = config_dict['pref_prob']
        dataset_success_cutoff_map = config_dict['dataset_success_cutoff_map']
        config_data = config_dict['config_data']
        
        # Unpack eval mode info
        eval_mode = eval_mode_info['eval_mode']
        task_to_indices = eval_mode_info.get('task_to_indices', {})
        tasks_with_both = eval_mode_info.get('tasks_with_both', [])
        dataset_list = eval_mode_info.get('dataset_list', [])
        
        # Select sample type
        if random.random() < pref_prob:
            sample_type = "preference"
        else:
            sample_type = "progress"
        
        # Generate sample
        if sample_type == "preference":
            if eval_mode and tasks_with_both:
                # Evaluation mode: use real quality differences
                task = random.choice(tasks_with_both)
                
                # Get optimal and suboptimal trajectories
                optimal_idx = random.choice(task_to_indices[task]['optimal'])
                suboptimal_idx = random.choice(task_to_indices[task]['suboptimal'])
                
                chosen_traj_dict = dataset_list[optimal_idx]
                rejected_traj_dict = dataset_list[suboptimal_idx]
                data_gen_strategy = "quality_preference"
                
                metadata = extract_preference_metadata_from_dicts(
                    chosen_traj_dict=chosen_traj_dict,
                    rejected_traj_dict=rejected_traj_dict,
                    data_gen_strategy=data_gen_strategy,
                    max_frames=max_frames,
                    config=config_data,
                    dataset_success_cutoff_map=dataset_success_cutoff_map,
                    eval_mode=True,
                )
            else:
                # Training mode: use augmentations (rewind)
                chosen_traj_dict = dataset_item
                rejected_traj_dict = dataset_item
                data_gen_strategy = "rewind"
                
                metadata = extract_preference_metadata_from_dicts(
                    chosen_traj_dict=chosen_traj_dict,
                    rejected_traj_dict=rejected_traj_dict,
                    data_gen_strategy=data_gen_strategy,
                    max_frames=max_frames,
                    config=config_data,
                    dataset_success_cutoff_map=dataset_success_cutoff_map,
                    eval_mode=False,
                )
            
            return metadata
            
        elif sample_type == "progress":
            # Generate progress sample
            data_gen_strategy = "forward_progress"
            
            metadata = extract_progress_metadata_from_dict(
                traj_dict=dataset_item,
                data_gen_strategy=data_gen_strategy,
                max_frames=max_frames,
                config=config_data,
                dataset_success_cutoff_map=dataset_success_cutoff_map,
            )
            
            return metadata
        
        return None
        
    except Exception as e:
        # Return error info
        return {'error': str(e), 'sample_idx': sample_idx}


def generate_dataset(
    config: ExperimentConfig,
    num_samples: int,
    seed: int = 42,
    eval_mode: bool = False,
    num_workers: int = 1,
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
    if eval_mode:
        rank_0_info("EVALUATION MODE: No augmentations, real quality differences")
    rank_0_info("=" * 100)

    # Resolve dataset keys
    if eval_mode:
        config.data.eval_datasets = resolve_dataset_keys(config.data.eval_datasets, split="eval")
        config.data.train_datasets = resolve_dataset_keys(config.data.eval_datasets, split="train")
        rank_0_info(f"Resolved eval datasets: {config.data.eval_datasets}")
    else:
        config.data.eval_datasets = resolve_dataset_keys(config.data.train_datasets, split="eval")
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

    # Build task-to-indices mapping for eval mode
    task_to_indices = {}
    tasks_with_both = []
    dataset_list = []
    
    if eval_mode:
        for idx, item in enumerate(dataset.dataset):
            task = item['task']
            quality = item.get('quality_label', 'unknown')
            if task not in task_to_indices:
                task_to_indices[task] = {'optimal': [], 'suboptimal': []}
            
            if quality == 'successful':
                task_to_indices[task]['optimal'].append(idx)
            else:
                task_to_indices[task]['suboptimal'].append(idx)
        
        # Find tasks with both optimal and suboptimal
        tasks_with_both = [t for t, indices in task_to_indices.items() 
                          if len(indices['optimal']) > 0 and len(indices['suboptimal']) > 0]
        
        rank_0_info(f"Eval mode: Found {len(tasks_with_both)} tasks with both optimal and suboptimal trajectories")
        
        # Convert dataset to list for multiprocessing
        dataset_list = list(dataset.dataset)
    
    # Determine number of workers
    if num_workers == -1:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()
        rank_0_info(f"Auto-detected {num_workers} CPU cores")
    
    rank_0_info(f"Using {num_workers} worker(s) for generation")
    
    # Prepare config dict for workers (must be picklable)
    config_dict = {
        'max_frames': config.data.max_frames,
        'pref_prob': pref_prob,
        'dataset_success_cutoff_map': dataset.dataset_success_cutoff_map,
        'config_data': config.data,
    }
    
    # Prepare eval mode info for workers
    eval_mode_info = {
        'eval_mode': eval_mode,
        'task_to_indices': task_to_indices,
        'tasks_with_both': tasks_with_both,
        'dataset_list': dataset_list if eval_mode else [],
    }
    
    # Generate samples
    generated_samples = []
    failed_samples = 0
    
    if num_workers <= 1:
        # Sequential generation (original behavior)
        rank_0_info("Sequential generation (num_workers=1)")
        for i in tqdm(range(num_samples), desc="Generating samples"):
            # Get a trajectory from the dataset
            dataset_idx = i % len(dataset.dataset)
            item = dataset.dataset[dataset_idx]
            
            # Prepare args for generation function
            args_tuple = (i, item, config_dict, eval_mode_info, seed)
            
            # Generate sample
            try:
                result = generate_single_sample(args_tuple)
                
                if result and 'error' not in result:
                    generated_samples.append(result)
                else:
                    if result and 'error' in result:
                        rank_0_info(f"Error generating sample {i}: {result['error']}")
                    failed_samples += 1
            except Exception as e:
                rank_0_info(f"Error generating sample {i}: {e}")
                import traceback
                traceback.print_exc()
                failed_samples += 1
    else:
        # Parallel generation with multiprocessing
        rank_0_info(f"Parallel generation with {num_workers} workers")
        
        # Prepare arguments for all samples
        sample_args = []
        for i in range(num_samples):
            dataset_idx = i % len(dataset.dataset)
            if eval_mode and dataset_list:
                item = dataset_list[dataset_idx]
            else:
                item = dataset.dataset[dataset_idx]
            
            args_tuple = (i, item, config_dict, eval_mode_info, seed)
            sample_args.append(args_tuple)
        
        # Generate samples in parallel with progress bar
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(generate_single_sample, args) for args in sample_args]
            
            # Process results as they complete with progress bar
            # Use explicit file=sys.stderr to avoid stdout buffering issues
            import sys
            completed = 0
            for future in tqdm(as_completed(futures), total=num_samples, 
                             desc="Generating samples", file=sys.stderr, 
                             ncols=100, mininterval=0.5):
                try:
                    result = future.result()
                    
                    if result and 'error' not in result:
                        generated_samples.append(result)
                    else:
                        if result and 'error' in result:
                            sample_idx = result.get('sample_idx', '?')
                            # Don't print every error in multiprocessing to avoid spam
                            pass
                        failed_samples += 1
                except Exception as e:
                    failed_samples += 1
                
                completed += 1
                # Print periodic updates to stdout as well
                if completed % 1000 == 0:
                    rank_0_info(f"Progress: {completed}/{num_samples} samples generated")
    
    # Sort samples by their original order if needed (multiprocessing may return out of order)
    # This is optional - comment out if you don't care about order
    if num_workers > 1 and 'sample_idx' not in generated_samples[0]:
        # Samples don't have indices, they're already in order
        pass
    

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
    parser.add_argument(
        "--eval_mode",
        action="store_true",
        help="Evaluation mode: no augmentations, use real quality differences for preferences",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers for generation (default: 8, set to -1 for auto)",
    )
    parser.add_argument(
        "--save_batch_size",
        type=int,
        default=10000,
        help="Save dataset incrementally every N samples to avoid OOM (default: 10000, set to -1 to save all at once)",
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
    rank_0_info(f"Number of workers: {args.num_workers if args.num_workers > 0 else 'auto'}")
    rank_0_info(f"Eval mode: {args.eval_mode}")
    rank_0_info("=" * 100)

    # Generate and save samples (incrementally if needed to avoid OOM)
    if args.save_batch_size > 0 and args.num_samples > args.save_batch_size:
        # Incremental saving for large datasets
        rank_0_info(f"Using incremental saving (batch size: {args.save_batch_size})")
        
        num_batches = (args.num_samples + args.save_batch_size - 1) // args.save_batch_size
        temp_output_dir = Path(args.output_path).parent / f"{Path(args.output_path).name}_temp"
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.save_batch_size
            end_idx = min((batch_idx + 1) * args.save_batch_size, args.num_samples)
            batch_size = end_idx - start_idx
            
            rank_0_info(f"Generating batch {batch_idx + 1}/{num_batches} ({start_idx}-{end_idx})...")
            
            # Generate batch
            batch_samples = generate_dataset(
                config, 
                batch_size,
                args.seed + start_idx,  # Different seed per batch for reproducibility
                eval_mode=args.eval_mode,
                num_workers=args.num_workers,
            )
            
            if not batch_samples:
                rank_0_info(f"Warning: Batch {batch_idx + 1} generated 0 samples")
                continue
            
            # Save batch
            batch_dataset = Dataset.from_list(batch_samples)
            batch_path = temp_output_dir / f"batch_{batch_idx:04d}"
            batch_dataset.save_to_disk(str(batch_path))
            rank_0_info(f"Saved batch {batch_idx + 1} with {len(batch_samples)} samples")
            
            # Clear memory
            del batch_samples
            del batch_dataset
        
        # Concatenate all batches
        rank_0_info("Concatenating batches...")
        from datasets import concatenate_datasets
        all_batches = []
        for batch_idx in range(num_batches):
            batch_path = temp_output_dir / f"batch_{batch_idx:04d}"
            if batch_path.exists():
                batch_ds = Dataset.load_from_disk(str(batch_path))
                all_batches.append(batch_ds)
        
        if not all_batches:
            rank_0_info("No batches were saved. Exiting.")
            return
        
        dataset = concatenate_datasets(all_batches)
        
        # Save final dataset
        rank_0_info(f"Saving final dataset to {args.output_path}...")
        os.makedirs(args.output_path, exist_ok=True)
        dataset.save_to_disk(args.output_path)
        
        # Cleanup temp directory
        rank_0_info("Cleaning up temporary files...")
        import shutil
        shutil.rmtree(temp_output_dir, ignore_errors=True)
        
        # Collect samples for statistics (sample first 1000 to avoid loading all into memory)
        samples = dataset.select(range(min(1000, len(dataset)))).to_list()
        dataset_len = len(dataset)
    else:
        # Single-shot generation for small datasets
        rank_0_info("Generating all samples at once...")
        samples = generate_dataset(
            config, 
            args.num_samples, 
            args.seed, 
            eval_mode=args.eval_mode,
            num_workers=args.num_workers,
        )

        if not samples:
            rank_0_info("No samples generated. Exiting.")
            return

        # Save as HuggingFace Dataset
        rank_0_info(f"Saving dataset to {args.output_path}")
        os.makedirs(args.output_path, exist_ok=True)
        
        dataset = Dataset.from_list(samples)
        dataset.save_to_disk(args.output_path)
        dataset_len = len(samples)

    # Also save config for reference
    config_path = os.path.join(args.output_path, "generation_config.json")
    config_dict = {
        "num_samples_requested": args.num_samples,
        "num_samples_actual": dataset_len,
        "seed": args.seed,
        "eval_mode": args.eval_mode,
        "num_workers": args.num_workers,
        "save_batch_size": args.save_batch_size,
        "max_frames": config.data.max_frames,
        "sample_type_ratio": config.data.sample_type_ratio,
        "train_datasets": config.data.train_datasets,
        "eval_datasets": config.data.eval_datasets,
        "preference_strategy_ratio": config.data.preference_strategy_ratio,
        "progress_strategy_ratio": config.data.progress_strategy_ratio,
    }
    
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    rank_0_info(f"Dataset saved successfully to {args.output_path}")
    rank_0_info(f"Config saved to {config_path}")
    rank_0_info("=" * 100)
    
    # Print sample statistics (from first 1000 samples as estimate)
    rank_0_info(f"Total samples: {dataset_len}")
    sample_types = {}
    for sample in samples:
        sample_type = sample["sample_type"]
        sample_types[sample_type] = sample_types.get(sample_type, 0) + 1
    
    rank_0_info("Sample statistics (from sample):")
    for sample_type, count in sample_types.items():
        rank_0_info(f"  {sample_type}: {count} ({100 * count / len(samples):.1f}%)")
    rank_0_info("=" * 100)


if __name__ == "__main__":
    main()
