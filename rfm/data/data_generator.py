#!/usr/bin/env python3
"""
DataGenerator class for producing batches of data for RFM model training with three prediction heads:
1. Preference prediction: Predict whether o^1 or o^2 are preferred
2. Progress prediction: Predict progress for a single trajectory  
3. Comparative scoring: Rank o^1 and o^2 against a reference trajectory o^ref

The generator allows controlling the ratio between different prediction types.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Iterator, Union
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
import shutil
import os
from pathlib import Path
import torch
from rfm.data.batch_collator import BaseSample, PreferenceSample, SimilaritySample, PairedVideoSample, BatchCollator
from datasets import concatenate_datasets
from rfm.utils.logging import rank_0_print
from datasets import Dataset
import json
from rfm.utils.logging import timer

class DataGenerator:
    """Data generator for producing batches of prediction data with controlled ratios."""
    
    def __init__(self, config, is_evaluation=False):
        """Initialize DataGenerator with configuration."""
        self.config = config
        self.is_evaluation = is_evaluation
        
        # Choose datasets based on whether this is for evaluation or training
        if is_evaluation and config.data.eval_datasets:
            self.datasets = config.data.eval_datasets
            self.subsets = config.data.eval_subsets
        else:
            self.datasets = config.data.train_datasets
            self.subsets = config.data.train_subsets
            
        self.force_reprocess = config.data.force_reprocess
        
        # Initialize dataset and index mappings
        self.dataset = None
        self.robot_trajectories = []
        self.human_trajectories = []
        self.optimal_by_task = {}
        self.suboptimal_by_task = {}
        self.quality_indices = {}
        self.task_indices = {}
        self.source_indices = {}
        
        self.preference_ratio = config.data.preference_ratio
        self.similarity_ratio = 1.0 - config.data.preference_ratio
        self.dataset_preference_ratio = config.data.dataset_preference_ratio

        # Load trajectory dataset
        self._load_trajectory_dataset()
        
        # Initialize preference and similarity datasets
        self._load_preference_dataset()
        self._load_similarity_dataset()
        
        rank_0_print(f"DataGenerator initialized with {len(self.dataset)} total trajectories")
        rank_0_print(f"  Robot trajectories: {len(self.robot_trajectories)}")
        rank_0_print(f"  Human trajectories: {len(self.human_trajectories)}")
        rank_0_print(f"  Tasks: {len(self.task_indices)}")
        rank_0_print(f"  Quality labels: {len(self.quality_indices)}")
        rank_0_print(f"  Data sources: {len(self.source_indices)}")
    
    def _load_trajectory_dataset(self):
        """Load trajectory dataset using preprocessed index-based cache."""
        if self.is_evaluation:
            cache_dir = f"./processed_datasets/eval_cache"
            cache_type = "evaluation"
        else:
            cache_dir = f"./processed_datasets/train_cache"
            cache_type = "training"
        
        # Check if preprocessed cache exists
        if os.path.exists(cache_dir) and not self.force_reprocess:
            rank_0_print(f"Found preprocessed {cache_type} cache at {cache_dir}, loading...")
            try:
                self._load_preprocessed_cache(cache_dir, is_training=not self.is_evaluation)
                rank_0_print(f"Successfully loaded preprocessed {cache_type} cache with {len(self.dataset)} trajectory indices")
            except Exception as e:
                rank_0_print(f"Failed to load preprocessed {cache_type} cache: {e}, will reprocess datasets")
                # Remove corrupted cache
                shutil.rmtree(cache_dir, ignore_errors=True)
                raise RuntimeError(
                    f"{cache_type.capitalize()} dataset preprocessing required. Please run:\n"
                    "python preprocess_datasets.py\n"
                    "This will create the necessary index-based cache for efficient data loading."
                )
        else:
            # If no cache exists, we need to run the preprocessor first
            rank_0_print(f"No preprocessed {cache_type} cache found. Please run preprocess_datasets.py first to create the cache.")
            raise RuntimeError(
                f"{cache_type.capitalize()} dataset preprocessing required. Please run:\n"
                "python preprocess_datasets.py\n"
                "This will create the necessary index-based cache for efficient data loading."
            )
    
    def _load_preprocessed_cache(self, cache_dir: str, is_training: bool = True):
        """Load the preprocessed cache with index mappings."""
        # Load the processed dataset
        dataset_cache_dir = os.path.join(cache_dir, "processed_dataset")
        self.dataset = Dataset.load_from_disk(dataset_cache_dir)

        # Load index mappings
        mappings_file = os.path.join(cache_dir, "index_mappings.json")
        with open(mappings_file, 'r') as f:
            index_mappings = json.load(f)
        
        # Store the index mappings for fast access
        self.robot_trajectories = index_mappings['robot_trajectories']
        self.human_trajectories = index_mappings['human_trajectories']
        self.optimal_by_task = index_mappings['optimal_by_task']
        self.suboptimal_by_task = index_mappings['suboptimal_by_task']
        self.quality_indices = index_mappings['quality_indices']
        self.task_indices = index_mappings['task_indices']
        self.source_indices = index_mappings['source_indices']
        
        dataset_type = "training" if is_training else "evaluation"
        rank_0_print(f"Loaded {len(self.dataset)} trajectory indices from preprocessed {dataset_type} cache")
    
    def _load_frames_from_npz(self, npz_filepath: str) -> np.ndarray:
        """Load frames on-demand from npz file.
        
        Args:
            npz_filepath: Path to the .npz file containing frames
            
        Returns:
            numpy array with shape (T, H, W, C) containing the video frames
        """
        if not npz_filepath or not os.path.exists(npz_filepath):
            raise ValueError(f"NPZ file not found: {npz_filepath}")
        
        try:
            # Load frames from npz file
            with np.load(npz_filepath) as data:
                frames = data['frames']
                # Verify the data structure
                if 'shape' in data:
                    expected_shape = tuple(data['shape'])
                    if frames.shape != expected_shape:
                        rank_0_print(f"Warning: Loaded frames shape {frames.shape} doesn't match expected {expected_shape}")
                
                return frames
        except Exception as e:
            rank_0_print(f"Error loading frames from {npz_filepath}: {e}")
            raise RuntimeError(f"Failed to load frames from {npz_filepath}: {e}")
    
    def _get_trajectory_frames(self, trajectory_idx: int) -> np.ndarray:
        """Get frames for a trajectory by index, loading from npz if needed.
        
        Args:
            trajectory_idx: Index of the trajectory in the dataset
            
        Returns:
            numpy array with shape (T, H, W, C) containing the video frames
        """
        trajectory = self.dataset[trajectory_idx]
        npz_filepath = trajectory.get('frames')
        
        if not npz_filepath:
            raise ValueError(f"No frames path found for trajectory {trajectory_idx}")
        
        return self._load_frames_from_npz(npz_filepath)
    
    def _get_batch_frames(self, trajectory_indices: List[int]) -> List[np.ndarray]:
        """Get frames for multiple trajectories efficiently.
        
        Args:
            trajectory_indices: List of trajectory indices to load
            
        Returns:
            List of numpy arrays with shapes (T, H, W, C) for each trajectory
        """
        frames_list = []
        for idx in trajectory_indices:
            try:
                frames = self._get_trajectory_frames(idx)
                frames_list.append(frames)
            except Exception as e:
                rank_0_print(f"Warning: Failed to load frames for trajectory {idx}: {e}")
                # Return empty frames as fallback
                frames_list.append(np.array([]))
        
        return frames_list
    
    def _create_rewind_trajectory(self, original_traj: Dict) -> Dict:
        """Create a suboptimal trajectory by rewinding the original trajectory."""
        # Load frames from npz file
        frames_data = self._load_frames_from_npz(original_traj['frames'])
        
        # Get the number of frames
        if hasattr(frames_data, 'shape'):
            num_frames = frames_data.shape[0]  # Use shape[0] for numpy array
        else:
            num_frames = len(frames_data)
        
        if num_frames < 4:
            # If trajectory is too short, just return the original
            return original_traj
        
        # Randomly select start and end points for the forward segment
        start_idx = random.randint(0, num_frames // 2)
        end_idx = random.randint(num_frames // 2, num_frames)
        
        # Ensure minimum segment length
        while end_idx - start_idx < 3:
            start_idx = random.randint(0, num_frames // 2)
            end_idx = random.randint(num_frames // 2, num_frames)
        
        # Extract forward segment
        forward_frames = frames_data[start_idx:end_idx]
        # Progress should be relative to original trajectory, not just the segment
        forward_progress = [(start_idx + i + 1) / num_frames for i in range(end_idx - start_idx)]
        
        # Create rewind segment (reverse the forward segment)
        selected_end_point = random.randint(2, len(forward_frames) - 1)
        reverse_frames = forward_frames[::-1][1:selected_end_point]
        # Reverse progress should also be relative to original trajectory
        # Calculate progress for the reversed frames in the correct order
        reverse_progress = [(end_idx - 1 - i) / num_frames for i in range(len(reverse_frames))]
        
        # Combine forward and reverse segments
        if isinstance(forward_frames, np.ndarray):
            # If frames are numpy arrays, use concatenate
            combined_frames = np.concatenate([forward_frames, reverse_frames], axis=0)
        else:
            # If frames are lists, use regular concatenation
            combined_frames = forward_frames + reverse_frames

        combined_progress = forward_progress + reverse_progress
        
        # Create new trajectory with rewind frames
        rewind_traj = original_traj.copy()
        rewind_traj['frames'] = combined_frames
        rewind_traj['frames_shape'] = combined_frames.shape  # Store shape for the rewind trajectory
        rewind_traj['id'] = f"{original_traj['id']}_rewind_{random.randint(1000, 9999)}"
        rewind_traj['quality_label'] = 'suboptimal'  # Mark as suboptimal
        rewind_traj['metadata'] = rewind_traj.get('metadata', {}).copy()
        rewind_traj['metadata']['rewind_generated'] = True
        rewind_traj['metadata']['original_traj_id'] = original_traj['id']
        rewind_traj['metadata']['rewind_progress'] = combined_progress
        
        return rewind_traj
    
    def _create_preference_sample_from_dataset(self) -> PreferenceSample:
        """Create a preference sample from the loaded preference dataset."""
        if not self.preferences:
            raise ValueError("No preferences loaded from dataset")
        
        # For now, return a simple preference sample
        # This can be enhanced later when we have actual preference data
        preference = random.choice(self.preferences)
        
        # This is a placeholder - would need to be implemented based on actual preference data structure
        raise NotImplementedError("Preference sample creation from dataset not yet implemented")
    
    def _load_preference_dataset(self):
        """Load the preference dataset from disk or hub if provided."""
        self.preferences = []
        
        # For now, we'll use empty preferences since the config structure has changed
        # This can be updated later if needed
        rank_0_print("No preference dataset provided, will use random sampling for preferences")
        return
    
    def _load_similarity_dataset(self):
        """Load the similarity dataset if provided."""
        # For now, we'll use empty similarity dataset
        # This can be updated later if needed
        rank_0_print("No similarity dataset provided, will use random sampling for similarities")
        return
    
    def _create_preference_sample(self) -> PreferenceSample:
        """Create a preference prediction sample: chosen vs rejected where chosen is preferred.
        
        This method implements three different strategies for generating negative trajectories
        to create diverse and robust preference learning data:
        
        **Strategy 1: Rewind Same Task (33%)**
        - Creates a suboptimal trajectory by rewinding the optimal trajectory
        - Same task, different trajectory ID
        - Good for learning task-specific failure modes and temporal dynamics
        
        **Strategy 2: Suboptimal/Failure Same Task (33%)**
        - Uses existing suboptimal/failure trajectories from the same task
        - Same task, different trajectory ID
        - Good for learning from real failure examples and task-specific suboptimal patterns
        
        **Strategy 3: Different Task (33%)**
        - Uses trajectories from completely different tasks
        - Different task, can be optimal or suboptimal
        - Good for learning cross-task generalization and what makes trajectories "good" 
          across different contexts
        
        The strategies are chosen with equal probability to ensure balanced learning
        across different types of negative examples. This helps the model learn robust
        preference patterns that generalize well across tasks and scenarios.
        
        Returns:
            PreferenceSample: A preference sample with chosen (optimal) vs rejected 
            (negative) trajectories and associated metadata
        """
        
        with timer("create_preference_sample", verbose=False):
            if random.random() < self.dataset_preference_ratio and self.preferences:
                # Use preference trajectories from dataset
                return self._create_preference_sample_from_dataset()
            else:
                return self._create_preference_sample_with_strategies()
        
    def _create_preference_sample_with_strategies(self) -> PreferenceSample:
        """Create a preference prediction sample using various negative generation strategies.
        
        Implements three strategies for generating negative trajectories to create diverse
        preference learning data. Each strategy is chosen with equal probability (33% each).
        """
        
        # Use preprocessed optimal trajectories from index maps
        if not self.optimal_by_task:
            raise ValueError("No optimal trajectories found for preference generation")
        
        # Get a random task and optimal trajectory from it
        task_name = random.choice(list(self.optimal_by_task.keys()))
        optimal_idx = random.choice(self.optimal_by_task[task_name])
        optimal_traj = self.dataset[optimal_idx]

        # Choose negative generation strategy (equal probability for three strategies)
        strategy = random.random()
        
        if strategy < 0.33:
            # Strategy 1: Use rewind-generated suboptimal trajectory from same task
            negative_traj = self._create_rewind_trajectory(optimal_traj)
            strategy_used = "rewind_same_task"
        elif strategy < 0.66:
            # Strategy 2: Use random suboptimal trajectory from same task
            same_task_suboptimal_indices = self.suboptimal_by_task.get(task_name, [])
            same_task_suboptimal = [
                self.dataset[idx] for idx in same_task_suboptimal_indices 
                if self.dataset[idx]['id'] != optimal_traj['id']
            ]
            if same_task_suboptimal:
                negative_traj = random.choice(same_task_suboptimal)
                strategy_used = "suboptimal_same_task"
            else:
                # Fall back to rewind if no same-task suboptimal trajectories
                negative_traj = self._create_rewind_trajectory(optimal_traj)
                strategy_used = "rewind_same_task_fallback"
        else:
            # Strategy 3: Use trajectory from different task (can be optimal or suboptimal)
            other_tasks = [task for task in self.optimal_by_task.keys() if task != optimal_traj['task']]
            if other_tasks:
                other_task = random.choice(other_tasks)
                # Get random index from other task and access dataset directly
                other_task_indices = self.optimal_by_task[other_task]
                if other_task_indices:
                    other_idx = random.choice(other_task_indices)
                    other_traj = self.dataset[other_idx]
                    # Check if it's not the same trajectory
                    if other_traj['id'] != optimal_traj['id']:
                        negative_traj = other_traj
                        strategy_used = "different_task"
                    else:
                        # Fall back to rewind if same trajectory
                        negative_traj = self._create_rewind_trajectory(optimal_traj)
                        strategy_used = "rewind_same_task_fallback"
                else:
                    # Fall back to rewind if no other trajectories available
                    negative_traj = self._create_rewind_trajectory(optimal_traj)
                    strategy_used = "rewind_same_task_fallback"
            else:
                # Fall back to rewind if only one task available
                negative_traj = self._create_rewind_trajectory(optimal_traj)
                strategy_used = "rewind_same_task_fallback"
        
        # Get frames from npz files
        optimal_frames = self._get_trajectory_frames(optimal_idx)
        
        # Handle negative trajectory frames - could be from dataset (npz) or rewind-generated (numpy)
        if isinstance(negative_traj, dict) and 'frames' in negative_traj:
            if isinstance(negative_traj['frames'], str) and negative_traj['frames'].endswith('.npz'):
                # Regular trajectory with npz path
                negative_frames = self._load_frames_from_npz(negative_traj['frames'])
            elif isinstance(negative_traj['frames'], np.ndarray):
                # Rewind trajectory with numpy array
                negative_frames = negative_traj['frames']
            else:
                raise ValueError(f"Unexpected frames format in negative trajectory: {type(negative_traj['frames'])}")
        else:
            raise ValueError(f"Invalid negative trajectory format: {type(negative_traj)}")
        
        # Calculate target progress for both trajectories
        target_progress_A = self._calculate_target_progress(optimal_traj, optimal_frames)
        target_progress_B = self._calculate_target_progress(negative_traj, negative_frames)
        
        # Get frame shapes
        optimal_frames_shape = optimal_traj.get('frames_shape')
        if isinstance(optimal_frames_shape, list):
            optimal_frames_shape = tuple(optimal_frames_shape)
        
        negative_frames_shape = negative_traj.get('frames_shape')
        if isinstance(negative_frames_shape, list):
            negative_frames_shape = tuple(negative_frames_shape)
    
        # Create preference sample structure
        sample = PreferenceSample(
            # Core HF dataset fields (from optimal trajectory)
            id=optimal_traj['id'],
            task=optimal_traj['task'],
            lang_vector=optimal_traj['lang_vector'],
            data_source=optimal_traj['data_source'],
            frames=np.array(optimal_traj['frames']),
            frames_shape=optimal_frames_shape,
            quality_label=optimal_traj.get('quality_label', 'successful'),
            is_robot=optimal_traj['is_robot'],
            metadata=optimal_traj.get('metadata', {}).copy() if optimal_traj.get('metadata') else {},
            # Preference-specific fields - using chosen/rejected naming
            chosen_frames=optimal_frames,
            rejected_frames=negative_frames,
            chosen_frames_shape=optimal_frames_shape,
            rejected_frames_shape=negative_frames_shape,
            preferred_trajectory="chosen",  # chosen is the optimal trajectory
            chosen_id=optimal_traj['id'],
            rejected_id=negative_traj['id'],
            # Rejected trajectory fields
            rejected_task=negative_traj['task'],
            rejected_lang_vector=negative_traj['lang_vector'],
            rejected_data_source=negative_traj['data_source'],
            rejected_quality_label=negative_traj.get('quality_label'),
            rejected_is_robot=negative_traj['is_robot'],
            # Progress fields
            target_progress_A=target_progress_A,
            target_progress_B=target_progress_B,
            sample_type=strategy_used
        )    
        return sample
    
    def _calculate_target_progress(self, trajectory: Dict, frames: np.ndarray = None) -> List[float]:
        """Calculate target progress values for a trajectory."""
        
        # Check if this is a rewind trajectory (has rewind progress in metadata)
        if (trajectory.get('metadata') and 
            trajectory['metadata'].get('rewind_generated') and 
            trajectory['metadata'].get('rewind_progress')):
            # Use the rewind progress that was calculated during rewind generation
            return trajectory['metadata']['rewind_progress']
        
        # If frames not provided, get them from trajectory
        if frames is None:
            frames = self._load_frames_from_npz(trajectory['frames'])
        
        # Calculate number of frames
        if hasattr(frames, 'shape'):
            num_frames = frames.shape[0]  # Use shape[0] for numpy array
        else:
            num_frames = len(frames)
        
        # For optimal trajectories, use linear progress (0.0 to 1.0)
        return [i / (num_frames - 1) for i in range(num_frames)]
    
    def _create_similarity_sample(self) -> SimilaritySample:
        """Create a similarity scoring sample: o^1 and o^2 ranked against o^ref.
        
        Two modes (50/50 split):
        1. Rewind mode: o^1 is rewound from same task, o^2 is from different task
        2. Optimal/Suboptimal mode: o^1 is optimal/suboptimal from same task, o^2 varies
        """
        
        # Randomly choose between rewind mode and optimal/suboptimal mode
        use_rewind_mode = random.choice([True, False])
        
        if use_rewind_mode:
            return self._create_rewind_similarity_sample()
        else:
            return self._create_optimal_similarity_sample()
    
    def _create_rewind_similarity_sample(self) -> SimilaritySample:
        """Create similarity sample using rewind logic.
        
        Rules:
        - traj_sim is rewound trajectory from same task as o^ref (different trajectory)
        - traj_diff MUST be from different task
        - traj_sim is preferred over traj_diff relative to o^ref
        
        Minimal requirements:
        - At least 2 trajectories in reference task (for o^ref and traj_sim rewind)
        - At least 2 tasks (for traj_diff from different task)
        """
        
        # Get available tasks
        task_names = list(self.optimal_by_task.keys())
        
        # Check minimal requirements
        if len(task_names) < 2:
            raise ValueError(f"Rewind similarity sample requires at least 2 tasks, but only {len(task_names)} available")
        
        # Select reference task
        task_ref = random.choice(task_names)
        ref_task_indices = self.optimal_by_task[task_ref]
        
        # Check minimal requirements for reference task
        if len(ref_task_indices) < 2:
            raise ValueError(f"Task '{task_ref}' has only {len(ref_task_indices)} trajectory(ies). "
                           f"Need at least 2 trajectories in reference task for rewind similarity sample.")
        
        # Select reference trajectory
        ref_idx = random.choice(ref_task_indices)
        ref_traj = self.dataset[ref_idx]
        
        # traj_sim is a rewound trajectory from same task (different from ref)
        available_sim_indices = [idx for idx in ref_task_indices if idx != ref_idx]
        if not available_sim_indices:
            raise ValueError(f"Cannot create rewound traj_sim: no trajectories available in task '{task_ref}' "
                           f"different from reference trajectory {ref_traj['id']}")
        sim_idx = random.choice(available_sim_indices)
        traj_sim = self.dataset[sim_idx]
        
        # traj_diff MUST be from different task
        other_task = random.choice([t for t in task_names if t != task_ref])
        other_task_indices = self.optimal_by_task[other_task]
        if not other_task_indices:
            raise ValueError(f"Task '{other_task}' has no trajectories available for traj_diff")
        diff_idx = random.choice(other_task_indices)
        traj_diff = self.dataset[diff_idx]
        
        # Final validation
        self._validate_similarity_trajectories(ref_traj, traj_sim, traj_diff)
        
        # Deserialize frames and create sample
        return self._build_similarity_sample(ref_traj, traj_sim, traj_diff, is_rewind=True, strategy_used="rewind_same_task")
    
    def _create_optimal_similarity_sample(self) -> SimilaritySample:
        """Create similarity sample using optimal/suboptimal logic.
        
        Rules:
        - traj_sim is always from the same task as o^ref (different trajectory)
        - traj_sim is always preferred over traj_diff relative to o^ref
        - If traj_sim is suboptimal, then traj_diff must be from a different task
        - If traj_sim is optimal, then traj_diff can be suboptimal from the same task or from a different task
        
        Minimal requirements:
        - At least 2 trajectories in the reference task (for o^ref and traj_sim)
        - At least 1 other trajectory available for traj_diff (same or different task)
        """
        
        # Get available tasks
        task_names = list(self.optimal_by_task.keys())
        
        # Select reference task
        task_ref = random.choice(task_names)
        
        # Get optimal and all trajectories from reference task
        ref_optimal_indices = self.optimal_by_task[task_ref]
        ref_all_indices = self.optimal_by_task[task_ref] + self.suboptimal_by_task.get(task_ref, [])
        
        # Check minimal requirements for reference task
        if len(ref_all_indices) < 2:
            raise ValueError(f"Task '{task_ref}' has only {len(ref_all_indices)} trajectory(ies). "
                           f"Need at least 2 trajectories in reference task for optimal similarity sample.")
        
        if not ref_optimal_indices:
            # Fall back to all trajectories if no optimal ones
            ref_optimal_indices = ref_all_indices
        
        # Select reference trajectory (can be optimal or suboptimal)
        ref_idx = random.choice(ref_all_indices)
        ref_traj = self.dataset[ref_idx]
        
        # Decide if traj_sim should be optimal or suboptimal
        use_optimal_sim = random.choice([True, False])
        
        if use_optimal_sim and ref_optimal_indices:
            traj_sim, traj_diff = self._select_optimal_sim_trajectories(
                task_ref, task_names, ref_traj, ref_optimal_indices, ref_all_indices
            )
        else:
            traj_sim, traj_diff = self._select_suboptimal_sim_trajectories(
                task_ref, task_names, ref_traj, ref_optimal_indices, ref_all_indices
            )
        
        # Final validation
        self._validate_similarity_trajectories(ref_traj, traj_sim, traj_diff)
        
        # Deserialize frames and create sample
        return self._build_similarity_sample(ref_traj, traj_sim, traj_diff, is_rewind=False, strategy_used="optimal_same_task")
    
    def _select_optimal_sim_trajectories(self, task_ref, task_names, ref_traj, ref_optimal_indices, ref_all_indices):
        """Select trajectories when traj_sim should be optimal."""
        # traj_sim is optimal from the same task as ref (must be different trajectory)
        available_sim_indices = [idx for idx in ref_optimal_indices if self.dataset[idx]['id'] != ref_traj['id']]
        if not available_sim_indices:
            raise ValueError(f"Cannot create optimal traj_sim: no optimal trajectories available in task '{task_ref}' "
                           f"different from reference trajectory {ref_traj['id']}")
        sim_idx = random.choice(available_sim_indices)
        traj_sim = self.dataset[sim_idx]
        
        # traj_diff can be suboptimal from same task OR from different task
        if len(task_names) > 1 and random.choice([True, False]):
            # Choose traj_diff from different task
            other_task = random.choice([t for t in task_names if t != task_ref])
            other_task_indices = self.optimal_by_task[other_task]
            if not other_task_indices:
                raise ValueError(f"Task '{other_task}' has no trajectories available for traj_diff")
            diff_idx = random.choice(other_task_indices)
            traj_diff = self.dataset[diff_idx]
        else:
            # Choose traj_diff as suboptimal from same task
            ref_suboptimal_indices = [idx for idx in ref_all_indices 
                                    if idx not in ref_optimal_indices 
                                    and self.dataset[idx]['id'] not in [ref_traj['id'], traj_sim['id']]]
            if not ref_suboptimal_indices:
                # Try any trajectory from same task that's different from ref and traj_sim
                available_same_task_indices = [idx for idx in ref_all_indices 
                                            if self.dataset[idx]['id'] not in [ref_traj['id'], traj_sim['id']]]
                if not available_same_task_indices:
                    # Must use different task
                    if len(task_names) < 2:
                        raise ValueError(f"Cannot create traj_diff: only one task available and no trajectories "
                                       f"in task '{task_ref}' different from ref {ref_traj['id']} and traj_sim {traj_sim['id']}")
                    other_task = random.choice([t for t in task_names if t != task_ref])
                    other_task_indices = self.optimal_by_task[other_task]
                    if not other_task_indices:
                        raise ValueError(f"Task '{other_task}' has no trajectories available for traj_diff")
                    diff_idx = random.choice(other_task_indices)
                    traj_diff = self.dataset[diff_idx]
                else:
                    diff_idx = random.choice(available_same_task_indices)
                    traj_diff = self.dataset[diff_idx]
            else:
                diff_idx = random.choice(ref_suboptimal_indices)
                traj_diff = self.dataset[diff_idx]
        
        return traj_sim, traj_diff
    
    def _select_suboptimal_sim_trajectories(self, task_ref, task_names, ref_traj, ref_optimal_indices, ref_all_indices):
        """Select trajectories when traj_sim should be suboptimal."""
        # traj_sim is suboptimal from the same task as ref (must be different trajectory)
        ref_suboptimal_indices = [idx for idx in ref_all_indices 
                                     if idx not in ref_optimal_indices 
                                     and self.dataset[idx]['id'] != ref_traj['id']]
        if not ref_suboptimal_indices:
            # Fallback to any trajectory from same task different from ref
            available_same_task_indices = [idx for idx in ref_all_indices if self.dataset[idx]['id'] != ref_traj['id']]
            if not available_same_task_indices:
                raise ValueError(f"Cannot create traj_sim: no trajectories available in task '{task_ref}' "
                               f"different from reference trajectory {ref_traj['id']}")
            sim_idx = random.choice(available_same_task_indices)
            traj_sim = self.dataset[sim_idx]
        else:
            sim_idx = random.choice(ref_suboptimal_indices)
            traj_sim = self.dataset[sim_idx]
        
        # traj_diff MUST be from different task (since traj_sim is suboptimal)
        if len(task_names) < 2:
            raise ValueError(f"Cannot create traj_diff: traj_sim is suboptimal so traj_diff must be from different task, "
                           f"but only one task '{task_ref}' is available")
        
        other_task = random.choice([t for t in task_names if t != task_ref])
        other_task_indices = self.optimal_by_task[other_task]
        if not other_task_indices:
            raise ValueError(f"Task '{other_task}' has no trajectories available for traj_diff")
        diff_idx = random.choice(other_task_indices)
        traj_diff = self.dataset[diff_idx]
        
        return traj_sim, traj_diff
    
    def _validate_similarity_trajectories(self, ref_traj, traj_sim, traj_diff):
        """Validate that all trajectories have unique IDs."""
        if traj_sim['id'] == ref_traj['id']:
            raise ValueError(f"traj_sim and o^ref have the same trajectory ID: {traj_sim['id']}")
        
        if traj_diff['id'] == ref_traj['id']:
            raise ValueError(f"traj_diff and o^ref have the same trajectory ID: {traj_diff['id']}")
        
        if traj_sim['id'] == traj_diff['id']:
            raise ValueError(f"traj_sim and traj_diff have the same trajectory ID: {traj_sim['id']}")
    
    def _build_similarity_sample(self, ref_traj, traj_sim, traj_diff, is_rewind=False, strategy_used=None):
        """Build the final similarity sample from trajectories."""
        # Get frames from npz files
        ref_frames = self._load_frames_from_npz(ref_traj['frames'])
        traj_sim_frames = self._load_frames_from_npz(traj_sim['frames'])
        traj_diff_frames = self._load_frames_from_npz(traj_diff['frames'])
        
        # Calculate target progress for all trajectories
        target_progress_A = self._calculate_target_progress(traj_sim, traj_sim_frames)
        target_progress_B = self._calculate_target_progress(traj_diff, traj_diff_frames)
        target_progress_ref = self._calculate_target_progress(ref_traj, ref_frames)

        # Get frame shapes and convert to tuples if needed
        ref_frames_shape = ref_traj.get('frames_shape')
        if isinstance(ref_frames_shape, list):
            ref_frames_shape = tuple(ref_frames_shape)
        
        traj_sim_frames_shape = traj_sim.get('frames_shape')
        if isinstance(traj_sim_frames_shape, list):
            traj_sim_frames_shape = tuple(traj_sim_frames_shape)
        
        traj_diff_frames_shape = traj_diff.get('frames_shape')
        if isinstance(traj_diff_frames_shape, list):
            traj_diff_frames_shape = tuple(traj_diff_frames_shape)
        
        # Create similarity sample structure
        sample = SimilaritySample(
            # Core HF dataset fields (from ref_traj)
            id=ref_traj['id'],
            task=ref_traj['task'],
            lang_vector=ref_traj['lang_vector'],
            data_source=ref_traj['data_source'],
            frames=np.array(ref_traj['frames']),
            frames_shape=ref_frames_shape,
            quality_label=ref_traj.get('quality_label', 'successful'),
            is_robot=ref_traj['is_robot'],
            metadata=ref_traj.get('metadata'),
            # Similarity-specific fields - using traj_sim/traj_diff naming
            reference_frames=np.array(ref_traj['frames']),  # o^ref
            traj_sim_frames=np.array(traj_sim['frames']),   # Similar trajectory
            traj_diff_frames=np.array(traj_diff['frames']), # Different trajectory
            reference_frames_shape=ref_frames_shape,
            traj_sim_frames_shape=traj_sim_frames_shape,
            traj_diff_frames_shape=traj_diff_frames_shape,
            task_ref=ref_traj['task'],
            task_sim=traj_sim['task'],
            task_diff=traj_diff['task'],
            ref_trajectory_id=ref_traj['id'],
            traj_sim_id=traj_sim['id'],
            traj_diff_id=traj_diff['id'],
            # Similar trajectory fields
            traj_sim_task=traj_sim['task'],
            traj_sim_lang_vector=traj_sim['lang_vector'],
            traj_sim_data_source=traj_sim['data_source'],
            traj_sim_quality_label=traj_sim.get('quality_label'),
            traj_sim_is_robot=traj_sim['is_robot'],
            # Different trajectory fields
            traj_diff_task=traj_diff['task'],
            traj_diff_lang_vector=traj_diff['lang_vector'],
            traj_diff_data_source=traj_diff['data_source'],
            traj_diff_quality_label=traj_diff.get('quality_label'),
            traj_diff_is_robot=traj_diff['is_robot'],
            # Progress fields
            target_progress_A=target_progress_A,
            target_progress_B=target_progress_B,
            target_progress_ref=target_progress_ref,
            sample_type=strategy_used
        )
        
        return sample
    


# Infinite dataset that generates samples on-demand
class InfiniteDataGeneratorDataset:
    """Infinite dataset that generates samples on-demand with controlled ratios."""
    
    def __init__(self, data_generator: DataGenerator, max_samples: int = 1000000):
        """
        Initialize the infinite dataset.
        
        Args:
            data_generator: DataGenerator instance to use for generating samples
            max_samples: Maximum number of samples to generate (for len() method)
        """
        self.data_generator = data_generator
        self.max_samples = max_samples
        self.current_idx = 0
    
    def __len__(self):
        """Return the maximum number of samples."""
        return self.max_samples
    
    def __getitem__(self, idx):
        """Generate a sample on-demand based on the configured ratios."""
        # Determine which type of sample to generate based on ratios
        rand_val = random.random()
        
        if rand_val < self.data_generator.preference_ratio:
            # Generate preference sample
            return self.data_generator._create_preference_sample()
        else:
            # Generate similarity sample
            return self.data_generator._create_similarity_sample()


class InfinitePairedVideoDataset:
    """Infinite dataset that generates simple paired video comparison samples.
    
    This dataset samples two random videos from the dataset and creates comparison samples:
    - Half the time: second video is rewound from the first video
    - Half the time: second video is a random other trajectory
    """
    
    def __init__(self, data_generator: DataGenerator, max_samples: int = 1000000):
        """
        Initialize the paired video dataset.
        
        Args:
            data_generator: DataGenerator instance to use for generating samples
            max_samples: Maximum number of samples to generate (for len() method)
        """
        self.data_generator = data_generator
        self.max_samples = max_samples
        self.current_idx = 0
    
    def __len__(self):
        """Return the maximum number of samples."""
        return self.max_samples
    
    def __getitem__(self, idx):
        """Generate a paired video sample on-demand."""
        # Randomly choose between rewind and random trajectory for second video
        use_rewind = random.choice([True, False])
        
        if use_rewind:
            return self._create_rewind_paired_sample()
        else:
            return self._create_random_paired_sample()
    
    def _create_rewind_paired_sample(self):
        """Create a sample where the second video is rewound from the first."""
        # Get a random trajectory
        if not self.data_generator.optimal_by_task:
            raise ValueError("No optimal trajectories found for paired video generation")
        
        task_name = random.choice(list(self.data_generator.optimal_by_task.keys()))
        first_idx = random.choice(self.data_generator.optimal_by_task[task_name])
        first_traj = self.data_generator.dataset[first_idx]
        
        # Create rewound version of the first trajectory
        second_traj = self.data_generator._create_rewind_trajectory(first_traj)
        
        # Create a simple paired sample
        return PairedVideoSample(
            id=f"paired_rewind_{first_traj['id']}",
            task=first_traj['task'],
            lang_vector=first_traj['lang_vector'],
            data_source=first_traj['data_source'],
            frames=first_traj['frames'],  # BaseSample requirement
            frames_shape=first_traj.get('frames_shape'),
            quality_label=first_traj.get('quality_label', 'successful'),
            is_robot=first_traj.get('is_robot', True),
            metadata=first_traj.get('metadata', {}),
            # Paired video specific fields
            traj_A_frames=self.data_generator._get_trajectory_frames(first_idx),
            traj_B_frames=second_traj['frames'],  # Already numpy array from rewind
            traj_A_frames_shape=self.data_generator._get_trajectory_frames(first_idx).shape,
            traj_B_frames_shape=second_traj['frames'].shape,
            traj_A_id=first_traj['id'],
            traj_B_id=second_traj['id'],
            traj_A_task=first_traj['task'],
            traj_B_task=second_traj['task'],
            traj_A_lang_vector=first_traj['lang_vector'],
            traj_B_lang_vector=second_traj.get('lang_vector'),
            traj_A_data_source=first_traj['data_source'],
            traj_B_data_source=second_traj.get('data_source'),
            traj_A_quality_label=first_traj.get('quality_label', 'successful'),
            traj_B_quality_label=second_traj.get('quality_label', 'suboptimal'),
            traj_A_is_robot=first_traj.get('is_robot', True),
            traj_B_is_robot=second_traj.get('is_robot', True),
            sample_type='rewind_paired'
        )
    
    def _create_random_paired_sample(self):
        """Create a sample where the second video is a random other trajectory."""
        # Get a random trajectory
        if not self.data_generator.optimal_by_task:
            raise ValueError("No optimal trajectories found for paired video generation")
        
        task_name = random.choice(list(self.data_generator.optimal_by_task.keys()))
        first_idx = random.choice(self.data_generator.optimal_by_task[task_name])
        first_traj = self.data_generator.dataset[first_idx]
        
        # Get a random different trajectory (could be from same or different task)
        all_trajectories = []
        for task_indices in self.data_generator.optimal_by_task.values():
            all_trajectories.extend(task_indices)
        
        # Remove the first trajectory from consideration
        available_indices = [idx for idx in all_trajectories if idx != first_idx]
        if not available_indices:
            raise ValueError("No other trajectories available for paired video generation")
        
        second_idx = random.choice(available_indices)
        second_traj = self.data_generator.dataset[second_idx]
        
        # Create a simple paired sample
        return PairedVideoSample(
            id=f"paired_random_{first_traj['id']}_{second_traj['id']}",
            task=first_traj['task'],
            lang_vector=first_traj['lang_vector'],
            data_source=first_traj['data_source'],
            frames=first_traj['frames'],  # BaseSample requirement
            frames_shape=first_traj.get('frames_shape'),
            quality_label=first_traj.get('quality_label', 'successful'),
            is_robot=first_traj.get('is_robot', True),
            metadata=first_traj.get('metadata', {}),
            # Paired video specific fields
            traj_A_frames=self.data_generator._get_trajectory_frames(first_idx),
            traj_B_frames=self.data_generator._get_trajectory_frames(second_idx),
            traj_A_frames_shape=self.data_generator._get_trajectory_frames(first_idx).shape,
            traj_B_frames_shape=self.data_generator._get_trajectory_frames(second_idx).shape,
            traj_A_id=first_traj['id'],
            traj_B_id=second_traj['id'],
            traj_A_task=first_traj['task'],
            traj_B_task=second_traj['task'],
            traj_A_lang_vector=first_traj['lang_vector'],
            traj_B_lang_vector=second_traj.get('lang_vector'),
            traj_A_data_source=first_traj['data_source'],
            traj_B_data_source=second_traj.get('data_source'),
            traj_A_quality_label=first_traj.get('quality_label', 'successful'),
            traj_B_quality_label=second_traj.get('quality_label', 'suboptimal'),
            traj_A_is_robot=first_traj.get('is_robot', True),
            traj_B_is_robot=second_traj.get('is_robot', True),
            sample_type='random_paired'
        )


def test():
    """Test the BatchCollator with generated samples."""
    from transformers import AutoProcessor
    
    # Create a mock config for testing
    from dataclasses import dataclass
    from typing import List
    
    @dataclass
    class MockDataConfig:
        train_datasets: List[str] = None
        train_subsets: List[str] = None
        eval_datasets: List[str] = None
        eval_subsets: List[str] = None
        preference_ratio: float = 1.0
        similarity_ratio: float = 0.0
        dataset_preference_ratio: float = 0.7
        shuffle: bool = True
        seed: int = 42
        num_proc: int = 4
        max_frames: int = 32
        force_reprocess: bool = False
        dataloader_pin_memory: bool = False
        dataloader_num_workers: int = 0
    
    @dataclass
    class MockConfig:
        data: MockDataConfig = None
        debug: bool = False
    
    # Create mock config
    mock_data_config = MockDataConfig(
        train_datasets=["abraranwar/libero_rfm"],
        train_subsets=["libero_10"],
        preference_ratio=1.0,
        similarity_ratio=0.0,
        shuffle=True,
        seed=42,
        num_proc=4,
        max_frames=32,
        force_reprocess=False
    )
    
    mock_config = MockConfig(data=mock_data_config, debug=False)
    
    # Create data generator with mock config
    generator = DataGenerator(config=mock_config)
    
    # Test the infinite dataset
    rank_0_print("Testing InfiniteDataGeneratorDataset...")
    infinite_dataset = InfiniteDataGeneratorDataset(generator, max_samples=10)
    
    preference_count = 0
    similarity_count = 0
    
    for i in range(10):
        sample = infinite_dataset[i]
        if sample.sample_type == "preference":
            preference_count += 1
        else:
            similarity_count += 1
        rank_0_print(f"Sample {i}: {sample.sample_type}")
    
    rank_0_print(f"Generated {preference_count} preference samples and {similarity_count} similarity samples")
    rank_0_print(f"Expected ratio: {generator.preference_ratio:.1f} preference, {generator.similarity_ratio:.1f} similarity")
    
    # Test the batch collator with infinite dataset
    rank_0_print("\nTesting batch collator with infinite dataset...")
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    batch_collator = BatchCollator(processor, max_length=1024)
    
    # Generate a batch from the infinite dataset
    batch = []
    for i in range(10):  # Generate 4 samples
        sample = infinite_dataset[i]
        batch.append(sample)
    
    processed_batch = batch_collator(batch)
    for key, value in processed_batch.items():
        rank_0_print(key)
        if key == "preference_inputs":
            for key2, value2 in value.items():
                if key2 != "sample_type":
                    rank_0_print(f"{key2} {value2.shape}")
        elif key == "similarity_inputs":
            for key2, value2 in value.items():
                if key2 != "sample_type":
                    rank_0_print(f"{key2} {value2.shape}")

    # Do a quick forward pass on RFMModel
    from transformers import Qwen2_5_VLModel, AutoProcessor
    from rfm.models.rfm import RFMModel
    
    # Load base model and create RFMModel
    base_model = Qwen2_5_VLModel.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    # Add RFM special tokens if they don't exist
    special_tokens = ["<|split_token|>", "<|reward_token|>", "<|pref_token|>"]
    for token in special_tokens:
        if token not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_special_tokens({"additional_special_tokens": [token]})
            rank_0_print(f"Added special token: {token}")
    
    # Resize token embeddings if new tokens were added
    if len(processor.tokenizer) != base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(processor.tokenizer))
        rank_0_print(f"Resized token embeddings to {len(processor.tokenizer)}")
    
    rfm_model = RFMModel(config=base_model.config, processor=processor)
    rfm_model.model.load_state_dict(base_model.state_dict())
    
    # Check if CUDA is available
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    rfm_model = rfm_model.to(device)
    inputs = processed_batch["preference_inputs"]

    import ipdb; ipdb.set_trace()
    # Debug video grid dimensions in test
    rank_0_print(f"TEST DEBUG: video_grid_thw shape: {inputs.get('video_grid_thw').shape if inputs.get('video_grid_thw') is not None else None}")
    rank_0_print(f"TEST DEBUG: pixel_values_videos shape: {inputs.get('pixel_values_videos').shape if inputs.get('pixel_values_videos') is not None else None}")
    rank_0_print(f"TEST DEBUG: second_per_grid_ts shape: {inputs.get('second_per_grid_ts').shape if inputs.get('second_per_grid_ts') is not None else None}")

    outputs = rfm_model(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        # pixel_values=inputs.get("pixel_values").to(device),
        pixel_values_videos=inputs.get("pixel_values_videos").to(device),
        # image_grid_thw=inputs.get("image_grid_thw").to(device),
        video_grid_thw=inputs.get("video_grid_thw").to(device),
        second_per_grid_ts=inputs.get("second_per_grid_ts").to(device),
        sample_type="preference",  # Test preference prediction
    )

    rank_0_print("RFM model output structure:")
    rank_0_print(f"  logits: {outputs.shape if outputs is not None else None}")
    rank_0_print(f"  output type: {type(outputs)}")


if __name__ == "__main__":
    test() 