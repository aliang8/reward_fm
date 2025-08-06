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
from rfm.data.batch_collator import BaseSample, PreferenceSample, SimilaritySample, BatchCollator
from datasets import concatenate_datasets

class DataGenerator:
    """Data generator for producing batches of prediction data with controlled ratios."""
    
    def __init__(
        self,
        dataset_path: str = "rfm_dataset",
        dataset_subsets: List[str] = ["libero"],
        preference_dataset_path: Optional[str] = None,
        preference_dataset_subset: Optional[str] = None,
        batch_size: int = 32,
        preference_ratio: float = 0.5,
        similarity_ratio: float = 0.5,
        max_frames: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
        dataset_preference_ratio: float = 0.7,
        num_proc: int = 1,
        debug: bool = False,
    ):
        """
        Initialize the data generator.
        
        Args:
            dataset_path: Path to the HuggingFace dataset or dataset name
            dataset_subsets: List of dataset names to load (e.g., ["libero", "droid"] or ["libero"])
            preference_dataset_path: Optional path to preference dataset
            preference_dataset_subset: Optional subset name for preference dataset
            batch_size: Number of samples per batch
            preference_ratio: Ratio of preference prediction samples (0.0 to 1.0)
            similarity_ratio: Ratio of similarity scoring samples (0.0 to 1.0)
            max_frames: Maximum frames per trajectory
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
            dataset_preference_ratio: Ratio of preferences from dataset vs generated (0.0 to 1.0)
            num_proc: Number of processes to use for dataset processing (default: 1)
            debug: Whether to enable debug mode (reduces dataset size, enables debug features)
        """
        self.dataset_path = dataset_path
        self.dataset_subsets = dataset_subsets
        self.preference_dataset_path = preference_dataset_path
        self.preference_dataset_subset = preference_dataset_subset
        self.batch_size = batch_size
        self.preference_ratio = preference_ratio
        self.similarity_ratio = similarity_ratio
        self.max_frames = max_frames
        self.shuffle = shuffle
        self.seed = seed
        self.dataset_preference_ratio = dataset_preference_ratio
        self.num_proc = num_proc
        self.debug = debug 
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Load trajectory dataset (includes frame processing)
        self._load_trajectory_dataset()
        
        # Load preference dataset if provided
        self._load_preference_dataset()
        
        # Group trajectories by task for efficient sampling
        self._group_trajectories_by_task()
        
        # Preprocess trajectory categories for efficient sampling
        self._preprocess_trajectory_categories()
        
        # Initialize sentence transformer model
        self.lang_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"DataGenerator initialized with {len(self.trajectories)} total trajectories")
        print(f"Loaded datasets: {dataset_subsets}")
        if self.preferences:
            print(f"Loaded {len(self.preferences)} preference pairs")
        print(f"Batch size: {batch_size}")
        print(f"Ratios - Preference: {preference_ratio}, Similarity: {similarity_ratio}")
        print(f"Dataset preference ratio: {dataset_preference_ratio}")
    
    def _serialize_frames(self, frames_array: np.ndarray) -> bytes:
        """
        Serialize frames array to bytes for efficient storage.
        
        Args:
            frames_array: numpy array with shape (T, H, W, C)
            
        Returns:
            Serialized bytes
        """
        if frames_array.size == 0:
            return b''
        return frames_array.tobytes()
    
    def _deserialize_frames(self, bytes_blob: bytes, shape: tuple = (32, 240, 240, 3)) -> np.ndarray:
        """
        Deserialize bytes back to frames array.
        
        Args:
            bytes_blob: Serialized bytes
            shape: Expected shape of the frames array (T, H, W, C)
            
        Returns:
            Deserialized numpy array
        """
        if not bytes_blob:
            return np.array([])
        return np.frombuffer(bytes_blob, dtype=np.uint8).reshape(shape)
    
    def _preprocess_videos(self, frames, num_frames: int = 32) -> np.ndarray:
        """
        Downsample frames to the specified number using uniform sampling and pad to max_frames.
        
        Args:
            frames: VideoReader object from HuggingFace Video feature
            num_frames: Number of frames to extract (default: 32)
            
        Returns:
            frames as numpy arrays with shape (max_frames, H, W, C) where T is time dimension
        """
        if not frames:
            return np.array([])
        
        # Convert VideoReader to list of frames
        all_frames = list(frames)
        total_frames = len(all_frames)
        
        if total_frames == 0:
            return np.array([])
        
        # Stack frames into tensor and convert to numpy
        # Each frame["data"] should be HxWxC, stacking gives TxHxWxC
        frames_tensor = torch.stack([frame["data"] for frame in all_frames])
        frames_array = frames_tensor.numpy()
        
        # Ensure we have the correct shape: (T, H, W, C)
        if len(frames_array.shape) != 4:
            raise ValueError(f"Expected 4D array (T, H, W, C), got shape {frames_array.shape}")
        
        # Convert from CxHxW to HxWxC
        if frames_array.shape[1] == 3:
            frames_array = np.transpose(frames_array, (0, 2, 3, 1))

        if total_frames <= num_frames:
            # If video has fewer frames than requested, pad with last frame
            if total_frames < num_frames:
                last_frame = frames_array[-1]  # Get the last frame
                padding_frames = np.repeat(last_frame[np.newaxis, :, :, :], num_frames - total_frames, axis=0)
                frames_array = np.concatenate([frames_array, padding_frames], axis=0)
            return frames_array
        else:
            # Uniform sampling across the video
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            return frames_array[frame_indices]
    
    def _process_dataset_videos_map(self, dataset):
        """
        Process dataset frames using .map() method for efficient on-the-fly processing.
        
        Args:
            dataset: HuggingFace dataset containing trajectories
            
        Returns:
            Dataset with processed frames
        """
        # Check if frames are already processed (serialized bytes or numpy arrays)
        sample_item = dataset[0]
        frames_data = sample_item.get('frames')
        
        if isinstance(frames_data, bytes) or isinstance(frames_data, np.ndarray):
            print("Frames already processed, skipping downsampling.")
            return dataset
        
        print("Downsampling video frames using .map()...")
        
        def process_videos(example):
            """Downsample frames in a single example and serialize."""
            frames = example.get('frames_path')
            frames_array = self._preprocess_videos(frames, self.max_frames)
            
            # Serialize frames to bytes for efficient storage
            frames_bytes = self._serialize_frames(frames_array)
            
            # Update the example with serialized frames
            del example["frames_path"]
            example["frames"] = frames_bytes
            return example
        
        # Apply the mapping function to the dataset
        processed_dataset = dataset.map(
            process_videos,
            desc="Processing videos",
            num_proc=self.num_proc
        )

        print(f"Frame downsampling complete. Each trajectory now has {self.max_frames} frames.")
        
        # Save the processed dataset to disk for future fast loading
        cache_dir = f"./processed_datasets/{self.dataset_path.replace('/', '_')}_{self.max_frames}frames"
        print(f"Saving processed dataset to {cache_dir} for future fast loading...")
        processed_dataset.save_to_disk(cache_dir)
        print(f"Processed dataset saved to {cache_dir}")
        
        return processed_dataset
    
    def _create_rewind_trajectory(self, original_traj: Dict) -> Dict:
        """Create a suboptimal trajectory by rewinding the original trajectory."""
        frames_data = original_traj['frames']
        
        # Deserialize frames if they're bytes
        if isinstance(frames_data, bytes):
            frames = self._deserialize_frames(frames_data)
        else:
            frames = frames_data
        
        if len(frames) < 4:
            # If trajectory is too short, just return the original
            return original_traj
        
        # Randomly select start and end points for the forward segment
        start_idx = random.randint(0, len(frames) // 2)
        end_idx = random.randint(len(frames) // 2, len(frames))
        
        # Ensure minimum segment length
        while end_idx - start_idx < 3:
            start_idx = random.randint(0, len(frames) // 2)
            end_idx = random.randint(len(frames) // 2, len(frames))
        
        # Extract forward segment
        forward_frames = frames[start_idx:end_idx]
        forward_progress = [(i + 1) / len(frames[start_idx:]) for i in range(len(forward_frames))]
        
        # Create rewind segment (reverse the forward segment)
        selected_end_point = random.randint(2, len(forward_frames) - 1)
        reverse_frames = forward_frames[::-1][1:selected_end_point]
        reverse_progress = forward_progress[::-1][1:selected_end_point]
        
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
        rewind_traj['id'] = f"{original_traj['id']}_rewind_{random.randint(1000, 9999)}"
        rewind_traj['optimal'] = False  # Mark as suboptimal
        rewind_traj['metadata'] = rewind_traj.get('metadata', {}).copy()
        rewind_traj['metadata']['rewind_generated'] = True
        rewind_traj['metadata']['original_traj_id'] = original_traj['id']
        rewind_traj['metadata']['rewind_progress'] = combined_progress
        
        return rewind_traj
    
    def _load_trajectory_dataset(self):
        """Load the trajectory dataset from disk or hub."""
        print(f"Loading trajectory dataset from: {self.dataset_path}")
        
        # Load multiple subsets and combine them
        all_datasets = []
        
        for dataset_name in self.dataset_subsets:
            print(f"Loading dataset: {dataset_name}")
            dataset = self._load_dataset_from_path(self.dataset_path, dataset_name)
            
            # Handle DatasetDict by accessing the train split
            if hasattr(dataset, 'keys') and 'train' in dataset:
                print(f"  Found DatasetDict with train split, accessing train data...")
                dataset = dataset['train']
            
            # Check if frames need processing by examining the first sample
            if len(dataset) > 0:
                sample_item = dataset[0]
                frames_data = sample_item.get('frames')
                
                # If frames are already numpy arrays, skip processing
                if isinstance(frames_data, np.ndarray):
                    print(f"  Frames already processed (numpy arrays), skipping frame processing")
                elif isinstance(frames_data, bytes):
                    print(f"  Frames are serialized bytes, will be deserialized on-demand")
                else:
                    print(f"  Frames need processing, applying .map()...")
                    dataset = self._process_dataset_videos_map(dataset)
            else:
                print(f"  Empty dataset, skipping frame processing")
            
            all_datasets.append(dataset)
            print(f"  Loaded {len(dataset)} trajectories from dataset '{dataset_name}'")
        
        # Combine all datasets
        if len(all_datasets) == 1:
            self.trajectories = all_datasets[0]
        else:
            self.trajectories = concatenate_datasets(all_datasets)
        
        print(f"Combined {len(self.trajectories)} total trajectories from {len(self.dataset_subsets)} datasets")
    
    def _load_preference_dataset(self):
        """Load the preference dataset from disk or hub if provided."""
        self.preferences = []
        
        if not self.preference_dataset_path:
            print("No preference dataset provided, will use random sampling for preferences")
            return
        
        print(f"Loading preference dataset from: {self.preference_dataset_path}")
        
        # Load preference dataset using helper method
        preference_dataset = self._load_dataset_from_path(
            self.preference_dataset_path, 
            self.preference_dataset_subset
        )
        
        # Convert to Preference objects
        for item in preference_dataset:
            from dataset_types import Preference
            preference = Preference(
                traj_id=item['traj_id'],
                chosen_id=item['chosen_id'],
                rejected_id=item['rejected_id']
            )
            self.preferences.append(preference)
        
        print(f"Loaded {len(self.preferences)} preference pairs from subset '{self.preference_dataset_subset}'")
        
    def _load_dataset_from_path(self, dataset_path: str, subset: str = None):
        """Helper method to load a dataset from path (local or hub)."""

        if '/' in dataset_path and not os.path.exists(dataset_path):
            # Check for cached processed dataset first
            cache_dir = f"./processed_datasets/{dataset_path.replace('/', '_')}_{self.max_frames}frames"
            if os.path.exists(cache_dir):
                print(f"Found cached processed dataset at {cache_dir}, loading...")
                ds = load_from_disk(cache_dir)
                return ds
            
            # Load from HuggingFace Hub
            from datasets import load_dataset, Video, Features
            print(f"Loading from HuggingFace Hub: {dataset_path}")

            def patch_path(old_path):
                root_dir = "/workspace/vlm_reward_model/rfm_dataset"
                return f"{root_dir}/{old_path}"       # e.g., "./videos/trajectory_0000.mp4"
            
            ds = load_dataset(dataset_path, name=subset, split="train")
            ds = ds.map(lambda x: {"frames_path": patch_path(x["frames"])})
            ds = ds.cast_column("frames_path", Video(decode=True))
            # Only select a small subset for debugging
            if self.debug:
                ds = ds.select(range(5))
                print("  Debug mode: Using only first 5 samples")
            return ds

        else:
            # Load from local disk
            if os.path.isdir(dataset_path):
                # It's a directory, load the specific subset
                dataset = load_from_disk(dataset_path)
                if subset:
                    return dataset[subset]
                else:
                    return dataset
            else:
                # It's a file, load directly
                return load_from_disk(dataset_path)
    
    def _group_trajectories_by_task(self):
        """Group trajectories by task name for efficient sampling."""
        self.task_groups = {}
        for traj in self.trajectories:
            task_name = traj['task']
            if task_name not in self.task_groups:
                self.task_groups[task_name] = []
            self.task_groups[task_name].append(traj)
        
        print(f"Grouped trajectories into {len(self.task_groups)} tasks")
        
        # Create trajectory ID mapping for quick lookup
        self.traj_id_to_traj = {traj['id']: traj for traj in self.trajectories}
    
    def _preprocess_trajectory_categories(self):
        """Preprocess trajectories into categories for efficient sampling."""
        
        # Categorize by robot vs human
        self.robot_trajectories = [traj for traj in self.trajectories if traj.get('is_robot', True)]
        self.human_trajectories = [traj for traj in self.trajectories if not traj.get('is_robot', True)]
        
        # Categorize by optimal vs suboptimal
        self.optimal_trajectories = [traj for traj in self.trajectories if traj.get('optimal', True)]
        self.suboptimal_trajectories = [traj for traj in self.trajectories if not traj.get('optimal', True)]
        
        # Categorize by task and optimality
        self.optimal_by_task = {}
        self.suboptimal_by_task = {}
        for task_name, task_trajectories in self.task_groups.items():
            self.optimal_by_task[task_name] = [traj for traj in task_trajectories if traj.get('optimal', True)]
            self.suboptimal_by_task[task_name] = [traj for traj in task_trajectories if not traj.get('optimal', True)]
        
        # Categorize by data source
        self.trajectories_by_source = {}
        for traj in self.trajectories:
            source = traj.get('data_source', 'unknown')
            if source not in self.trajectories_by_source:
                self.trajectories_by_source[source] = []
            self.trajectories_by_source[source].append(traj)
        
        print(f"Preprocessed trajectory categories:")
        print(f"  Robot trajectories: {len(self.robot_trajectories)}")
        print(f"  Human trajectories: {len(self.human_trajectories)}")
        print(f"  Optimal trajectories: {len(self.optimal_trajectories)}")
        print(f"  Suboptimal trajectories: {len(self.suboptimal_trajectories)}")
        print(f"  Data sources: {list(self.trajectories_by_source.keys())}")
    
    def get_optimal_trajectories_by_task(self, task_name: str) -> List[Dict]:
        """Get optimal trajectories for a specific task."""
        return self.optimal_by_task.get(task_name, [])
    
    def get_suboptimal_trajectories_by_task(self, task_name: str) -> List[Dict]:
        """Get suboptimal trajectories for a specific task."""
        return self.suboptimal_by_task.get(task_name, [])
    
    def get_trajectories_by_source(self, source: str) -> List[Dict]:
        """Get trajectories from a specific data source."""
        return self.trajectories_by_source.get(source, [])
    
    def get_robot_trajectories(self) -> List[Dict]:
        """Get all robot trajectories."""
        return self.robot_trajectories
    
    def get_human_trajectories(self) -> List[Dict]:
        """Get all human trajectories."""
        return self.human_trajectories
    
    def get_optimal_trajectories(self) -> List[Dict]:
        """Get all optimal trajectories."""
        return self.optimal_trajectories
    
    def get_suboptimal_trajectories(self) -> List[Dict]:
        """Get all suboptimal trajectories."""
        return self.suboptimal_trajectories
    
    def _create_preference_sample(self) -> PreferenceSample:
        """Create a preference prediction sample: o^1 vs o^2 where o^1 is preferred."""
        
        # Decide whether to use dataset preferences or generated preferences
        use_dataset_preference = (
            self.preferences and 
            random.random() < self.dataset_preference_ratio
        )
        
        if use_dataset_preference:
            # Use preference dataset
            preference = random.choice(self.preferences)
            
            # Get the trajectories by ID
            chosen_traj = self.traj_id_to_traj.get(preference.chosen_id)
            rejected_traj = self.traj_id_to_traj.get(preference.rejected_id)
            
            if chosen_traj is None or rejected_traj is None:
                raise ValueError(f"Could not find trajectories for preference {preference.traj_id}")
            
            # Calculate target progress for both trajectories
            target_progress_A = self._calculate_target_progress(chosen_traj)
            target_progress_B = self._calculate_target_progress(rejected_traj)
            
            # Create preference sample structure
            sample = PreferenceSample(
                # Core HF dataset fields (from chosen trajectory)
                id=chosen_traj['id'],
                task=chosen_traj['task'],
                lang_vector=chosen_traj['lang_vector'],
                data_source=chosen_traj['data_source'],
                frames=chosen_traj['frames'],
                optimal=chosen_traj['optimal'],
                is_robot=chosen_traj['is_robot'],
                metadata=chosen_traj.get('metadata'),
                # Preference-specific fields
                trajectory_A_frames=chosen_traj['frames'],
                trajectory_B_frames=rejected_traj['frames'],
                preferred_trajectory="A",  # A is the chosen trajectory
                trajectory_A_id=chosen_traj['id'],
                trajectory_B_id=rejected_traj['id'],
                # Trajectory B fields
                trajectory_B_task=rejected_traj['task'],
                trajectory_B_lang_vector=rejected_traj['lang_vector'],
                trajectory_B_data_source=rejected_traj['data_source'],
                trajectory_B_optimal=rejected_traj['optimal'],
                trajectory_B_is_robot=rejected_traj['is_robot'],
                # Progress fields
                target_progress_A=target_progress_A,
                target_progress_B=target_progress_B,
            )
        else:
            sample = self._create_subopt_rewind_traj()
        
        return sample
    
    def _create_subopt_rewind_traj(self) -> PreferenceSample:
        """Create a preference prediction sample using various negative generation strategies."""
        
        # Use preprocessed optimal trajectories
        if not self.optimal_trajectories:
            raise ValueError("No optimal trajectories found for preference generation")
        
        optimal_traj = random.choice(self.optimal_trajectories)
        
        # Choose negative generation strategy (equal probability for each)
        strategy = random.random()
        
        if strategy < 0.33:
            # Strategy 1: Use rewind-generated suboptimal trajectory
            negative_traj = self._create_rewind_trajectory(optimal_traj)
        elif strategy < 0.66:
            # Strategy 2: Use random trajectory from different task
            different_task_trajectories = [
                traj for traj in self.trajectories 
                if traj['task'] != optimal_traj['task']
            ]
            if different_task_trajectories:
                negative_traj = random.choice(different_task_trajectories)
            else:
                # Fall back to rewind if no different task trajectories
                negative_traj = self._create_rewind_trajectory(optimal_traj)
        else:
            # Strategy 3: Use random suboptimal trajectory from same task
            same_task_suboptimal = [
                traj for traj in self.suboptimal_trajectories 
                if traj['task'] == optimal_traj['task'] and traj['id'] != optimal_traj['id']
            ]
            if same_task_suboptimal:
                negative_traj = random.choice(same_task_suboptimal)
            else:
                # Fall back to rewind if no same-task suboptimal trajectories
                negative_traj = self._create_rewind_trajectory(optimal_traj)
        
        # Calculate target progress for both trajectories
        target_progress_A = self._calculate_target_progress(optimal_traj)
        target_progress_B = self._calculate_target_progress(negative_traj)
        
        # Create preference sample structure
        sample = PreferenceSample(
            # Core HF dataset fields (from optimal trajectory)
            id=optimal_traj['id'],
            task=optimal_traj['task'],
            lang_vector=optimal_traj['lang_vector'],
            data_source=optimal_traj['data_source'],
            frames=optimal_traj['frames'],
            optimal=optimal_traj['optimal'],
            is_robot=optimal_traj['is_robot'],
            metadata=optimal_traj.get('metadata'),
            # Preference-specific fields
            trajectory_A_frames=optimal_traj['frames'],
            trajectory_B_frames=negative_traj['frames'],
            preferred_trajectory="A",  # A is the optimal trajectory
            trajectory_A_id=optimal_traj['id'],
            trajectory_B_id=negative_traj['id'],
            # Trajectory B fields
            trajectory_B_task=negative_traj['task'],
            trajectory_B_lang_vector=negative_traj['lang_vector'],
            trajectory_B_data_source=negative_traj['data_source'],
            trajectory_B_optimal=negative_traj['optimal'],
            trajectory_B_is_robot=negative_traj['is_robot'],
            # Progress fields
            target_progress_A=target_progress_A,
            target_progress_B=target_progress_B,
        )
        
        return sample
    
    def _calculate_target_progress(self, trajectory: Dict) -> List[float]:
        """Calculate target progress values for a trajectory."""
        frames_data = trajectory['frames']
        
        # Deserialize frames if they're bytes
        if isinstance(frames_data, bytes):
            frames = self._deserialize_frames(frames_data)
            num_frames = frames.shape[0]  # Use shape[0] for numpy array
        else:
            frames = frames_data
            num_frames = len(frames)
        
        # Check if this is a rewind trajectory (has rewind progress in metadata)
        if (trajectory.get('metadata') and 
            trajectory['metadata'].get('rewind_generated') and 
            trajectory['metadata'].get('rewind_progress')):
            # Use the rewind progress that was calculated during rewind generation
            return trajectory['metadata']['rewind_progress']
        else:
            # For optimal trajectories, use linear progress (0.0 to 1.0)
            return [i / (num_frames - 1) for i in range(num_frames)]
    
    def _create_similarity_sample(self) -> SimilaritySample:
        """Create a similarity scoring sample: o^1 and o^2 ranked against o^ref."""
        
        # Get available tasks
        task_names = list(self.task_groups.keys())
        
        if len(task_names) < 2:
            # If only one task, use same task for all
            task_ref = task_names[0]
            task_other = task_names[0]
        else:
            # Randomly select two different tasks
            task_ref, task_other = random.sample(task_names, 2)
        
        # Get optimal trajectories from reference task
        ref_trajectories = self.get_optimal_trajectories_by_task(task_ref)
        if not ref_trajectories:
            # Fall back to all trajectories if no optimal ones
            ref_trajectories = self.task_groups[task_ref]
        ref_traj = random.choice(ref_trajectories)
        
        # Get trajectories from other task
        other_trajectories = self.task_groups[task_other]
        other_traj = random.choice(other_trajectories)
        
        # For o^1, use a trajectory from the same task as reference (optimal)
        if task_ref == task_other:
            # Same task case - pick different trajectory
            same_task_trajectories = [t for t in ref_trajectories if t['id'] != ref_traj['id']]
            if same_task_trajectories:
                traj_1 = random.choice(same_task_trajectories)
            else:
                traj_1 = ref_traj  # Fallback
        else:
            # Different task case - use trajectory from other task
            traj_1 = other_traj
        
        # For o^2, use a trajectory from different task (suboptimal)
        if task_ref != task_other:
            traj_2 = other_traj
        else:
            # Same task case - pick another trajectory
            same_task_trajectories = [t for t in ref_trajectories if t['id'] not in [ref_traj['id'], traj_1['id']]]
            if same_task_trajectories:
                traj_2 = random.choice(same_task_trajectories)
            else:
                traj_2 = traj_1  # Fallback
        
        # Calculate target progress for all trajectories
        target_progress_A = self._calculate_target_progress(traj_1)
        target_progress_B = self._calculate_target_progress(traj_2)
        
        # Create similarity sample structure
        sample = SimilaritySample(
            # Core HF dataset fields (from ref_traj)
            id=ref_traj['id'],
            task=ref_traj['task'],
            lang_vector=ref_traj['lang_vector'],
            data_source=ref_traj['data_source'],
            frames=ref_traj['frames'],
            optimal=ref_traj['optimal'],
            is_robot=ref_traj['is_robot'],
            metadata=ref_traj.get('metadata'),
            # Comparative-specific fields
            reference_frames=ref_traj['frames'],  # o^ref
            trajectory_A_frames=traj_1['frames'],  # o^1
            trajectory_B_frames=traj_2['frames'],  # o^2
            task_ref=task_ref,
            task_A=traj_1['task'],
            task_B=traj_2['task'],
            ref_trajectory_id=ref_traj['id'],
            trajectory_A_id=traj_1['id'],
            trajectory_B_id=traj_2['id'],
            # Trajectory A fields (o^1)
            trajectory_A_task=traj_1['task'],
            trajectory_A_lang_vector=traj_1['lang_vector'],
            trajectory_A_data_source=traj_1['data_source'],
            trajectory_A_optimal=traj_1['optimal'],
            trajectory_A_is_robot=traj_1['is_robot'],
            # Trajectory B fields (o^2)
            trajectory_B_task=traj_2['task'],
            trajectory_B_lang_vector=traj_2['lang_vector'],
            trajectory_B_data_source=traj_2['data_source'],
            trajectory_B_optimal=traj_2['optimal'],
            trajectory_B_is_robot=traj_2['is_robot'],
            # Progress fields
            target_progress_A=target_progress_A,
            target_progress_B=target_progress_B,
        )
        
        return sample
    
    def _determine_batch_composition(self) -> Dict[str, int]:
        """Determine how many samples of each type to include in the batch."""
        
        # Calculate target counts based on ratios
        total_samples = self.batch_size
        
        # For preference and similarity, use the specified ratios
        preference_count = int(total_samples * self.preference_ratio)
        similarity_count = int(total_samples * self.similarity_ratio)
        
        # Ensure we don't exceed batch size
        actual_total = preference_count + similarity_count
        if actual_total > total_samples:
            # Scale down proportionally
            scale_factor = total_samples / actual_total
            preference_count = int(preference_count * scale_factor)
            similarity_count = int(similarity_count * scale_factor)
        
        return {
            "preference": preference_count,
            "similarity": similarity_count
        }
    
    def generate_batch(self) -> List[Union[PreferenceSample, SimilaritySample]]:
        """Generate a single batch of data."""
        
        # Determine batch composition
        composition = self._determine_batch_composition()
        
        samples = []
        
        # Generate preference samples
        for _ in range(composition["preference"]):
            sample = self._create_preference_sample()
            samples.append(sample)
        
        # Generate similarity samples
        for _ in range(composition["similarity"]):
            sample = self._create_similarity_sample()
            samples.append(sample)
        
        # Shuffle if requested
        if self.shuffle:
            random.shuffle(samples)
        
        return samples
    
    def generate_batches(self, num_batches: int) -> Iterator[List[Union[PreferenceSample, SimilaritySample]]]:
        """Generate multiple batches."""
        for i in range(num_batches):
            yield self.generate_batch()

# Dataset wrapper for HuggingFace Trainer compatibility
class DataGeneratorDataset:
    """Dataset wrapper that uses DataGenerator to provide samples for HuggingFace Trainer."""
    
    def __init__(self, data_generator: DataGenerator, num_batches: int = 1000):
        """
        Initialize the dataset wrapper.
        
        Args:
            data_generator: DataGenerator instance to use for generating samples
            num_batches: Number of batches to pre-generate
        """
        self.data_generator = data_generator
        self.num_batches = num_batches
        # Pre-generate some batches for the dataset
        self.samples = []
        for _ in range(num_batches):
            batch = data_generator.generate_batch()
            self.samples.extend(batch)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample by index."""
        return self.samples[idx]



def test():
    """Test the BatchCollator with generated samples."""
    from transformers import AutoProcessor
    
    # Create data generator
    generator = DataGenerator(
        dataset_path="aliangdw/rfm",
        dataset_subsets=["libero_10"],
        batch_size=4,  # Small batch for testing
        preference_ratio=0.5,
        similarity_ratio=0.5,
        shuffle=True,
        seed=42,
        num_proc=4
    )
    
    # Generate a batch
    print("Generating test batch...")
    batch = generator.generate_batch()
    print(f"Generated batch with {len(batch)} samples")
    
    # Test the batch collator
    print("\nTesting batch collator...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    batch_collator = BatchCollator(processor, max_length=1024)
    
    processed_batch = batch_collator(batch)
    for key, value in processed_batch.items():
        print(key)
        if key == "preference_inputs":
            for key2, value2 in value.items():
                if key2 != "prediction_type":
                    print(key2, value2.shape)
        elif key == "similarity_inputs":
            for key2, value2 in value.items():
                if key2 != "prediction_type":
                    print(key2, value2.shape)


if __name__ == "__main__":
    # main()
    test() 