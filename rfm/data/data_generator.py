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
from rfm.utils.logging import rank_0_print

class DataGenerator:
    """Data generator for producing batches of prediction data with controlled ratios."""
    
    def __init__(
        self,
        dataset_path: str = "rfm_dataset",
        dataset_subsets: List[str] = ["libero"],
        preference_dataset_path: Optional[str] = None,
        preference_dataset_subset: Optional[str] = None,
        preference_ratio: float = 0.5,
        similarity_ratio: float = 0.5,
        max_frames: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
        dataset_preference_ratio: float = 0.7,
        num_proc: int = 1,
        debug: bool = False,
        force_reprocess: bool = False
    ):
        """
        Initialize the data generator.
        
        Args:
            dataset_path: Path to the HuggingFace dataset or dataset name
            dataset_subsets: List of dataset names to load (e.g., ["libero", "droid"] or ["libero"])
            preference_dataset_path: Optional path to preference dataset
            preference_dataset_subset: Optional subset name for preference dataset
            preference_ratio: Ratio of preference prediction samples (0.0 to 1.0)
            similarity_ratio: Ratio of similarity scoring samples (0.0 to 1.0)
            max_frames: Maximum frames per trajectory
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
            dataset_preference_ratio: Ratio of preferences from dataset vs generated (0.0 to 1.0)
            num_proc: Number of processes to use for dataset processing (default: 1)
            debug: Whether to enable debug mode (reduces dataset size, enables debug features)
            force_reprocess: Whether to force reprocessing of dataset even if cached version exists
        """
        self.dataset_path = dataset_path
        self.dataset_subsets = dataset_subsets
        self.preference_dataset_path = preference_dataset_path
        self.preference_dataset_subset = preference_dataset_subset
        self.preference_ratio = preference_ratio
        self.similarity_ratio = similarity_ratio
        self.max_frames = max_frames
        self.shuffle = shuffle
        self.seed = seed
        self.dataset_preference_ratio = dataset_preference_ratio
        self.num_proc = num_proc
        self.debug = debug 
        self.force_reprocess = force_reprocess
        
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
        
        rank_0_print(f"DataGenerator initialized with {len(self.trajectories)} total trajectories")
        rank_0_print(f"Loaded datasets: {dataset_subsets}")
        if self.preferences:
            rank_0_print(f"Loaded {len(self.preferences)} preference pairs")
        rank_0_print(f"Ratios - Preference: {preference_ratio}, Similarity: {similarity_ratio}")
        rank_0_print(f"Dataset preference ratio: {dataset_preference_ratio}")
    
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
    
    def _deserialize_frames(self, bytes_blob: bytes, shape: tuple = None) -> np.ndarray:
        """
        Deserialize bytes back to frames array.
        
        Args:
            bytes_blob: Serialized bytes
            shape: Expected shape of the frames array (T, H, W, C). If None, will try to infer.
            
        Returns:
            Deserialized numpy array
        """
        if not bytes_blob:
            return np.array([])
        
        # If shape is provided, use it directly
        if shape is not None and len(shape) > 0:
            try:
                # Validate shape
                if any(dim <= 0 for dim in shape):
                    print(f"Warning: Invalid shape dimensions {shape}, falling back to inference")
                else:
                    return np.frombuffer(bytes_blob, dtype=np.uint8).reshape(shape)
            except Exception as e:
                print(f"Warning: Failed to reshape with provided shape {shape}: {e}")
                # Fall back to inference
        
        # Try to infer shape if not provided (backward compatibility)
        try:
            total_elements = len(bytes_blob)
            # Try to find a reasonable shape - assume square frames and 3 channels
            for T in range(1, min(33, total_elements // 3 + 1)):
                remaining = total_elements // T
                if remaining % 3 == 0:  # Must be divisible by 3 for RGB
                    H = W = int(np.sqrt(remaining // 3))
                    if H * H * 3 * T == total_elements:
                        return np.frombuffer(bytes_blob, dtype=np.uint8).reshape(T, H, W, 3)
            
            # If no reasonable shape found, return as 1D array
            return np.frombuffer(bytes_blob, dtype=np.uint8)
        except Exception as e:
            print(f"Warning: Failed to infer shape: {e}")
            return np.frombuffer(bytes_blob, dtype=np.uint8)
    
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
        
        if (isinstance(frames_data, bytes) or isinstance(frames_data, np.ndarray)) and not self.force_reprocess:
            rank_0_print("Frames already processed, skipping downsampling.")
            return dataset
        elif self.force_reprocess and (isinstance(frames_data, bytes) or isinstance(frames_data, np.ndarray)):
            rank_0_print("Force reprocessing enabled. Reprocessing frames despite being already processed.")
        
        rank_0_print("Downsampling video frames using .map()...")
        
        def process_videos(example):
            """Downsample frames in a single example and serialize."""
            frames = example.get('frames_path')
            frames_array = self._preprocess_videos(frames, self.max_frames)
            
            # Store the shape information for later deserialization
            frames_shape = frames_array.shape
            
            # Serialize frames to bytes for efficient storage
            frames_bytes = self._serialize_frames(frames_array)
            
            # Update the example with serialized frames and shape information
            del example["frames_path"]
            example["frames"] = frames_bytes
            example["frames_shape"] = frames_shape  # Store shape for later use
            return example
        
        # Apply the mapping function to the dataset
        processed_dataset = dataset.map(
            process_videos,
            desc="Processing videos",
            num_proc=self.num_proc
        )

        rank_0_print(f"Frame downsampling complete. Each trajectory now has {self.max_frames} frames.")
        
        # Save the processed dataset to disk for future fast loading
        cache_dir = f"./processed_datasets/{self.dataset_path.replace('/', '_')}_{self.max_frames}frames"
        rank_0_print(f"Saving processed dataset to {cache_dir} for future fast loading...")
        processed_dataset.save_to_disk(cache_dir)
        rank_0_print(f"Processed dataset saved to {cache_dir}")
        
        return processed_dataset
    
    def _create_rewind_trajectory(self, original_traj: Dict) -> Dict:
        """Create a suboptimal trajectory by rewinding the original trajectory."""
        frames_data = original_traj['frames']
        
        # Deserialize frames if they're bytes
        if isinstance(frames_data, bytes):
            # Get the stored shape if available
            frames_shape = original_traj.get('frames_shape')
            # Convert to tuple if it's a list
            if isinstance(frames_shape, list):
                frames_shape = tuple(frames_shape)
            frames = self._deserialize_frames(frames_data, shape=frames_shape)
        else:
            frames = frames_data
        
        # Get the number of frames
        if hasattr(frames, 'shape'):
            num_frames = frames.shape[0]  # Use shape[0] for numpy array
        else:
            num_frames = len(frames)
        
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
        forward_frames = frames[start_idx:end_idx]
        forward_progress = [(i + 1) / (end_idx - start_idx) for i in range(end_idx - start_idx)]
        
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
        rewind_traj['frames_shape'] = combined_frames.shape  # Store shape for the rewind trajectory
        rewind_traj['id'] = f"{original_traj['id']}_rewind_{random.randint(1000, 9999)}"
        rewind_traj['optimal'] = False  # Mark as suboptimal
        rewind_traj['metadata'] = rewind_traj.get('metadata', {}).copy()
        rewind_traj['metadata']['rewind_generated'] = True
        rewind_traj['metadata']['original_traj_id'] = original_traj['id']
        rewind_traj['metadata']['rewind_progress'] = combined_progress
        
        return rewind_traj
    
    def _load_trajectory_dataset(self):
        """Load the trajectory dataset from disk or hub."""
        rank_0_print(f"Loading trajectory dataset from: {self.dataset_path}")
        
        # Load multiple subsets and combine them
        all_datasets = []
        
        for dataset_name in self.dataset_subsets:
            rank_0_print(f"Loading dataset: {dataset_name}")
            dataset = self._load_dataset_from_path(self.dataset_path, dataset_name)
            
            # Handle DatasetDict by accessing the train split
            if hasattr(dataset, 'keys') and 'train' in dataset:
                rank_0_print(f"  Found DatasetDict with train split, accessing train data...")
                dataset = dataset['train']
            
            # Check if frames need processing by examining the first sample
            if len(dataset) > 0:
                sample_item = dataset[0]
                frames_data = sample_item.get('frames')
                
                # If frames are already numpy arrays, skip processing
                if isinstance(frames_data, np.ndarray):
                    rank_0_print(f"  Frames already processed (numpy arrays), skipping frame processing")
                elif isinstance(frames_data, bytes):
                    rank_0_print(f"  Frames are serialized bytes, will be deserialized on-demand")
                else:
                    rank_0_print(f"  Frames need processing, applying .map()...")
                    dataset = self._process_dataset_videos_map(dataset)
            else:
                rank_0_print(f"  Empty dataset, skipping frame processing")
            
            all_datasets.append(dataset)
            rank_0_print(f"  Loaded {len(dataset)} trajectories from dataset '{dataset_name}'")
        
        # Combine all datasets
        if len(all_datasets) == 1:
            self.trajectories = all_datasets[0]
        else:
            self.trajectories = concatenate_datasets(all_datasets)
        
        rank_0_print(f"Combined {len(self.trajectories)} total trajectories from {len(self.dataset_subsets)} datasets")
    
    def _load_preference_dataset(self):
        """Load the preference dataset from disk or hub if provided."""
        self.preferences = []
        
        if not self.preference_dataset_path:
            rank_0_print("No preference dataset provided, will use random sampling for preferences")
            return
        
        rank_0_print(f"Loading preference dataset from: {self.preference_dataset_path}")
        
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
        
        rank_0_print(f"Loaded {len(self.preferences)} preference pairs from subset '{self.preference_dataset_subset}'")
        
    def _load_dataset_from_path(self, dataset_path: str, subset: str = None):
        """Helper method to load a dataset from path (local or hub)."""

        if '/' in dataset_path and not os.path.exists(dataset_path):
            # Check for cached processed dataset first
            cache_dir = f"./processed_datasets/{dataset_path.replace('/', '_')}_{self.max_frames}frames"
            if os.path.exists(cache_dir) and not self.force_reprocess:
                rank_0_print(f"Found cached processed dataset at {cache_dir}, loading...")
                ds = load_from_disk(cache_dir)
                return ds
            elif self.force_reprocess and os.path.exists(cache_dir):
                rank_0_print(f"Force reprocessing enabled. Removing cached dataset at {cache_dir}...")
                shutil.rmtree(cache_dir)
                rank_0_print(f"Removed cached dataset. Will reprocess and save to {cache_dir}")
            
            # Load from HuggingFace Hub
            from datasets import load_dataset, Video, Features
            rank_0_print(f"Loading from HuggingFace Hub: {dataset_path}")

            def patch_path(old_path):
                root_dir = "/workspace/vlm_reward_model/rfm_dataset"
                return f"{root_dir}/{old_path}"       # e.g., "./videos/trajectory_0000.mp4"
            
            ds = load_dataset(dataset_path, name=subset, split="train")
            ds = ds.map(lambda x: {"frames_path": patch_path(x["frames"])})
            ds = ds.cast_column("frames_path", Video(decode=True))
            # Only select a small subset for debugging
            if self.debug:
                ds = ds.select(range(5))
                rank_0_print("  Debug mode: Using only first 5 samples")
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
        
        rank_0_print(f"Grouped trajectories into {len(self.task_groups)} tasks")
        
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
        
        rank_0_print(f"Preprocessed trajectory categories:")
        rank_0_print(f"  Robot trajectories: {len(self.robot_trajectories)}")
        rank_0_print(f"  Human trajectories: {len(self.human_trajectories)}")
        rank_0_print(f"  Optimal trajectories: {len(self.optimal_trajectories)}")
        rank_0_print(f"  Suboptimal trajectories: {len(self.suboptimal_trajectories)}")
        rank_0_print(f"  Data sources: {list(self.trajectories_by_source.keys())}")
    
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
            
            # Deserialize frames once for both trajectories
            chosen_frames_shape = chosen_traj.get('frames_shape')
            if isinstance(chosen_frames_shape, list):
                chosen_frames_shape = tuple(chosen_frames_shape)
            chosen_frames = self._deserialize_frames(chosen_traj['frames'], shape=chosen_frames_shape) if isinstance(chosen_traj['frames'], bytes) else chosen_traj['frames']
            
            rejected_frames_shape = rejected_traj.get('frames_shape')
            if isinstance(rejected_frames_shape, list):
                rejected_frames_shape = tuple(rejected_frames_shape)
            rejected_frames = self._deserialize_frames(rejected_traj['frames'], shape=rejected_frames_shape) if isinstance(rejected_traj['frames'], bytes) else rejected_traj['frames']
            
            # Calculate target progress for both trajectories using pre-deserialized frames
            target_progress_A = self._calculate_target_progress(chosen_traj, chosen_frames)
            target_progress_B = self._calculate_target_progress(rejected_traj, rejected_frames)
            
            # Get frame shapes and convert to tuples if needed
            chosen_frames_shape = chosen_traj.get('frames_shape')
            if isinstance(chosen_frames_shape, list):
                chosen_frames_shape = tuple(chosen_frames_shape)
            
            rejected_frames_shape = rejected_traj.get('frames_shape')
            if isinstance(rejected_frames_shape, list):
                rejected_frames_shape = tuple(rejected_frames_shape)
            
            # Create preference sample structure
            sample = PreferenceSample(
                # Core HF dataset fields (from chosen trajectory)
                id=chosen_traj['id'],
                task=chosen_traj['task'],
                lang_vector=chosen_traj['lang_vector'],
                data_source=chosen_traj['data_source'],
                frames=chosen_traj['frames'],
                frames_shape=chosen_frames_shape,
                optimal=chosen_traj['optimal'],
                is_robot=chosen_traj['is_robot'],
                metadata=chosen_traj.get('metadata'),
                # Preference-specific fields
                trajectory_A_frames=chosen_traj['frames'],
                trajectory_B_frames=rejected_traj['frames'],
                trajectory_A_frames_shape=chosen_frames_shape,
                trajectory_B_frames_shape=rejected_frames_shape,
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
        
        # Choose negative generation strategy (equal probability for rewind vs same-task suboptimal)
        strategy = random.random()
        
        if strategy < 0.5:
            # Strategy 1: Use rewind-generated suboptimal trajectory
                negative_traj = self._create_rewind_trajectory(optimal_traj)
        else:
            # Strategy 2: Use random suboptimal trajectory from same task
            same_task_suboptimal = [
                traj for traj in self.suboptimal_trajectories 
                if traj['task'] == optimal_traj['task'] and traj['id'] != optimal_traj['id']
            ]
            if same_task_suboptimal:
                negative_traj = random.choice(same_task_suboptimal)
            else:
                # Fall back to rewind if no same-task suboptimal trajectories
                negative_traj = self._create_rewind_trajectory(optimal_traj)
        
        # Deserialize frames once for both trajectories
        optimal_frames_shape = optimal_traj.get('frames_shape')
        if isinstance(optimal_frames_shape, list):
            optimal_frames_shape = tuple(optimal_frames_shape)
        optimal_frames = self._deserialize_frames(optimal_traj['frames'], shape=optimal_frames_shape) if isinstance(optimal_traj['frames'], bytes) else optimal_traj['frames']
        
        negative_frames_shape = negative_traj.get('frames_shape')
        if isinstance(negative_frames_shape, list):
            negative_frames_shape = tuple(negative_frames_shape)
        negative_frames = self._deserialize_frames(negative_traj['frames'], shape=negative_frames_shape) if isinstance(negative_traj['frames'], bytes) else negative_traj['frames']
        
        # Calculate target progress for both trajectories using pre-deserialized frames
        target_progress_A = self._calculate_target_progress(optimal_traj, optimal_frames)
        target_progress_B = self._calculate_target_progress(negative_traj, negative_frames)
        
        # Get frame shapes and convert to tuples if needed
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
            frames=optimal_traj['frames'],
            frames_shape=optimal_frames_shape,
            optimal=optimal_traj['optimal'],
            is_robot=optimal_traj['is_robot'],
            metadata=optimal_traj.get('metadata'),
            # Preference-specific fields
            trajectory_A_frames=optimal_traj['frames'],
            trajectory_B_frames=negative_traj['frames'],
            trajectory_A_frames_shape=optimal_frames_shape,
            trajectory_B_frames_shape=negative_frames_shape,
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
    
    def _calculate_target_progress(self, trajectory: Dict, frames: np.ndarray = None) -> List[float]:
        """Calculate target progress values for a trajectory."""
        
        # Check if this is a rewind trajectory (has rewind progress in metadata)
        if (trajectory.get('metadata') and 
            trajectory['metadata'].get('rewind_generated') and 
            trajectory['metadata'].get('rewind_progress')):
            # Use the rewind progress that was calculated during rewind generation
            return trajectory['metadata']['rewind_progress']
        
        # If frames not provided, deserialize them
        if frames is None:
            frames_data = trajectory['frames']
            if isinstance(frames_data, bytes):
                frames_shape = trajectory.get('frames_shape')
                if isinstance(frames_shape, list):
                    frames_shape = tuple(frames_shape)
                frames = self._deserialize_frames(frames_data, shape=frames_shape)
            else:
                frames = frames_data
        
        # Calculate number of frames
        if hasattr(frames, 'shape'):
            num_frames = frames.shape[0]  # Use shape[0] for numpy array
        else:
            num_frames = len(frames)
        
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
        
        # Deserialize frames once for all trajectories
        traj_1_frames_shape = traj_1.get('frames_shape')
        if isinstance(traj_1_frames_shape, list):
            traj_1_frames_shape = tuple(traj_1_frames_shape)
        traj_1_frames = self._deserialize_frames(traj_1['frames'], shape=traj_1_frames_shape) if isinstance(traj_1['frames'], bytes) else traj_1['frames']
        
        traj_2_frames_shape = traj_2.get('frames_shape')
        if isinstance(traj_2_frames_shape, list):
            traj_2_frames_shape = tuple(traj_2_frames_shape)
        traj_2_frames = self._deserialize_frames(traj_2['frames'], shape=traj_2_frames_shape) if isinstance(traj_2['frames'], bytes) else traj_2['frames']
        
        # Calculate target progress for all trajectories using pre-deserialized frames
        target_progress_A = self._calculate_target_progress(traj_1, traj_1_frames)
        target_progress_B = self._calculate_target_progress(traj_2, traj_2_frames)
        
        # Get frame shapes and convert to tuples if needed
        ref_frames_shape = ref_traj.get('frames_shape')
        if isinstance(ref_frames_shape, list):
            ref_frames_shape = tuple(ref_frames_shape)
        
        traj_1_frames_shape = traj_1.get('frames_shape')
        if isinstance(traj_1_frames_shape, list):
            traj_1_frames_shape = tuple(traj_1_frames_shape)
        
        traj_2_frames_shape = traj_2.get('frames_shape')
        if isinstance(traj_2_frames_shape, list):
            traj_2_frames_shape = tuple(traj_2_frames_shape)
        
        # Create similarity sample structure
        sample = SimilaritySample(
            # Core HF dataset fields (from ref_traj)
            id=ref_traj['id'],
            task=ref_traj['task'],
            lang_vector=ref_traj['lang_vector'],
            data_source=ref_traj['data_source'],
            frames=ref_traj['frames'],
            frames_shape=ref_frames_shape,
            optimal=ref_traj['optimal'],
            is_robot=ref_traj['is_robot'],
            metadata=ref_traj.get('metadata'),
            # Comparative-specific fields
            reference_frames=ref_traj['frames'],  # o^ref
            trajectory_A_frames=traj_1['frames'],  # o^1
            trajectory_B_frames=traj_2['frames'],  # o^2
            reference_frames_shape=ref_frames_shape,
            trajectory_A_frames_shape=traj_1_frames_shape,
            trajectory_B_frames_shape=traj_2_frames_shape,
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


def test():
    """Test the BatchCollator with generated samples."""
    from transformers import AutoProcessor
    
    # Create data generator
    generator = DataGenerator(
        dataset_path="aliangdw/rfm",
        dataset_subsets=["libero_10"],
        preference_ratio=0.5,
        similarity_ratio=0.5,
        shuffle=True,
        seed=42,
        num_proc=4
    )
    
    # Test the infinite dataset
    rank_0_print("Testing InfiniteDataGeneratorDataset...")
    infinite_dataset = InfiniteDataGeneratorDataset(generator, max_samples=10)
    
    preference_count = 0
    similarity_count = 0
    
    for i in range(10):
        sample = infinite_dataset[i]
        if sample.prediction_type == "preference":
            preference_count += 1
        else:
            similarity_count += 1
        rank_0_print(f"Sample {i}: {sample.prediction_type}")
    
    rank_0_print(f"Generated {preference_count} preference samples and {similarity_count} similarity samples")
    rank_0_print(f"Expected ratio: {generator.preference_ratio:.1f} preference, {generator.similarity_ratio:.1f} similarity")
    
    # Test the batch collator with infinite dataset
    rank_0_print("\nTesting batch collator with infinite dataset...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    batch_collator = BatchCollator(processor, max_length=1024)
    
    # Generate a batch from the infinite dataset
    batch = []
    for i in range(4):  # Generate 4 samples
        sample = infinite_dataset[i]
        batch.append(sample)
    
    processed_batch = batch_collator(batch)
    for key, value in processed_batch.items():
        rank_0_print(key)
        if key == "preference_inputs":
            for key2, value2 in value.items():
                if key2 != "prediction_type":
                    rank_0_print(f"{key2} {value2.shape}")
        elif key == "similarity_inputs":
            for key2, value2 in value.items():
                if key2 != "prediction_type":
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
    
    rfm_model = RFMModel(config=base_model.config, tokenizer=processor.tokenizer)
    rfm_model.model.load_state_dict(base_model.state_dict())
    
    # Check if CUDA is available
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    rfm_model = rfm_model.to(device)
    inputs = processed_batch["preference_inputs"]

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
        prediction_type="preference",  # Test preference prediction
    )

    rank_0_print("RFM model output structure:")
    rank_0_print(f"  logits: {outputs.shape if outputs is not None else None}")
    rank_0_print(f"  output type: {type(outputs)}")


if __name__ == "__main__":
    test() 