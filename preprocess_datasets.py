#!/usr/bin/env python3
"""
Dataset preprocessing script that creates index-based caches for fast trajectory access.
Uses HuggingFace's .map() for efficient processing and saves trajectory indices.
Handles both training and evaluation datasets separately.
"""

import os
import json
import shutil
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from datasets import Dataset, DatasetDict, load_dataset, Video, concatenate_datasets
import numpy as np
from tqdm import tqdm

from rfm.utils.logging import rank_0_print
from evals.eval_utils import load_experiment_config_from_yaml


class BaseDatasetPreprocessor:
    """Base class for dataset preprocessing with common functionality."""
    
    def __init__(self, config, cache_dir: str, dataset_type: str):
        self.config = config
        self.cache_dir = cache_dir
        self.dataset_type = dataset_type
        
        # Add attributes for video processing
        self.max_frames = config.data.max_frames
        self.resized_height = config.data.resized_height
        self.resized_width = config.data.resized_width
        self.force_reprocess = config.data.force_reprocess
        self.num_proc = config.data.num_proc
        
        # Dataset storage
        self.dataset: Optional[Dataset] = None
        self.robot_trajectories: List[int] = []
        self.human_trajectories: List[int] = []
        self.optimal_by_task: Dict[str, List[int]] = {}
        self.suboptimal_by_task: Dict[str, List[int]] = {}
        self.quality_indices: Dict[str, List[int]] = {}
        self.task_indices: Dict[str, List[int]] = {}
        self.source_indices: Dict[str, List[int]] = {}
    
    def preprocess_datasets(self):
        """Preprocess datasets and create index-based cache."""
        rank_0_print(f"\n🔧 Preprocessing {self.dataset_type} datasets...")
        
        # Load and concatenate all datasets first
        all_datasets = []
        
        if self.dataset_type == "training":
            datasets = self.config.data.train_datasets
            subsets = self.config.data.train_subsets
        else:
            datasets = self.config.data.eval_datasets
            subsets = self.config.data.eval_subsets
        
        for i, (dataset_path, subset) in enumerate(zip(datasets, subsets)):
            rank_0_print(f"Loading {self.dataset_type} dataset {i+1}/{len(datasets)}: {dataset_path}/{subset}")
            
            # Load dataset
            dataset = self._load_dataset_from_path(dataset_path, subset)
            
            # Handle DatasetDict
            if isinstance(dataset, DatasetDict):
                if 'train' in dataset:
                    dataset = dataset['train']
                else:
                    rank_0_print(f"Warning: No 'train' split found in {dataset_path}/{subset}")
                    continue
            
            all_datasets.append(dataset)
            rank_0_print(f"  Loaded {len(dataset)} trajectories from {dataset_path}/{subset}")
        
        if not all_datasets:
            if self.dataset_type == "training":
                raise ValueError("No training datasets were successfully loaded")
            else:
                rank_0_print("Warning: No evaluation datasets were successfully loaded")
                return
        
        # Combine all datasets first
        if len(all_datasets) == 1:
            combined_dataset = all_datasets[0]
        else:
            rank_0_print(f"Concatenating {len(all_datasets)} {self.dataset_type} datasets...")
            combined_dataset = concatenate_datasets(all_datasets)
        
        rank_0_print(f"Combined {len(combined_dataset)} total {self.dataset_type} trajectories")
        
        # Now process the combined dataset with video processing
        # First cast the frames_path column to Video feature
        rank_0_print("Casting frames_path column to Video feature...")
        combined_dataset = combined_dataset.cast_column("frames_path", Video(decode=True))
        
        # Now process the videos
        self.dataset = self._process_dataset_videos_map(combined_dataset)
        
        # Save cache
        self._save_cache()
    
    def _preprocess_videos(self, frames, num_frames: int = None) -> np.ndarray:
        """
        Process video frames from VideoReader objects into numpy arrays with downsampling.
        
        Args:
            frames: VideoReader object from HuggingFace Video feature
            num_frames: Number of frames to extract (default: uses self.max_frames)
            
        Returns:
            frames as numpy arrays with shape (max_frames, H, W, C) where T is time dimension
        """
        if num_frames is None:
            num_frames = self.max_frames
            
        if frames is None:
            return np.array([])
        
        try:
            # Convert VideoReader to list of frames
            all_frames = list(frames)
            total_frames = len(all_frames)
            
            if total_frames == 0:
                return np.array([])
            
            # Extract frame data from VideoReader objects
            frames_list = []
            for frame in all_frames:
                if hasattr(frame, 'get') and 'data' in frame:
                    # VideoReader frame format
                    frame_data = frame['data']
                    if hasattr(frame_data, 'numpy'):
                        frame_data = frame_data.numpy()
                    frames_list.append(frame_data)
                else:
                    # Direct frame data
                    if hasattr(frame, 'numpy'):
                        frame_data = frame.numpy()
                    else:
                        frame_data = frame
                    frames_list.append(frame_data)
            
            # Stack frames into numpy array
            if frames_list and hasattr(frames_list[0], 'shape'):
                frames_array = np.stack(frames_list)
            else:
                frames_array = np.array(frames_list)
            
            # Ensure we have the correct shape: (T, H, W, C)
            if len(frames_array.shape) != 4:
                raise ValueError(f"Expected 4D array (T, H, W, C), got shape {frames_array.shape}")
            
            # Convert from CxHxW to HxWxC if needed
            if frames_array.shape[1] == 3:
                frames_array = np.transpose(frames_array, (0, 2, 3, 1))
            
            # Downsample frames to max_frames
            if total_frames <= num_frames:
                # If video has fewer frames than requested, pad with last frame
                if total_frames < num_frames:
                    last_frame = frames_array[-1]  # Get the last frame
                    padding_frames = np.repeat(last_frame[np.newaxis, :, :, :], num_frames - total_frames, axis=0)
                    frames_array = np.concatenate([frames_array, padding_frames], axis=0)
            else:
                # Uniform sampling across the video
                frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
                frames_array = frames_array[frame_indices]
            
            return frames_array
            
        except Exception as e:
            rank_0_print(f"Error in _preprocess_videos: {e}")
            return np.array([])
    
    def _process_dataset_videos_map(self, dataset):
        """
        Process dataset frames using .map() method for efficient on-the-fly processing.
        Also builds index mappings during the same pass to avoid multiple iterations.
        Frames are saved as .npz files and only file paths are stored in the dataset.
        
        Args:
            dataset: HuggingFace dataset containing trajectories
            
        Returns:
            Dataset with processed frame paths and metadata
        """
        # Check if frames are already processed (npz file paths)
        sample_item = dataset[0]
        frames_data = sample_item.get('frames')
        
        if isinstance(frames_data, str) and frames_data.endswith('.npz') and not self.force_reprocess:
            rank_0_print("Frames already processed as npz file paths, skipping processing.")
            return dataset
        elif self.force_reprocess and isinstance(frames_data, str) and frames_data.endswith('.npz'):
            rank_0_print("Force reprocessing enabled. Reprocessing frames despite being already processed.")
        
        rank_0_print("Processing video frames into npz files and building index mappings...")
        
        # Debug: Check the dataset structure
        sample_item = dataset[0]
        rank_0_print(f"Sample dataset item keys: {list(sample_item.keys())}")
        rank_0_print(f"Sample item structure: {sample_item}")
        
        # Initialize index mappings
        self.robot_trajectories = []
        self.human_trajectories = []
        self.optimal_by_task = {}
        self.suboptimal_by_task = {}
        self.quality_indices = {}
        self.task_indices = {}
        self.source_indices = {}
        
        # Create frames directory for npz files
        frames_dir = os.path.join(self.cache_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        def process_videos_and_build_indices(example, idx):
            """Process frames and build index mappings in a single pass."""
            # Debug: Log what we're processing
            if idx < 5:  # Only log first 5 examples to avoid spam
                rank_0_print(f"Processing example {idx}: {example.get('id', 'unknown')} - {example.get('task', 'unknown')}")
            
            # Get the video reader object from the Video feature
            frames = example.get('frames_path')
            if frames is None:
                rank_0_print(f"Warning: No frames_path for example {idx}")
                return {"frames": None, "frames_processed": False}
            
            try:
                # Process video frames using the _preprocess_videos method
                frames_array = self._preprocess_videos(frames, self.config.data.max_frames)
                
                if frames_array.size == 0:
                    rank_0_print(f"Warning: No frames processed for example {idx}")
                    return {"frames": None, "frames_processed": False}
                
                # Save frames as npz file
                frames_filename = f"trajectory_{idx:06d}_{example.get('id', 'unknown')}.npz"
                frames_filepath = os.path.join(frames_dir, frames_filename)
                
                # Save frames with metadata
                np.savez_compressed(
                    frames_filepath,
                    frames=frames_array,
                    shape=frames_array.shape,
                    num_frames=frames_array.shape[0] if len(frames_array.shape) > 0 else 0
                )
                
                # Store file path and metadata in dataset (not the actual frames)
                example["frames"] = frames_filepath  # Store path to npz file
                example["frames_shape"] = frames_array.shape
                example["num_frames"] = frames_array.shape[0] if len(frames_array.shape) > 0 else 0
                example["frames_processed"] = True
                
                # BUILD INDEX MAPPINGS DURING THE SAME PASS
                # Debug: Log the values we're extracting
                if idx < 5:
                    rank_0_print(f"  Example {idx} - is_robot: {example.get('is_robot', True)}, task: {example.get('task', 'unknown')}, quality: {example.get('quality_label', 'successful')}")
                
                # Robot/Human trajectories
                if example.get('is_robot', True):
                    self.robot_trajectories.append(idx)
                else:
                    self.human_trajectories.append(idx)
                
                # Quality-based indices
                quality = example.get('quality_label', 'successful')
                if quality not in self.quality_indices:
                    self.quality_indices[quality] = []
                self.quality_indices[quality].append(idx)
                
                # Task-based indices
                task = example.get('task', 'unknown')
                if task not in self.task_indices:
                    self.task_indices[task] = []
                self.task_indices[task].append(idx)
                
                # Source-based indices
                source = example.get('data_source', 'unknown')
                if source not in self.source_indices:
                    self.source_indices[source] = []
                self.source_indices[source].append(idx)
                
                # Optimal/Suboptimal by task
                if task not in self.optimal_by_task:
                    self.optimal_by_task[task] = []
                    self.suboptimal_by_task[task] = []
                
                if quality in ['successful', 'optimal']:
                    self.optimal_by_task[task].append(idx)
                elif quality in ['suboptimal', 'failed']:
                    self.suboptimal_by_task[task].append(idx)
                
            except Exception as e:
                rank_0_print(f"Warning: Failed to process video frames for example {idx}: {e}")
                example["frames"] = None
                example["frames_processed"] = False
            
            # Remove the frames_path since we don't need it anymore
            if "frames_path" in example:
                del example["frames_path"]
            
            return example
        
        # Apply the mapping function to the dataset
        processed_dataset = dataset.map(
            process_videos_and_build_indices,
            with_indices=True,
            desc="Processing video frames and building indices",
            num_proc=self.config.data.num_proc
        )
        
        # Log the built indices
        rank_0_print(f"Built {self.dataset_type} index mappings:")
        rank_0_print(f"  Robot trajectories: {len(self.robot_trajectories)}")
        rank_0_print(f"  Human trajectories: {len(self.human_trajectories)}")
        rank_0_print(f"  Tasks: {len(self.task_indices)}")
        rank_0_print(f"  Quality labels: {len(self.quality_indices)}")
        rank_0_print(f"  Data sources: {len(self.source_indices)}")
        
        return processed_dataset
    
    def _save_cache(self):
        """Save the dataset and index mappings to cache."""
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Save the processed dataset
        dataset_cache_dir = os.path.join(self.cache_dir, "processed_dataset")
        self.dataset.save_to_disk(dataset_cache_dir)
        
        # Save index mappings
        index_mappings = {
            'robot_trajectories': self.robot_trajectories,
            'human_trajectories': self.human_trajectories,
            'optimal_by_task': self.optimal_by_task,
            'suboptimal_by_task': self.suboptimal_by_task,
            'quality_indices': self.quality_indices,
            'task_indices': self.task_indices,
            'source_indices': self.source_indices,
        }
        
        mappings_file = os.path.join(self.cache_dir, "index_mappings.json")
        with open(mappings_file, 'w') as f:
            json.dump(index_mappings, f, indent=2)
        
        # Save dataset info
        if self.dataset_type == "training":
            datasets = self.config.data.train_datasets
            subsets = self.config.data.train_subsets
        else:
            datasets = self.config.data.eval_datasets
            subsets = self.config.data.eval_subsets
            
        dataset_info = {
            'datasets': datasets,
            'subsets': subsets,
            'total_trajectories': len(self.dataset),
            'cache_timestamp': str(datetime.datetime.now()),
            'config_hash': self._get_config_hash()
        }
        
        info_file = os.path.join(self.cache_dir, "dataset_info.json")
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        rank_0_print(f"{self.dataset_type.capitalize()} cache saved to {self.cache_dir}")
    
    def _get_config_hash(self) -> str:
        """Generate a hash of the relevant config parameters."""
        import hashlib
        
        if self.dataset_type == "training":
            # Create a string representation of relevant training config parameters
            config_str = f"train_{self.config.data.train_datasets}_{self.config.data.train_subsets}_{self.config.data.max_frames}_{self.config.data.resized_height}_{self.config.data.resized_width}"
        else:
            # Create a string representation of relevant evaluation config parameters
            config_str = f"eval_{self.config.data.eval_datasets}_{self.config.data.eval_subsets}_{self.config.data.max_frames}_{self.config.data.resized_height}_{self.config.data.resized_width}"
        
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _load_dataset_from_path(self, dataset_path: str, subset: str = None):
        """Load dataset from path with proper video handling."""
        if '/' in dataset_path and not os.path.exists(dataset_path):
            # Loading from HuggingFace Hub - handle video paths
            rank_0_print(f"Loading from HuggingFace Hub: {dataset_path}")
            
            # Check if RFM_DATASET_PATH is set
            rfm_dataset_path = os.environ.get('RFM_DATASET_PATH')
            if not rfm_dataset_path:
                raise ValueError(
                    "RFM_DATASET_PATH environment variable not set. "
                    "Please set it to the directory containing your downloaded datasets. "
                    "Example: export RFM_DATASET_PATH=/path/to/your/datasets"
                )
            
            dataset_name = dataset_path.split("/")[-1]
            
            def patch_path(old_path):
                # RFM_DATASET_PATH is set in the environment variable
                root_dir = f"{rfm_dataset_path}/{dataset_name}"
                return f"{root_dir}/{old_path}"  # e.g., "./videos/trajectory_0000.mp4"
            
            # Load dataset with subset
            if subset:
                dataset = load_dataset(dataset_path, name=subset, split="train")
            else:
                dataset = load_dataset(dataset_path, split="train")
            
            # Just patch the paths, don't decode videos yet
            dataset = dataset.map(lambda x: {"frames_path": patch_path(x["frames"])})
            return dataset
        else:
            # Load from local disk
            if subset:
                dataset = load_dataset(dataset_path, subset)
            else:
                dataset = load_dataset(dataset_path)
            return dataset


class TrainingDatasetPreprocessor(BaseDatasetPreprocessor):
    """Preprocessor specifically for training datasets."""
    
    def __init__(self, config):
        super().__init__(config, "./processed_datasets/train_cache", "training")


class EvaluationDatasetPreprocessor(BaseDatasetPreprocessor):
    """Preprocessor specifically for evaluation datasets."""
    
    def __init__(self, config):
        super().__init__(config, "./processed_datasets/eval_cache", "evaluation")


def main():
    """Main preprocessing function."""
    # Load config
    config_path = "rfm/configs/config.yaml"  # Adjust path as needed
    config = load_experiment_config_from_yaml(config_path)

    # Create separate preprocessors for training and evaluation
    train_preprocessor = TrainingDatasetPreprocessor(config)
    eval_preprocessor = EvaluationDatasetPreprocessor(config)
    
    # Preprocess training datasets
    print("\n=== Processing Training Datasets ===")
    train_preprocessor.preprocess_datasets()
    
    # Preprocess evaluation datasets
    print("\n=== Processing Evaluation Datasets ===")
    eval_preprocessor.preprocess_datasets()
    
    # Test the caches
    print("\n=== Testing Caches ===")
    
    # Training cache
    if train_preprocessor.dataset:
        print(f"\n📚 Training Dataset:")
        print(f"  Total trajectories: {len(train_preprocessor.dataset)}")
        print(f"  Robot trajectories: {len(train_preprocessor.robot_trajectories)}")
        print(f"  Human trajectories: {len(train_preprocessor.human_trajectories)}")
        print(f"  Tasks: {list(train_preprocessor.task_indices.keys())}")
        
        # Test direct access
        if len(train_preprocessor.dataset) > 0:
            test_traj = train_preprocessor.dataset[0]
            print(f"  Sample trajectory: {test_traj['id']} - {test_traj['task']}")
    
    # Evaluation cache
    if eval_preprocessor.dataset:
        print(f"\n📚 Evaluation Dataset:")
        print(f"  Total trajectories: {len(eval_preprocessor.dataset)}")
        print(f"  Robot trajectories: {len(eval_preprocessor.robot_trajectories)}")
        print(f"  Human trajectories: {len(eval_preprocessor.human_trajectories)}")
        print(f"  Tasks: {list(eval_preprocessor.task_indices.keys())}")
        
        # Test direct access
        if len(eval_preprocessor.dataset) > 0:
            test_traj = eval_preprocessor.dataset[0]
            print(f"  Sample trajectory: {test_traj['id']} - {test_traj['task']}")
    
    print("\n✅ Dataset preprocessing complete!")
    print(f"Training cache: {train_preprocessor.cache_dir}")
    print(f"Evaluation cache: {eval_preprocessor.cache_dir}")


if __name__ == "__main__":
    main() 