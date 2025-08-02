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
from typing import List, Dict, Tuple, Optional, Iterator
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
import shutil
import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Sample:
    """A unified sample structure that can handle all prediction types."""
    
    # Core HF dataset fields
    prediction_type: str
    id: str
    task: str
    lang_vector: np.ndarray
    data_source: str
    frames: List[str]
    optimal: bool
    ranking: int
    preference_embedding: np.ndarray
    is_robot: bool
    
    # Preference-specific fields
    trajectory_A_frames: Optional[List[str]] = None
    trajectory_B_frames: Optional[List[str]] = None
    preferred_trajectory: Optional[str] = None  # "A" or "B"
    trajectory_A_id: Optional[str] = None
    trajectory_B_id: Optional[str] = None
    trajectory_B_task: Optional[str] = None
    trajectory_B_lang_vector: Optional[np.ndarray] = None
    trajectory_B_data_source: Optional[str] = None
    trajectory_B_optimal: Optional[bool] = None
    trajectory_B_ranking: Optional[int] = None
    trajectory_B_preference_embedding: Optional[np.ndarray] = None
    trajectory_B_is_robot: Optional[bool] = None
    
    # Comparative-specific fields
    reference_frames: Optional[List[str]] = None  # o^ref
    trajectory_A_frames: Optional[List[str]] = None  # o^1
    trajectory_B_frames: Optional[List[str]] = None  # o^2
    ranking_list: Optional[List[int]] = None
    task_ref: Optional[str] = None
    task_A: Optional[str] = None
    task_B: Optional[str] = None
    ref_trajectory_id: Optional[str] = None
    trajectory_A_id: Optional[str] = None
    trajectory_B_id: Optional[str] = None
    trajectory_A_task: Optional[str] = None
    trajectory_A_lang_vector: Optional[np.ndarray] = None
    trajectory_A_data_source: Optional[str] = None
    trajectory_A_optimal: Optional[bool] = None
    trajectory_A_ranking: Optional[int] = None
    trajectory_A_preference_embedding: Optional[np.ndarray] = None
    trajectory_A_is_robot: Optional[bool] = None
    trajectory_B_task: Optional[str] = None
    trajectory_B_lang_vector: Optional[np.ndarray] = None
    trajectory_B_data_source: Optional[str] = None
    trajectory_B_optimal: Optional[bool] = None
    trajectory_B_ranking: Optional[int] = None
    trajectory_B_preference_embedding: Optional[np.ndarray] = None
    trajectory_B_is_robot: Optional[bool] = None
    
    # Progress-specific fields
    trajectory_frames: Optional[List[str]] = None
    trajectory_id: Optional[str] = None
    success: Optional[bool] = None
    
    # Metadata field
    metadata: Optional[Dict] = None


@dataclass
class Batch:
    """A batch of samples with all prediction types mixed together."""
    
    samples: List[Sample] = field(default_factory=list)
    
    def __len__(self) -> int:
        """Return the number of samples in the batch."""
        return len(self.samples)
    
    def __iter__(self):
        """Iterate over samples."""
        return iter(self.samples)


class DataGenerator:
    """Data generator for producing batches of prediction data with controlled ratios."""
    
    def __init__(
        self,
        dataset_path: str = "libero_hf_dataset/libero_hf_dataset",
        batch_size: int = 32,
        preference_ratio: float = 0.5,
        comparative_ratio: float = 0.5,
        progress_ratio: float = 1.0,  # All data can be used for progress prediction
        max_frames: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize the data generator.
        
        Args:
            dataset_path: Path to the HuggingFace dataset
            batch_size: Number of samples per batch
            preference_ratio: Ratio of preference prediction samples (0.0 to 1.0)
            comparative_ratio: Ratio of comparative scoring samples (0.0 to 1.0)
            progress_ratio: Ratio of progress prediction samples (0.0 to 1.0)
            max_frames: Maximum frames per trajectory
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.preference_ratio = preference_ratio
        self.comparative_ratio = comparative_ratio
        self.progress_ratio = progress_ratio
        self.max_frames = max_frames
        self.shuffle = shuffle
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Load the dataset
        self._load_dataset()
        
        # Group trajectories by task for efficient sampling
        self._group_trajectories_by_task()
        
        # Initialize sentence transformer model
        self.lang_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"DataGenerator initialized with {len(self.trajectories)} total trajectories")
        print(f"Batch size: {batch_size}")
        print(f"Ratios - Preference: {preference_ratio}, Comparative: {comparative_ratio}, Progress: {progress_ratio}")
    
    def _load_dataset(self):
        """Load the HuggingFace dataset from disk or hub."""
        print(f"Loading dataset from: {self.dataset_path}")
        
        # Check if it's a hub path (contains '/') and not a local path
        if '/' in self.dataset_path and not os.path.exists(self.dataset_path):
            # Load from HuggingFace Hub
            from datasets import load_dataset
            print(f"Loading from HuggingFace Hub: {self.dataset_path}")
            self.dataset = load_dataset(self.dataset_path, split="train")
        else:
            # Load from local disk
            self.dataset = load_from_disk(self.dataset_path)
        
        self.trajectories = list(self.dataset)
        print(f"Loaded {len(self.trajectories)} trajectories")
    
    def _group_trajectories_by_task(self):
        """Group trajectories by task name for efficient sampling."""
        self.task_groups = {}
        for traj in self.trajectories:
            task_name = traj['task']
            if task_name not in self.task_groups:
                self.task_groups[task_name] = []
            self.task_groups[task_name].append(traj)
        
        print(f"Grouped trajectories into {len(self.task_groups)} tasks")
    
    def _create_preference_sample(self) -> Sample:
        """Create a preference prediction sample: o^1 vs o^2 where o^1 is preferred."""
        
        # Find tasks with at least 2 trajectories
        available_tasks = [task for task, trajectories in self.task_groups.items() if len(trajectories) >= 2]
        if not available_tasks:
            raise ValueError("No tasks with at least 2 trajectories found")
        
        # Randomly select a task
        task_name = random.choice(available_tasks)
        task_trajectories = self.task_groups[task_name]
        
        # Select two different trajectories from the same task
        traj_1, traj_2 = random.sample(task_trajectories, 2)
        
        # Create unified sample structure with all fields
        sample = Sample(
            prediction_type="preference",
            # Core HF dataset fields (from primary trajectory)
            id=traj_1['id'],
            task=traj_1['task'],
            lang_vector=traj_1['lang_vector'],
            data_source=traj_1['data_source'],
            frames=traj_1['frames'],
            optimal=traj_1['optimal'],
            ranking=traj_1['ranking'],
            preference_embedding=traj_1['preference_embedding'],
            is_robot=traj_1['is_robot'],
            metadata=traj_1.get('metadata'),
            # Preference-specific fields
            trajectory_A_frames=traj_1['frames'],
            trajectory_B_frames=traj_2['frames'],
            preferred_trajectory="A",  # A is preferred
            trajectory_A_id=traj_1['id'],
            trajectory_B_id=traj_2['id'],
            # Trajectory B fields
            trajectory_B_task=traj_2['task'],
            trajectory_B_lang_vector=traj_2['lang_vector'],
            trajectory_B_data_source=traj_2['data_source'],
            trajectory_B_optimal=traj_2['optimal'],
            trajectory_B_ranking=traj_2['ranking'],
            trajectory_B_preference_embedding=traj_2['preference_embedding'],
            trajectory_B_is_robot=traj_2['is_robot'],
        )
        
        return sample
    
    def _create_progress_sample(self) -> Sample:
        """Create a progress prediction sample: single trajectory for progress prediction."""
        
        # Randomly select a trajectory
        traj = random.choice(self.trajectories)
        
        # Create unified sample structure with all fields
        sample = Sample(
            prediction_type="progress",
            # Core HF dataset fields
            id=traj['id'],
            task=traj['task'],
            lang_vector=traj['lang_vector'],
            data_source=traj['data_source'],
            frames=traj['frames'],
            optimal=traj['optimal'],
            ranking=traj['ranking'],
            preference_embedding=traj['preference_embedding'],
            is_robot=traj['is_robot'],
            metadata=traj.get('metadata'),
            # Progress-specific fields
            trajectory_frames=traj['frames'],
            trajectory_id=traj['id'],
            success=traj.get('optimal', True),
        )
        
        return sample
    
    def _create_comparative_sample(self) -> Sample:
        """Create a comparative scoring sample: o^1 and o^2 ranked against o^ref."""
        
        # Get available tasks
        task_names = list(self.task_groups.keys())
        
        if len(task_names) < 2:
            # If only one task, use same task for all
            task_ref = task_names[0]
            task_other = task_names[0]
        else:
            # Randomly select two different tasks
            task_ref, task_other = random.sample(task_names, 2)
        
        # Get trajectories from reference task
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
        
        # Determine ranking (reference is best, o^1 is better than o^2)
        ranking = [1, 2, 3]  # reference=1 (best), o^1=2, o^2=3 (worst)
        
        # Create unified sample structure with all fields
        sample = Sample(
            prediction_type="comparative",
            # Core HF dataset fields (from ref_traj)
            id=ref_traj['id'],
            task=ref_traj['task'],
            lang_vector=ref_traj['lang_vector'],
            data_source=ref_traj['data_source'],
            frames=ref_traj['frames'],
            optimal=ref_traj['optimal'],
            ranking=ref_traj['ranking'],
            preference_embedding=ref_traj['preference_embedding'],
            is_robot=ref_traj['is_robot'],
            metadata=ref_traj.get('metadata'),
            # Comparative-specific fields
            reference_frames=ref_traj['frames'],  # o^ref
            trajectory_A_frames=traj_1['frames'],  # o^1
            trajectory_B_frames=traj_2['frames'],  # o^2
            ranking_list=ranking,  # [reference_rank, o1_rank, o2_rank]
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
            trajectory_A_ranking=traj_1['ranking'],
            trajectory_A_preference_embedding=traj_1['preference_embedding'],
            trajectory_A_is_robot=traj_1['is_robot'],
            # Trajectory B fields (o^2)
            trajectory_B_task=traj_2['task'],
            trajectory_B_lang_vector=traj_2['lang_vector'],
            trajectory_B_data_source=traj_2['data_source'],
            trajectory_B_optimal=traj_2['optimal'],
            trajectory_B_ranking=traj_2['ranking'],
            trajectory_B_preference_embedding=traj_2['preference_embedding'],
            trajectory_B_is_robot=traj_2['is_robot'],
        )
        
        return sample
    
    def _determine_batch_composition(self) -> Dict[str, int]:
        """Determine how many samples of each type to include in the batch."""
        
        # Calculate target counts based on ratios
        total_samples = self.batch_size
        
        # For preference and comparative, use the specified ratios
        preference_count = int(total_samples * self.preference_ratio)
        comparative_count = int(total_samples * self.comparative_ratio)
        
        # For progress, we can use all data (both preference and comparative samples can be used for progress)
        # So we calculate how many additional progress samples we need
        progress_from_preference = int(preference_count * self.progress_ratio)
        progress_from_comparative = int(comparative_count * self.progress_ratio)
        additional_progress = max(0, total_samples - preference_count - comparative_count)
        
        # Ensure we don't exceed batch size
        actual_total = preference_count + comparative_count + additional_progress
        if actual_total > total_samples:
            # Scale down proportionally
            scale_factor = total_samples / actual_total
            preference_count = int(preference_count * scale_factor)
            comparative_count = int(comparative_count * scale_factor)
            additional_progress = total_samples - preference_count - comparative_count
        
        return {
            "preference": preference_count,
            "comparative": comparative_count,
            "progress": additional_progress,
            "progress_from_preference": progress_from_preference,
            "progress_from_comparative": progress_from_comparative
        }
    
    def generate_batch(self) -> 'Batch':
        """Generate a single batch of data."""
        
        # Determine batch composition
        composition = self._determine_batch_composition()
        
        samples = []
        
        # Generate preference samples
        for _ in range(composition["preference"]):
            try:
                sample = self._create_preference_sample()
                samples.append(sample)
            except Exception as e:
                print(f"Error creating preference sample: {e}")
                continue
        
        # Generate comparative samples
        for _ in range(composition["comparative"]):
            try:
                sample = self._create_comparative_sample()
                samples.append(sample)
            except Exception as e:
                print(f"Error creating comparative sample: {e}")
                continue
        
        # Generate additional progress samples if needed
        while len(samples) < self.batch_size:
            try:
                sample = self._create_progress_sample()
                samples.append(sample)
            except Exception as e:
                print(f"Error creating progress sample: {e}")
                continue
        
        # Shuffle if requested
        if self.shuffle:
            random.shuffle(samples)
        
        return Batch(samples=samples)
    
    def generate_batches(self, num_batches: int) -> Iterator[Batch]:
        """Generate multiple batches."""
        for i in range(num_batches):
            yield self.generate_batch()

def main():
    """Example usage of the DataGenerator."""
    
    # Create data generator with 50/50 preference/comparative ratio
    generator = DataGenerator(
        dataset_path="libero_hf_dataset/libero_hf_dataset",
        batch_size=32,
        preference_ratio=0.5,
        comparative_ratio=0.5,
        progress_ratio=1.0,  # All data can be used for progress
        shuffle=True,
        seed=42
    )
    
    # Generate a single batch
    print("Generating a single batch...")
    batch = generator.generate_batch()

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
            self.samples.extend(batch.samples)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample by index."""
        return self.samples[idx]


if __name__ == "__main__":
    main() 