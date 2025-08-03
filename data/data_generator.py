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
import torch


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
    trajectory_B_is_robot: Optional[bool] = None
    
    # Comparative-specific fields
    reference_frames: Optional[List[str]] = None  # o^ref
    trajectory_A_frames: Optional[List[str]] = None  # o^1
    trajectory_B_frames: Optional[List[str]] = None  # o^2
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
    trajectory_A_is_robot: Optional[bool] = None
    trajectory_B_task: Optional[str] = None
    trajectory_B_lang_vector: Optional[np.ndarray] = None
    trajectory_B_data_source: Optional[str] = None
    trajectory_B_optimal: Optional[bool] = None
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


@dataclass
class ProcessedBatch:
    """A batch of processed samples with model inputs."""
    
    # Text inputs
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    
    # Vision inputs (if present)
    pixel_values: Optional[torch.Tensor] = None
    pixel_values_videos: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    video_grid_thw: Optional[torch.Tensor] = None
    
    # Trajectory A inputs (for preference/comparative samples)
    input_ids_A: Optional[torch.Tensor] = None
    attention_mask_A: Optional[torch.Tensor] = None
    pixel_values_A: Optional[torch.Tensor] = None
    pixel_values_videos_A: Optional[torch.Tensor] = None
    image_grid_thw_A: Optional[torch.Tensor] = None
    video_grid_thw_A: Optional[torch.Tensor] = None
    
    # Trajectory B inputs (for preference/comparative samples)
    input_ids_B: Optional[torch.Tensor] = None
    attention_mask_B: Optional[torch.Tensor] = None
    pixel_values_B: Optional[torch.Tensor] = None
    pixel_values_videos_B: Optional[torch.Tensor] = None
    image_grid_thw_B: Optional[torch.Tensor] = None
    video_grid_thw_B: Optional[torch.Tensor] = None
    
    # Reference inputs (for comparative samples)
    input_ids_ref: Optional[torch.Tensor] = None
    attention_mask_ref: Optional[torch.Tensor] = None
    pixel_values_ref: Optional[torch.Tensor] = None
    pixel_values_videos_ref: Optional[torch.Tensor] = None
    image_grid_thw_ref: Optional[torch.Tensor] = None
    video_grid_thw_ref: Optional[torch.Tensor] = None
    
    # Sample metadata
    samples: List[Sample] = None


class BatchCollator:
    """Simple batch collator that processes Sample objects."""
    
    def __init__(self, processor=None, max_length: int = 1024):
        """
        Initialize the batch collator.
        
        Args:
            processor: HuggingFace processor for text and vision processing
            max_length: Maximum sequence length for text
        """
        self.processor = processor
        self.max_length = max_length
    
    def __call__(self, samples: List[Sample]) -> ProcessedBatch:
        """
        Collate a list of samples into a ProcessedBatch object.
        
        Args:
            samples: List of Sample objects
            
        Returns:
            ProcessedBatch object containing the processed tensors
        """
        # For now, create dummy tensors for testing
        batch_size = len(samples)
        
        # Create dummy input_ids and attention_mask
        input_ids = torch.randint(0, 1000, (batch_size, self.max_length))
        attention_mask = torch.ones(batch_size, self.max_length)
        
        # Create dummy tensors for trajectory A and B if needed
        input_ids_A = None
        input_ids_B = None
        input_ids_ref = None
        
        # Check if we have preference or comparative samples
        has_preference = any(s.prediction_type == "preference" for s in samples)
        has_comparative = any(s.prediction_type == "comparative" for s in samples)
        
        if has_preference or has_comparative:
            input_ids_A = torch.randint(0, 1000, (batch_size, self.max_length))
            input_ids_B = torch.randint(0, 1000, (batch_size, self.max_length))
        
        if has_comparative:
            input_ids_ref = torch.randint(0, 1000, (batch_size, self.max_length))
        
        return ProcessedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_ids_A=input_ids_A,
            input_ids_B=input_ids_B,
            input_ids_ref=input_ids_ref,
            samples=samples
        )


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
        comparative_ratio: float = 0.5,
        max_frames: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
        rewind_ratio: float = 0.3,
        dataset_preference_ratio: float = 0.7
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
            comparative_ratio: Ratio of comparative scoring samples (0.0 to 1.0)
            max_frames: Maximum frames per trajectory
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducibility
            rewind_ratio: Ratio of rewind-generated preferences (0.0 to 1.0)
            dataset_preference_ratio: Ratio of preferences from dataset vs rewind (0.0 to 1.0)
        """
        self.dataset_path = dataset_path
        self.dataset_subsets = dataset_subsets
        self.preference_dataset_path = preference_dataset_path
        self.preference_dataset_subset = preference_dataset_subset
        self.batch_size = batch_size
        self.preference_ratio = preference_ratio
        self.comparative_ratio = comparative_ratio
        self.max_frames = max_frames
        self.shuffle = shuffle
        self.seed = seed
        self.rewind_ratio = rewind_ratio
        self.dataset_preference_ratio = dataset_preference_ratio
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Load trajectory dataset
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
        print(f"Ratios - Preference: {preference_ratio}, Comparative: {comparative_ratio}")
        print(f"Rewind ratio: {rewind_ratio}, Dataset preference ratio: {dataset_preference_ratio}")
    
    def _create_rewind_trajectory(self, original_traj: Dict) -> Dict:
        """Create a suboptimal trajectory by rewinding the original trajectory."""
        frames = original_traj['frames']
        
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
        all_trajectories = []
        
        for dataset_name in self.dataset_subsets:
            print(f"Loading dataset: {dataset_name}")
            dataset = self._load_dataset_from_path(self.dataset_path, dataset_name)
            
            # Handle DatasetDict by accessing the train split
            if hasattr(dataset, 'keys') and 'train' in dataset:
                print(f"  Found DatasetDict with train split, accessing train data...")
                dataset = dataset['train']
            
            dataset_trajectories = list(dataset)
            all_trajectories.extend(dataset_trajectories)
            print(f"  Loaded {len(dataset_trajectories)} trajectories from dataset '{dataset_name}'")
        
        self.trajectories = all_trajectories
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
            # Load from HuggingFace Hub
            from datasets import load_dataset
            print(f"Loading from HuggingFace Hub: {dataset_path}")
            if subset:
                return load_dataset(dataset_path, name=subset)
            else:
                return load_dataset(dataset_path)
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
    
    def _create_preference_sample(self) -> Sample:
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
            
            # Create unified sample structure with all fields
            sample = Sample(
                prediction_type="preference",
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
            )
        else:
            # Use generated preference (rewind or random)
            sample = self._create_generated_preference_sample()
        
        return sample
    
    def _create_generated_preference_sample(self) -> Sample:
        """Create a preference prediction sample using various negative generation strategies."""
        
        # Use preprocessed optimal trajectories
        if not self.optimal_trajectories:
            raise ValueError("No optimal trajectories found for preference generation")
        
        optimal_traj = random.choice(self.optimal_trajectories)
        
        # Choose negative generation strategy
        strategy = random.random()
        
        if strategy < self.rewind_ratio:
            # Strategy 1: Use rewind-generated suboptimal trajectory
            negative_traj = self._create_rewind_trajectory(optimal_traj)
        elif strategy < 0.7:
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
        
        # Create unified sample structure with all fields
        sample = Sample(
            prediction_type="preference",
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
        )
        
        return sample
    
    def _create_progress_sample(self) -> Sample:
        """Create a progress prediction sample: single trajectory for progress prediction."""
        
        # Use preprocessed optimal trajectories for progress prediction
        if not self.optimal_trajectories:
            raise ValueError("No optimal trajectories found for progress prediction")
        
        traj = random.choice(self.optimal_trajectories)
        
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
        )
        
        return sample
    
    def _determine_batch_composition(self) -> Dict[str, int]:
        """Determine how many samples of each type to include in the batch."""
        
        # Calculate target counts based on ratios
        total_samples = self.batch_size
        
        # For preference and comparative, use the specified ratios
        preference_count = int(total_samples * self.preference_ratio)
        comparative_count = int(total_samples * self.comparative_ratio)
        
        # Ensure we don't exceed batch size
        actual_total = preference_count + comparative_count
        if actual_total > total_samples:
            # Scale down proportionally
            scale_factor = total_samples / actual_total
            preference_count = int(preference_count * scale_factor)
            comparative_count = int(comparative_count * scale_factor)
        
        return {
            "preference": preference_count,
            "comparative": comparative_count
        }
    
    def generate_batch(self) -> 'Batch':
        """Generate a single batch of data."""
        
        # Determine batch composition
        composition = self._determine_batch_composition()
        
        samples = []
        
        # Generate preference samples
        for _ in range(composition["preference"]):
            sample = self._create_preference_sample()
            samples.append(sample)
        
        # Generate comparative samples
        for _ in range(composition["comparative"]):
            sample = self._create_comparative_sample()
            samples.append(sample)
        
        # Generate additional progress samples if needed
        while len(samples) < self.batch_size:
            sample = self._create_progress_sample()
            samples.append(sample)
        
        # Shuffle if requested
        if self.shuffle:
            random.shuffle(samples)
        
        return Batch(samples=samples)
    
    def generate_batches(self, num_batches: int) -> Iterator[Batch]:
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
            self.samples.extend(batch.samples)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample by index."""
        return self.samples[idx]


def test():
    """Test the BatchCollator with generated samples."""
    
    # Create data generator
    generator = DataGenerator(
        dataset_path="aliangdw/rfm",
        dataset_subsets=["libero_90"],
        batch_size=4,  # Small batch for testing
        preference_ratio=0.5,
        comparative_ratio=0.5,
        shuffle=True,
        seed=42
    )
    
    # Generate a batch
    print("Generating test batch...")
    batch = generator.generate_batch()
    print(f"Generated batch with {len(batch)} samples")
    
    # Print sample types
    for i, sample in enumerate(batch.samples):
        print(f"Sample {i}: {sample.prediction_type}")
    
    # Test the batch collator
    print("\nTesting batch collator...")
    batch_collator = BatchCollator(max_length=1024)
    
    try:
        processed_batch = batch_collator(batch.samples)
        print("✅ Successfully processed batch!")
        print(f"Processed batch type: {type(processed_batch)}")
        print(f"Input IDs shape: {processed_batch.input_ids.shape}")
        
        # Check for trajectory-specific inputs
        if processed_batch.input_ids_A is not None:
            print(f"Trajectory A input IDs shape: {processed_batch.input_ids_A.shape}")
        if processed_batch.input_ids_B is not None:
            print(f"Trajectory B input IDs shape: {processed_batch.input_ids_B.shape}")
        if processed_batch.input_ids_ref is not None:
            print(f"Reference input IDs shape: {processed_batch.input_ids_ref.shape}")
            
    except Exception as e:
        print(f"❌ Error processing batch: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # main()
    test() 