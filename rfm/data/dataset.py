#!/usr/bin/env python3
"""
Custom dataset classes for RFM data generation.

This module contains specialized dataset classes that generate different types of samples:
- InfiniteDataGeneratorDataset: Generates preference/similarity samples
- RewoundDataset: Generates preference samples where original is chosen and rewound is rejected
- PairedSuccessFailureDataset: Generates preference samples by pairing successful/failed trajectories
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Iterator, Union
from tqdm import tqdm
from rfm.data.batch_collator import BaseSample, PreferenceSample, SimilaritySample
from rfm.utils.logging import rank_0_print


class InfiniteDataGeneratorDataset:
    """Dataset that generates preference and similarity samples infinitely."""
    
    def __init__(self, data_generator, max_samples=100):
        self.data_generator = data_generator
        self.max_samples = max_samples
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.max_samples

    def __next__(self):
        """Generate the next sample."""
        # Randomly choose between preference and similarity
        if random.random() < self.data_generator.preference_ratio:
            sample = self.data_generator._create_preference_sample_with_strategies()
        else:
            sample = self.data_generator._create_similarity_sample()
        
        return sample

    def __getitem__(self, idx):
        return self.__next__()

class RewoundDataset:
    """Dataset that generates preference samples where original trajectory is chosen and rewound is rejected."""
    
    def __init__(self, data_generator, **kwargs):
        self.data_generator = data_generator
        
        # Get rewind parameters from config
        self.rewind_lengths = getattr(data_generator.config.data, "rewind_lengths", None)
        self.samples_per_trajectory = getattr(data_generator.config.data, "samples_per_trajectory", 1)
        
        # If no rewind lengths specified, use 1 to max_frames - 1
        if self.rewind_lengths is None:
            max_frames = getattr(data_generator.config.data, "max_frames", 8)
            self.rewind_lengths = list(range(1, max_frames))
        
        # Generate all possible sample indices upfront (not the actual samples)
        self.sample_indices = self._generate_all_sample_indices()
        self.current_idx = 0
        
        rank_0_print(f"Generated {len(self.sample_indices)} rewound sample indices")
    
    def _generate_all_sample_indices(self) -> List[Dict]:
        """Generate all possible rewound sample indices (not the actual samples)."""
        sample_indices = []
        
        for traj_idx in self.data_generator.robot_trajectories:
            original_traj = self.data_generator.dataset[traj_idx]
            
            for rewind_length in self.rewind_lengths:
                for _ in range(self.samples_per_trajectory):
                    # Store just the indices and parameters needed to generate the sample later
                    sample_indices.append({
                        'original_traj_idx': traj_idx,
                        'rewind_length': rewind_length,
                        'original_traj_id': original_traj.get('id', f'traj_{traj_idx}')
                    })
        
        return sample_indices
    
    def _generate_sample_from_indices(self, sample_idx_info: Dict) -> PreferenceSample:
        """Generate a single sample from stored indices."""
        original_traj_idx = sample_idx_info['original_traj_idx']
        rewind_length = sample_idx_info['rewind_length']
        
        # Get the original trajectory
        original_traj = self.data_generator.dataset[original_traj_idx]
        
        # Create rewound trajectory
        rewound_traj = self.data_generator._create_rewind_trajectory(
            original_traj, rewind_length=rewind_length
        )
        
        # Get frames
        original_frames = self.data_generator._get_trajectory_frames(original_traj_idx)
        rewound_frames = rewound_traj["frames"]  # Already numpy array
        
        # Create preference sample (original is chosen, rewound is rejected)
        sample = PreferenceSample(
            chosen_frames=original_frames,
            rejected_frames=rewound_frames,
            chosen_id=original_traj["id"],
            rejected_id=rewound_traj["id"],
            chosen_task=original_traj["task"],
            rejected_task=rewound_traj["task"],
            chosen_lang_vector=original_traj["lang_vector"],
            rejected_lang_vector=rewound_traj["lang_vector"],
            chosen_data_source=original_traj["data_source"],
            rejected_data_source=rewound_traj["data_source"],
            chosen_quality_label=original_traj["quality_label"],
            rejected_quality_label=rewound_traj["quality_label"],
            chosen_is_robot=original_traj["is_robot"],
            rejected_is_robot=rewound_traj["is_robot"],
            chosen_frames_shape=original_traj["frames_shape"],
            rejected_frames_shape=rewound_traj["frames_shape"],
            sample_type="preference",
            id=original_traj["id"],
            task=original_traj["task"],
            lang_vector=original_traj["lang_vector"],
            data_source=original_traj["data_source"],
            quality_label=original_traj["quality_label"],
            is_robot=original_traj["is_robot"],
            frames_shape=original_traj["frames_shape"],
            num_frames_rewound=rewound_traj.get("num_frames_rewound", rewind_length),
        )
        
        return sample
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self):
        """Get the next sample by generating it from stored indices."""
        if self.current_idx >= len(self.sample_indices):
            raise StopIteration
        
        # Get the sample indices for this sample
        sample_idx_info = self.sample_indices[self.current_idx]
        
        # Generate the actual sample on-demand
        sample = self._generate_sample_from_indices(sample_idx_info)
        
        self.current_idx += 1
        return sample
    
    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self.__next__()


class PairedSuccessFailureDataset:
    """Dataset that generates preference samples by pairing successful and failed trajectories for the same task."""
    
    def __init__(self, data_generator, **kwargs):
        self.data_generator = data_generator
        
        # Generate all possible sample indices upfront (not the actual samples)
        self.sample_indices = self._generate_all_sample_indices()
        self.current_idx = 0
        
        rank_0_print(f"Generated {len(self.sample_indices)} success-failure sample indices")
    
    def _generate_all_sample_indices(self) -> List[Dict]:
        """Generate all possible success-failure sample indices (not the actual samples)."""
        sample_indices = []
        
        # Group trajectories by task and success status
        task_success_trajs = {}
        task_failure_trajs = {}
        
        print(f"Generating success-failure samples for {len(self.data_generator.robot_trajectories)} trajectories")
        
        for traj_idx in self.data_generator.robot_trajectories:
            traj = self.data_generator.dataset[traj_idx]
            task = traj.get("task", "unknown")
            quality_label = traj.get("quality_label", "unknown")
            
            if task not in task_success_trajs:
                task_success_trajs[task] = []
                task_failure_trajs[task] = []
            
            if quality_label == "successful":
                task_success_trajs[task].append(traj_idx)
            elif quality_label == "failure":
                task_failure_trajs[task].append(traj_idx)
        
        print(f"Generated {len(task_success_trajs)} success tasks and {len(task_failure_trajs)} failure tasks")
        
        # Generate all possible pairs
        for task in tqdm(task_success_trajs, desc="Generating success-failure samples"):
            success_indices = task_success_trajs[task]
            failure_indices = task_failure_trajs[task]
            
            if not success_indices or not failure_indices:
                continue
            
            # Create all possible pairs (successful is chosen, failed is rejected)
            for success_idx in success_indices:
                for failure_idx in failure_indices:
                    # Store just the indices needed to generate the sample later
                    sample_indices.append({
                        'success_traj_idx': success_idx,
                        'failure_traj_idx': failure_idx,
                        'task': task
                    })
        
        return sample_indices
    
    def _generate_sample_from_indices(self, sample_idx_info: Dict) -> PreferenceSample:
        """Generate a single sample from stored indices."""
        success_idx = sample_idx_info['success_traj_idx']
        failure_idx = sample_idx_info['failure_traj_idx']
        
        # Get the trajectories
        success_traj = self.data_generator.dataset[success_idx]
        failure_traj = self.data_generator.dataset[failure_idx]
        
        # Get frames
        success_frames = self.data_generator._get_trajectory_frames(success_idx)
        failure_frames = self.data_generator._get_trajectory_frames(failure_idx)
        
        # Create preference sample (successful is chosen, failed is rejected)
        sample = PreferenceSample(
            chosen_frames=success_frames,
            rejected_frames=failure_frames,
            chosen_id=success_traj["id"],
            rejected_id=failure_traj["id"],
            chosen_task=success_traj["task"],
            rejected_task=failure_traj["task"],
            chosen_lang_vector=success_traj["lang_vector"],
            rejected_lang_vector=failure_traj["lang_vector"],
            chosen_data_source=success_traj["data_source"],
            rejected_data_source=failure_traj["data_source"],
            chosen_quality_label=success_traj["quality_label"],
            rejected_quality_label=failure_traj["quality_label"],
            chosen_is_robot=success_traj["is_robot"],
            rejected_is_robot=failure_traj["is_robot"],
            chosen_frames_shape=success_traj["frames_shape"],
            rejected_frames_shape=failure_traj["frames_shape"],
            id=success_traj["id"],
            task=success_traj["task"],
            lang_vector=success_traj["lang_vector"],
            data_source=success_traj["data_source"],
            quality_label=success_traj["quality_label"],
            is_robot=success_traj["is_robot"],
            frames_shape=success_traj["frames_shape"],
            sample_type="preference",
            num_frames_rewound=None,  # Not applicable for success-failure pairs
        )
        
        return sample
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self):
        """Get the next sample by generating it from stored indices."""
        if self.current_idx >= len(self.sample_indices):
            raise StopIteration
        
        # Get the sample indices for this sample
        sample_idx_info = self.sample_indices[self.current_idx]
        
        # Generate the actual sample on-demand
        sample = self._generate_sample_from_indices(sample_idx_info)
        
        self.current_idx += 1
        return sample
    
    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self.__next__()