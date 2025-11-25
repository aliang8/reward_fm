#!/usr/bin/env python3
"""
Data generator for similarity score evaluation.

This generator creates similarity samples for evaluation:
- For each paired human-robot trajectory pair (same task), creates similarity samples
- For each pairing, samples N negative trajectories from other tasks
- Creates similarity samples: ref=human, traj_sim=robot (same task), traj_diff=negative (different task)
"""

import random
import torch
from tqdm import tqdm

from rfm.data.dataset_types import SimilaritySample, Trajectory
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.datasets.helpers import (
    linspace_subsample_frames,
    pad_trajectory_to_max_frames_np,
    pad_trajectory_to_max_frames_torch,
    load_embeddings_from_path,
    load_frames_from_npz,
)
from rfm.utils.distributed import rank_0_print


class SimilarityScoreSampler(RFMBaseSampler):
    """
    Data generator that creates similarity samples for evaluation.
    
    For each paired human-robot trajectory pair (same task):
    - Creates similarity samples with ref=human, traj_sim=robot, traj_diff=negative (from different task)
    - Samples N negative trajectories from other tasks for each pairing
    """

    def __init__(
        self,
        config,
        dataset,
        combined_indices,
        dataset_success_cutoff_map=None,
        is_evaluation=False,
        verbose=True,
        num_negatives: int = 5,
        **kwargs,
    ):
        super().__init__(config, dataset, combined_indices, dataset_success_cutoff_map, verbose=verbose)
        
        self.num_negatives = num_negatives
        self.sample_indices = self._generate_all_sample_indices()
        
        rank_0_print(
            f"Generated {len(self.sample_indices)} similarity score sample indices from {len(self.paired_human_robot_by_task)} tasks",
            verbose=self.verbose,
        )

    def _generate_all_sample_indices(self) -> list[dict]:
        """Generate all possible similarity score sample indices."""
        sample_indices = []
        
        import ipdb; ipdb.set_trace()
        # Iterate through all tasks with paired human-robot data
        for task, paired_info in self.paired_human_robot_by_task.items():
            human_indices = paired_info["human"]
            robot_indices = paired_info["robot"]
            
            if not human_indices or not robot_indices:
                continue
            
            # Get all tasks except the current one for negative sampling
            other_tasks = [t for t in self.task_indices.keys() if t != task]
            
            if not other_tasks:
                continue
            
            # For each human-robot pair, create N samples (one per negative)
            for human_idx in human_indices:
                for robot_idx in robot_indices:
                    # Sample N negative tasks (with replacement if needed)
                    negative_tasks = random.choices(other_tasks, k=self.num_negatives)
                    
                    # Create one sample index entry per negative
                    for negative_task in negative_tasks:
                        negative_task_indices = self.task_indices.get(negative_task, [])
                        if not negative_task_indices:
                            continue
                        
                        # Store the negative task, we'll sample a specific negative during generation
                        sample_indices.append({
                            "human_idx": human_idx,
                            "robot_idx": robot_idx,
                            "task": task,
                            "negative_task": negative_task,
                            "negative_task_indices": negative_task_indices,
                        })
        
        return sample_indices

    def _generate_sample_from_indices(self, sample_idx_info: dict) -> SimilaritySample:
        """Generate a single similarity sample from stored indices."""
        human_idx = sample_idx_info["human_idx"]
        robot_idx = sample_idx_info["robot_idx"]
        task = sample_idx_info["task"]
        negative_task = sample_idx_info["negative_task"]
        negative_task_indices = sample_idx_info["negative_task_indices"]
        
        # Get human and robot trajectories
        human_traj = self.dataset[human_idx]
        robot_traj = self.dataset[robot_idx]
        
        # Sample a negative trajectory from the specified different task
        if not negative_task_indices:
            return None
        
        negative_idx = random.choice(negative_task_indices)
        negative_traj = self.dataset[negative_idx]
        
        # Create trajectories for the similarity sample
        ref_traj = self._create_trajectory_from_data(human_traj)
        sim_traj = self._create_trajectory_from_data(robot_traj)
        diff_traj = self._create_trajectory_from_data(negative_traj)
        
        # Create metadata
        metadata = {
            "task": task,
            "negative_task": negative_task,
            "human_id": human_traj["id"],
            "robot_id": robot_traj["id"],
            "negative_id": negative_traj["id"],
            "data_gen_strategy": "similarity_score_eval",
        }
        
        # Add metadata to trajectories
        if ref_traj.metadata is None:
            ref_traj.metadata = {}
        ref_traj.metadata.update(metadata)
        
        sample = SimilaritySample(
            ref_trajectory=ref_traj,
            sim_trajectory=sim_traj,
            diff_trajectory=diff_traj,
            data_gen_strategy="similarity_score_eval",
        )
        
        return sample

    def _create_trajectory_from_data(self, traj_data: dict) -> Trajectory:
        """Create a Trajectory object from dataset entry."""
        # Get frames or embeddings
        frames = None
        video_embeddings = None
        text_embedding = None
        
        if self.config.load_embeddings and traj_data.get("embeddings_path"):
            embeddings = load_embeddings_from_path(traj_data["embeddings_path"])
            video_embeddings = embeddings["video_embeddings"]
            text_embedding = embeddings["text_embedding"]
            
            # Subsample frames
            subsequence_video_embeddings, frame_indices = linspace_subsample_frames(
                video_embeddings, self.config.max_frames
            )
            frames_shape_orig = subsequence_video_embeddings.shape
            
            # Pad trajectory
            video_embeddings, padded_progress = pad_trajectory_to_max_frames_torch(
                subsequence_video_embeddings, [1.0] * len(frame_indices), self.config.max_frames
            )
        else:
            frames = load_frames_from_npz(traj_data["frames"])
            if frames is None or len(frames) == 0:
                return None
            
            # Subsample frames
            subsequence_frames, frame_indices = linspace_subsample_frames(frames, self.config.max_frames)
            frames_shape_orig = subsequence_frames.shape
            
            # Pad trajectory
            frames, padded_progress = pad_trajectory_to_max_frames_np(
                subsequence_frames, [1.0] * len(frame_indices), self.config.max_frames
            )
        
        # Create trajectory
        trajectory = Trajectory(
            id=traj_data["id"],
            task=traj_data["task"],
            frames=frames,
            frames_shape=frames_shape_orig,
            video_embeddings=video_embeddings,
            text_embedding=text_embedding,
            data_source=traj_data["data_source"],
            lang_vector=traj_data.get("lang_vector"),
            is_robot=traj_data["is_robot"],
            quality_label=traj_data.get("quality_label"),
            data_gen_strategy="similarity_score_eval",
            target_progress=padded_progress,
            partial_success=traj_data.get("partial_success"),
            metadata={},
        )
        
        return trajectory

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])

