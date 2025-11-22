import numpy as np
import torch
import random
from itertools import combinations
from tqdm import tqdm

from rfm.data.dataset_types import PreferenceSample, Trajectory
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.datasets.helpers import (
    linspace_subsample_frames,
    pad_trajectory_to_max_frames_np,
    pad_trajectory_to_max_frames_torch,
    load_frames_from_npz,
    load_embeddings_from_path,
    convert_absolute_to_relative_progress,
)
from rfm.utils.distributed import rank_0_print


class RoboArenaQualityPreferenceSampler(RFMBaseSampler):
    """Dataset that generates preference samples by pairing trajectories with different partial_rewards for the same task.
    
    For RoboArena dataset, pairs trajectories from the same task where the chosen trajectory
    has a higher partial_reward (partial_success) than the rejected trajectory.
    """

    def __init__(
        self,
        config,
        dataset,
        combined_indices,
        dataset_success_cutoff_map=None,
        is_evaluation=False,
        verbose=True,
        **kwargs,
    ):
        super().__init__(config, dataset, combined_indices, dataset_success_cutoff_map, verbose=verbose)

        self._cached_tasks = self.dataset["task"]
        self._cached_partial_success = self.dataset.get("partial_success")

        # Generate all possible sample indices upfront (not the actual samples)
        self.sample_indices = self._generate_all_sample_indices()
        rank_0_print(f"Generated {len(self.sample_indices)} RoboArena quality preference sample indices", verbose=self.verbose)

    def _generate_all_sample_indices(self) -> list[dict]:
        """Generate all possible quality preference sample indices based on partial_reward (partial_success)."""
        sample_indices = []

        # Group trajectories by task
        task_to_trajs = {}

        rank_0_print(
            f"Generating RoboArena quality preference samples for {len(self.robot_trajectories)} trajectories",
            verbose=self.verbose,
        )

        for traj_idx in self.robot_trajectories:
            # Use cached arrays for efficient access
            task = self._cached_tasks[traj_idx]
            partial_success = self._cached_partial_success[traj_idx] if traj_idx < len(self._cached_partial_success) else None
            
            # Ensure partial_success exists
            if partial_success is None:
                rank_0_print(
                    f"Warning: Trajectory {traj_idx} (task: {task}) missing partial_success, skipping",
                    verbose=self.verbose,
                )
                continue

            if task not in task_to_trajs:
                task_to_trajs[task] = []

            task_to_trajs[task].append({
                "traj_idx": traj_idx,
                "partial_success": float(partial_success),
            })

        # Generate pairs for each task
        for task in tqdm(task_to_trajs, desc="Generating RoboArena quality preference samples"):
            trajs = task_to_trajs[task]

            # Need at least 2 trajectories to create pairs
            if len(trajs) < 2:
                continue

            # Create all pairs of trajectories
            for i, traj1 in enumerate(trajs):
                for j, traj2 in enumerate(trajs):
                    if i >= j:  # Avoid duplicates and self-pairs
                        continue

                    partial1 = traj1["partial_success"]
                    partial2 = traj2["partial_success"]

                    # Skip if partial_success values are equal (can't determine preference)
                    if partial1 == partial2:
                        continue

                    # Determine which trajectory is chosen (higher partial_success)
                    if partial1 > partial2:
                        chosen_traj_idx = traj1["traj_idx"]
                        rejected_traj_idx = traj2["traj_idx"]
                        chosen_partial = partial1
                        rejected_partial = partial2
                    else:
                        chosen_traj_idx = traj2["traj_idx"]
                        rejected_traj_idx = traj1["traj_idx"]
                        chosen_partial = partial2
                        rejected_partial = partial1

                    sample_indices.append({
                        "chosen_traj_idx": chosen_traj_idx,
                        "rejected_traj_idx": rejected_traj_idx,
                        "task": task,
                        "chosen_partial_success": chosen_partial,
                        "rejected_partial_success": rejected_partial,
                    })

        return sample_indices

    def _generate_sample_from_indices(self, sample_idx_info: dict) -> PreferenceSample:
        """Generate a single sample from stored indices."""
        chosen_idx = sample_idx_info["chosen_traj_idx"]
        rejected_idx = sample_idx_info["rejected_traj_idx"]

        # Get the trajectories
        chosen_traj = self.dataset[chosen_idx]
        rejected_traj = self.dataset[rejected_idx]

        # Initialize variables
        chosen_frames = None
        chosen_video_embeddings = None
        chosen_text_embedding = None
        rejected_frames = None
        rejected_video_embeddings = None
        rejected_text_embedding = None

        max_frames = self.config.max_frames

        # Load and process chosen trajectory
        if self.config.load_embeddings and chosen_traj.get("embeddings_path"):
            embeddings = load_embeddings_from_path(chosen_traj["embeddings_path"])
            chosen_video_embeddings = embeddings["video_embeddings"]
            chosen_text_embedding = embeddings["text_embedding"]
            data = chosen_video_embeddings
            total_frames = chosen_video_embeddings.shape[0] if hasattr(chosen_video_embeddings, "shape") else len(chosen_video_embeddings)
            use_embeddings = True
        else:
            chosen_frames = load_frames_from_npz(chosen_traj["frames"])
            data = chosen_frames
            total_frames = len(chosen_frames)
            use_embeddings = False

        data, frame_indices = linspace_subsample_frames(data, max_frames)
        chosen_frames_shape_orig = data.shape
        progress_abs = [idx / (total_frames - 1) for idx in frame_indices]

        if use_embeddings:
            chosen_video_embeddings, progress_abs = pad_trajectory_to_max_frames_torch(data, progress_abs, max_frames)
        else:
            chosen_frames, progress_abs = pad_trajectory_to_max_frames_np(data, progress_abs, max_frames)

        if self.config.progress_pred_type == "relative":
            chosen_progress = convert_absolute_to_relative_progress(progress_abs)
        else:
            chosen_progress = progress_abs

        chosen_metadata = {
            "quality_label": chosen_traj["quality_label"],
            "data_source": chosen_traj["data_source"],
            "task": chosen_traj["task"],
            "id": chosen_traj["id"],
            "video_path": chosen_traj["frames"],
            "partial_success": chosen_traj.get("partial_success"),
        }

        # Load and process rejected trajectory
        if self.config.load_embeddings and rejected_traj.get("embeddings_path"):
            embeddings = load_embeddings_from_path(rejected_traj["embeddings_path"])
            rejected_video_embeddings = embeddings["video_embeddings"]
            rejected_text_embedding = embeddings["text_embedding"]
            data = rejected_video_embeddings
            total_frames = rejected_video_embeddings.shape[0] if hasattr(rejected_video_embeddings, "shape") else len(rejected_video_embeddings)
            use_embeddings = True
        else:
            rejected_frames = load_frames_from_npz(rejected_traj["frames"])
            data = rejected_frames
            total_frames = len(rejected_frames)
            use_embeddings = False

        data, frame_indices = linspace_subsample_frames(data, max_frames)
        rejected_frames_shape_orig = data.shape
        progress_abs = [idx / (total_frames - 1) for idx in frame_indices]

        if use_embeddings:
            rejected_video_embeddings, progress_abs = pad_trajectory_to_max_frames_torch(data, progress_abs, max_frames)
        else:
            rejected_frames, progress_abs = pad_trajectory_to_max_frames_np(data, progress_abs, max_frames)

        if self.config.progress_pred_type == "relative":
            rejected_progress = convert_absolute_to_relative_progress(progress_abs)
        else:
            rejected_progress = progress_abs

        rejected_metadata = {
            "quality_label": rejected_traj["quality_label"],
            "data_source": rejected_traj["data_source"],
            "task": rejected_traj["task"],
            "id": rejected_traj["id"],
            "video_path": rejected_traj["frames"],
            "partial_success": rejected_traj.get("partial_success"),
        }

        # Create trajectory objects
        chosen_trajectory = Trajectory(
            frames=chosen_frames,
            frames_shape=chosen_frames_shape_orig,
            video_embeddings=chosen_video_embeddings,
            text_embedding=chosen_text_embedding,
            id=chosen_traj["id"],
            task=chosen_traj["task"],
            lang_vector=np.array(chosen_traj["lang_vector"]),
            data_source=chosen_traj["data_source"],
            quality_label=chosen_traj["quality_label"],
            is_robot=chosen_traj["is_robot"],
            target_progress=chosen_progress,
            partial_success=chosen_traj.get("partial_success"),
            metadata=chosen_metadata,
        )

        rejected_trajectory = Trajectory(
            frames=rejected_frames,
            frames_shape=rejected_frames_shape_orig,
            video_embeddings=rejected_video_embeddings,
            text_embedding=rejected_text_embedding,
            id=rejected_traj["id"],
            task=rejected_traj["task"],
            lang_vector=np.array(rejected_traj["lang_vector"]),
            data_source=rejected_traj["data_source"],
            quality_label=rejected_traj["quality_label"],
            is_robot=rejected_traj["is_robot"],
            target_progress=rejected_progress,
            partial_success=rejected_traj.get("partial_success"),
            metadata=rejected_metadata,
        )

        # Create preference sample
        sample = PreferenceSample(
            chosen_trajectory=chosen_trajectory,
            rejected_trajectory=rejected_trajectory,
            data_gen_strategy="quality_preference_roboarena",
        )

        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])

