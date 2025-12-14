import numpy as np
import torch
import random
from collections import defaultdict
from rfm.data.dataset_types import ProgressSample, Trajectory
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.datasets.helpers import (
    linspace_subsample_frames,
    pad_trajectory_to_max_frames_np,
    pad_trajectory_to_max_frames_torch,
    load_embeddings_from_path,
    load_frames_from_npz,
    convert_absolute_to_relative_progress,
    compute_success_labels,
)
from rfm.utils.logger import get_logger

logger = get_logger()


class ProgressPolicyRankingSampler(RFMBaseSampler):
    """Dataset that generates progress samples for policy ranking by selecting N trajectories per quality label for tasks with multiple quality labels."""

    def __init__(
        self,
        config,
        dataset,
        combined_indices,
        dataset_success_cutoff_map=None,
        is_evaluation=False,
        verbose=True,
        num_examples_per_quality_pr: int = 5,
        **kwargs,
    ):
        super().__init__(config, dataset, combined_indices, dataset_success_cutoff_map, verbose=verbose)

        self.num_examples_per_quality_pr = num_examples_per_quality_pr
        logger.info(
            f"ProgressPolicyRankingSampler initialized with {len(self.robot_trajectories)} trajectories"
        )

        self.sample_indices = self._generate_all_sample_indices()

        logger.info(f"Generated {len(self.sample_indices)} sample indices")

    def _generate_all_sample_indices(self) -> list[dict]:
        """Generate sample indices by selecting tasks with multiple quality labels and sampling N trajectories per quality label."""
        # Group trajectories by task and quality label
        task_to_quality_to_trajs = defaultdict(lambda: defaultdict(list))
        
        for traj_idx in self.robot_trajectories:
            traj = self.dataset[traj_idx]
            task = traj["task"]
            quality_label = traj["quality_label"]
            task_to_quality_to_trajs[task][quality_label].append(traj_idx)

        # Filter to tasks that have multiple quality labels
        tasks_with_multiple_qualities = {
            task: quality_to_trajs
            for task, quality_to_trajs in task_to_quality_to_trajs.items()
            if len(quality_to_trajs) > 1
        }

        logger.info(
            f"Found {len(tasks_with_multiple_qualities)} tasks with multiple quality labels"
        )

        # Sample N trajectories per quality label for each task
        sample_indices = []
        for task, quality_to_trajs in tasks_with_multiple_qualities.items():
            for quality_label, traj_indices in quality_to_trajs.items():
                # Sample up to num_examples_per_quality_pr trajectories for this quality label
                num_to_sample = min(self.num_examples_per_quality_pr, len(traj_indices))
                sampled_traj_indices = random.sample(traj_indices, num_to_sample)
                
                for traj_idx in sampled_traj_indices:
                    sample_indices.append({
                        "traj_idx": traj_idx,
                        "video_path": self.dataset[traj_idx]["frames"],
                        "id": self.dataset[traj_idx]["id"]
                    })

        logger.info(
            f"Sampled {len(sample_indices)} trajectories across {len(tasks_with_multiple_qualities)} tasks"
        )

        return sample_indices

    def _generate_sample_from_indices(self, sample_idx_info: dict) -> ProgressSample:
        """Generate a single progress sample from trajectory index."""
        traj_idx = sample_idx_info["traj_idx"]
        video_path = sample_idx_info["video_path"]

        traj = self.dataset[traj_idx]

        # Initialize variables
        frames = None
        video_embeddings = None
        text_embedding = None

        # Use linspace sampling to get max_frames
        max_frames = self.config.max_frames

        # Load data (embeddings or frames)
        if self.config.load_embeddings and traj.get("embeddings_path"):
            embeddings = load_embeddings_from_path(traj["embeddings_path"])
            video_embeddings = embeddings["video_embeddings"]
            text_embedding = embeddings["text_embedding"]
            data = video_embeddings
            total_frames = video_embeddings.shape[0] if hasattr(video_embeddings, "shape") else len(video_embeddings)
            use_embeddings = True
        else:
            frames = load_frames_from_npz(traj["frames"])
            data = frames
            total_frames = len(frames)
            use_embeddings = False

        data, frame_indices = linspace_subsample_frames(data, max_frames)
        frames_shape_orig = data.shape

        # Compute progress based on type
        if self.config.progress_pred_type == "absolute_wrt_total_frames":
            progress_abs = [(idx + 1) / total_frames for idx in frame_indices]
        elif self.config.progress_pred_type.startswith("absolute"):
            # absolute_first_frame: use linspace logic
            progress_abs = [idx / (total_frames - 1) for idx in frame_indices]
        else:  # relative_first_frame
            # For relative, we still compute absolute first, then convert
            progress_abs = [idx / (total_frames - 1) for idx in frame_indices]

        if use_embeddings:
            video_embeddings, progress_abs = pad_trajectory_to_max_frames_torch(data, progress_abs, max_frames)
        else:
            frames, progress_abs = pad_trajectory_to_max_frames_np(data, progress_abs, max_frames)

        if self.config.progress_pred_type == "relative_first_frame":
            progress = convert_absolute_to_relative_progress(progress_abs)
        else:
            progress = progress_abs

        metadata = {
            "quality_label": traj["quality_label"],
            "data_source": traj["data_source"],
            "task": traj["task"],
            "id": traj["id"],
            "video_path": video_path,
        }

        # Compute success labels
        success_label = compute_success_labels(
            target_progress=progress,
            data_source=traj["data_source"],
            dataset_success_percent=self.dataset_success_cutoff_map,
            max_success=self.config.max_success,
        )

        # Create trajectory for the progress sample
        trajectory = Trajectory(
            frames=frames,
            frames_shape=frames_shape_orig,
            video_embeddings=video_embeddings,
            text_embedding=text_embedding,
            id=traj["id"],
            task=traj["task"],
            lang_vector=np.array(traj["lang_vector"]),
            data_source=traj["data_source"],
            quality_label=traj["quality_label"],
            is_robot=traj["is_robot"],
            target_progress=progress,
            partial_success=traj.get("partial_success"),
            success_label=success_label,
            metadata=metadata,
        )

        # Create progress sample
        sample = ProgressSample(trajectory=trajectory)

        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])

