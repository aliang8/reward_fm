from typing import Dict, List, Any, Optional

import numpy as np
import random
from collections import defaultdict
from rfm.data.dataset_types import ProgressSample
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.datasets.helpers import (
    linspace_subsample_frames,
    load_embeddings_from_path,
    load_frames_from_npz,
    convert_absolute_to_relative_progress,
    create_trajectory_from_dict,
)
from rfm.utils.logger import get_logger

logger = get_logger()


class ProgressPolicyRankingSampler(RFMBaseSampler):
    """Dataset that generates progress samples for policy ranking by selecting N trajectories per quality label for tasks with multiple quality labels."""

    def __init__(
        self,
        num_examples_per_quality_pr: int = 5,
        frame_step: int = 1,
        use_frame_steps: bool = True,
        max_tasks: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_examples_per_quality_pr = num_examples_per_quality_pr
        self.frame_step = frame_step
        self.use_frame_steps = True
        self.max_tasks = max_tasks
        logger.info(f"ProgressPolicyRankingSampler initialized with {len(self.robot_trajectories)} trajectories")

        self.sample_indices = self._generate_all_sample_indices()

        logger.info(f"Generated {len(self.sample_indices)} sample indices")

    def _generate_all_sample_indices(self) -> List[Dict[str, Any]]:
        """Generate sample indices by selecting tasks with multiple quality labels/partial_success values and sampling N trajectories per group.

        For non-RoboArena: Groups by task and quality_label.
        For RoboArena: Groups by task and partial_success values.

        If use_frame_steps=True, generates subsequence samples like reward_alignment (0:frame_step, 0:2*frame_step, etc.).
        If use_frame_steps=False, generates one sample per trajectory (whole trajectory).
        """
        # Check if this is RoboArena (has partial_success)
        is_roboarena = False
        if self.robot_trajectories:
            first_traj = self.dataset[self.robot_trajectories[0]]
            is_roboarena = first_traj.get("partial_success") is not None

        # Group trajectories by task and grouping key (quality_label or partial_success)
        task_to_key_to_trajs = defaultdict(lambda: defaultdict(list))

        for traj_idx in self.robot_trajectories:
            traj = self.dataset[traj_idx]
            task = traj["task"]

            if is_roboarena:
                # RoboArena: Use rounded partial_success as key to group similar values
                partial_success = traj.get("partial_success")
                if partial_success is not None:
                    grouping_key = round(float(partial_success), 2)
                    task_to_key_to_trajs[task][grouping_key].append(traj_idx)
            else:
                # Non-RoboArena: Use quality_label
                quality_label = traj["quality_label"]
                task_to_key_to_trajs[task][quality_label].append(traj_idx)

        # Filter to tasks that have multiple grouping values
        tasks_with_multiple_values = {
            task: key_to_trajs for task, key_to_trajs in task_to_key_to_trajs.items() if len(key_to_trajs) > 1
        }

        dataset_type_str = "partial_success values" if is_roboarena else "quality labels"
        logger.info(f"Found {len(tasks_with_multiple_values)} tasks with multiple {dataset_type_str}")

        # Limit number of tasks if max_tasks is specified
        if self.max_tasks is not None and self.max_tasks > 0:
            # Convert to list, shuffle, and take first max_tasks
            tasks_list = list(tasks_with_multiple_values.items())
            random.shuffle(tasks_list)
            tasks_with_multiple_values = dict(tasks_list[: self.max_tasks])
            logger.info(f"Limited to {len(tasks_with_multiple_values)} tasks (max_tasks={self.max_tasks})")

        # Sample trajectories for each task
        sample_indices = []
        for task, key_to_trajs in tasks_with_multiple_values.items():
            if is_roboarena:
                # RoboArena: Sample N trajectories per partial_success value (grouping_key)
                for grouping_key, traj_indices in key_to_trajs.items():
                    if traj_indices:
                        # Sample up to num_examples_per_quality_pr trajectories for this partial_success value
                        num_to_sample = min(self.num_examples_per_quality_pr, len(traj_indices))
                        sampled_traj_indices = random.sample(traj_indices, num_to_sample)

                        for traj_idx in sampled_traj_indices:
                            traj = self.dataset[traj_idx]
                            sample_indices.extend(self._generate_indices_for_trajectory(traj_idx, traj))
            else:
                # Non-RoboArena: Sample N trajectories per quality label
                for grouping_key, traj_indices in key_to_trajs.items():
                    # Sample up to num_examples_per_quality_pr trajectories for this quality label
                    num_to_sample = min(self.num_examples_per_quality_pr, len(traj_indices))
                    sampled_traj_indices = random.sample(traj_indices, num_to_sample)

                    for traj_idx in sampled_traj_indices:
                        traj = self.dataset[traj_idx]
                        sample_indices.extend(self._generate_indices_for_trajectory(traj_idx, traj))

        logger.info(f"Sampled {len(sample_indices)} samples across {len(tasks_with_multiple_values)} tasks")

        return sample_indices

    def _generate_indices_for_trajectory(self, traj_idx: int, traj: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate sample indices for a single trajectory.

        Args:
            traj_idx: Index of the trajectory in the dataset
            traj: Trajectory dictionary

        Returns:
            List of sample index dictionaries
        """
        num_frames = traj["num_frames"]
        indices = []

        if self.use_frame_steps:
            # Generate subsequence indices like reward_alignment: 0:frame_step, 0:2*frame_step, etc.
            for end_idx in range(self.frame_step, num_frames + 1, self.frame_step):
                indices.append({
                    "traj_idx": traj_idx,
                    "end_idx": end_idx,
                    "num_frames": num_frames,
                    "video_path": traj["frames"],
                    "id": traj["id"],
                    "use_frame_steps": True,
                })
        else:
            # Generate one sample per trajectory (whole trajectory)
            indices.append({
                "traj_idx": traj_idx,
                "video_path": traj["frames"],
                "id": traj["id"],
                "use_frame_steps": False,
            })

        return indices

    def _generate_sample_from_indices(self, sample_idx_info: dict) -> ProgressSample:
        """Generate a single progress sample from trajectory index."""
        traj_idx = sample_idx_info["traj_idx"]
        use_frame_steps = sample_idx_info.get("use_frame_steps", True)

        traj = self.dataset[traj_idx]

        # Initialize variables
        frames = None
        video_embeddings = None
        text_embedding = None

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

        if use_frame_steps:
            # Frame steps mode: create subsequence like reward_alignment
            end_idx = sample_idx_info["end_idx"]
            num_frames = sample_idx_info["num_frames"]

            # Create subsequence
            if use_embeddings:
                subsequence_data = data[:end_idx]
                subsequence_data, frame_indices = linspace_subsample_frames(subsequence_data, self.config.max_frames)
                frames_shape_orig = subsequence_data.shape
            else:
                subsequence_frames = data[:end_idx]
                subsequence_frames, frame_indices = linspace_subsample_frames(
                    subsequence_frames, self.config.max_frames
                )
                frames_shape_orig = subsequence_frames.shape

            # Ground truth progress: linear from 0 to 1
            if self.config.progress_pred_type.startswith("absolute"):
                gt_progress = (end_idx - 1) / (num_frames - 1)
            else:  # relative_first_frame
                gt_progress = 1 / num_frames

            # Create progress values for each subsampled frame
            num_subsampled = len(frame_indices)
            if num_subsampled > 1:
                progress_values = [gt_progress * (idx / (num_subsampled - 1)) for idx in range(num_subsampled)]
            else:
                progress_values = [gt_progress]

            metadata = {
                "subsequence_end": end_idx,
                "ground_truth_progress": gt_progress,
                "quality_label": traj["quality_label"],
                "data_source": traj["data_source"],
                "task": traj["task"],
                "id": traj["id"],
                "video_path": sample_idx_info["video_path"],
            }

            trajectory = create_trajectory_from_dict(
                traj,
                overrides={
                    "frames": subsequence_frames if not use_embeddings else None,
                    "frames_shape": frames_shape_orig,
                    "video_embeddings": subsequence_data if use_embeddings else None,
                    "text_embedding": text_embedding,
                    "data_gen_strategy": "policy_ranking",
                    "target_progress": progress_values,
                    "metadata": metadata,
                },
            )
        else:
            # Whole trajectory mode: use linspace sampling
            max_frames = self.config.max_frames
            data, frame_indices = linspace_subsample_frames(data, max_frames)
            frames_shape_orig = data.shape

            # Compute progress based on type
            if self.config.progress_pred_type == "absolute_wrt_total_frames":
                progress_abs = [(idx + 1) / total_frames for idx in frame_indices]
            elif self.config.progress_pred_type.startswith("absolute"):
                progress_abs = [idx / (total_frames - 1) for idx in frame_indices]
            else:  # relative_first_frame
                progress_abs = [idx / (total_frames - 1) for idx in frame_indices]

            if self.config.progress_pred_type == "relative_first_frame":
                progress = convert_absolute_to_relative_progress(progress_abs)
            else:
                progress = progress_abs

            metadata = {
                "quality_label": traj["quality_label"],
                "data_source": traj["data_source"],
                "task": traj["task"],
                "id": traj["id"],
                "video_path": sample_idx_info["video_path"],
            }

            trajectory = create_trajectory_from_dict(
                traj,
                overrides={
                    "frames": data if not use_embeddings else None,
                    "frames_shape": frames_shape_orig,
                    "video_embeddings": data if use_embeddings else None,
                    "text_embedding": text_embedding,
                    "lang_vector": np.array(traj["lang_vector"]),
                    "target_progress": progress,
                    "metadata": metadata,
                },
            )

        trajectory = self._post_process_trajectory(trajectory)
        sample = ProgressSample(trajectory=trajectory)
        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])
