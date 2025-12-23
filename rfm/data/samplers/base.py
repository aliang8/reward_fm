#!/usr/bin/env python3
from typing import Optional, Dict, Any, List, Set, Tuple

import numpy as np
import random
import torch
from datasets import Dataset

from rfm.configs.experiment_configs import DataConfig
from rfm.data.datasets.helpers import (
    load_frames_from_npz,
    subsample_segment_frames,
    compute_progress_from_segment,
    pad_trajectory_to_max_frames_torch,
    pad_trajectory_to_max_frames_np,
    subsample_pairs_and_progress,
    compute_success_labels,
    convert_continuous_to_discrete_bins,
    convert_continuous_to_discrete_bin,
    create_trajectory_from_dict,
    load_embeddings_from_path,
    create_rewind_trajectory,
)
from rfm.data.dataset_types import Trajectory
from rfm.utils.logger import get_logger

logger = get_logger()


class RFMBaseSampler:
    """Base sampler class that provides trajectory retrieval functions for generating samples."""

    def __init__(
        self,
        config: DataConfig,
        dataset: Dataset,
        combined_indices: Dict[str, Any],
        dataset_success_cutoff_map: Optional[Dict[str, float]] = None,
        verbose: bool = True,
    ):
        """Initialize sampler with dataset and indices.

        Args:
            config: Configuration object
            dataset: The loaded dataset
            combined_indices: Dictionary of combined indices from dataset loading
            dataset_success_cutoff_map: Dictionary mapping dataset names to success cutoff percentages
            verbose: Verbose flag
        """
        self.config = config
        self.dataset = dataset
        self.verbose = verbose
        self.dataset_success_cutoff_map = dataset_success_cutoff_map or {}

        self._cached_ids = self.dataset["id"]
        self._cached_is_robot = self.dataset["is_robot"]

        # Build indices from combined_indices
        self._build_indices(combined_indices)

    def _build_indices(self, combined_indices):
        """Build all index mappings from combined_indices.

        Args:
            combined_indices: Dictionary of combined indices from dataset loading
        """
        # Initialize index mappings from the loaded indices
        self.robot_trajectories = combined_indices["robot_trajectories"]
        self.human_trajectories = combined_indices["human_trajectories"]
        self.optimal_by_task = combined_indices["optimal_by_task"]
        self.suboptimal_by_task = combined_indices["suboptimal_by_task"]
        self.quality_indices = combined_indices["quality_indices"]
        self.task_indices = combined_indices["task_indices"]
        self.source_indices = combined_indices["source_indices"]
        self.partial_success_indices = combined_indices["partial_success_indices"]
        self.paired_human_robot_by_task = combined_indices["paired_human_robot_by_task"]
        self.tasks_with_multiple_quality_labels = combined_indices["tasks_with_multiple_quality_labels"]

        # Build mapping from data source -> available task instructions
        self._build_tasks_by_data_source()

    def _build_tasks_by_data_source(self):
        """Cache mapping from data_source to available task instructions."""
        self.tasks_by_data_source: Dict[str, List[str]] = {}

        all_tasks = self.dataset["task"]
        all_sources = self.dataset["data_source"]

        source_to_tasks: Dict[str, Set[str]] = {}
        for task, source in zip(all_tasks, all_sources):
            if task is None or source is None:
                continue
            if source not in source_to_tasks:
                source_to_tasks[source] = set()
            source_to_tasks[source].add(task)

        self.tasks_by_data_source = {source: list(tasks) for source, tasks in source_to_tasks.items()}

    def _post_process_trajectory(self, trajectory: Trajectory, skip_padding: bool = False) -> Trajectory:
        """Post-process a trajectory: pad frames/embeddings and progress, compute success labels, convert partial_success.

        Args:
            trajectory: The trajectory to post-process
            skip_padding: Whether to skip padding (e.g., for pairwise progress)

        Returns:
            A new Trajectory instance with post-processed fields
        """
        frames = trajectory.frames
        video_embeddings = trajectory.video_embeddings
        target_progress = trajectory.target_progress or []

        # Pad trajectory and progress if not skipped
        if not skip_padding and target_progress:
            if self.config.load_embeddings and video_embeddings is not None:
                video_embeddings, target_progress = pad_trajectory_to_max_frames_torch(
                    video_embeddings, target_progress, self.config.max_frames
                )
            elif frames is not None:
                frames, target_progress = pad_trajectory_to_max_frames_np(
                    frames, target_progress, self.config.max_frames
                )

        # Compute success labels
        success_label = compute_success_labels(
            target_progress=target_progress,
            data_source=trajectory.data_source,
            dataset_success_percent=self.dataset_success_cutoff_map,
            max_success=self.config.max_success,
        )

        # Convert partial_success to discrete bins if in discrete mode
        partial_success = trajectory.partial_success
        if partial_success is not None and self.config.progress_loss_type.lower() == "discrete":
            num_bins = self.config.progress_discrete_bins
            partial_success = convert_continuous_to_discrete_bin(partial_success, num_bins)

        # Create new Trajectory with updated fields using model_copy (cleaner than manual construction)
        update_dict = {
            "frames": frames,
            "video_embeddings": video_embeddings,
            "target_progress": target_progress,
            "success_label": success_label,
            "partial_success": partial_success,
        }
        return trajectory.model_copy(update=update_dict)

    def _generate_sample(self, item):
        """Generate a sample from an item.

        This method should be overridden by subclasses to implement their specific
        sample generation logic.

        Args:
            item: An item from the dataset (typically a trajectory dict)

        Returns:
            A sample object (e.g., PreferenceSample, SimilaritySample, ProgressSample)
        """
        raise NotImplementedError("Subclasses must implement _generate_sample")

    def _get_same_task_optimal(self, ref_traj: dict) -> dict | None:
        """Get optimal trajectory from same task (different from ref).

        Args:
            ref_traj: Reference trajectory

        Returns:
            Same task optimal trajectory dict or None if not available
        """
        task_name = ref_traj["task"]
        same_task_optimal_indices = self.optimal_by_task.get(task_name, [])
        if not same_task_optimal_indices:
            logger.trace(f"[BASE SAMPLER] _get_same_task_optimal: No optimal indices for task '{task_name}'")
            return None

        # Use cached IDs to check without loading full trajectories
        chosen_id = ref_traj["id"]
        random_idx = random.choice(same_task_optimal_indices)

        # Retry if the selected trajectory has the same ID as ref
        max_retries = min(10, len(same_task_optimal_indices))
        retries = 0
        while self._cached_ids[random_idx] == chosen_id and retries < max_retries:
            random_idx = random.choice(same_task_optimal_indices)
            retries += 1

        # If still matches after retries, fall back to filtering
        if self._cached_ids[random_idx] == chosen_id:
            filtered_indices = [idx for idx in same_task_optimal_indices if self._cached_ids[idx] != chosen_id]
            if filtered_indices:
                random_idx = random.choice(filtered_indices)
            else:
                # No other trajectories available
                logger.trace(
                    f"[BASE SAMPLER] _get_same_task_optimal: All trajectories have same ID '{chosen_id}' for task '{task_name}'"
                )
                return None

        result = self.dataset[random_idx]
        logger.trace(
            f"[BASE SAMPLER] _get_same_task_optimal: Found trajectory {result.get('id', 'unknown')} for task '{task_name}'"
        )
        return result

    def _get_same_task_suboptimal(self, ref_traj: dict) -> dict | None:
        """Get suboptimal trajectory from same task.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Suboptimal trajectory dict or None if not available
        """
        task_name = ref_traj["task"]
        same_task_suboptimal_indices = self.suboptimal_by_task.get(task_name, [])
        if not same_task_suboptimal_indices:
            logger.trace(f"[BASE SAMPLER] _get_same_task_suboptimal: No suboptimal indices for task '{task_name}'")
            return None

        # Use cached IDs to check without loading full trajectories
        chosen_id = ref_traj["id"]
        random_idx = random.choice(same_task_suboptimal_indices)

        # Retry if the selected trajectory has the same ID as ref
        max_retries = min(10, len(same_task_suboptimal_indices))
        retries = 0
        while self._cached_ids[random_idx] == chosen_id and retries < max_retries:
            random_idx = random.choice(same_task_suboptimal_indices)
            retries += 1

        # If still matches after retries, fall back to filtering
        if self._cached_ids[random_idx] == chosen_id:
            filtered_indices = [idx for idx in same_task_suboptimal_indices if self._cached_ids[idx] != chosen_id]
            if filtered_indices:
                random_idx = random.choice(filtered_indices)
            else:
                # No other trajectories available
                logger.trace(
                    f"[BASE SAMPLER] _get_same_task_suboptimal: All trajectories have same ID '{chosen_id}' for task '{task_name}'"
                )
                return None

        result = self.dataset[random_idx]
        logger.trace(
            f"[BASE SAMPLER] _get_same_task_suboptimal: Found trajectory {result.get('id', 'unknown')} for task '{task_name}'"
        )
        return result

    def _get_different_video_traj(self, ref_traj: dict) -> dict | None:
        """Get trajectory from different task.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Different task trajectory dict or None if not available
        """
        other_tasks = [task for task in self.optimal_by_task.keys() if task != ref_traj["task"]]
        if not other_tasks:
            logger.trace(
                f"[BASE SAMPLER] _get_different_video_traj: No other tasks available (ref task: '{ref_traj['task']}')"
            )
            return None

        other_task = random.choice(other_tasks)
        other_task_indices = self.optimal_by_task[other_task]
        if not other_task_indices:
            logger.trace(f"[BASE SAMPLER] _get_different_video_traj: Task '{other_task}' has no optimal indices")
            return None

        other_idx = random.choice(other_task_indices)
        result = self.dataset[other_idx]
        logger.trace(
            f"[BASE SAMPLER] _get_different_video_traj: Found trajectory {result.get('id', 'unknown')} from task '{other_task}'"
        )
        return result

    def _get_different_task_instruction(self, ref_traj: dict) -> dict | None:
        """Get the same trajectory but with a different task instruction.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Trajectory dict with different task instruction or None if not available
        """
        same_source_prob = self.config.task_instruction_same_source_prob
        data_source = ref_traj.get("data_source")
        candidate_tasks = []

        if data_source and data_source in self.tasks_by_data_source and random.random() < same_source_prob:
            candidate_tasks = [task for task in self.tasks_by_data_source[data_source] if task != ref_traj["task"]]

        if not candidate_tasks:
            candidate_tasks = [task for task in self.optimal_by_task.keys() if task != ref_traj["task"]]

        if not candidate_tasks:
            logger.trace(
                f"[BASE SAMPLER] _get_different_task_instruction: No candidate tasks available (ref task: '{ref_traj['task']}')"
            )
            return None

        other_task = random.choice(candidate_tasks)

        # Get embeddings_path and lang_vector from a random trajectory with the other_task
        other_task_indices = self.optimal_by_task.get(other_task, [])
        if not other_task_indices:
            logger.trace(f"[BASE SAMPLER] _get_different_task_instruction: Task '{other_task}' has no optimal indices")
            return None

        other_task_idx = random.choice(other_task_indices)
        other_task_traj = self.dataset[other_task_idx]

        # Create a copy of the trajectory with the task changed
        # Use embeddings_path and lang_vector from the other_task trajectory
        new_traj = ref_traj.copy()
        new_traj["task"] = other_task
        # Get embeddings_path and lang_vector from a random trajectory with the other_task
        if "embeddings_path" in other_task_traj:
            new_traj["embeddings_path"] = other_task_traj["embeddings_path"]
        if "lang_vector" in other_task_traj:
            new_traj["lang_vector"] = other_task_traj["lang_vector"]
        return new_traj

    def _get_paired_human_robot_traj(self, ref_traj: dict) -> dict | None:
        """Get paired human/robot trajectory for the same task.

        Given a reference trajectory, if it's a robot trajectory, returns a human trajectory
        from the same task. If it's a human trajectory, returns a robot trajectory from the
        same task.

        Args:
            ref_traj: Reference trajectory (can be robot or human)

        Returns:
            Paired trajectory dict (opposite type) or None if not available
        """
        task = ref_traj["task"]
        is_robot = ref_traj.get("is_robot", True)

        if task not in self.paired_human_robot_by_task:
            logger.trace(
                f"[BASE SAMPLER] _get_paired_human_robot_traj: Task '{task}' not in paired_human_robot_by_task"
            )
            return None

        task_pairs = self.paired_human_robot_by_task[task]

        # Get opposite type
        opposite_key = "human" if is_robot else "robot"
        opposite_indices = task_pairs.get(opposite_key, [])

        if not opposite_indices:
            logger.trace(f"[BASE SAMPLER] _get_paired_human_robot_traj: No {opposite_key} indices for task '{task}'")
            return None

        # Sample a paired trajectory and verify it's different from reference
        chosen_id = ref_traj["id"]
        available_indices = opposite_indices.copy()
        paired_traj = None

        # Add retry limit to prevent infinite loops
        max_retries = min(len(available_indices), 10)
        retries = 0

        logger.trace(
            f"[BASE SAMPLER] _get_paired_human_robot_traj: Looking for {opposite_key} trajectory (chosen_id: {chosen_id}, available: {len(available_indices)})"
        )

        while (paired_traj is None or paired_traj.get("id") == chosen_id) and retries < max_retries:
            retries += 1

            if not available_indices:
                logger.trace(
                    f"[BASE SAMPLER] _get_paired_human_robot_traj: No more available indices after {retries} retries"
                )
                return None

            paired_idx = random.choice(available_indices)
            paired_traj = self.dataset[paired_idx]

            # If it matches, remove this index and try again
            if paired_traj.get("id") == chosen_id:
                available_indices = [idx for idx in available_indices if idx != paired_idx]
                paired_traj = None
                continue

        # If we exhausted retries without finding a valid trajectory, return None
        if paired_traj is None or paired_traj.get("id") == chosen_id:
            logger.trace(
                f"[BASE SAMPLER] _get_paired_human_robot_traj: Failed to find valid paired trajectory after {max_retries} retries"
            )
            return None

        logger.trace(
            f"[BASE SAMPLER] _get_paired_human_robot_traj: Found paired trajectory {paired_traj.get('id', 'unknown')} on retry {retries}"
        )
        return paired_traj

    def _get_different_partial_success_traj(self, ref_traj: dict) -> dict | None:
        """Get trajectory from same task with different partial_success (for RoboArena).

        Finds trajectories with either higher or lower partial_success than the reference,
        using absolute difference for threshold checking.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Trajectory dict with different partial_success from same task or None if not available
        """
        task_name = ref_traj["task"]
        ref_partial_success = ref_traj.get("partial_success")

        # Check if partial_success is available
        if ref_partial_success is None:
            logger.trace(
                f"[BASE SAMPLER] _get_different_partial_success_traj: No partial_success for trajectory {ref_traj.get('id', 'unknown')}"
            )
            return None

        # Get minimum threshold from config (default to 0.0 if not set)
        min_threshold = self.config.roboarena_partial_success_threshold

        # Get all trajectories from the same task
        same_task_indices = self.task_indices.get(task_name, [])
        if not same_task_indices:
            logger.trace(
                f"[BASE SAMPLER] _get_different_partial_success_traj: No trajectories found for task '{task_name}'"
            )
            return None

        # Filter to trajectories with different partial_success that meet the threshold requirement
        # Uses absolute difference to allow both higher and lower partial_success
        chosen_id = ref_traj["id"]
        candidate_indices = []

        for idx in same_task_indices:
            # Skip if same trajectory
            if self._cached_ids[idx] == chosen_id:
                continue

            # Get partial_success for this trajectory
            traj_dict = self.dataset[idx]
            traj_partial_success = traj_dict.get("partial_success", None)

            if traj_partial_success is None:
                logger.trace(
                    f"[BASE SAMPLER] _get_different_partial_success_traj: No partial_success for trajectory {traj_dict.get('id', 'unknown')}, task '{task_name}'"
                )
                continue

            # Include if partial_success differs from reference by at least the threshold (using abs)
            partial_success_diff = abs(ref_partial_success - traj_partial_success)
            if partial_success_diff >= min_threshold:
                candidate_indices.append(idx)

        if not candidate_indices:
            logger.trace(
                f"[BASE SAMPLER] _get_different_partial_success_traj: No trajectories with different partial_success (threshold: {min_threshold}) for task '{task_name}' (ref: {ref_partial_success})"
            )
            return None

        # Randomly select from candidates
        selected_idx = random.choice(candidate_indices)
        result = self.dataset[selected_idx]
        result_partial_success = result.get("partial_success")
        direction = "higher" if result_partial_success > ref_partial_success else "lower"
        logger.trace(
            f"[BASE SAMPLER] _get_different_partial_success_traj: Found trajectory {result.get('id', 'unknown')} with partial_success {result_partial_success} ({direction} than {ref_partial_success}, abs diff: {abs(ref_partial_success - result_partial_success):.3f}, threshold: {min_threshold})"
        )
        return result

    def _get_rewound_traj(self, ref_traj: dict) -> Trajectory:
        """Get rewound trajectory from reference trajectory.

        Args:
            ref_traj: Reference trajectory

        Returns:
            Rewound trajectory as Trajectory object (already processed)
        """
        traj_id = ref_traj.get("id", "unknown")
        logger.trace(f"[BASE SAMPLER] _get_rewound_traj: Creating rewound trajectory for ID: {traj_id}")

        ds_key = ref_traj["data_source"]
        success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)
        result = create_rewind_trajectory(
            ref_traj,
            max_frames=self.config.max_frames,
            use_embeddings=self.config.load_embeddings,
            progress_pred_type=getattr(self.config, "progress_pred_type", "absolute"),
            success_cutoff=success_cutoff,
            dataset_success_percent=self.dataset_success_cutoff_map,
            max_success=self.config.max_success,
        )
        logger.trace(f"[BASE SAMPLER] _get_rewound_traj: Successfully created rewound trajectory for ID: {traj_id}")
        return result

    def _get_uniform_sample_indices(self, data, direction: str = "bidirectional") -> Optional[Tuple[int, int]]:
        """Get start and end indices for uniform_sample strategy.

        Samples two random frames from the trajectory based on the specified direction.

        Args:
            data: Trajectory data (frames or embeddings) to sample from
            direction: Sampling direction - "forward" (second frame after first),
                      "reverse" (second frame before first), or "bidirectional" (either direction)

        Returns:
            Tuple of (start_idx, end_idx) for the segment, or None if insufficient frames
        """
        num_frames_total = len(data) if hasattr(data, "__len__") else data.shape[0]

        if num_frames_total < 2:
            logger.trace(f"[BASE SAMPLER] _get_uniform_sample_indices: Not enough frames ({num_frames_total})")
            return None

        # Sample first random frame
        frame1_idx = random.randint(0, num_frames_total - 1)

        # Sample second frame based on direction
        if direction == "forward":
            # Second frame must be after the first
            if frame1_idx == num_frames_total - 1:
                # Can't sample forward from last frame, need to adjust
                frame1_idx = random.randint(0, num_frames_total - 2)
            frame2_idx = random.randint(frame1_idx + 1, num_frames_total - 1)
        elif direction == "reverse":
            # Second frame must be before the first
            if frame1_idx == 0:
                # Can't sample reverse from first frame, need to adjust
                frame1_idx = random.randint(1, num_frames_total - 1)
            frame2_idx = random.randint(0, frame1_idx - 1)
        else:  # bidirectional (default)
            # Randomly choose to sample from before or after
            if frame1_idx == 0:
                # Can only sample after
                frame2_idx = random.randint(1, num_frames_total - 1)
            elif frame1_idx == num_frames_total - 1:
                # Can only sample before
                frame2_idx = random.randint(0, frame1_idx - 1)
            else:
                # Can sample from either side
                if random.random() < 0.5:
                    # Sample from before
                    frame2_idx = random.randint(0, frame1_idx - 1)
                else:
                    # Sample from after
                    frame2_idx = random.randint(frame1_idx + 1, num_frames_total - 1)

        # Ensure start_idx < end_idx (end_idx is exclusive)
        start_idx = min(frame1_idx, frame2_idx)
        end_idx = max(frame1_idx, frame2_idx) + 1

        logger.trace(
            f"[BASE SAMPLER] _get_uniform_sample_indices: Selected segment [{start_idx}, {end_idx}) "
            f"from {num_frames_total} total frames (direction: {direction})"
        )
        return start_idx, end_idx

    def _get_traj_from_data(self, traj: dict | Trajectory, subsample_strategy: str | None = None) -> Trajectory:
        """Load, subsample, and optionally pad trajectory data and create a Trajectory object.

        When pairwise_progress is enabled, padding is skipped as it's not needed.

        Args:
            traj: Trajectory dict or Trajectory object
            subsample_strategy: Optional strategy for subsampling ("successful", "subsequence", or None for default)

        Returns:
            Trajectory object with loaded and subsampled data (padded only if not using pairwise_progress)
        """
        if isinstance(traj, Trajectory):
            return traj

        frames = None
        video_embeddings = None
        text_embedding = None

        # Use pairwise subsampling if pairwise_progress is enabled
        use_pairwise = self.config.pairwise_progress

        if self.config.load_embeddings and traj.get("embeddings_path"):
            embeddings = load_embeddings_from_path(traj["embeddings_path"])
            video_embeddings = embeddings["video_embeddings"]
            text_embedding = embeddings["text_embedding"]
            data = video_embeddings
        else:
            if isinstance(traj["frames"], str):
                frames = load_frames_from_npz(traj["frames"])
            else:
                frames = traj["frames"]
            data = frames

        # Get total frames for progress computation
        if hasattr(data, "shape"):
            num_frames_total = data.shape[0]
        else:
            num_frames_total = len(data)

        if use_pairwise:
            # Use pairwise subsampling for pairwise progress
            subsampled, progress, metadata = subsample_pairs_and_progress(
                data,
                self.config.max_frames,
                progress_pred_type=self.config.progress_pred_type,
            )
            frames_shape = subsampled.shape
            if self.config.load_embeddings:
                video_embeddings = subsampled
            else:
                frames = subsampled
        else:
            ds_key = traj["data_source"]
            success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

            # Handle uniform_sample strategy: pick two random frames as segment bounds
            start_idx = None
            end_idx = None
            if subsample_strategy == "uniform_sample":
                uniform_indices = self._get_uniform_sample_indices(data, direction="bidirectional")
                if uniform_indices is None:
                    # Not enough frames for uniform_sample, fall back to subsequence
                    subsample_strategy = "subsequence"
                else:
                    start_idx, end_idx = uniform_indices
            elif subsample_strategy == "uniform_sample_forward":
                uniform_indices = self._get_uniform_sample_indices(data, direction="forward")
                if uniform_indices is None:
                    # Not enough frames for uniform_sample, fall back to subsequence
                    subsample_strategy = "subsequence"
                else:
                    start_idx, end_idx = uniform_indices
            elif subsample_strategy == "uniform_sample_reverse":
                uniform_indices = self._get_uniform_sample_indices(data, direction="reverse")
                if uniform_indices is None:
                    # Not enough frames for uniform_sample, fall back to subsequence
                    subsample_strategy = "subsequence"
                else:
                    start_idx, end_idx = uniform_indices

            logger.trace(
                f"[BASE SAMPLER] _get_traj_from_data: Subsampling trajectory with strategy: {subsample_strategy}, start_idx: {start_idx}, end_idx: {end_idx}"
            )
            perc_end = success_cutoff if subsample_strategy == "successful" else 2.0 / 3.0
            subsampled, start_idx, end_idx, indices = subsample_segment_frames(
                data, self.config.max_frames, method="linspace", perc_end=perc_end, start_idx=start_idx, end_idx=end_idx
            )
            frames_shape = subsampled.shape
            progress = compute_progress_from_segment(
                num_frames_total=num_frames_total,
                start_idx=start_idx,
                end_idx=end_idx,
                frame_indices=indices,
                progress_pred_type=self.config.progress_pred_type,
                success_cutoff=success_cutoff,
            )

            metadata = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "subsampled_indices": indices,
                "strategy": subsample_strategy or "subsequence",
            }
            if self.config.load_embeddings:
                video_embeddings = subsampled
            else:
                frames = subsampled

        if subsample_strategy and "reverse" in subsample_strategy:
            # Reverse frames/embeddings along time dimension
            if frames is not None:
                if isinstance(frames, np.ndarray):
                    frames = np.flip(frames, axis=0)
                elif isinstance(frames, torch.Tensor):
                    frames = torch.flip(frames, dims=[0])
            if video_embeddings is not None:
                if isinstance(video_embeddings, np.ndarray):
                    video_embeddings = np.flip(video_embeddings, axis=0)
                elif isinstance(video_embeddings, torch.Tensor):
                    video_embeddings = torch.flip(video_embeddings, dims=[0])
            # Reverse progress
            progress = list(reversed(progress))

        if self.config.progress_loss_type.lower() == "discrete":
            num_bins = self.config.progress_discrete_bins
            # Convert continuous progress [0, 1] to discrete bins [0, num_bins-1]
            progress = convert_continuous_to_discrete_bins(progress, num_bins)

        skip_padding = use_pairwise
        trajectory = create_trajectory_from_dict(
            traj,
            overrides={
                "frames": frames,
                "frames_shape": frames_shape,
                "video_embeddings": video_embeddings,
                "text_embedding": text_embedding,
                "target_progress": progress,
                "metadata": metadata,
            },
        )
        trajectory = self._post_process_trajectory(trajectory, skip_padding=skip_padding)

        # Reverse success labels if we reversed progress (use model_copy for cleaner update)
        if subsample_strategy and "reverse" in subsample_strategy and trajectory.success_label is not None:
            reversed_success_label = list(reversed(trajectory.success_label))
            trajectory = trajectory.model_copy(update={"success_label": reversed_success_label})

        return trajectory
