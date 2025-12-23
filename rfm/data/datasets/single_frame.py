from typing import Any, Dict, Optional

import random
import torch
from datasets import Dataset

from rfm.configs.experiment_configs import DataConfig
from rfm.data.datasets.base import BaseDataset
from rfm.data.dataset_types import ProgressSample, PreferenceSample, Trajectory
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.datasets.helpers import (
    load_frames_from_npz,
    load_embeddings_from_path,
    compute_success_labels,
    DataGenStrat,
)
from rfm.utils.logger import get_logger, rank_0_info

logger = get_logger()


class SingleFrameProgressSampler(RFMBaseSampler):
    """Sampler that samples a single random frame from a trajectory."""

    def __init__(
        self,
        config: DataConfig,
        dataset: Dataset,
        combined_indices: Dict[str, Any],
        dataset_success_cutoff_map: Optional[Dict[str, float]] = None,
        is_evaluation: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(config, dataset, combined_indices, dataset_success_cutoff_map, verbose=verbose)

    def _generate_sample(self, item: dict):
        """Generate a single-frame progress sample from a trajectory."""
        return self._create_single_frame_sample(item)

    def _create_single_frame_sample(self, traj: dict):
        """Create a progress sample with a single random frame from the trajectory."""
        # Load all frames from the trajectory (handle both string paths and preloaded frames)
        if isinstance(traj["frames"], str):
            frames = load_frames_from_npz(traj["frames"])
        else:
            frames = traj["frames"]
        num_frames = frames.shape[0]

        if num_frames == 0:
            logger.trace(f"[SingleFrameProgressSampler] No frames in trajectory {traj.get('id', 'unknown')}")
            return None

        # Randomly select a single frame index
        frame_idx = random.randint(0, num_frames - 1)

        # Get the progress value at this frame
        target_progress_list = traj.get("target_progress", [])
        if target_progress_list:
            # Use the progress value at the selected frame
            if frame_idx < len(target_progress_list):
                progress_value = target_progress_list[frame_idx]
            else:
                # If frame index is out of range, use the last progress value
                progress_value = target_progress_list[-1]
        else:
            # Default progress if not available
            progress_value = float(frame_idx) / max(num_frames - 1, 1)

        # Extract single frame (keep the time dimension with size 1)
        single_frame = frames[frame_idx : frame_idx + 1]  # (1, H, W, C)
        frames_shape = single_frame.shape  # Store shape for collator

        # Load embeddings if available
        video_embeddings = None
        text_embedding = None
        if self.config.load_embeddings and traj.get("embeddings_path"):
            embeddings = load_embeddings_from_path(traj["embeddings_path"])
            video_embeddings = embeddings.get("video_embeddings")
            text_embedding = embeddings.get("text_embedding")
            if video_embeddings is not None:
                # Extract embedding for the selected frame
                if frame_idx < len(video_embeddings):
                    video_embeddings = video_embeddings[frame_idx : frame_idx + 1]  # Keep time dimension

        # Compute success labels for the single frame
        success_label = compute_success_labels(
            target_progress=[progress_value],
            data_source=traj.get("data_source"),
            dataset_success_percent=self.dataset_success_cutoff_map,
            max_success=self.config.max_success,
        )

        # Create trajectory object with single frame
        progress_traj = Trajectory(
            id=traj["id"],
            task=traj["task"],
            frames=single_frame,
            frames_shape=frames_shape,
            target_progress=[progress_value],  # Single progress value
            metadata=traj.get("metadata", {}),
            video_embeddings=video_embeddings,
            text_embedding=text_embedding,
            lang_vector=traj.get("lang_vector"),
            data_source=traj.get("data_source"),
            quality_label=traj.get("quality_label"),
            success_label=success_label,
        )

        # Create progress sample
        sample = ProgressSample(
            trajectory=progress_traj,
            sample_type="progress",
            data_gen_strategy="single_frame",  # No data strategy for single frame
        )
        sample.resample_attempts = 1
        return sample


class SingleFrameDataset(BaseDataset):
    """Dataset that samples single frames from trajectories for progress and preference prediction."""

    def __init__(self, config, is_evaluation=False, max_samples=None, **kwargs):
        super().__init__(config, is_evaluation, **kwargs)

        # Override max_frames to 1 for single frame sampling
        original_max_frames = config.max_frames
        config.max_frames = 1

        self.progress_sampler = None
        self.preference_sampler = None

        # Initialize samplers based on sample_type_ratio
        if config.sample_type_ratio[1] > 0:  # Progress
            self.progress_sampler = SingleFrameProgressSampler(
                config,
                self.dataset,
                self._combined_indices,
                self.dataset_success_cutoff_map,
                is_evaluation,
                verbose=False,
                **kwargs,
            )

        if config.sample_type_ratio[0] > 0:  # Preference
            self.preference_sampler = SingleFramePreferenceSampler(
                config,
                self.dataset,
                self._combined_indices,
                self.dataset_success_cutoff_map,
                is_evaluation,
                verbose=False,
                **kwargs,
            )

        # Restore original max_frames (in case it's used elsewhere)
        config.max_frames = original_max_frames
        self.max_samples = max_samples
        self.data_len = len(self.dataset)
        self.sample_type_ratio = config.sample_type_ratio

        rank_0_info(f"SingleFrameDataset initialized with {self.data_len} trajectories")

    def __len__(self):
        if self.max_samples is None:
            return self.data_len
        else:
            return self.max_samples

    def _generate_sample_from_item(self, item):
        """Generate a sample (progress or preference) from an item."""
        traj_id = item.get("id", "unknown")
        quality_label = item.get("quality_label", "successful")

        # Available samplers with their probabilities
        samplers = [
            ("pref", self.sample_type_ratio[0], self.preference_sampler),
            ("progress", self.sample_type_ratio[1], self.progress_sampler),
        ]

        # Remove samplers with zero probability or None samplers
        available_samplers = [
            (name, prob, sampler) for name, prob, sampler in samplers if prob > 0 and sampler is not None
        ]

        if not available_samplers:
            raise ValueError("No samplers available")

        # Select sampler based on probabilities
        total_prob = sum(prob for _, prob, _ in available_samplers)
        if total_prob == 0:
            raise ValueError("No samplers with positive probability")

        normalized_samplers = [(name, prob / total_prob, sampler) for name, prob, sampler in available_samplers]

        prob = random.random()
        cumulative_prob = 0.0
        selected_sampler = None
        selected_name = None

        for name, normalized_prob, sampler in normalized_samplers:
            cumulative_prob += normalized_prob
            if prob <= cumulative_prob:
                selected_sampler = sampler
                selected_name = name
                break

        # Fallback: select first sampler
        if selected_sampler is None:
            selected_name, _, selected_sampler = available_samplers[0]

        # Generate sample
        sample = selected_sampler._generate_sample(item)
        if sample is None:
            raise ValueError(f"Failed to generate {selected_name} sample for trajectory {traj_id}")

        return sample

    def __getitem__(self, idx):
        """Create a single-frame sample (progress or preference) from the dataset."""
        idx = idx % self.data_len
        logger.trace(f"[SingleFrameDataset] __getitem__: Starting for idx={idx}")

        # Get the item from the filtered dataset
        item = self.dataset[idx]
        traj_id = item.get("id", "unknown")
        logger.trace(f"[SingleFrameDataset] __getitem__: Got item with ID={traj_id}")

        # Generate sample (progress or preference)
        sample = self._generate_sample_from_item(item)

        logger.trace(f"[SingleFrameDataset] __getitem__: Successfully generated sample for idx={idx}, ID={traj_id}")
        return sample


class SingleFramePreferenceSampler(RFMBaseSampler):
    """Sampler that creates preference samples with single frames from trajectories."""

    def __init__(
        self,
        config: DataConfig,
        dataset: Dataset,
        combined_indices: Dict[str, Any],
        dataset_success_cutoff_map: Optional[Dict[str, float]] = None,
        is_evaluation: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(config, dataset, combined_indices, dataset_success_cutoff_map, verbose=verbose)

    def _generate_sample(self, item: dict):
        """Generate a single-frame preference sample from a trajectory."""
        return self._create_single_frame_preference_sample(item)

    def _create_single_frame_trajectory(self, traj: dict, frame_idx: int):
        """Helper to create a single-frame trajectory from a trajectory dict."""
        # Load all frames from the trajectory
        if isinstance(traj["frames"], str):
            frames = load_frames_from_npz(traj["frames"])
        else:
            frames = traj["frames"]

        num_frames = frames.shape[0]
        if num_frames == 0:
            return None

        # Ensure frame_idx is valid
        frame_idx = min(frame_idx, num_frames - 1)

        # Get progress value at this frame
        target_progress_list = traj.get("target_progress", [])
        if target_progress_list:
            if frame_idx < len(target_progress_list):
                progress_value = target_progress_list[frame_idx]
            else:
                progress_value = target_progress_list[-1]
        else:
            progress_value = float(frame_idx) / max(num_frames - 1, 1)

        # Extract single frame
        single_frame = frames[frame_idx : frame_idx + 1]  # (1, H, W, C)
        frames_shape = single_frame.shape

        # Load embeddings if available
        video_embeddings = None
        text_embedding = None
        if self.config.load_embeddings and traj.get("embeddings_path"):
            embeddings = load_embeddings_from_path(traj["embeddings_path"])
            video_embeddings = embeddings.get("video_embeddings")
            text_embedding = embeddings.get("text_embedding")
            if video_embeddings is not None:
                if frame_idx < len(video_embeddings):
                    video_embeddings = video_embeddings[frame_idx : frame_idx + 1]

        # Compute success labels
        success_label = compute_success_labels(
            target_progress=[progress_value],
            data_source=traj.get("data_source"),
            dataset_success_percent=self.dataset_success_cutoff_map,
            max_success=self.config.max_success,
        )

        return Trajectory(
            id=traj["id"],
            task=traj["task"],
            frames=single_frame,
            frames_shape=frames_shape,
            target_progress=[progress_value],
            metadata=traj.get("metadata", {}),
            video_embeddings=video_embeddings,
            text_embedding=text_embedding,
            lang_vector=traj.get("lang_vector"),
            data_source=traj.get("data_source"),
            quality_label=traj.get("quality_label"),
            success_label=success_label,
        )

    def _create_single_frame_preference_sample(self, traj: dict):
        """Create a preference sample with single frames using two strategies:
        1. Different task: one frame from current trajectory, one from different task
        2. Two frames: one from first half, one from second half (successful trajectories only)
        """
        quality_label = traj.get("quality_label", "successful")

        # Strategy 1: Two frames from same trajectory (only for successful)
        # Strategy 2: Different task

        # For successful trajectories, randomly choose between two strategies
        if quality_label == "successful":
            # Randomly choose strategy: 0 = two frames, 1 = different task
            strategy = random.randint(0, 1)
        else:
            # For non-successful, only use different task
            strategy = 1

        if strategy == 0:
            # Two frames strategy: one from first half, one from second half
            return self._create_two_frames_preference(traj)
        else:
            # Different task strategy
            return self._create_different_task_preference(traj)

    def _create_two_frames_preference(self, traj: dict):
        """Create preference sample with two frames from same trajectory (first half vs second half)."""
        # Load frames
        if isinstance(traj["frames"], str):
            frames = load_frames_from_npz(traj["frames"])
        else:
            frames = traj["frames"]

        num_frames = frames.shape[0]
        if num_frames < 2:
            logger.trace(f"[SingleFramePreferenceSampler] Not enough frames for two-frames strategy: {num_frames}")
            return None

        # Sample one frame from first half, one from second half
        mid_point = num_frames // 2
        first_half_idx = random.randint(0, max(mid_point - 1, 0))
        second_half_idx = random.randint(mid_point, num_frames - 1)

        # Create trajectories for chosen (second half, higher progress) and rejected (first half, lower progress)
        chosen_traj = self._create_single_frame_trajectory(traj, second_half_idx)
        rejected_traj = self._create_single_frame_trajectory(traj, first_half_idx)

        if chosen_traj is None or rejected_traj is None:
            return None

        # Create preference sample
        sample = PreferenceSample(
            chosen_trajectory=chosen_traj,
            rejected_trajectory=rejected_traj,
            sample_type="preference",
            data_gen_strategy="single_frame_two_frames",
        )
        sample.resample_attempts = 1
        return sample

    def _create_different_task_preference(self, traj: dict):
        """Create preference sample with frames from different tasks."""
        current_task = traj["task"]

        # Get a different task trajectory
        other_tasks = [task for task in self.optimal_by_task.keys() if task != current_task]
        if not other_tasks:
            logger.trace(f"[SingleFramePreferenceSampler] No other tasks available for different task strategy")
            return None

        other_task = random.choice(other_tasks)
        other_task_indices = self.optimal_by_task.get(other_task, [])
        if not other_task_indices:
            logger.trace(f"[SingleFramePreferenceSampler] No optimal indices for task '{other_task}'")
            return None

        # Get a trajectory from the different task
        other_idx = random.choice(other_task_indices)
        other_traj = self.dataset[other_idx]

        # Sample one frame from each trajectory
        # Load frames to determine frame count
        if isinstance(traj["frames"], str):
            current_frames = load_frames_from_npz(traj["frames"])
        else:
            current_frames = traj["frames"]

        if isinstance(other_traj["frames"], str):
            other_frames = load_frames_from_npz(other_traj["frames"])
        else:
            other_frames = other_traj["frames"]

        current_num_frames = current_frames.shape[0]
        other_num_frames = other_frames.shape[0]

        if current_num_frames == 0 or other_num_frames == 0:
            logger.trace(f"[SingleFramePreferenceSampler] One of the trajectories has no frames")
            return None

        # Sample random frames from each
        current_frame_idx = random.randint(0, current_num_frames - 1)
        other_frame_idx = random.randint(0, other_num_frames - 1)

        # Current trajectory is chosen (successful), other task is rejected (progress = 0)
        chosen_traj = self._create_single_frame_trajectory(traj, current_frame_idx)
        rejected_traj = self._create_single_frame_trajectory(other_traj, other_frame_idx)

        if chosen_traj is None or rejected_traj is None:
            return None

        # Set rejected trajectory progress to 0 (different task)
        rejected_traj.target_progress = [0.0]

        # Create preference sample
        sample = PreferenceSample(
            chosen_trajectory=chosen_traj,
            rejected_trajectory=rejected_traj,
            sample_type="preference",
            data_gen_strategy="single_frame_different_task",
        )
        sample.resample_attempts = 1
        return sample
