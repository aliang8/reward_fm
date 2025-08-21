#!/usr/bin/env python3
"""
Custom dataset classes for RFM data generation.

This module contains specialized dataset classes that generate different types of samples:
- InfiniteDataGeneratorDataset: Generates preference/similarity samples
- RewoundDataset: Generates preference samples where original is chosen and rewound is rejected
- PairedSuccessFailureDataset: Generates preference samples by pairing successful/failed trajectories
- VideoBinnedDataset: Generates preference samples from video files by binning frames and pairing subsequences
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Iterator, Union
from tqdm import tqdm
from rfm.data.batch_collator import BaseSample, PreferenceSample, SimilaritySample
from rfm.utils.logging import rank_0_print
import cv2
import os
from pathlib import Path


def calulate_target_progress(frames: np.ndarray) -> List[float]:
    num_frames = frames.shape[0]
    return [i / (num_frames - 1) for i in range(num_frames)]


def extract_frames_from_video(video_path: str, fps: int = 1) -> np.ndarray:
    """
    Extract frames from video file at specified FPS.

    Args:
        video_path: Path to the .mp4 file
        fps: Frames per second to extract (default: 1)

    Returns:
        numpy array of shape (num_frames, H, W, C) with extracted frames
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate frame interval for target FPS
    frame_interval = max(1, int(video_fps / fps))

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        frame_count += 1

    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return np.array(frames)


def create_binned_subsequences(frames: np.ndarray, num_bins: int = 10) -> List[Dict]:
    """
    Create binned subsequences from frames.

    Args:
        frames: numpy array of shape (num_frames, H, W, C)
        num_bins: number of bins to create

    Returns:
        List of dictionaries with 'start_frame', 'end_frame', 'frames', 'bin_idx', 'progress'
    """
    num_frames = frames.shape[0]

    if num_frames < num_bins:
        # If fewer frames than bins, create one frame per bin
        bin_size = 1
        num_bins = num_frames
    else:
        bin_size = num_frames // num_bins

    binned_subsequences = []

    for bin_idx in range(num_bins):
        start_frame = bin_idx * bin_size
        end_frame = min(start_frame + bin_size, num_frames)

        # Handle last bin to include remaining frames
        if bin_idx == num_bins - 1:
            end_frame = num_frames

        if start_frame >= end_frame:
            continue

        bin_frames = frames[start_frame:end_frame]
        progress = start_frame / (num_frames - 1) if num_frames > 1 else 0.0

        binned_subsequences.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frames": bin_frames,
                "bin_idx": bin_idx,
                "progress": progress,
            }
        )

    return binned_subsequences


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
                    sample_indices.append(
                        {
                            "original_traj_idx": traj_idx,
                            "rewind_length": rewind_length,
                            "original_traj_id": original_traj.get("id", f"traj_{traj_idx}"),
                        }
                    )

        return sample_indices

    def _generate_sample_from_indices(self, sample_idx_info: Dict) -> PreferenceSample:
        """Generate a single sample from stored indices."""
        original_traj_idx = sample_idx_info["original_traj_idx"]
        rewind_length = sample_idx_info["rewind_length"]

        # Get the original trajectory
        original_traj = self.data_generator.dataset[original_traj_idx]

        # Create rewound trajectory
        rewound_traj = self.data_generator._create_rewind_trajectory(original_traj, rewind_length=rewind_length)

        # Get frames
        original_frames = self.data_generator._get_trajectory_frames(original_traj_idx)
        rewound_frames = rewound_traj["frames"]  # Already numpy array

        target_progress_chosen = calulate_target_progress(original_frames)
        target_progress_rejected = rewound_traj["metadata"]["rewind_progress"]

        # Create preference sample (original is chosen, rewound is rejected)
        sample = PreferenceSample(
            # chosen metadata
            chosen_frames=original_frames,
            chosen_frames_shape=original_traj["frames_shape"],
            chosen_id=original_traj["id"],
            chosen_task=original_traj["task"],
            chosen_lang_vector=original_traj["lang_vector"],
            chosen_data_source=original_traj["data_source"],
            chosen_quality_label=original_traj["quality_label"],
            chosen_is_robot=original_traj["is_robot"],
            # rejected metadata
            rejected_frames=rewound_frames,
            rejected_frames_shape=rewound_traj["frames_shape"],
            rejected_id=rewound_traj["id"],
            rejected_task=rewound_traj["task"],
            rejected_lang_vector=rewound_traj["lang_vector"],
            rejected_data_source=rewound_traj["data_source"],
            rejected_quality_label=rewound_traj["quality_label"],
            rejected_is_robot=rewound_traj["is_robot"],
            num_frames_rewound=rewound_traj.get("num_frames_rewound", rewind_length),
            data_gen_strategy="rewound",
            target_progress_chosen=target_progress_chosen,
            target_progress_rejected=target_progress_rejected,
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
                    sample_indices.append(
                        {"success_traj_idx": success_idx, "failure_traj_idx": failure_idx, "task": task}
                    )

        return sample_indices

    def _generate_sample_from_indices(self, sample_idx_info: Dict) -> PreferenceSample:
        """Generate a single sample from stored indices."""
        success_idx = sample_idx_info["success_traj_idx"]
        failure_idx = sample_idx_info["failure_traj_idx"]

        # Get the trajectories
        success_traj = self.data_generator.dataset[success_idx]
        failure_traj = self.data_generator.dataset[failure_idx]

        # Get frames
        success_frames = self.data_generator._get_trajectory_frames(success_idx)
        failure_frames = self.data_generator._get_trajectory_frames(failure_idx)

        target_progress_chosen = calulate_target_progress(success_frames)
        target_progress_rejected = None

        # Create preference sample (successful is chosen, failed is rejected)
        sample = PreferenceSample(
            # chosen metadata
            chosen_frames=success_frames,
            chosen_frames_shape=success_traj["frames_shape"],
            chosen_id=success_traj["id"],
            chosen_task=success_traj["task"],
            chosen_lang_vector=success_traj["lang_vector"],
            chosen_data_source=success_traj["data_source"],
            chosen_quality_label=success_traj["quality_label"],
            chosen_is_robot=success_traj["is_robot"],
            # rejected metadata
            rejected_frames=failure_frames,
            rejected_frames_shape=failure_traj["frames_shape"],
            rejected_id=failure_traj["id"],
            rejected_task=failure_traj["task"],
            rejected_lang_vector=failure_traj["lang_vector"],
            rejected_data_source=failure_traj["data_source"],
            rejected_quality_label=failure_traj["quality_label"],
            rejected_is_robot=failure_traj["is_robot"],
            data_gen_strategy="success_failure",
            num_frames_rewound=None,  # Not applicable for success-failure pairs
            target_progress_chosen=target_progress_chosen,
            target_progress_rejected=None,  # not applicable for failure trajectories
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


class VideoBinnedDataset:
    """
    Dataset that generates preference samples from video files by binning frames.

    For each video, it extracts frames at 1 FPS, bins them, and creates preference pairs
    where the trajectory further along in the task is always preferred.
    """

    def __init__(self, data_generator, num_bins: int = 10, fps: int = 10, **kwargs):
        self.data_generator = data_generator
        self.num_bins = num_bins
        self.fps = fps

        # Store video data once, reference by indices
        self.video_data = {}  # video_path -> binned_subsequences
        self.sample_indices = []

        # Generate all possible sample indices upfront
        self._generate_all_sample_indices()
        self.current_idx = 0

        # Log detailed statistics
        total_videos = len(self.video_data)
        rank_0_print(f"VideoBinnedDataset: {len(self.sample_indices)} preference pairs from {total_videos} videos")
        rank_0_print(f"  - Bins per video: {self.num_bins}")
        rank_0_print(f"  - FPS extraction: {self.fps}")
        rank_0_print(f"  - Average pairs per video: {len(self.sample_indices) / max(1, total_videos):.1f}")
        rank_0_print(f"  - Total samples: {len(self.sample_indices)}")

    def _generate_all_sample_indices(self) -> None:
        """Generate all possible video binned sample indices."""
        for traj_idx in self.data_generator.robot_trajectories:
            traj = self.data_generator.dataset[traj_idx]

            # Get the video path from frames_path
            frames_path = traj.get("frames_path", "")
            if not frames_path or not frames_path.endswith(".mp4"):
                continue

            # Extract frames from video to determine number of bins
            frames = extract_frames_from_video(frames_path, fps=self.fps)
            binned_subsequences = create_binned_subsequences(frames, num_bins=self.num_bins)

            if len(binned_subsequences) < 2:
                # Need at least 2 bins to create preference pairs
                continue

            # Store video data once
            self.video_data[frames_path] = {
                "binned_subsequences": binned_subsequences,
                "traj_info": {
                    "traj_idx": traj_idx,
                    "traj_id": traj["id"],
                    "task": traj["task"],
                    "data_source": traj["data_source"],
                    "quality_label": traj["quality_label"],
                    "is_robot": traj["is_robot"],
                    "lang_vector": traj["lang_vector"],
                    "num_frames": len(frames),
                },
            }

            # Generate all possible pairwise combinations
            for i in range(len(binned_subsequences)):
                for j in range(i + 1, len(binned_subsequences)):
                    # Store just the indices and video path reference
                    self.sample_indices.append({"frames_path": frames_path, "bin1_idx": i, "bin2_idx": j})

    def _generate_sample_from_indices(self, sample_idx_info: Dict) -> PreferenceSample:
        """Generate a single sample from stored indices."""
        frames_path = sample_idx_info["frames_path"]
        bin1_idx = sample_idx_info["bin1_idx"]
        bin2_idx = sample_idx_info["bin2_idx"]
        entry = self.video_data.get(frames_path)

        if not entry:
            rank_0_print(f"Video data not found for path: {frames_path}")
            return None

        # Use the pre-computed binned subsequences (no need to re-extract frames)
        if len(entry["binned_subsequences"]) < 2:
            return None

        # Validate bin indices
        if (
            bin1_idx >= len(entry["binned_subsequences"])
            or bin2_idx >= len(entry["binned_subsequences"])
            or bin1_idx < 0
            or bin2_idx < 0
        ):
            rank_0_print(
                f"Invalid bin indices: {bin1_idx}, {bin2_idx} for video with {len(entry['binned_subsequences'])} bins"
            )
            return None

        # Use the pre-computed bin pair
        bin1 = entry["binned_subsequences"][bin1_idx]
        bin2 = entry["binned_subsequences"][bin2_idx]

        # Determine which bin is further along (higher progress)
        if bin1["progress"] > bin2["progress"]:
            chosen_bin = bin1
            rejected_bin = bin2
        else:
            chosen_bin = bin2
            rejected_bin = bin1

        # Calculate target progress for each bin
        chosen_frames = chosen_bin["frames"]
        rejected_frames = rejected_bin["frames"]

        target_progress_chosen = calulate_target_progress(chosen_frames)
        target_progress_rejected = calulate_target_progress(rejected_frames)

        # Create preference sample (further along is chosen)
        sample = PreferenceSample(
            # chosen metadata (further along)
            chosen_frames=chosen_frames,
            chosen_frames_shape=chosen_frames.shape,
            chosen_id=f"{entry['traj_info']['traj_id']}_bin{chosen_bin['bin_idx']}",
            chosen_task=entry["traj_info"]["task"],
            chosen_lang_vector=entry["traj_info"]["lang_vector"],
            chosen_data_source=entry["traj_info"]["data_source"],
            chosen_quality_label=entry["traj_info"]["quality_label"],
            chosen_is_robot=entry["traj_info"]["is_robot"],
            # rejected metadata (earlier in task)
            rejected_frames=rejected_frames,
            rejected_frames_shape=rejected_frames.shape,
            rejected_id=f"{entry['traj_info']['traj_id']}_bin{rejected_bin['bin_idx']}",
            rejected_task=entry["traj_info"]["task"],
            rejected_lang_vector=entry["traj_info"]["lang_vector"],
            rejected_data_source=entry["traj_info"]["data_source"],
            rejected_quality_label=entry["traj_info"]["quality_label"],
            rejected_is_robot=entry["traj_info"]["is_robot"],
            data_gen_strategy="video_binned",
            num_frames_rewound=None,  # Not applicable for video binning
            target_progress_chosen=target_progress_chosen,
            target_progress_rejected=target_progress_rejected,
            bin_idx_chosen=chosen_bin["bin_idx"],
            bin_idx_rejected=rejected_bin["bin_idx"],
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

        # Skip invalid samples
        while sample is None and self.current_idx < len(self.sample_indices):
            self.current_idx += 1
            if self.current_idx >= len(self.sample_indices):
                raise StopIteration

            sample_idx_info = self.sample_indices[self.current_idx]
            sample = self._generate_sample_from_indices(sample_idx_info)

        if sample is None:
            raise StopIteration

        self.current_idx += 1
        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self.__next__()
