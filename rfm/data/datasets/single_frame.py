import random
import torch

from rfm.data.datasets.base import BaseDataset
from rfm.data.dataset_types import ProgressSample, Trajectory
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.datasets.helpers import (
    load_frames_from_npz,
    load_embeddings_from_path,
    compute_success_labels,
)
from rfm.utils.logger import get_logger, rank_0_info

logger = get_logger()


class SingleFrameProgressSampler(RFMBaseSampler):
    """Sampler that samples a single random frame from a trajectory."""

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
        single_frame = frames[frame_idx:frame_idx+1]  # (1, H, W, C)
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
                    video_embeddings = video_embeddings[frame_idx:frame_idx+1]  # Keep time dimension

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
    """Dataset that samples single frames from trajectories for progress prediction."""

    def __init__(self, config, is_evaluation=False, max_samples=None, **kwargs):
        super().__init__(config, is_evaluation, **kwargs)

        # Override max_frames to 1 for single frame sampling
        original_max_frames = config.max_frames
        config.max_frames = 1

        self.progress_sampler = SingleFrameProgressSampler(
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

        rank_0_info(f"SingleFrameDataset initialized with {self.data_len} trajectories")

    def __len__(self):
        if self.max_samples is None:
            return self.data_len
        else:
            return self.max_samples

    def __getitem__(self, idx):
        """Create a single-frame progress sample from the dataset."""
        idx = idx % self.data_len
        logger.trace(f"[SingleFrameDataset] __getitem__: Starting for idx={idx}")

        # Get the item from the filtered dataset
        item = self.dataset[idx]
        traj_id = item.get("id", "unknown")
        logger.trace(f"[SingleFrameDataset] __getitem__: Got item with ID={traj_id}")

        # Generate single frame sample
        sample = self.progress_sampler._generate_sample(item)
        if sample is None:
            # If sampling fails, raise an error
            raise ValueError(f"Failed to generate single frame sample for trajectory {traj_id}")
        
        logger.trace(f"[SingleFrameDataset] __getitem__: Successfully generated sample for idx={idx}, ID={traj_id}")
        return sample