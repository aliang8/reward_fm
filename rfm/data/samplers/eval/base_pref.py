import numpy as np

from rfm.data.dataset_types import PreferenceSample, Trajectory
from rfm.data.samplers.base import RFMBaseSampler
from rfm.data.datasets.helpers import (
    linspace_subsample_frames,
    load_frames_from_npz,
    load_embeddings_from_path,
    convert_absolute_to_relative_progress,
    create_trajectory_from_dict,
)


class BaseQualityPreferenceSampler(RFMBaseSampler):
    """Base class for quality preference samplers.

    Subclasses should implement `_generate_all_sample_indices` to define how
    trajectories are paired. This base class provides the common `_generate_sample_from_indices`
    method that loads and processes the trajectories.
    """

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
            total_frames = (
                chosen_video_embeddings.shape[0]
                if hasattr(chosen_video_embeddings, "shape")
                else len(chosen_video_embeddings)
            )
            use_embeddings = True
        else:
            chosen_frames = load_frames_from_npz(chosen_traj["frames"])
            data = chosen_frames
            total_frames = len(chosen_frames)
            use_embeddings = False

        data, frame_indices = linspace_subsample_frames(data, max_frames)
        chosen_frames_shape_orig = data.shape

        # Compute progress based on type
        if self.config.progress_pred_type == "absolute_wrt_total_frames":
            progress_abs = [(idx + 1) / total_frames for idx in frame_indices]
        elif self.config.progress_pred_type.startswith("absolute"):
            # absolute_first_frame: use linspace logic
            progress_abs = [idx / (total_frames - 1) for idx in frame_indices]
        else:  # relative_first_frame
            # For relative, we still compute absolute first, then convert
            progress_abs = [idx / (total_frames - 1) for idx in frame_indices]

        if self.config.progress_pred_type == "relative_first_frame":
            chosen_progress = convert_absolute_to_relative_progress(progress_abs)
        else:
            chosen_progress = progress_abs

        chosen_metadata = {
            "quality_label": chosen_traj["quality_label"],
            "data_source": chosen_traj["data_source"],
            "task": chosen_traj["task"],
            "id": chosen_traj["id"],
            "video_path": chosen_traj["frames"],
        }
        # Add partial_success if available
        if chosen_traj.get("partial_success") is not None:
            chosen_metadata["partial_success"] = chosen_traj.get("partial_success")

        chosen_trajectory = create_trajectory_from_dict(
            chosen_traj,
            overrides={
                "frames": data if not use_embeddings else None,
                "frames_shape": chosen_frames_shape_orig,
                "video_embeddings": data if use_embeddings else None,
                "text_embedding": chosen_text_embedding,
                "lang_vector": np.array(chosen_traj["lang_vector"]),
                "target_progress": chosen_progress,
                "metadata": chosen_metadata,
            },
        )
        chosen_trajectory = self._post_process_trajectory(chosen_trajectory)

        # Load and process rejected trajectory
        if self.config.load_embeddings and rejected_traj.get("embeddings_path"):
            embeddings = load_embeddings_from_path(rejected_traj["embeddings_path"])
            rejected_video_embeddings = embeddings["video_embeddings"]
            rejected_text_embedding = embeddings["text_embedding"]
            data = rejected_video_embeddings
            total_frames = (
                rejected_video_embeddings.shape[0]
                if hasattr(rejected_video_embeddings, "shape")
                else len(rejected_video_embeddings)
            )
            use_embeddings = True
        else:
            rejected_frames = load_frames_from_npz(rejected_traj["frames"])
            data = rejected_frames
            total_frames = len(rejected_frames)
            use_embeddings = False

        data, frame_indices = linspace_subsample_frames(data, max_frames)
        rejected_frames_shape_orig = data.shape

        # Compute progress based on type
        if self.config.progress_pred_type == "absolute_wrt_total_frames":
            progress_abs = [(idx + 1) / total_frames for idx in frame_indices]
        elif self.config.progress_pred_type.startswith("absolute"):
            # absolute_first_frame: use linspace logic
            progress_abs = [idx / (total_frames - 1) for idx in frame_indices]
        else:  # relative_first_frame
            # For relative, we still compute absolute first, then convert
            progress_abs = [idx / (total_frames - 1) for idx in frame_indices]

        if self.config.progress_pred_type == "relative_first_frame":
            rejected_progress = convert_absolute_to_relative_progress(progress_abs)
        else:
            rejected_progress = progress_abs

        rejected_metadata = {
            "quality_label": rejected_traj["quality_label"],
            "data_source": rejected_traj["data_source"],
            "task": rejected_traj["task"],
            "id": rejected_traj["id"],
            "video_path": rejected_traj["frames"],
        }
        # Add partial_success if available
        if rejected_traj.get("partial_success") is not None:
            rejected_metadata["partial_success"] = rejected_traj.get("partial_success")

        rejected_trajectory = create_trajectory_from_dict(
            rejected_traj,
            overrides={
                "frames": data if not use_embeddings else None,
                "frames_shape": rejected_frames_shape_orig,
                "video_embeddings": data if use_embeddings else None,
                "text_embedding": rejected_text_embedding,
                "lang_vector": np.array(rejected_traj["lang_vector"]),
                "target_progress": rejected_progress,
                "metadata": rejected_metadata,
            },
        )
        rejected_trajectory = self._post_process_trajectory(rejected_trajectory)

        data_gen_strategy = getattr(self, "data_gen_strategy", "quality_preference")

        # Create preference sample
        sample = PreferenceSample(
            chosen_trajectory=chosen_trajectory,
            rejected_trajectory=rejected_trajectory,
            data_gen_strategy=data_gen_strategy,
        )

        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])
