#!/usr/bin/env python3
"""
SimilarityDataGenerator class for producing batches of similarity scoring data.
"""

import random

from rfm.data.dataset_types import SimilaritySample, Trajectory
from .base import RFMBaseDataset
from .helpers import (
    DataGenStrat,
    create_rewind_trajectory,
    load_embeddings_from_path,
    load_frames_from_npz,
    linspace_subsample_frames,
    pad_trajectory_to_max_frames_np,
    pad_trajectory_to_max_frames_torch,
    subsample_segment_frames,
    compute_progress_from_segment,
)
from rfm.utils.distributed import rank_0_print


class SimilarityDataset(RFMBaseDataset):
    """Data generator for producing batches of similarity scoring data."""

    def __init__(self, config, is_evaluation=False, verbose=True, **kwargs):
        """Initialize SimilarityDataset with configuration."""
        super().__init__(config, is_evaluation, verbose=verbose, **kwargs)
        self.similarity_strategy_ratio: list[float] = config.similarity_strategy_ratio

        if self.verbose:
            rank_0_print(f"SimilarityDataset initialized with {len(self.dataset)} total trajectories")

    def __getitem__(self, idx):
        return self._create_similarity_sample()

    def _create_similarity_sample(self, ref_traj: dict | None = None) -> SimilaritySample:
        """Create a similarity scoring sample: o^1 and o^2 ranked against o^ref.

        Two modes:
        1. Rewind mode: o^1 is rewound from same task, o^2 is from different task
            - here o^1 is preferred and should be ranked higher than o^2
        2. Optimal/Suboptimal mode: o^1 is optimal/suboptimal from same task, o^2 varies
            - here o^1 is preferred and should be ranked higher than o^2

        Args:
            ref_traj: Optional reference trajectory. If None, samples from optimal trajectories.
        """

        # Use provided reference trajectory if given; otherwise sample one
        if ref_traj is None:
            # Use preprocessed optimal trajectories from index maps
            if not self.optimal_by_task:
                raise ValueError("No optimal trajectories found for similarity sample generation")

            # Get a random task and optimal trajectory from it
            task_name = random.choice(list(self.optimal_by_task.keys()))
            optimal_indices = self.optimal_by_task[task_name]
            while not optimal_indices:
                task_name = random.choice(list(self.optimal_by_task.keys()))
                optimal_indices = self.optimal_by_task[task_name]

            optimal_idx = random.choice(optimal_indices)
            ref_traj = self.dataset[optimal_idx]

        # Initialize variables for strategy selection
        selected_sample = None
        strategy_used = None

        # Strategy selection with rebalancing on failure
        strategies = [
            (DataGenStrat.REWIND_SAME_TASK, self.similarity_strategy_ratio[0]),
            (DataGenStrat.SUBOPTIMAL_SAME_TASK, self.similarity_strategy_ratio[1]),
        ]

        # Remove strategies with zero probability
        strategies = [(strat, prob) for strat, prob in strategies if prob > 0]

        max_attempts = 3  # Limit retry attempts to prevent infinite loops
        attempt = 0

        while selected_sample is None and attempt < max_attempts:
            attempt += 1

            # Rebalance probabilities based on remaining strategies
            total_prob = sum(prob for _, prob in strategies)
            if total_prob == 0:
                # All strategies have zero probability, fallback to rewind
                selected_sample = self._create_rewind_similarity_sample(ref_traj)
                strategy_used = DataGenStrat.REWIND_SAME_TASK
                break

            # Normalize probabilities
            normalized_strategies = [(strat, prob / total_prob) for strat, prob in strategies]

            # Select strategy based on rebalanced probabilities
            prob = random.random()
            cumulative_prob = 0.0
            selected_strategy = None

            for strat, normalized_prob in normalized_strategies:
                cumulative_prob += normalized_prob
                if prob <= cumulative_prob:
                    selected_strategy = strat
                    break

            # Execute selected strategy
            if selected_strategy == DataGenStrat.REWIND_SAME_TASK:
                selected_sample = self._create_rewind_similarity_sample(ref_traj)
                strategy_used = DataGenStrat.REWIND_SAME_TASK

            elif selected_strategy == DataGenStrat.SUBOPTIMAL_SAME_TASK:
                try:
                    selected_sample = self._create_optimal_similarity_sample(ref_traj)
                    strategy_used = DataGenStrat.SUBOPTIMAL_SAME_TASK
                except Exception as e:
                    rank_0_print(f"Optimal similarity sampling failed: {e}, removing from available strategies")
                    # Strategy failed, remove it from future attempts
                    strategies = [
                        (strat, prob) for strat, prob in strategies if strat != DataGenStrat.SUBOPTIMAL_SAME_TASK
                    ]

        # Final fallback: If all strategies failed, use rewind
        if selected_sample is None:
            selected_sample = self._create_rewind_similarity_sample(ref_traj)
            strategy_used = DataGenStrat.REWIND_SAME_TASK

        return selected_sample

    def _create_rewind_similarity_sample(self, ref_traj: dict) -> SimilaritySample:
        """Create similarity sample using rewind logic.

        Rules:
        - traj_sim is rewound trajectory from same task as o^ref (different trajectory)
        - traj_diff MUST be from different task than o^ref
        - o^ref is optimal trajectory from a random task

        Args:
            ref_traj: Reference trajectory to use as the base for similarity comparison
        """

        # Extract task name from the provided reference trajectory
        task_name = ref_traj["task"]

        # Create traj_sim: rewound trajectory from same task as o^ref
        # Get success cutoff from pre-loaded map
        ds_key = ref_traj["data_source"]
        success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

        traj_sim = create_rewind_trajectory(
            ref_traj,
            max_frames=self.config.max_frames,
            use_embeddings=self.config.load_embeddings,
            progress_pred_type=getattr(self.config, "progress_pred_type", "absolute"),
            success_cutoff=success_cutoff,
        )

        # Create traj_diff: trajectory from different task than o^ref
        other_tasks = [task for task in self.optimal_by_task.keys() if task != ref_traj["task"]]
        if not other_tasks:
            # If only one task available, use suboptimal from same task
            same_task_suboptimal_indices = self.suboptimal_by_task.get(task_name, [])
            if same_task_suboptimal_indices:
                traj_diff_idx = random.choice(same_task_suboptimal_indices)
                traj_diff = self.dataset[traj_diff_idx]
            else:
                # Fallback: create another rewind trajectory
                traj_diff = create_rewind_trajectory(
                    ref_traj,
                    max_frames=self.config.max_frames,
                    use_embeddings=self.config.load_embeddings,
                    progress_pred_type=getattr(self.config, "progress_pred_type", "absolute"),
                    success_cutoff=success_cutoff,
                )
        else:
            # Use trajectory from different task
            other_task = random.choice(other_tasks)
            other_task_indices = self.optimal_by_task[other_task]
            if other_task_indices:
                other_idx = random.choice(other_task_indices)
                traj_diff = self.dataset[other_idx]
            else:
                # Fallback: use suboptimal from same task
                same_task_suboptimal_indices = self.suboptimal_by_task.get(task_name, [])
                if same_task_suboptimal_indices:
                    traj_diff_idx = random.choice(same_task_suboptimal_indices)
                    traj_diff = self.dataset[traj_diff_idx]
                else:
                    # Final fallback: create another rewind trajectory
                    traj_diff = create_rewind_trajectory(
                        ref_traj,
                        max_frames=self.config.max_frames,
                        use_embeddings=self.config.load_embeddings,
                        progress_pred_type=getattr(self.config, "progress_pred_type", "absolute"),
                        success_cutoff=success_cutoff,
                    )

        # ===============================================================
        # Load and subsample reference trajectory
        # ===============================================================
        ref_frames = None
        ref_video_embeddings = None
        ref_text_embedding = None

        if self.config.load_embeddings and ref_traj.get("embeddings_path"):
            ref_video_embeddings = load_embeddings_from_path(ref_traj["embeddings_path"], "video_embeddings")
            ref_text_embedding = load_embeddings_from_path(ref_traj["embeddings_path"], "text_embedding")

            # Get success cutoff from pre-loaded map
            ds_key = ref_traj["data_source"]
            success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

            subsampled, start_idx, end_idx, indices = subsample_segment_frames(
                ref_video_embeddings, self.config.max_frames
            )
            ref_frames_shape_orig = subsampled.shape
            ref_progress = compute_progress_from_segment(
                num_frames_total=len(ref_video_embeddings),
                start_idx=start_idx,
                end_idx=end_idx,
                frame_indices=indices,
                progress_pred_type=self.config.progress_pred_type,
                success_cutoff=success_cutoff,
            )
            ref_metadata = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "subsampled_indices": indices,
            }
            ref_video_embeddings = subsampled
        else:
            if isinstance(ref_traj["frames"], str):
                ref_frames = load_frames_from_npz(ref_traj["frames"])
            else:
                ref_frames = ref_traj["frames"]

            # Get success cutoff from pre-loaded map
            ds_key = ref_traj["data_source"]
            success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

            subsampled, start_idx, end_idx, indices = subsample_segment_frames(ref_frames, self.config.max_frames)
            ref_frames_shape_orig = subsampled.shape
            ref_progress = compute_progress_from_segment(
                num_frames_total=len(ref_frames),
                start_idx=start_idx,
                end_idx=end_idx,
                frame_indices=indices,
                progress_pred_type=self.config.progress_pred_type,
                success_cutoff=success_cutoff,
            )
            ref_metadata = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "subsampled_indices": indices,
            }
            ref_frames = subsampled

        # ===============================================================
        # Load and subsample traj_sim trajectory (rewound)
        # ===============================================================
        traj_sim_frames = None
        traj_sim_video_embeddings = None
        traj_sim_text_embedding = None

        if self.config.load_embeddings and traj_sim.get("embeddings_path"):
            traj_sim_video_embeddings = load_embeddings_from_path(traj_sim["embeddings_path"], "video_embeddings")
            traj_sim_text_embedding = load_embeddings_from_path(traj_sim["embeddings_path"], "text_embedding")
            # Rewound trajectory already has frames and progress from create_rewind_trajectory
            traj_sim_video_embeddings = traj_sim["frames"]
            traj_sim_frames_shape_orig = traj_sim["frames_shape"]
            traj_sim_progress = traj_sim["target_progress"]
            traj_sim_metadata = traj_sim["metadata"]
        else:
            if isinstance(traj_sim["frames"], str):
                traj_sim_frames = load_frames_from_npz(traj_sim["frames"])
            else:
                traj_sim_frames = traj_sim["frames"]

            # Rewound trajectory already has frames and progress from create_rewind_trajectory
            traj_sim_frames = traj_sim["frames"]
            traj_sim_frames_shape_orig = traj_sim["frames_shape"]
            traj_sim_progress = traj_sim["target_progress"]
            traj_sim_metadata = traj_sim["metadata"]

        # ===============================================================
        # Load and subsample traj_diff trajectory
        # ===============================================================
        traj_diff_frames = None
        traj_diff_video_embeddings = None
        traj_diff_text_embedding = None

        if self.config.load_embeddings and traj_diff.get("embeddings_path"):
            traj_diff_video_embeddings = load_embeddings_from_path(traj_diff["embeddings_path"], "video_embeddings")
            traj_diff_text_embedding = load_embeddings_from_path(traj_diff["embeddings_path"], "text_embedding")

            # Get success cutoff from pre-loaded map
            ds_key = traj_diff["data_source"]
            success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

            subsampled, start_idx, end_idx, indices = subsample_segment_frames(
                traj_diff_video_embeddings, self.config.max_frames
            )
            traj_diff_frames_shape_orig = subsampled.shape
            traj_diff_progress = compute_progress_from_segment(
                num_frames_total=len(traj_diff_video_embeddings),
                start_idx=start_idx,
                end_idx=end_idx,
                frame_indices=indices,
                progress_pred_type=self.config.progress_pred_type,
                success_cutoff=success_cutoff,
            )
            traj_diff_metadata = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "subsampled_indices": indices,
            }
            traj_diff_video_embeddings = subsampled
        else:
            if isinstance(traj_diff["frames"], str):
                traj_diff_frames = load_frames_from_npz(traj_diff["frames"])
            else:
                traj_diff_frames = traj_diff["frames"]

            # Get success cutoff from pre-loaded map
            ds_key = traj_diff["data_source"]
            success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

            subsampled, start_idx, end_idx, indices = subsample_segment_frames(traj_diff_frames, self.config.max_frames)
            traj_diff_frames_shape_orig = subsampled.shape
            traj_diff_progress = compute_progress_from_segment(
                num_frames_total=len(traj_diff_frames),
                start_idx=start_idx,
                end_idx=end_idx,
                frame_indices=indices,
                progress_pred_type=self.config.progress_pred_type,
                success_cutoff=success_cutoff,
            )
            traj_diff_metadata = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "subsampled_indices": indices,
            }
            traj_diff_frames = subsampled

        # ===============================================================
        # Pad all trajectories to max_frames if needed
        # ===============================================================
        if self.config.load_embeddings:
            ref_video_embeddings, ref_progress = pad_trajectory_to_max_frames_torch(
                ref_video_embeddings, ref_progress, self.config.max_frames
            )
            traj_sim_video_embeddings, traj_sim_progress = pad_trajectory_to_max_frames_torch(
                traj_sim_video_embeddings, traj_sim_progress, self.config.max_frames
            )
            traj_diff_video_embeddings, traj_diff_progress = pad_trajectory_to_max_frames_torch(
                traj_diff_video_embeddings, traj_diff_progress, self.config.max_frames
            )
        else:
            ref_frames, ref_progress = pad_trajectory_to_max_frames_np(ref_frames, ref_progress, self.config.max_frames)
            traj_sim_frames, traj_sim_progress = pad_trajectory_to_max_frames_np(
                traj_sim_frames, traj_sim_progress, self.config.max_frames
            )
            traj_diff_frames, traj_diff_progress = pad_trajectory_to_max_frames_np(
                traj_diff_frames, traj_diff_progress, self.config.max_frames
            )

        # Create SimilaritySample
        sample = SimilaritySample(
            ref_trajectory=Trajectory(
                frames=ref_frames,
                frames_shape=ref_frames_shape_orig,
                video_embeddings=ref_video_embeddings,
                text_embedding=ref_text_embedding,
                id=ref_traj["id"],
                task=ref_traj["task"],
                lang_vector=ref_traj["lang_vector"],
                data_source=ref_traj["data_source"],
                quality_label=ref_traj.get("quality_label"),
                is_robot=ref_traj["is_robot"],
                target_progress=ref_progress,
                metadata=ref_metadata,
            ),
            sim_trajectory=Trajectory(
                frames=traj_sim_frames,
                frames_shape=traj_sim_frames_shape_orig,
                video_embeddings=traj_sim_video_embeddings,
                text_embedding=traj_sim_text_embedding,
                id=traj_sim["id"],
                task=traj_sim["task"],
                lang_vector=traj_sim["lang_vector"],
                data_source=traj_sim["data_source"],
                quality_label=traj_sim["quality_label"],
                is_robot=traj_sim["is_robot"],
                target_progress=traj_sim_progress,
                metadata=traj_sim_metadata,
            ),
            diff_trajectory=Trajectory(
                frames=traj_diff_frames,
                frames_shape=traj_diff_frames_shape_orig,
                video_embeddings=traj_diff_video_embeddings,
                text_embedding=traj_diff_text_embedding,
                id=traj_diff["id"],
                task=traj_diff["task"],
                lang_vector=traj_diff["lang_vector"],
                data_source=traj_diff["data_source"],
                quality_label=traj_diff["quality_label"],
                is_robot=traj_diff["is_robot"],
                target_progress=traj_diff_progress,
                metadata=traj_diff_metadata,
            ),
            data_gen_strategy="rewind_similarity",
        )

        return sample

    def _create_optimal_similarity_sample(self, ref_traj: dict) -> SimilaritySample:
        """Create similarity sample using optimal/suboptimal logic.

        Rules:
        - o^ref is optimal trajectory from a random task
        - traj_sim is optimal trajectory from same task as o^ref (different trajectory)
        - traj_diff is suboptimal trajectory from same task as o^ref

        Args:
            ref_traj: Reference trajectory to use as the base for similarity comparison
        """

        # Extract task name from the provided reference trajectory
        task_name = ref_traj["task"]

        # Create traj_sim: optimal trajectory from same task as o^ref (different trajectory)
        same_task_optimal_indices = [
            idx for idx in self.optimal_by_task[task_name] if self.dataset[idx]["id"] != ref_traj["id"]
        ]
        if same_task_optimal_indices:
            traj_sim_idx = random.choice(same_task_optimal_indices)
            traj_sim = self.dataset[traj_sim_idx]
        else:
            # If no other optimal trajectories, use the same one
            traj_sim = ref_traj

        # Create traj_diff: suboptimal trajectory from same task as o^ref
        same_task_suboptimal_indices = self.suboptimal_by_task.get(task_name, [])
        if same_task_suboptimal_indices:
            traj_diff_idx = random.choice(same_task_suboptimal_indices)
            traj_diff = self.dataset[traj_diff_idx]
        else:
            # If no suboptimal trajectories, create a rewind trajectory
            # Get success cutoff from pre-loaded map
            ds_key = ref_traj["data_source"]
            success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

            traj_diff = create_rewind_trajectory(
                ref_traj,
                max_frames=self.config.max_frames,
                use_embeddings=self.config.load_embeddings,
                progress_pred_type=getattr(self.config, "progress_pred_type", "absolute"),
                success_cutoff=success_cutoff,
            )

        # ===============================================================
        # Load and subsample reference trajectory
        # ===============================================================
        ref_frames = None
        ref_video_embeddings = None
        ref_text_embedding = None

        if self.config.load_embeddings and ref_traj.get("embeddings_path"):
            ref_video_embeddings = load_embeddings_from_path(ref_traj["embeddings_path"], "video_embeddings")
            ref_text_embedding = load_embeddings_from_path(ref_traj["embeddings_path"], "text_embedding")

            # Get success cutoff from pre-loaded map
            ds_key = ref_traj["data_source"]
            success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

            subsampled, start_idx, end_idx, indices = subsample_segment_frames(
                ref_video_embeddings, self.config.max_frames
            )
            ref_frames_shape_orig = subsampled.shape
            ref_progress = compute_progress_from_segment(
                num_frames_total=len(ref_video_embeddings),
                start_idx=start_idx,
                end_idx=end_idx,
                frame_indices=indices,
                progress_pred_type=self.config.progress_pred_type,
                success_cutoff=success_cutoff,
            )
            ref_metadata = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "subsampled_indices": indices,
            }
            ref_video_embeddings = subsampled
        else:
            if isinstance(ref_traj["frames"], str):
                ref_frames = load_frames_from_npz(ref_traj["frames"])
            else:
                ref_frames = ref_traj["frames"]

            # Get success cutoff from pre-loaded map
            ds_key = ref_traj["data_source"]
            success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

            subsampled, start_idx, end_idx, indices = subsample_segment_frames(ref_frames, self.config.max_frames)
            ref_frames_shape_orig = subsampled.shape
            ref_progress = compute_progress_from_segment(
                num_frames_total=len(ref_frames),
                start_idx=start_idx,
                end_idx=end_idx,
                frame_indices=indices,
                progress_pred_type=self.config.progress_pred_type,
                success_cutoff=success_cutoff,
            )
            ref_metadata = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "subsampled_indices": indices,
            }
            ref_frames = subsampled

        # ===============================================================
        # Load and subsample traj_sim trajectory
        # ===============================================================
        traj_sim_frames = None
        traj_sim_video_embeddings = None
        traj_sim_text_embedding = None

        if self.config.load_embeddings and traj_sim.get("embeddings_path"):
            traj_sim_video_embeddings = load_embeddings_from_path(traj_sim["embeddings_path"], "video_embeddings")
            traj_sim_text_embedding = load_embeddings_from_path(traj_sim["embeddings_path"], "text_embedding")

            # Get success cutoff from pre-loaded map
            ds_key = traj_sim["data_source"]
            success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

            subsampled, start_idx, end_idx, indices = subsample_segment_frames(
                traj_sim_video_embeddings, self.config.max_frames
            )
            traj_sim_frames_shape_orig = subsampled.shape
            traj_sim_progress = compute_progress_from_segment(
                num_frames_total=len(traj_sim_video_embeddings),
                start_idx=start_idx,
                end_idx=end_idx,
                frame_indices=indices,
                progress_pred_type=self.config.progress_pred_type,
                success_cutoff=success_cutoff,
            )
            traj_sim_metadata = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "subsampled_indices": indices,
            }
            traj_sim_video_embeddings = subsampled
        else:
            if isinstance(traj_sim["frames"], str):
                traj_sim_frames = load_frames_from_npz(traj_sim["frames"])
            else:
                traj_sim_frames = traj_sim["frames"]

            # Get success cutoff from pre-loaded map
            ds_key = traj_sim["data_source"]
            success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

            subsampled, start_idx, end_idx, indices = subsample_segment_frames(traj_sim_frames, self.config.max_frames)
            traj_sim_frames_shape_orig = subsampled.shape
            traj_sim_progress = compute_progress_from_segment(
                num_frames_total=len(traj_sim_frames),
                start_idx=start_idx,
                end_idx=end_idx,
                frame_indices=indices,
                progress_pred_type=self.config.progress_pred_type,
                success_cutoff=success_cutoff,
            )
            traj_sim_metadata = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "subsampled_indices": indices,
            }
            traj_sim_frames = subsampled

        # ===============================================================
        # Load and subsample traj_diff trajectory
        # ===============================================================
        traj_diff_frames = None
        traj_diff_video_embeddings = None
        traj_diff_text_embedding = None

        if self.config.load_embeddings and traj_diff.get("embeddings_path"):
            traj_diff_video_embeddings = load_embeddings_from_path(traj_diff["embeddings_path"], "video_embeddings")
            traj_diff_text_embedding = load_embeddings_from_path(traj_diff["embeddings_path"], "text_embedding")

            # Check if traj_diff is a rewound trajectory
            if traj_diff.get("target_progress"):
                # Rewound trajectory already has embeddings and progress
                traj_diff_video_embeddings = traj_diff["frames"]
                traj_diff_frames_shape_orig = traj_diff["frames_shape"]
                traj_diff_progress = traj_diff["target_progress"]
                traj_diff_metadata = traj_diff["metadata"]
            else:
                # Get success cutoff from pre-loaded map
                ds_key = traj_diff["data_source"]
                success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

                subsampled, start_idx, end_idx, indices = subsample_segment_frames(
                    traj_diff_video_embeddings, self.config.max_frames
                )
                traj_diff_frames_shape_orig = subsampled.shape
                traj_diff_progress = compute_progress_from_segment(
                    num_frames_total=len(traj_diff_video_embeddings),
                    start_idx=start_idx,
                    end_idx=end_idx,
                    frame_indices=indices,
                    progress_pred_type=self.config.progress_pred_type,
                    success_cutoff=success_cutoff,
                )
                traj_diff_metadata = {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "subsampled_indices": indices,
                }
                traj_diff_video_embeddings = subsampled
        else:
            if isinstance(traj_diff["frames"], str):
                traj_diff_frames = load_frames_from_npz(traj_diff["frames"])
            else:
                traj_diff_frames = traj_diff["frames"]

            # Check if traj_diff is a rewound trajectory
            if traj_diff.get("target_progress"):
                # Rewound trajectory already has frames and progress
                traj_diff_frames = traj_diff["frames"]
                traj_diff_frames_shape_orig = traj_diff["frames_shape"]
                traj_diff_progress = traj_diff["target_progress"]
                traj_diff_metadata = traj_diff["metadata"]
            else:
                # Get success cutoff from pre-loaded map
                ds_key = traj_diff["data_source"]
                success_cutoff = self.dataset_success_cutoff_map.get(ds_key, self.config.max_success)

                subsampled, start_idx, end_idx, indices = subsample_segment_frames(
                    traj_diff_frames, self.config.max_frames
                )
                traj_diff_frames_shape_orig = subsampled.shape
                traj_diff_progress = compute_progress_from_segment(
                    num_frames_total=len(traj_diff_frames),
                    start_idx=start_idx,
                    end_idx=end_idx,
                    frame_indices=indices,
                    progress_pred_type=self.config.progress_pred_type,
                    success_cutoff=success_cutoff,
                )
                traj_diff_metadata = {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "subsampled_indices": indices,
                }
                traj_diff_frames = subsampled

        # ===============================================================
        # Pad all trajectories to max_frames if needed
        # ===============================================================
        if self.config.load_embeddings:
            ref_video_embeddings, ref_progress = pad_trajectory_to_max_frames_torch(
                ref_video_embeddings, ref_progress, self.config.max_frames
            )
            traj_sim_video_embeddings, traj_sim_progress = pad_trajectory_to_max_frames_torch(
                traj_sim_video_embeddings, traj_sim_progress, self.config.max_frames
            )
            traj_diff_video_embeddings, traj_diff_progress = pad_trajectory_to_max_frames_torch(
                traj_diff_video_embeddings, traj_diff_progress, self.config.max_frames
            )
        else:
            ref_frames, ref_progress = pad_trajectory_to_max_frames_np(ref_frames, ref_progress, self.config.max_frames)
            traj_sim_frames, traj_sim_progress = pad_trajectory_to_max_frames_np(
                traj_sim_frames, traj_sim_progress, self.config.max_frames
            )
            traj_diff_frames, traj_diff_progress = pad_trajectory_to_max_frames_np(
                traj_diff_frames, traj_diff_progress, self.config.max_frames
            )

        # Create SimilaritySample
        sample = SimilaritySample(
            ref_trajectory=Trajectory(
                frames=ref_frames,
                frames_shape=ref_frames_shape_orig,
                video_embeddings=ref_video_embeddings,
                text_embedding=ref_text_embedding,
                id=ref_traj["id"],
                task=ref_traj["task"],
                lang_vector=ref_traj["lang_vector"],
                data_source=ref_traj["data_source"],
                quality_label=ref_traj.get("quality_label"),
                is_robot=ref_traj["is_robot"],
                target_progress=ref_progress,
                metadata=ref_metadata,
            ),
            sim_trajectory=Trajectory(
                frames=traj_sim_frames,
                frames_shape=traj_sim_frames_shape_orig,
                video_embeddings=traj_sim_video_embeddings,
                text_embedding=traj_sim_text_embedding,
                id=traj_sim["id"],
                task=traj_sim["task"],
                lang_vector=traj_sim["lang_vector"],
                data_source=traj_sim["data_source"],
                quality_label=traj_sim["quality_label"],
                is_robot=traj_sim["is_robot"],
                target_progress=traj_sim_progress,
                metadata=traj_sim_metadata,
            ),
            diff_trajectory=Trajectory(
                frames=traj_diff_frames,
                frames_shape=traj_diff_frames_shape_orig,
                video_embeddings=traj_diff_video_embeddings,
                text_embedding=traj_diff_text_embedding,
                id=traj_diff["id"],
                task=traj_diff["task"],
                lang_vector=traj_diff["lang_vector"],
                data_source=traj_diff["data_source"],
                quality_label=traj_diff["quality_label"],
                is_robot=traj_diff["is_robot"],
                target_progress=traj_diff_progress,
                metadata=traj_diff_metadata,
            ),
            data_gen_strategy="optimal_similarity",
        )

        return sample
