import random

from rfm.data.dataset_types import ProgressSample, Trajectory
from .base import RFMBaseDataset
from .helpers import (
    DataGenStrat,
    create_rewind_trajectory,
    load_frames_from_npz,
    load_embeddings_from_path,
    pad_trajectory_to_max_frames_torch,
    pad_trajectory_to_max_frames_np,
    subsample_frames_and_progress,
    subsample_pairs_and_progress,
)


class ProgressDataset(RFMBaseDataset):
    """Data generator for progress samples."""

    def __getitem__(self, idx):
        """Iterate over one sample per trajectory in the dataset."""
        dataset_len = len(self.dataset)
        traj = self.dataset[idx % dataset_len]
        sample = self._create_progress_sample(traj)
        return sample

    def _create_progress_sample(self, traj: dict):
        """Create a progress sample using normalized and rebalanced strategy selection.

        Implements three strategies:
        1. Successful: Use original trajectory as-is
        2. Rewind Same Task: Create rewound trajectory from same task
        3. Different Task: Use trajectory from different task (progress set to 0.0)
        """

        # Initialize variables for strategy selection
        processed_traj = None
        strategy_used = None

        # Strategy setup with rebalancing on failure
        strategies = [
            ("successful", self.config.progress_strategy_ratio[0]),
            (DataGenStrat.REWIND_SAME_TASK, self.config.progress_strategy_ratio[1]),
            (DataGenStrat.DIFFERENT_TASK, self.config.progress_strategy_ratio[2]),
        ]

        if self.config.pairwise_progress:
            strategies[1] = (DataGenStrat.REWIND_SAME_TASK, 0.0) # remove rewind same task strategy for pairwise progress

        # Remove strategies with zero probability
        strategies = [(strat, prob) for strat, prob in strategies if prob > 0]

        max_attempts = 3  # Limit retry attempts to prevent infinite loops
        attempt = 0

        while processed_traj is None and attempt < max_attempts:
            attempt += 1

            # Rebalance probabilities based on remaining strategies
            total_prob = sum(prob for _, prob in strategies)
            if total_prob == 0:
                # All strategies have zero probability, fallback to successful
                processed_traj = traj.copy()
                strategy_used = "successful"
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
            if selected_strategy == "successful":
                processed_traj = traj.copy()
                strategy_used = "successful"

            elif selected_strategy == DataGenStrat.REWIND_SAME_TASK:
                processed_traj = create_rewind_trajectory(
                    traj,
                    max_frames=self.config.max_frames,
                    use_embeddings=self.config.load_embeddings,
                    progress_pred_type=self.config.progress_pred_type,
                )
                strategy_used = DataGenStrat.REWIND_SAME_TASK

            elif selected_strategy == DataGenStrat.DIFFERENT_TASK:
                other_traj = self._create_different_task_trajectory(traj)
                if other_traj is not None:
                    processed_traj = other_traj.copy()
                    strategy_used = DataGenStrat.DIFFERENT_TASK
                else:
                    # Strategy failed, remove it from future attempts
                    strategies = [(strat, prob) for strat, prob in strategies if strat != DataGenStrat.DIFFERENT_TASK]

        # Final fallback: If all strategies failed, use successful
        if processed_traj is None:
            processed_traj = traj.copy()
            strategy_used = "successful"

        frames = None
        video_embeddings = None
        text_embedding = None
        task = processed_traj["task"]
        lang_vector = processed_traj["lang_vector"]

        if self.config.load_embeddings and processed_traj.get("embeddings_path"):
            # We are loading precomputed image/text embeddings
            if strategy_used == DataGenStrat.REWIND_SAME_TASK:
                video_embeddings = processed_traj["frames"]  # For rewind, "frames" contains embeddings
                progress = processed_traj["target_progress"]
                metadata = processed_traj["metadata"]
            else:
                video_embeddings = load_embeddings_from_path(processed_traj["embeddings_path"], "video_embeddings")
                if self.config.pairwise_progress:
                    video_embeddings, progress, metadata = subsample_pairs_and_progress(
                        video_embeddings, self.config.max_frames, progress_pred_type=self.config.progress_pred_type
                    )
                else:
                    video_embeddings, progress, metadata = subsample_frames_and_progress(
                        video_embeddings, self.config.max_frames, progress_pred_type=self.config.progress_pred_type
                    )

            text_embedding = load_embeddings_from_path(processed_traj["embeddings_path"], "text_embedding")
            if strategy_used == DataGenStrat.DIFFERENT_TASK:
                # we need to use the original task embeddings instead of the different task embeddings
                text_embedding = load_embeddings_from_path(traj["embeddings_path"], "text_embedding")
                lang_vector = traj["lang_vector"]
                task = traj["task"]

            if not self.config.pairwise_progress:
                video_embeddings, progress = pad_trajectory_to_max_frames_torch(
                    video_embeddings, progress, self.config.max_frames
                )
        else:
            # We are using the image frames
            if strategy_used == DataGenStrat.REWIND_SAME_TASK:
                frames = processed_traj["frames"]
                progress = processed_traj["target_progress"]
                metadata = processed_traj["metadata"]
            else:
                frames = load_frames_from_npz(processed_traj["frames"])
                if self.config.pairwise_progress:
                    frames, progress, metadata = subsample_pairs_and_progress(
                        frames, self.config.max_frames, progress_pred_type=self.config.progress_pred_type
                    )
                else:
                    frames, progress, metadata = subsample_frames_and_progress(
                        frames, self.config.max_frames, progress_pred_type=self.config.progress_pred_type
                    )

            if strategy_used == DataGenStrat.DIFFERENT_TASK:
                # for different task, we use original language instruction, but
                # the video is from a different task
                lang_vector = traj["lang_vector"]
                task = traj["task"]

            if not self.config.pairwise_progress:
                frames, progress = pad_trajectory_to_max_frames_np(frames, progress, self.config.max_frames)

        if strategy_used == DataGenStrat.DIFFERENT_TASK:
            progress = [0.0] * len(progress)
    
        progress_traj = Trajectory(
            frames=frames,
            target_progress=progress,
            frames_shape=frames.shape if frames is not None else None,
            video_embeddings=video_embeddings,
            text_embedding=text_embedding,
            id=processed_traj["id"],
            task=task,
            lang_vector=lang_vector,
            data_source=processed_traj["data_source"],
            quality_label=processed_traj["quality_label"],
            is_robot=processed_traj["is_robot"],
            data_gen_strategy=strategy_used,
            metadata=metadata,
        )

        return ProgressSample(
            trajectory=progress_traj,
            sample_type="progress",
        )

    def _create_different_task_trajectory(self, chosen_traj: dict) -> dict | None:
        """Create a trajectory from a different task than the chosen trajectory.

        This function tries to find trajectories from different tasks.
        Returns None if no other tasks are available.

        Args:
            chosen_traj: The chosen trajectory dictionary

        Returns:
            Optional[Dict]: The rejected trajectory, or None if none available
        """
        # Find other tasks
        other_tasks = [task for task in self.optimal_by_task.keys() if task != chosen_traj["task"]]
        if other_tasks:
            other_task = random.choice(other_tasks)
            other_task_indices = self.optimal_by_task[other_task]

            if other_task_indices:
                other_idx = random.choice(other_task_indices)
                other_traj = self.dataset[other_idx]

                # Check if it's not the same trajectory
                if other_traj["id"] != chosen_traj["id"]:
                    return other_traj
        else:
            # Only one task available
            return None

        return None
