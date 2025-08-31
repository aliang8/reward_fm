#!/usr/bin/env python3
"""
PreferenceDataGenerator class for producing batches of preference prediction data.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Iterator, Union
from rfm.data.dataset_types import PreferenceSample, SimilaritySample, ProgressSample
from rfm.data.generators.base import BaseDataGenerator
from rfm.utils.logging import rank_0_print, timer


class PreferenceDataGenerator(BaseDataGenerator):
    """Data generator for producing batches of preference prediction data."""

    def __init__(self, config, is_evaluation=False, verbose=True):
        """Initialize PreferenceDataGenerator with configuration."""
        self.dataset_preference_ratio = config.dataset_preference_ratio
        self.preference_strategy_ratio: List[float] = config.preference_strategy_ratio

        super().__init__(config, is_evaluation, verbose=verbose)

        # Initialize preference dataset
        self._load_preference_dataset()

        rank_0_print(f"PreferenceDataGenerator initialized with {len(self.dataset)} total trajectories")

    def _create_rewind_trajectory(self, original_traj: Dict, rewind_length: Optional[int] = None) -> Dict:
        """Create a suboptimal trajectory by rewinding the original trajectory.

        This method creates a trajectory that goes forward then rewinds back:
        1. Selects start index in the first half of the original trajectory
        2. Selects end index in the latter half of the original trajectory
        3. Picks a rewind index between start and end
        4. Creates a forward segment from start index to end-1 (avoiding repetition)
        5. Creates a rewind segment by reversing from end-2 back to rewind_point (completely avoiding repetition)
        6. Concatenates forward + rewind to create the full trajectory
        7. Applies uniform subsampling to get the final num_frames
        8. Calculates progress relative to start index but out of total 64 frames

        Example:
        Original frames: [0, 1, 2, ... 63]
        Start index: 10
        End index: 30
        Rewind point: 25
        Rewind length: 5
        Forward frames: [9, 10, 11, ..., 28, 29] # we include the start index, but not the end index
        Rewind frames: [28, 27, 26, 25] # we include the rewind point, but not the last frame of the forward segment
        Combined frames: [9, 10, 11, ..., 28, 29, 28, 27, 26, 25]

        # Note: always start at 1, the denominator is (num_frames - start_idx)
        Forward progress: [1/54, 2/54, 3/54, ..., 29/54, 30/54]
        Rewind progress: [29/54, 28/54, 27/54, 26/54]
        Combined progress: [1/54, 2/54, 3/54, ..., 29/54, 29/54, 28/54, 27/54, 26/54]

        # We then apply subsampling to get num_frames frames
        # We use linspace subsampling to get evenly spaced frames, including the first and last frame

        Args:
            original_traj: Original trajectory dictionary
            rewind_length: Number of frames to rewind (default: random 1 to max_frames)
        """
        # Load frames from npz file
        frames_data = self._load_frames_from_npz(original_traj["frames"])

        # Get the number of frames
        if hasattr(frames_data, "shape"):
            num_frames = frames_data.shape[0]  # Use shape[0] for numpy array
        else:
            num_frames = len(frames_data)

        # Step 1: Select start and end indices
        # Start index is in the first half of the trajectory
        start_idx = random.randint(0, num_frames // 2 - 1)
        # End index is in the latter half of the trajectory
        end_idx = random.randint(num_frames // 2, num_frames)

        # Ensure we have enough frames between start and end
        while end_idx - start_idx < 5:
            start_idx = random.randint(0, num_frames // 2 - 1)
            end_idx = random.randint(num_frames // 2, num_frames)

        # Step 2: Select rewind index between start and end
        if rewind_length is None:
            # Pick rewind point randomly between start+1 and end-1
            # We want at least 1 frame forward and at least 1 frame rewind
            rewind_point = random.randint(start_idx + 1, end_idx - 1)
            rewind_length = end_idx - rewind_point
        else:
            # Ensure rewind_length is valid
            max_rewind = end_idx - start_idx - 1
            if rewind_length >= max_rewind:
                rewind_length = max_rewind
            if rewind_length < 1:
                rewind_length = 1
            rewind_point = start_idx + rewind_length

        # Step 3: Extract forward segment
        # Does not include end index to avoid
        forward_frames = frames_data[start_idx:end_idx]
        forward_indices = list(range(start_idx, end_idx))  # start to end-1

        # Step 4: Create rewind segment
        # NOTE: progress is relative to start index
        # Example: If start=10, rewind_point=25, end=40 (assuming 64 total frames):
        # Forward: [10, 11, 12, ..., 38, 39] (start to end-1, avoiding repetition)
        # Forward progress: [1/54, 2/54, 3/54, ..., 29/54, 30/54]
        # Rewind: [38, 37, 36, ..., 26, 25] (end-2 back to rewind_point+1)
        # Rewind progress: [29/54, 28/54, 27/54, ...] (going backwards from where forward left off)
        # Combined: [10, 11, 12, ..., 38, 39, 38, 37, ..., 26, 25]
        # Combined progress: [1/54, 2/54, 3/54, ..., 30/54, 29/54, 28/54, ...]

        # start from end-2 because we don't want to include the last frame of forward segment
        # end at rewind_point-1 because we want to include the first frame of rewind segment
        reverse_frames = frames_data[end_idx - 2 : rewind_point - 1 : -1]

        # Step 5: Combine forward and reverse segments
        if isinstance(forward_frames, np.ndarray):
            # If frames are numpy arrays, use concatenate
            combined_frames = np.concatenate([forward_frames, reverse_frames], axis=0)
        else:
            # If frames are lists, use regular concatenation
            combined_frames = forward_frames + reverse_frames

        # Step 6: Calculate progress for each frame position in the combined trajectory
        # Progress should represent position within the selected segment, starting from 1/64
        forward_progress = []
        for i in range(len(forward_indices)):  # 0 to len(forward_indices)-1
            # Progress starts at 1/(num_frames - start_idx) for first frame, increments by 1/(num_frames - start_idx) for each frame
            forward_progress.append((i + 1) / (num_frames - start_idx))  # Progress: 1/64, 2/64, 3/64, ...

        rewind_progress = forward_progress[::-1][1:rewind_length]

        # Combine progress values
        combined_progress = forward_progress + rewind_progress

        # Step 7: Apply linspace subsampling to get final num_frames
        # Use linspace for rewound trajectories to get predictable, evenly spaced frames
        # must include the first and last frame
        num_frames_to_sample = self.config.max_frames
        subsampled_frames, subsampled_indices = self._linspace_subsample_frames(combined_frames, num_frames_to_sample)

        # Step 8: Map the subsampled indices to the corresponding progress values
        # The subsampled_indices tell us which frames from the combined trajectory we're using
        subsampled_progress = [combined_progress[idx] for idx in subsampled_indices]

        metadata = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "rewind_point": rewind_point,
            "rewind_length": rewind_length,
            "subsampled_indices": subsampled_indices,
        }

        # Create new trajectory with rewind frames
        rewind_traj = original_traj.copy()
        rewind_traj["frames"] = subsampled_frames
        rewind_traj["frames_shape"] = subsampled_frames.shape
        rewind_traj["target_progress"] = subsampled_progress
        rewind_traj["metadata"] = metadata
        rewind_traj["quality_label"] = "rewound"
        return rewind_traj

    def _create_video_binned_trajectory(self, original_traj: Dict, num_bins: int = 10) -> Tuple[Dict, Dict]:
        """Create a preference sample by splitting a video into temporal bins and sampling from different bins.

        This strategy creates preference samples by:
        1. Splitting the original video into N temporal bins (e.g., 4 bins for a 32-frame video)
        2. Randomly selecting two different bins from the same video
        3. Creating a preference sample where one bin represents progress and the other represents regression

        **Example:**
        ```
        Original video: 32 frames
        Bins: [0-7], [8-15], [16-23], [24-31]

        Strategy 1: Compare early progress vs late progress
        - Chosen: frames [16-23] (bin 2, middle progress)
        - Rejected: frames [0-7] (bin 0, early progress)

        Strategy 2: Compare progress vs regression
        - Chosen: frames [24-31] (bin 3, final progress)
        - Rejected: frames [16-23] (bin 2, middle progress, but shown in reverse)

        Strategy 3: Compare adjacent bins with different progress
        - Chosen: frames [8-15] (bin 1, early-mid progress)
        - Rejected: frames [0-7] (bin 0, early progress)
        ```

        **Benefits:**
        - Teaches the model to recognize temporal progress within the same task
        - Helps distinguish between early, middle, and late stages of task completion
        - Creates diverse preference pairs from the same video without external data
        - Useful for learning fine-grained temporal dynamics and progress indicators

        Args:
            original_traj: Original trajectory dictionary containing video frames
            num_bins: Number of temporal bins to split the video into (default: 10)

        Returns:
            Tuple[Dict, Dict]: (chosen_trajectory, rejected_trajectory) where both are modified
            trajectories with frames from different bins and updated metadata

        Raises:
            ValueError: If video is too short to create meaningful bins
            RuntimeError: If video binning fails for any reason
        """
        # Load frames from npz file
        frames_data = self._load_frames_from_npz(original_traj["frames"])

        # Get the number of frames
        if hasattr(frames_data, "shape"):
            num_frames = frames_data.shape[0]
        else:
            num_frames = len(frames_data)

        if num_frames < num_bins * 2:
            raise ValueError(f"Video too short ({num_frames} frames) to create {num_bins} meaningful bins")

        # Calculate bin size and boundaries
        bin_size = num_frames // num_bins
        bin_boundaries = []
        for i in range(num_bins):
            start = i * bin_size
            end = start + bin_size if i < num_bins - 1 else num_frames
            bin_boundaries.append((start, end))

        # Randomly select two different bins
        bin_indices = list(range(num_bins))
        chosen_bin_idx = random.choice(bin_indices)
        bin_indices.remove(chosen_bin_idx)
        rejected_bin_idx = random.choice(bin_indices)

        # Extract frames from the chosen bin (this will be the "chosen" trajectory)
        chosen_start, chosen_end = bin_boundaries[chosen_bin_idx]
        chosen_frames = frames_data[chosen_start:chosen_end]

        chosen_progress = []
        for i in range(len(chosen_frames)):
            chosen_progress.append((i + 1) / (len(frames_data) - chosen_start))

        # Extract frames from the rejected bin (this will be the "rejected" trajectory)
        rejected_start, rejected_end = bin_boundaries[rejected_bin_idx]
        rejected_frames = frames_data[rejected_start:rejected_end]

        rejected_progress = []
        for i in range(len(rejected_frames)):
            rejected_progress.append((i + 1) / (len(frames_data) - rejected_start))

        # Apply uniform subsampling to both bins to ensure consistent frame counts
        # Use uniform subsampling for real trajectories (not rewound)
        num_frames_to_sample = self.config.max_frames
        chosen_frames, chosen_indices = self._linspace_subsample_frames(chosen_frames, num_frames_to_sample)
        rejected_frames, rejected_indices = self._linspace_subsample_frames(rejected_frames, num_frames_to_sample)

        # Calculate progress for each bin relative to the original trajectory
        chosen_progress = [chosen_progress[idx] for idx in chosen_indices]
        rejected_progress = [rejected_progress[idx] for idx in rejected_indices]

        # Create the chosen trajectory (from chosen bin)
        chosen_traj = original_traj.copy()
        chosen_traj["frames"] = chosen_frames
        chosen_traj["frames_shape"] = chosen_frames.shape
        chosen_traj["target_progress"] = chosen_progress
        chosen_traj["metadata"] = {
            "start_idx": chosen_start,
            "end_idx": chosen_end,
            "chosen_bin_idx": chosen_bin_idx,
            "rejected_bin_idx": rejected_bin_idx,
        }

        rejected_traj = original_traj.copy()
        rejected_traj["frames"] = rejected_frames
        rejected_traj["frames_shape"] = rejected_frames.shape
        rejected_traj["target_progress"] = rejected_progress
        rejected_traj["metadata"] = {
            "start_idx": rejected_start,
            "end_idx": rejected_end,
            "chosen_bin_idx": chosen_bin_idx,
            "rejected_bin_idx": rejected_bin_idx,
        }

        return chosen_traj, rejected_traj

    def _create_preference_sample_from_dataset(self) -> PreferenceSample:
        """Create a preference sample from the loaded preference dataset."""
        if not self.preferences:
            raise ValueError("No preferences loaded from dataset")

        # For now, return a simple preference sample
        # This can be enhanced later when we have actual preference data
        preference = random.choice(self.preferences)

        # This is a placeholder - would need to be implemented based on actual preference data structure
        raise NotImplementedError("Preference sample creation from dataset not yet implemented")

    def _load_preference_dataset(self):
        """Load the preference dataset from disk or hub if provided."""
        self.preferences = []

        # For now, we'll use empty preferences since the config structure has changed
        # This can be updated later if needed
        rank_0_print("No preference dataset provided, will use random sampling for preferences")
        return

    def _create_preference_sample(self) -> PreferenceSample:
        """Create a preference prediction sample: chosen vs rejected where chosen is preferred.

        This method can create preference samples from two sources:

        **Dataset Source:**
        - Uses pre-existing preference data from the loaded preference dataset
        - Good for learning from curated, high-quality preference examples
        - Controlled by config.dataset_preference_ratio

        **Data Augmentation Strategies:**
        When not using dataset preferences, delegates to _create_preference_sample_with_strategies()
        which implements various strategies for generating rejected trajectories.

        Returns:
            PreferenceSample: A preference sample with chosen (preferred) vs rejected
            (suboptimal) trajectories and associated metadata
        """

        with timer("create_preference_sample", verbose=False):
            if random.random() < self.dataset_preference_ratio and self.preferences:
                # Use preference trajectories from dataset
                return self._create_preference_sample_from_dataset()
            else:
                return self._create_preference_sample_with_strategies()

    def _create_preference_sample_with_strategies(self) -> PreferenceSample:
        """Create a preference prediction sample using various rejected trajectory generation strategies.

        This method implements four different strategies for generating rejected trajectories
        to create diverse and robust preference learning data. The strategy is chosen
        probabilistically according to self.preference_strategy_ratio.

        **Strategy 1: Rewind Same Task**
        - Creates a suboptimal trajectory by rewinding the chosen trajectory
        - Same task, different trajectory ID, artificially generated suboptimal behavior
        - Good for learning task-specific failure modes and temporal dynamics
        - Example: Forward progress [0→1→2→3] + rewind [2→1] = [0→1→2→3→2→1]

        **Strategy 2: Suboptimal/Failure Same Task**
        - Uses existing suboptimal/failure trajectories from the same task
        - Same task, different trajectory ID, real failure examples
        - Good for learning from actual failure patterns and task-specific suboptimal behaviors
        - Example: Compare successful "open door" vs failed "open door" attempts

        **Strategy 3: Different Task**
        - Uses trajectories from completely different tasks
        - Different task, can be chosen or suboptimal
        - Good for learning cross-task generalization and what makes trajectories "good"
          across different contexts
        - Example: Compare "open door" (successful) vs "press button" (successful)

        **Strategy 4: Video Binned**
        - Splits a single video into temporal bins and compares different bins
        - Same task, same video, different temporal segments
        - Good for learning temporal progress within the same task and fine-grained
          temporal dynamics
        - Example: Compare early progress [frames 0-7] vs late progress [frames 24-31]

        **Fallback Behavior:**
        If any strategy fails (e.g., no suboptimal trajectories available, video too short),
        the system automatically falls back to the rewind strategy to ensure robust
        data generation.

        Returns:
            PreferenceSample: A preference sample with chosen (preferred) vs rejected
            (suboptimal) trajectories and associated metadata

        Raises:
            ValueError: If no chosen trajectories are available for preference generation
            RuntimeError: If all strategies fail and fallback rewind also fails
        """

        # Use preprocessed chosen trajectories from index maps
        if not self.optimal_by_task:
            raise ValueError("No chosen trajectories found for preference generation")

        # Get a random task and chosen trajectory from it
        task_name = random.choice(list(self.optimal_by_task.keys()))

        optimal_indices = self.optimal_by_task[task_name]
        while not optimal_indices:
            task_name = random.choice(list(self.optimal_by_task.keys()))
            optimal_indices = self.optimal_by_task[task_name]

        chosen_idx = random.choice(optimal_indices)
        chosen_traj = self.dataset[chosen_idx]

        # Initialize variables for strategy selection
        rejected_traj = None
        strategy_used = None

        if random.random() < self.preference_strategy_ratio[0]:
            # Strategy 1: Use rewind-generated suboptimal trajectory from same task
            rejected_traj = self._create_rewind_trajectory(chosen_traj)
            strategy_used = "rewind_same_task"

        elif random.random() < self.preference_strategy_ratio[0] + self.preference_strategy_ratio[1]:
            # Strategy 2: Use random suboptimal trajectory from same task
            rejected_traj = self._create_same_task_suboptimal_trajectory(chosen_traj)
            if rejected_traj is not None:
                strategy_used = "suboptimal_same_task"

        elif (
            random.random()
            < self.preference_strategy_ratio[0] + self.preference_strategy_ratio[1] + self.preference_strategy_ratio[2]
        ):
            # Strategy 3: Use trajectory from different task (can be chosen or suboptimal)
            rejected_traj = self._create_different_task_trajectory(chosen_traj)
            if rejected_traj is not None:
                strategy_used = "different_task"

        else:
            # Strategy 4: Create preference sample from different bins of the same video
            try:
                chosen_traj, rejected_traj = self._create_video_binned_trajectory(
                    chosen_traj, num_bins=self.config.num_bins
                )
                strategy_used = "video_binned"
            except Exception as e:
                rank_0_print(f"Video binning failed: {e}, will fall back to rewind")

        # Fallback: If any strategy failed to produce a rejected trajectory, use rewind
        if rejected_traj is None:
            rejected_traj = self._create_rewind_trajectory(chosen_traj)
            strategy_used = "rewind_same_task"

        # ===============================================================
        # Subsample the chosen trajectory to max_frames
        # ===============================================================
        if isinstance(chosen_traj["frames"], str):
            chosen_traj["frames"] = self._load_frames_from_npz(chosen_traj["frames"])

        chosen_frames, chosen_progress, chosen_metadata = self._subsample_frames_and_progress(chosen_traj["frames"])
        if "metadata" in chosen_traj:
            chosen_metadata.update(chosen_traj["metadata"])

        # ===============================================================
        # Subsample the rejected trajectory to max_frames
        # ===============================================================

        if isinstance(rejected_traj["frames"], str):
            rejected_traj["frames"] = self._load_frames_from_npz(rejected_traj["frames"])

        if "rewind" not in strategy_used:
            # try subsampling the rejected trajectory
            rejected_frames, rejected_progress, rejected_metadata = self._subsample_frames_and_progress(
                rejected_traj["frames"]
            )
            if "metadata" in rejected_traj:
                rejected_metadata.update(rejected_traj["metadata"])

        else:
            rejected_frames = rejected_traj["frames"]
            rejected_progress = rejected_traj["target_progress"]
            rejected_metadata = rejected_traj["metadata"]

        # If our strategy is different task, make sure the rejected trajectory has 0 progress
        if strategy_used == "different_task":
            rejected_progress = [0.0] * len(rejected_progress)

        # Create preference sample structure
        sample = PreferenceSample(
            # Create Trajectory objects for chosen and rejected
            chosen_trajectory=Trajectory(
                frames=chosen_frames,
                frames_shape=chosen_frames.shape,
                id=chosen_traj["id"],
                task=chosen_traj["task"],
                lang_vector=chosen_traj["lang_vector"],
                data_source=chosen_traj["data_source"],
                quality_label=chosen_traj.get("quality_label"),
                is_robot=chosen_traj["is_robot"],
                target_progress=chosen_progress,
                data_gen_strategy="subsample_task",
                metadata=chosen_metadata,
            ),
            rejected_trajectory=Trajectory(
                frames=rejected_frames,
                frames_shape=rejected_frames.shape,
                id=rejected_traj["id"],
                task=rejected_traj["task"],
                lang_vector=rejected_traj["lang_vector"],
                data_source=rejected_traj["data_source"],
                quality_label=rejected_traj["quality_label"],
                is_robot=rejected_traj["is_robot"],
                target_progress=rejected_progress,
                data_gen_strategy=strategy_used,
                metadata=rejected_metadata,
            ),
        )
        return sample

    def _create_same_task_suboptimal_trajectory(self, chosen_traj: Dict) -> Optional[Dict]:
        """Create a suboptimal trajectory from the same task as the chosen trajectory.

        This function tries to find an existing suboptimal/failure trajectory from the same task.
        Returns None if no suboptimal trajectories are available.

        Args:
            chosen_traj: The chosen (optimal) trajectory dictionary

        Returns:
            Optional[Dict]: The rejected trajectory, or None if none available
        """
        task_name = chosen_traj["task"]

        # Try to find suboptimal trajectories from the same task
        same_task_suboptimal_indices = self.suboptimal_by_task.get(task_name, [])
        same_task_suboptimal = [
            self.dataset[idx] for idx in same_task_suboptimal_indices if self.dataset[idx]["id"] != chosen_traj["id"]
        ]

        if same_task_suboptimal:
            return random.choice(same_task_suboptimal)
        else:
            return None

    def _create_different_task_trajectory(self, chosen_traj: Dict) -> Optional[Dict]:
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


class VQADataGenerator(PreferenceDataGenerator):
    def __init__(self, config, is_evaluation=False):
        self.progress_ratio = config.progress_ratio
        super().__init__(config, is_evaluation)

    def _create_progress_sample(self) -> ProgressSample:
        """Create a progress sample."""
        # Get a random task and optimal trajectory from it
        task_name = random.choice(list(self.optimal_by_task.keys()))
        optimal_idx = random.choice(self.optimal_by_task[task_name])
        traj = self.dataset[optimal_idx]

        # Choose negative generation strategy using configured ratios
        r = random.random()
        rewind_ratio, subopt_ratio, diff_ratio = 0.2, 0.2, 0.2

        strategy_used = None
        if r < 0.6:
            if r < rewind_ratio:
                strategy_choice = 0
            elif r < rewind_ratio + subopt_ratio:
                strategy_choice = 1
            else:
                strategy_choice = 2

            if strategy_choice == 0:
                # Strategy 1: Use rewind-generated suboptimal trajectory from same task
                traj = self._create_rewind_trajectory(traj)
                strategy_used = "rewind_same_task"
            elif strategy_choice == 1:
                # Strategy 2: Use random suboptimal trajectory from same task
                traj = self._create_same_task_suboptimal_trajectory(traj)
                if traj is not None:
                    strategy_used = "suboptimal_same_task"
                else:
                    # Fall back to rewind if no same-task suboptimal trajectories
                    traj = self._create_rewind_trajectory(traj)
                    strategy_used = "rewind_same_task"
            else:
                # Strategy 3: Use trajectory from different task (can be optimal or suboptimal)
                traj = self._create_different_task_trajectory(traj)
                if traj is not None:
                    strategy_used = "different_task"
                else:
                    # Fall back to rewind if no other tasks available
                    traj = self._create_rewind_trajectory(traj)
                    strategy_used = "rewind_same_task"

            # Handle negative trajectory frames - could be from dataset (npz) or rewind-generated (numpy)
            if isinstance(traj, dict) and "frames" in traj:
                if isinstance(traj["frames"], str) and traj["frames"].endswith(".npz"):
                    # Regular trajectory with npz path
                    traj_frames = self._load_frames_from_npz(traj["frames"])
                elif isinstance(traj["frames"], np.ndarray):
                    # Rewind trajectory with numpy array
                    traj_frames = traj["frames"]
                else:
                    raise ValueError(f"Unexpected frames format in negative trajectory: {type(traj['frames'])}")
            else:
                raise ValueError(f"Invalid negative trajectory format: {type(traj)}")

        else:
            # Get frames from npz files and uniformly subsample
            traj_frames_full = self._get_trajectory_frames(optimal_idx)

            # Uniformly subsample the trajectory to num_frames (default 8)
            num_frames_to_sample = getattr(self.config, "max_frames", 8)
            traj_frames, traj_indices = self._linspace_subsample_frames(traj_frames_full, num_frames_to_sample)

            # Calculate progress relative to the original trajectory (64 frames)
            traj_progress = [idx / (len(traj_frames_full) - 1) for idx in traj_indices]

            # Store original frame positions for reference
            traj_original_positions = [idx for idx in traj_indices]

            # Update traj with subsampled frames and progress
            traj = traj.copy()
            traj["frames"] = traj_frames
            traj["frames_shape"] = traj_frames.shape
            traj["metadata"] = traj.get("metadata", {}).copy()
            traj["metadata"]["subsampled_generated"] = True
            traj["metadata"]["subsampled_progress"] = traj_progress
            traj["metadata"]["num_frames_subsampled"] = num_frames_to_sample
            traj["metadata"]["original_num_frames"] = len(traj_frames_full)
            traj["metadata"]["original_frame_positions"] = traj_original_positions

            # Ensure trajectory has exactly max_frames by padding if needed
            traj_frames_padded, traj_progress_padded = self._pad_trajectory_to_max_frames(
                traj_frames, traj_progress, num_frames_to_sample
            )

            # Update traj with padded frames and progress
            traj["frames"] = traj_frames_padded
            traj["frames_shape"] = traj_frames_padded.shape
            traj["metadata"]["subsampled_progress"] = traj_progress_padded

        # Calculate target progress for the trajectory
        # Use subsampled progress if available, otherwise calculate from frames
        if traj.get("metadata", {}).get("subsampled_generated"):
            target_progress = traj["metadata"]["subsampled_progress"]
        else:
            target_progress = self._calculate_target_progress(traj, traj_frames)

        # Get frame shapes from the trajectory (already padded if needed)
        traj_frames_shape = traj.get("frames_shape")
        if isinstance(traj_frames_shape, list):
            traj_frames_shape = tuple(traj_frames_shape)

        # Create progress sample
        sample = ProgressSample(
            frames=traj["frames"],
            frames_shape=traj_frames_shape,
            task=traj["task"],
            target_progress=target_progress,
            quality_label=traj.get("quality_label"),
            sample_type="progress",
        )

        return sample
