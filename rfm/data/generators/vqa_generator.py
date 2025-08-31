import random
import numpy as np
from rfm.data.vqa_batch_collator import ProgressSample
from rfm.data.generators.pref import PreferenceDataGenerator
from rfm.data.generators.sim import SimilarityDataGenerator
from rfm.data.dataset_types import SimilaritySample, ProgressSample, Trajectory
from rfm.data.generators.generator import DataGenerator
from rfm.data.generators.base import BaseDataGenerator
from rfm.utils.logging import rank_0_print
from typing import Dict, Optional, Tuple, List


class ProgressDataGenerator(BaseDataGenerator):
    """Data generator for producing batches of progress data."""

    def __init__(self, config, is_evaluation=False):
        """Initialize ProgressDataGenerator with configuration."""
        # Progress strategy ratio hardcoded for now
        self.progress_strategy_ratio = [0.2, 0.2, 0.2]  # [rewind, suboptimal_same_task, different_task]
        super().__init__(config, is_evaluation)

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

        if num_frames < 4:
            # If trajectory is too short, just return the original
            return original_traj

        # Step 1: Select start and end indices
        # Start index is in the first half of the trajectory
        start_idx = random.randint(0, num_frames // 2 - 1)
        # End index is in the latter half of the trajectory
        end_idx = random.randint(num_frames // 2, num_frames)

        # Ensure we have enough frames between start and end
        while end_idx - start_idx < 3:
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
        num_frames_to_sample = getattr(self.config, "max_frames", 8)
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

    def _subsample_frames_and_progress(self, frames: np.ndarray, max_frames: int) -> Tuple[np.ndarray, List[float]]:
        # For chosen trajectory, sample start and end indices to create a segment
        # This makes the progress calculation consistent with rewind trajectories
        num_frames_total = len(frames)

        # Select start and end indices for the chosen trajectory segment
        # Start index is in the first half of the trajectory
        start_idx = random.randint(0, num_frames_total // 2 - 1)
        # End index is in the latter half of the trajectory
        end_idx = random.randint(num_frames_total // 2, num_frames_total)

        # Ensure we have enough frames between start and end
        while end_idx - start_idx < 3:
            start_idx = random.randint(0, num_frames_total // 2 - 1)
            end_idx = random.randint(num_frames_total // 2, num_frames_total)

        # Extract the chosen segment
        segment_frames = frames[start_idx:end_idx]
        segment_indices = list(range(start_idx, end_idx))

        # Calculate progress for the full segment first (like forward indices in rewind)
        # Progress should represent position within the selected segment, starting from 1/64
        segment_progress = []
        for i in range(len(segment_indices)):
            segment_progress.append((i + 1) / (num_frames_total - start_idx))

        # Uniformly subsample the chosen trajectory segment to num_frames
        frames, indices = self._linspace_subsample_frames(segment_frames, self.config.max_frames)

        # Map the subsampled indices to the corresponding progress values from the full segment
        # The chosen_indices tell us which frames from the segment we're using
        progress = [segment_progress[idx] for idx in indices]

        # Ensure both trajectories have exactly max_frames by padding if needed
        # Pad by repeating the first frame and first progress value
        frames, progress = self._pad_trajectory_to_max_frames(frames, progress, self.config.max_frames)

        metadata = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "subsampled_indices": indices,
        }
        return frames, progress, metadata

    def _create_progress_sample(self):
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
        p = random.random()
        if p < self.progress_strategy_ratio[0]:
            # Strategy 1: Use rewind-generated suboptimal trajectory from same task
            rejected_traj = self._create_rewind_trajectory(chosen_traj)
            strategy_used = "rewind_same_task"

        elif p < self.progress_strategy_ratio[0] + self.progress_strategy_ratio[1]:
            # Strategy 2: Use random suboptimal trajectory from same task
            same_task_suboptimal_indices = self.suboptimal_by_task.get(task_name, [])
            same_task_suboptimal = [
                self.dataset[idx]
                for idx in same_task_suboptimal_indices
                if self.dataset[idx]["id"] != chosen_traj["id"]
            ]
            if same_task_suboptimal:
                rejected_traj = random.choice(same_task_suboptimal)
                strategy_used = "suboptimal_same_task"

        # elif random.random() < self.progress_strategy_ratio[0] + self.progress_strategy_ratio[1] + self.progress_strategy_ratio[2]:
        else:
            # Strategy 3: Use trajectory from different task (can be chosen or suboptimal)
            other_tasks = [task for task in self.optimal_by_task.keys() if task != chosen_traj["task"]]
            if other_tasks:
                other_task = random.choice(other_tasks)
                other_task_indices = self.optimal_by_task[other_task]
                if other_task_indices:
                    other_idx = random.choice(other_task_indices)
                    other_traj = self.dataset[other_idx]
                    # Check if it's not the same trajectory
                    if other_traj["id"] != chosen_traj["id"]:
                        rejected_traj = other_traj
                        strategy_used = "different_task"

        # Video binning strategy is not used for progress data for now
        # else:
        #     # Strategy 4: Create preference sample from different bins of the same video
        #     try:
        #         chosen_traj, rejected_traj = self._create_video_binned_trajectory(chosen_traj, num_bins=self.config.num_bins)
        #         strategy_used = "video_binned"
        #     except Exception as e:
        #         rank_0_print(f"Video binning failed: {e}, will fall back to rewind")

        # Fallback: If any strategy failed to produce a rejected trajectory, use rewind
        if rejected_traj is None:
            rejected_traj = self._create_rewind_trajectory(chosen_traj)
            strategy_used = "rewind_same_task"

        # ===============================================================
        # Subsample the chosen trajectory to max_frames
        # ===============================================================
        if isinstance(chosen_traj["frames"], str):
            chosen_traj["frames"] = self._load_frames_from_npz(chosen_traj["frames"])

        chosen_frames, chosen_progress, chosen_metadata = self._subsample_frames_and_progress(
            chosen_traj["frames"], self.config.max_frames
        )
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
                rejected_traj["frames"], self.config.max_frames
            )
            if "metadata" in rejected_traj:
                rejected_metadata.update(rejected_traj["metadata"])

        else:
            rejected_frames = rejected_traj["frames"]
            rejected_progress = rejected_traj["target_progress"]
            rejected_metadata = rejected_traj["metadata"]

        if p > sum(self.progress_strategy_ratio):
            # Use chosen trajectory
            sample = ProgressSample(
                trajectory=Trajectory(
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
                )
            )
        else:
            # Use rejected trajectory
            sample = ProgressSample(
                trajectory=Trajectory(
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


class VQADataGenerator(DataGenerator):
    """Data generator for producing batches of VQA data."""

    def __init__(self, config, is_evaluation=False):
        """Initialize DataGenerator with configuration."""
        self.config = config
        self.is_evaluation = is_evaluation

        # Initialize the individual generators
        self.preference_generator = PreferenceDataGenerator(config, is_evaluation)
        self.similarity_generator = SimilarityDataGenerator(config, is_evaluation)
        self.progress_generator = ProgressDataGenerator(config, is_evaluation)

        # Set the ratio for sampling between preference and similarity
        self.preference_ratio = config.preference_ratio
        self.progress_ratio = config.progress_ratio
        self.similarity_ratio = 1.0 - config.preference_ratio - config.progress_ratio

        rank_0_print(
            f"DataGenerator initialized with preference_ratio={self.preference_ratio:.2f}, progress_ratio={self.progress_ratio:.2f}, similarity_ratio={self.similarity_ratio:.2f}"
        )

    def _create_sample(self):
        """Create a sample based on the configured ratios."""
        if random.random() < self.progress_ratio:
            return self._create_progress_sample()
        elif random.random() < self.preference_ratio + self.progress_ratio:
            return self.preference_generator._create_preference_sample()
        else:
            return self.similarity_generator._create_similarity_sample()

    def _create_progress_sample(self):
        """Create a progress sample using the progress generator."""
        return self.progress_generator._create_progress_sample()


def test():
    """Test the VQABatchCollator with generated samples."""
    # Create a mock config for testing
    from dataclasses import dataclass
    from typing import List

    @dataclass
    class MockDataConfig:
        train_datasets: List[str] = None
        train_subsets: List[str] = None
        eval_datasets: List[str] = None
        eval_subsets: List[str] = None
        preference_ratio: float = 1.0
        progress_ratio: float = 0.0
        similarity_ratio: float = 0.0
        dataset_preference_ratio: float = 0.7
        shuffle: bool = True
        seed: int = 42
        num_proc: int = 4
        max_frames: int = 8  # Use 8 frames for testing the new subsampling logic
        force_reprocess: bool = False
        dataloader_pin_memory: bool = False
        dataloader_num_workers: int = 0
        model_type: str = "default"
        preference_strategy_ratio: List[float] = None

    @dataclass
    class MockConfig:
        data: MockDataConfig = None
        debug: bool = False

    # Create mock config
    mock_data_config = MockDataConfig(
        train_datasets=["abraranwar/libero_rfm"],
        train_subsets=["libero_90"],
        preference_ratio=0.5,
        progress_ratio=0.5,
        similarity_ratio=0.0,
        preference_strategy_ratio=[0.8, 0.1, 0.1],
        shuffle=True,
        seed=42,
        num_proc=4,
        max_frames=8,  # Use 8 frames for testing the new subsampling logic
        force_reprocess=False,
        model_type="default",
    )

    mock_config = MockConfig(data=mock_data_config, debug=False)

    # Create data generator with mock config
    generator = VQADataGenerator(config=mock_data_config)

    # Test the infinite dataset
    rank_0_print("Testing InfiniteDataGeneratorDataset...")
    from rfm.data.dataset import InfiniteDataGeneratorDataset

    infinite_dataset = InfiniteDataGeneratorDataset(generator)

    preference_count = 0
    similarity_count = 0
    progress_count = 0

    for i in range(50):
        sample = infinite_dataset[i]
        if sample.sample_type == "preference":
            preference_count += 1
        elif sample.sample_type == "similarity":
            similarity_count += 1
        elif sample.sample_type == "progress":
            progress_count += 1
        rank_0_print(f"Sample {i}: {sample.sample_type}")

    rank_0_print(
        f"Generated {preference_count} preference samples, {progress_count} progress samples, and {similarity_count} similarity samples"
    )
    rank_0_print(
        f"Expected ratio: {generator.preference_ratio:.1f} preference, {generator.progress_ratio:.1f} progress, {generator.similarity_ratio:.1f} similarity"
    )


if __name__ == "__main__":
    test()
