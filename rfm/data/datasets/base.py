#!/usr/bin/env python3
import collections
import json
import os
from typing import Any
import torch

from datasets import Dataset, concatenate_datasets
from rfm.data.datasets.helpers import load_dataset_success_percent
from rfm.data.dataset_category import DATASET_MAP, DATA_SOURCE_CATEGORY, get_paired_ds
from rfm.utils.distributed import banner
from rfm.utils.logger import get_logger

logger = get_logger()


def resolve_dataset_keys(dataset_keys: list[str] | list[list[str]], split: str) -> list[str] | list[list[str]]:
    """
    Resolve dataset keys through DATASET_MAP.

    Args:
        dataset_keys: List of dataset keys (e.g., ["mw", "oxe"]) or actual dataset names.
                     Can also be a list of lists for cases like similarity_score where multiple
                     datasets need to be grouped together (e.g., [["human_ds", "robot_ds"], ["other_ds"]]).
        split: Either "train" or "eval"

    Returns:
        List of resolved dataset names, combining all datasets from the keys.
        If input is a list of lists, returns a list of lists (each inner list resolved separately).
        If input is a flat list, returns a flat list.
    """
    # Check if this is a list of lists (nested structure)
    if dataset_keys and isinstance(dataset_keys[0], list):
        # Handle nested lists: resolve each inner list separately
        resolved_datasets = []
        for key_group in dataset_keys:
            resolved_group = []
            for key in key_group:
                if key in DATASET_MAP:
                    # Key found in DATASET_MAP, resolve it
                    if split in DATASET_MAP[key]:
                        resolved_group.extend(DATASET_MAP[key][split])
                    else:
                        logger.warning(f"Key '{key}' found in DATASET_MAP but no '{split}' split defined. Skipping.")
                else:
                    # Not a key, assume it's already a dataset name
                    resolved_group.append(key)
            resolved_datasets.append(resolved_group)
        return resolved_datasets

    # Handle flat list (original behavior)
    resolved_datasets = []
    for key in dataset_keys:
        if key in DATASET_MAP:
            # Key found in DATASET_MAP, resolve it
            if split in DATASET_MAP[key]:
                resolved_datasets.extend(DATASET_MAP[key][split])
            else:
                logger.warning(f"Key '{key}' found in DATASET_MAP but no '{split}' split defined. Skipping.")
        else:
            # Not a key, assume it's already a dataset name
            resolved_datasets.append(key)
    return resolved_datasets


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, config, is_evaluation=False):
        self.config = config
        self.is_evaluation = is_evaluation

        # Choose datasets based on whether this is for evaluation or training
        if is_evaluation and config.eval_datasets:
            dataset_keys = config.eval_datasets
            split = "eval"
        else:
            dataset_keys = config.train_datasets
            split = "train"

        self.datasets = resolve_dataset_keys(dataset_keys, split)

        # Load dataset-specific success cutoff map if available
        self.dataset_success_cutoff_map = {}
        if hasattr(config, "dataset_success_cutoff_file") and config.dataset_success_cutoff_file:
            self.dataset_success_cutoff_map = load_dataset_success_percent(config.dataset_success_cutoff_file)

        # Load trajectory dataset
        self.dataset, self._combined_indices = self._load_all_datasets()

        # Apply all filters simultaneously
        excluded_keywords = ["rings", "hands", "flick"]
        min_frames = config.min_frames_per_trajectory

        # Check if we're in progress_only mode (sample_type_ratio == [0, 1, 0])
        # In progress_only mode, filter to only include successful trajectories
        filter_successful_only = False
        if config.sample_type_ratio == [0, 1, 0] and not is_evaluation:
            filter_successful_only = True
            logger.info(
                "Progress-only mode detected (sample_type_ratio=[0, 1, 0]), filtering to only successful trajectories"
            )

        logger.info(f"Filtering dataset with {len(self.dataset)} total trajectories")
        self.dataset, self._combined_indices = self._filter_dataset(
            excluded_keywords=excluded_keywords,
            min_frames=min_frames,
            dataset=self.dataset,
            combined_indices=self._combined_indices,
            filter_successful_only=filter_successful_only,
        )
        logger.info(f"Dataset filtered with {len(self.dataset)} total trajectories")

        # Filter out trajectories based on multiple criteria (build indices first, then filter once)
        self.dataset, self._combined_indices = self._filter_task_based_criteria(
            dataset=self.dataset,
            combined_indices=self._combined_indices,
        )

        # Set cached fields after filtering
        self._cached_ids = self.dataset["id"]
        self._cached_is_robot = self.dataset["is_robot"]

        logger.info(f"Dataset loaded with {len(self.dataset)} total trajectories")

        # Initialize resampling stats containers shared by subclasses
        self._resample_attempt_stats: dict[str, collections.defaultdict[str, list[int]]] = {
            "preference": collections.defaultdict(list),
            "progress": collections.defaultdict(list),
            "similarity": collections.defaultdict(list),
        }
        self._resample_dataset_attempt_stats: dict[str, collections.defaultdict[str, list[int]]] = {
            "preference": collections.defaultdict(list),
            "progress": collections.defaultdict(list),
            "similarity": collections.defaultdict(list),
        }

    def __len__(self):
        return len(self.dataset)

    def _load_all_datasets(self):
        """Load trajectory dataset using preprocessed index-based cache.

        Returns:
            tuple: (dataset, combined_indices)
                - dataset: The loaded and concatenated dataset
                - combined_indices: Dictionary of combined indices
        """
        cache_dir = os.environ.get("RFM_PROCESSED_DATASETS_PATH", "")
        if not cache_dir:
            raise ValueError(
                "RFM_PROCESSED_DATASETS_PATH environment variable not set. Please set it to the directory containing your processed datasets."
            )
        cache_type = "evaluation" if self.is_evaluation else "training"

        # Check if preprocessed cache exists
        if os.path.exists(cache_dir):
            logger.debug(f"Found preprocessed cache at {cache_dir}, loading {cache_type} datasets...")

            dataset, combined_indices = self._load_preprocessed_cache(cache_dir, is_training=not self.is_evaluation)

            logger.debug(
                f"Successfully loaded preprocessed {cache_type} datasets with {len(dataset)} trajectory indices"
            )

            return dataset, combined_indices
        else:
            # If no cache exists, we need to run the preprocessor first
            logger.warning("No preprocessed cache found. Please run preprocess_datasets.py first to create the cache.")
            raise RuntimeError(
                "Dataset preprocessing required. Please run:\n"
                "uv run scripts/preprocess_datasets.py\n"
                "This will create the necessary index-based cache for efficient data loading."
            )

    def _get_available_datasets(self, cache_dir: str):
        """Check which datasets are available in the cache.

        Returns:
            tuple: (available_datasets, missing_datasets)
                - available_datasets: List of (dataset_path, individual_cache_dir) tuples
                - missing_datasets: List of dataset_path strings that are missing
        """
        available_datasets = []
        missing_datasets = []

        for dataset_path in self.datasets:
            # The preprocessing script creates individual cache directories for each dataset
            individual_cache_dir = os.path.join(cache_dir, dataset_path.replace("/", "_").replace(":", "_"))

            if os.path.exists(individual_cache_dir):
                info_file = os.path.join(individual_cache_dir, "dataset_info.json")
                if os.path.exists(info_file):
                    try:
                        with open(info_file) as f:
                            json.load(f)

                        available_datasets.append((dataset_path, individual_cache_dir))
                        logger.debug(f"Found cache: {individual_cache_dir}")
                    except:
                        logger.warning(f"Cache info file corrupted, skipping: {individual_cache_dir}")
                        continue
                else:
                    logger.debug(f"No info file found, skipping: {individual_cache_dir}")
                    continue
            else:
                missing_datasets.append(dataset_path)
                logger.debug(f"Missing cache: {individual_cache_dir}")

        # Warn about missing datasets
        if missing_datasets:
            logger.warning("⚠️  Warning: The following configured datasets are not available in the cache:")
            for dataset_path in missing_datasets:
                logger.warning(f"    ❌ {dataset_path}")
            logger.warning("  Available datasets will be loaded, but some configured data may be missing.")

        if not available_datasets:
            raise RuntimeError(
                f"No configured datasets are available in the cache. "
                f"Please run preprocess_datasets.py to create the cache for: {self.datasets}"
            )

        logger.debug(f"Summary: {len(available_datasets)} available, {len(missing_datasets)} missing")

        return available_datasets, missing_datasets

    def _load_datasets(self, available_datasets: list[tuple[str, str]]):
        """Load datasets from cache and return them along with per-dataset index mappings.

        Args:
            available_datasets: List of (dataset_path, individual_cache_dir) tuples

        Returns:
            tuple: (loaded_datasets, dataset_indices_list)
                - loaded_datasets: List of loaded Dataset objects
                - dataset_indices_list: List of index dictionaries, one per dataset (with original indices, not offset)
        """
        loaded_datasets = []
        dataset_indices_list = []

        for dataset_path, individual_cache_dir in available_datasets:
            # Load the processed dataset
            dataset_cache_dir = os.path.join(individual_cache_dir, "processed_dataset")
            if not os.path.exists(dataset_cache_dir):
                logger.warning(f"Processed dataset not found at {dataset_cache_dir}, skipping...")
                continue

            dataset = Dataset.load_from_disk(dataset_cache_dir, keep_in_memory=True)
            loaded_datasets.append(dataset)

            # Load index mappings
            indices = {}
            mappings_file = os.path.join(individual_cache_dir, "index_mappings.json")
            if os.path.exists(mappings_file):
                with open(mappings_file) as f:
                    indices = json.load(f)

            dataset_indices_list.append(indices)

            logger.debug(f"Loaded {len(dataset)} trajectories from {dataset_path}")

        if not loaded_datasets:
            raise RuntimeError("No datasets could be loaded from the cache")

        return loaded_datasets, dataset_indices_list

    def _build_indices(self, loaded_datasets: list[Dataset], dataset_indices_list: list[dict], cached_is_robot: list):
        """Build combined indices from loaded datasets and their index mappings.

        Args:
            loaded_datasets: List of loaded Dataset objects
            dataset_indices_list: List of index dictionaries, one per dataset (with original indices, not offset)
            cached_is_robot: List of is_robot flags for the concatenated dataset

        Returns:
            dict: Combined indices dictionary with all index mappings
        """
        combined_indices = {
            "robot_trajectories": [],
            "human_trajectories": [],
            "optimal_by_task": {},
            "suboptimal_by_task": {},
            "quality_indices": {},
            "task_indices": {},
            "source_indices": {},
            "partial_success_indices": {},
        }

        offset = 0

        for dataset, indices in zip(loaded_datasets, dataset_indices_list):
            # Adjust indices by adding offset and combine
            for key in combined_indices:
                if key in indices:
                    if isinstance(indices[key], list):
                        # For list indices, add offset
                        combined_indices[key].extend([idx + offset for idx in indices[key]])
                    elif isinstance(indices[key], dict):
                        # For regular dict indices, add offset to values
                        for subkey, subindices in indices[key].items():
                            if subkey not in combined_indices[key]:
                                combined_indices[key][subkey] = []
                            combined_indices[key][subkey].extend([idx + offset for idx in subindices])

            offset += len(dataset)

        # Build paired human robot index (needs cached_is_robot)
        combined_indices["paired_human_robot_by_task"] = self._build_paired_human_robot_index(
            combined_indices, cached_is_robot
        )

        # Find tasks that have both optimal and suboptimal trajectories
        tasks_with_multiple_quality_labels = set(combined_indices["optimal_by_task"].keys()) & set(
            combined_indices["suboptimal_by_task"].keys()
        )
        # Add it to combined_indices so samplers can access it
        combined_indices["tasks_with_multiple_quality_labels"] = list(tasks_with_multiple_quality_labels)

        return combined_indices

    def _load_preprocessed_cache(self, cache_dir: str, is_training: bool = True):
        """Load the preprocessed cache with index mappings for datasets.

        Returns:
            tuple: (dataset, combined_indices)
                - dataset: The loaded and concatenated dataset
                - combined_indices: Dictionary of combined indices
        """
        # Get available datasets
        available_datasets, missing_datasets = self._get_available_datasets(cache_dir)

        # Load datasets
        loaded_datasets, dataset_indices_list = self._load_datasets(available_datasets)

        # Concatenate datasets if multiple
        if len(loaded_datasets) == 1:
            dataset = loaded_datasets[0]
        else:
            dataset = concatenate_datasets(loaded_datasets)

        # Build combined indices
        combined_indices = self._build_indices(loaded_datasets, dataset_indices_list, dataset["is_robot"])

        dataset_type = "training" if is_training else "evaluation"
        logger.info(f"✅ Loaded {len(dataset)} total trajectories from preprocessed {dataset_type} datasets")
        logger.debug(f"Available datasets: {len(available_datasets)}/{len(missing_datasets) + len(available_datasets)}")
        logger.debug(f"Missing datasets: {len(missing_datasets)}")
        banner("Dataset statistics")
        logger.debug(f"Robot trajectories: {len(combined_indices['robot_trajectories'])}")
        logger.debug(f"Human trajectories: {len(combined_indices['human_trajectories'])}")
        logger.debug(f"Number of different tasks: {len(combined_indices['task_indices'])}")
        logger.debug(f"Data sources: {len(combined_indices['source_indices'])}")
        logger.debug(f"Tasks available: {list[Any](combined_indices['task_indices'].keys())[:10]} ...")
        logger.debug(f"Number of quality labels: {len(combined_indices['quality_indices'])}")
        for quality_label in combined_indices["quality_indices"]:
            logger.debug(f"  {quality_label}: {len(combined_indices['quality_indices'][quality_label])}")
        logger.debug(f"Data sources available: {combined_indices['source_indices'].keys()}")
        logger.debug(f"Number of paired tasks: {len(combined_indices['paired_human_robot_by_task'])}")
        logger.debug(
            f"Number of tasks with both multiple quality labels: {len(combined_indices['tasks_with_multiple_quality_labels'])}"
        )

        return dataset, combined_indices

    def _filter_dataset(
        self,
        excluded_keywords: list[str],
        min_frames: int,
        dataset,
        combined_indices: dict,
        filter_successful_only: bool = False,
    ):
        """Filter dataset based on multiple criteria simultaneously.

        Filters out trajectories that:
        - Have tasks containing excluded keywords
        - Have <= min_frames frames
        - (If filter_successful_only=True) Have quality_label != "successful"
        - RoboArena trajectories from tasks with only one partial_success category

        Uses batched map operations for efficient parallel processing.

        Args:
            excluded_keywords: List of keywords to exclude from task names (case-insensitive)
            min_frames: Minimum number of frames required (trajectories with frames > min_frames are kept)
            dataset: The dataset to filter
            combined_indices: Dictionary of combined indices to update after filtering
            filter_successful_only: If True, only keep trajectories with quality_label == "successful"

        Returns:
            tuple: (filtered_dataset, filtered_combined_indices)
                - filtered_dataset: The filtered dataset
                - filtered_combined_indices: The filtered combined indices
        """
        excluded_keywords_lower = [kw.lower() for kw in excluded_keywords]

        # Pre-compute RoboArena tasks with only one partial_success category
        all_tasks = dataset["task"]
        data_sources = dataset["data_source"]
        # Handle case where partial_success column might not exist
        if "partial_success" in dataset.column_names:
            partial_successes = dataset["partial_success"]
        else:
            partial_successes = [None] * len(dataset)

        # Group RoboArena trajectories by task
        roboarena_tasks_to_partial_success = collections.defaultdict(set)

        for task, data_source, partial_success in zip(all_tasks, data_sources, partial_successes):
            if task is None:
                continue
            # Check if this is a RoboArena trajectory
            if data_source and "roboarena" in str(data_source).lower():
                if partial_success is not None:
                    roboarena_tasks_to_partial_success[task].add(partial_success)

        # Find tasks with only one unique partial_success category
        roboarena_tasks_with_single_partial_success = {
            task
            for task, partial_success_set in roboarena_tasks_to_partial_success.items()
            if len(partial_success_set) == 1
        }

        def add_filter_flags(batch):
            """Add filter flags to batch for efficient filtering."""
            tasks = batch["task"]
            frames_shapes = batch["frames_shape"]
            quality_labels = batch["quality_label"]
            data_sources_batch = batch.get("data_source", [None] * len(tasks))

            drop_kw = []
            drop_frames = []
            drop_quality = []
            drop_roboarena = []

            for task, fs, quality_label, data_source in zip(tasks, frames_shapes, quality_labels, data_sources_batch):
                dkw = False
                dfr = False
                dq = False
                dr = False

                # Check excluded keywords
                if task is not None:
                    t = task.lower()
                    if any(kw in t for kw in excluded_keywords_lower):
                        dkw = True

                # Check minimum frames (only if not dropped by keywords)
                if not dkw and fs is not None and not (isinstance(fs, (list, tuple)) and len(fs) == 0):
                    if isinstance(fs, (list, tuple)):
                        num_frames = fs[0]
                    else:
                        num_frames = fs
                    if num_frames <= min_frames:
                        dfr = True

                # Check quality_label for successful-only filter (only if not dropped by other filters)
                if filter_successful_only and not dkw and not dfr:
                    if quality_label != "successful":
                        dq = True

                # Check RoboArena tasks with single partial_success (only if not dropped by other filters)
                if not dkw and not dfr and not dq:
                    if task is not None and task in roboarena_tasks_with_single_partial_success:
                        # Check if this is a RoboArena trajectory
                        if data_source and "roboarena" in str(data_source).lower():
                            dr = True

                drop_kw.append(dkw)
                drop_frames.append(dfr)
                drop_quality.append(dq)
                drop_roboarena.append(dr)

            return {
                "drop_by_keywords": drop_kw,
                "drop_by_frames": drop_frames,
                "drop_by_quality": drop_quality,
                "drop_by_roboarena": drop_roboarena,
            }

        # 1) Compute flags in a single batched pass
        dataset_with_flags = dataset.map(
            add_filter_flags,
            batched=True,
            num_proc=8,  # Can be increased for parallel processing if needed
            desc="Computing filter flags",
        )

        # 2) Compute counts and build keep_indices from flags
        drop_kw_list = dataset_with_flags["drop_by_keywords"]
        drop_frames_list = dataset_with_flags["drop_by_frames"]
        drop_quality_list = dataset_with_flags["drop_by_quality"]
        drop_roboarena_list = dataset_with_flags["drop_by_roboarena"]

        filtered_by_keywords = int(sum(drop_kw_list))
        filtered_by_frames = int(sum(drop_frames_list))
        filtered_by_quality = int(sum(drop_quality_list))
        filtered_by_roboarena = int(sum(drop_roboarena_list))
        total_filtered = filtered_by_keywords + filtered_by_frames + filtered_by_quality + filtered_by_roboarena

        # 3) Filter using precomputed flags (efficient)
        if total_filtered > 0:
            filter_messages = []
            if filtered_by_keywords > 0:
                filter_messages.append(f"{filtered_by_keywords} with excluded task keywords: {excluded_keywords}")
            if filtered_by_frames > 0:
                filter_messages.append(f"{filtered_by_frames} with <= {min_frames} frames")
            if filtered_by_quality > 0:
                filter_messages.append(f"{filtered_by_quality} with quality_label != 'successful'")
            if filtered_by_roboarena > 0:
                filter_messages.append(
                    f"{filtered_by_roboarena} RoboArena trajectories from {len(roboarena_tasks_with_single_partial_success)} tasks with only one partial_success category"
                )

            logger.info(f"Filtering out {total_filtered} trajectories ({', '.join(filter_messages)})")

            # Build keep_indices from flags (before filtering)
            keep_indices = [
                i
                for i, (dkw, dfr, dq, dr) in enumerate(
                    zip(drop_kw_list, drop_frames_list, drop_quality_list, drop_roboarena_list)
                )
                if not (dkw or dfr or dq or dr)
            ]

            removed_indices = set(range(len(dataset))) - set(keep_indices)

            logger.trace(f"Removed indices: {removed_indices}")

            # Filter dataset using select with keep_indices (more efficient than filter)
            filtered_dataset = dataset.select(keep_indices)

            # Update combined_indices using the shared helper
            filtered_combined_indices = self._update_indices_after_filtering(combined_indices, keep_indices)
        else:
            # No filtering needed, return original dataset and indices
            filtered_dataset = dataset
            filtered_combined_indices = combined_indices

        return filtered_dataset, filtered_combined_indices

    def _filter_task_based_criteria(
        self,
        dataset,
        combined_indices: dict,
    ):
        """Filter out suboptimal/failed trajectories that don't have optimal counterparts with the same task name.
        Also filter out tasks that only have failed/suboptimal trajectories.

        Args:
            dataset: The dataset to filter
            combined_indices: Dictionary of combined indices to update after filtering

        Returns:
            tuple: (filtered_dataset, filtered_combined_indices)
                - filtered_dataset: The filtered dataset
                - filtered_combined_indices: The filtered combined indices
        """
        # Get tasks that have optimal trajectories
        tasks_with_optimal = set(combined_indices.get("optimal_by_task", {}).keys())

        # Get all tasks in the dataset
        all_tasks = dataset["task"]
        quality_labels = dataset["quality_label"]

        # Identify trajectories to remove:
        # All trajectories from tasks that have no optimal trajectories
        indices_to_remove = set()
        tasks_removed = set()

        for idx, (task, quality_label) in enumerate(zip(all_tasks, quality_labels)):
            if task is None:
                # Skip trajectories with None task
                continue
            if task not in tasks_with_optimal:
                # This task has no optimal trajectories
                # Remove all trajectories from this task (they're all suboptimal/failed with no optimal counterparts)
                indices_to_remove.add(idx)
                tasks_removed.add(task)

        if indices_to_remove:
            # Build keep_indices
            keep_indices = [i for i in range(len(dataset)) if i not in indices_to_remove]

            filtered_by_no_optimal = len(indices_to_remove)
            tasks_removed_count = len(tasks_removed)

            logger.info(
                f"Filtering out {filtered_by_no_optimal} trajectories from {tasks_removed_count} tasks "
                f"that have no optimal trajectories (only suboptimal/failed)"
            )

            # Filter dataset using select with keep_indices
            filtered_dataset = dataset.select(keep_indices)

            # Update combined_indices using the shared helper
            filtered_combined_indices = self._update_indices_after_filtering(combined_indices, keep_indices)
        else:
            # No filtering needed
            filtered_dataset = dataset
            filtered_combined_indices = combined_indices

        return filtered_dataset, filtered_combined_indices

    def _update_indices_after_filtering(self, combined_indices: dict, keep_indices: list[int]) -> dict:
        """Update combined_indices after filtering the dataset.

        Args:
            combined_indices: Dictionary of combined indices before filtering
            keep_indices: List of indices to keep (from the original dataset)

        Returns:
            dict: Updated combined_indices with remapped indices
        """
        # Create a mapping from old index to new index
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}

        # Create a copy of combined_indices to avoid mutating the input
        filtered_combined_indices = {}

        # Filter all index lists and dicts
        for key in combined_indices:
            if isinstance(combined_indices[key], list):
                # Filter list indices
                filtered_combined_indices[key] = [old_to_new[idx] for idx in combined_indices[key] if idx in old_to_new]
            elif isinstance(combined_indices[key], dict):
                # Filter dict indices
                filtered_dict = {}
                for subkey, subindices in combined_indices[key].items():
                    # Handle nested dict structure (e.g., paired_human_robot_by_task has {"robot": [...], "human": [...]})
                    if isinstance(subindices, dict):
                        # This is a nested dict like {"robot": [...], "human": [...]}
                        filtered_nested_dict = {}
                        for nested_key, nested_list in subindices.items():
                            if isinstance(nested_list, list):
                                filtered_nested_list = [old_to_new[idx] for idx in nested_list if idx in old_to_new]
                                if filtered_nested_list:  # Only keep if there are remaining indices
                                    filtered_nested_dict[nested_key] = filtered_nested_list
                        # Only keep the task if it has at least one non-empty list (robot or human)
                        if filtered_nested_dict:
                            filtered_dict[subkey] = filtered_nested_dict
                    elif isinstance(subindices, list):
                        # Regular dict with list values (e.g., task_indices, source_indices)
                        filtered_indices = [old_to_new[idx] for idx in subindices if idx in old_to_new]
                        if filtered_indices:  # Only keep keys with remaining indices
                            filtered_dict[subkey] = filtered_indices
                filtered_combined_indices[key] = filtered_dict
            elif isinstance(combined_indices[key], set):
                # Filter set indices
                filtered_combined_indices[key] = {old_to_new[idx] for idx in combined_indices[key] if idx in old_to_new}
            else:
                # Keep other types as-is (e.g., strings, numbers)
                filtered_combined_indices[key] = combined_indices[key]

        return filtered_combined_indices

    # ------------------------------------------------------------------
    # Shared resample helpers for subclasses
    # ------------------------------------------------------------------
    def _record_resample_attempt(
        self, sample_type: str, strategy: str, sample_attempts: int, dataset_attempts: int
    ) -> None:
        if sample_type not in self._resample_attempt_stats:
            return

        self._resample_attempt_stats[sample_type][strategy].append(sample_attempts)
        self._resample_dataset_attempt_stats[sample_type][strategy].append(dataset_attempts)

    def _set_resample_attempts(self, sample, dataset_attempts: int):
        if sample is None:
            return None

        dataset_attempts = max(1, int(dataset_attempts))
        sample_attempts = int(getattr(sample, "resample_attempts", dataset_attempts))
        sample_attempts = max(1, sample_attempts)
        sample.resample_attempts = sample_attempts

        sample_type = getattr(sample, "sample_type", "unknown")
        strategy = str(getattr(sample, "data_gen_strategy", "unknown"))
        self._record_resample_attempt(sample_type, strategy, sample_attempts, dataset_attempts)

        return sample

    def get_resample_attempt_stats(self):
        return self._resample_attempt_stats

    def get_resample_dataset_attempt_stats(self):
        return self._resample_dataset_attempt_stats

    def _build_paired_human_robot_index(self, combined_indices: dict, cached_is_robot: list):
        """Build paired_human_robot_by_task index from task_indices by checking is_robot field.

        This builds the index after concatenation by iterating through task_indices
        and checking the is_robot field for each trajectory. Only includes trajectories
        from PAIRED data sources.

        Args:
            combined_indices: Dictionary of combined indices
            cached_is_robot: List of is_robot flags for the concatenated dataset

        Returns:
            dict: paired_human_robot_by_task dictionary
        """
        paired_human_robot_by_task = {}

        # Filter indices for paired data sources
        paired_data_source_indices = set()
        for data_source in get_paired_ds():
            if data_source in combined_indices["source_indices"]:
                paired_data_source_indices.update(combined_indices["source_indices"][data_source])

        if not paired_data_source_indices:
            logger.debug("No paired data sources found, skipping paired index building")
            return {}

        # Build index from task_indices using cached is_robot field, but only for paired data sources
        for task, indices in combined_indices["task_indices"].items():
            # Filter to only paired data sources
            task_indices_paired = [idx for idx in indices if idx in paired_data_source_indices]

            if not task_indices_paired:
                continue

            paired_human_robot_by_task[task] = {"robot": [], "human": []}

            for idx in task_indices_paired:
                is_robot = cached_is_robot[idx] if idx < len(cached_is_robot) else True
                if is_robot:
                    paired_human_robot_by_task[task]["robot"].append(idx)
                else:
                    paired_human_robot_by_task[task]["human"].append(idx)

        # Count tasks with both robot and human trajectories
        tasks_with_pairs = [
            task for task, task_dict in paired_human_robot_by_task.items() if task_dict["robot"] and task_dict["human"]
        ]
        num_tasks_with_pairs = len(tasks_with_pairs)
        logger.debug(
            f"Built paired_human_robot_by_task index: {num_tasks_with_pairs} tasks with both robot and human trajectories (from paired data sources only)"
        )

        return paired_human_robot_by_task
