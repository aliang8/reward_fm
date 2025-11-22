#!/usr/bin/env python3
import json
import os
import torch

from datasets import Dataset, concatenate_datasets
from rfm.data.datasets.helpers import load_dataset_success_percent
from rfm.data.dataset_category import DATASET_MAP
from rfm.utils.distributed import rank_0_print

def resolve_dataset_keys(dataset_keys: list[str], split: str) -> list[str]:
    """
    Resolve dataset keys through DATASET_MAP.

    Args:
        dataset_keys: List of dataset keys (e.g., ["mw", "oxe"]) or actual dataset names
        split: Either "train" or "eval"

    Returns:
        List of resolved dataset names, combining all datasets from the keys
    """
    resolved_datasets = []
    for key in dataset_keys:
        if key in DATASET_MAP:
            # Key found in DATASET_MAP, resolve it
            if split in DATASET_MAP[key]:
                resolved_datasets.extend(DATASET_MAP[key][split])
            else:
                rank_0_print(f"Warning: Key '{key}' found in DATASET_MAP but no '{split}' split defined. Skipping.")
        else:
            # Not a key, assume it's already a dataset name
            resolved_datasets.append(key)
    return resolved_datasets


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, config, is_evaluation=False, verbose=True, **kwargs):
        self.config = config
        self.is_evaluation = is_evaluation
        self.verbose = verbose

        # Choose datasets based on whether this is for evaluation or training
        if is_evaluation and config.eval_datasets:
            dataset_keys = config.eval_datasets
            split = "eval"
        else:
            dataset_keys = config.train_datasets
            split = "train"

        self.datasets = resolve_dataset_keys(dataset_keys, split)

        # Initialize dataset
        self.dataset = None

        # Load dataset-specific success cutoff map if available
        self.dataset_success_cutoff_map = {}
        if hasattr(config, "dataset_success_cutoff_file") and config.dataset_success_cutoff_file:
            self.dataset_success_cutoff_map = load_dataset_success_percent(config.dataset_success_cutoff_file)

        # Load trajectory dataset
        self._load_trajectory_dataset()

        # Filter dataset
        # We only want to iterate through successful trajectories
        # self.filtered_dataset = self.dataset.filter(lambda x: x["quality_label"] == "successful")
        self.filtered_dataset = self.dataset

        if verbose:
            rank_0_print(f"Dataset loaded with {len(self.dataset)} total trajectories", verbose=self.verbose)

    def __len__(self):
        return len(self.filtered_dataset)

    def _load_trajectory_dataset(self):
        """Load trajectory dataset using preprocessed index-based cache."""
        cache_dir = os.environ.get("RFM_PROCESSED_DATASETS_PATH", "")
        if not cache_dir:
            raise ValueError(
                "RFM_PROCESSED_DATASETS_PATH environment variable not set. Please set it to the directory containing your processed datasets."
            )
        cache_type = "evaluation" if self.is_evaluation else "training"

        # Check if preprocessed cache exists
        if os.path.exists(cache_dir):
            rank_0_print(
                f"Found preprocessed cache at {cache_dir}, loading {cache_type} datasets...", verbose=self.verbose
            )

            self._load_preprocessed_cache(cache_dir, is_training=not self.is_evaluation)

            rank_0_print(
                f"Successfully loaded preprocessed {cache_type} datasets with {len(self.dataset)} trajectory indices",
                verbose=self.verbose,
            )
        else:
            # If no cache exists, we need to run the preprocessor first
            rank_0_print(
                "No preprocessed cache found. Please run preprocess_datasets.py first to create the cache.",
                verbose=self.verbose,
            )
            raise RuntimeError(
                "Dataset preprocessing required. Please run:\n"
                "uv run scripts/preprocess_datasets.py\n"
                "This will create the necessary index-based cache for efficient data loading."
            )

    def _load_preprocessed_cache(self, cache_dir: str, is_training: bool = True):
        """Load the preprocessed cache with index mappings for datasets."""
        # Check which datasets are available
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
                        rank_0_print(f"       Found cache: {individual_cache_dir}", verbose=self.verbose)
                    except:
                        rank_0_print(
                            f"       Cache info file corrupted, skipping: {individual_cache_dir}", verbose=self.verbose
                        )
                        continue
                else:
                    rank_0_print(f"       No info file found, skipping: {individual_cache_dir}", verbose=self.verbose)
                    continue
            else:
                missing_datasets.append(dataset_path)
                rank_0_print(f"      ❌ Missing cache: {individual_cache_dir}", verbose=self.verbose)

        # Warn about missing datasets
        if missing_datasets:
            rank_0_print(
                "\n⚠️  Warning: The following configured datasets are not available in the cache:", verbose=self.verbose
            )
            for dataset_path in missing_datasets:
                rank_0_print(f"    ❌ {dataset_path}", verbose=self.verbose)
            rank_0_print(
                "  Available datasets will be loaded, but some configured data may be missing.", verbose=self.verbose
            )

        if not available_datasets:
            raise RuntimeError(
                f"No configured datasets are available in the cache. "
                f"Please run preprocess_datasets.py to create the cache for: {self.datasets}"
            )

        rank_0_print(
            f"\nSummary: {len(available_datasets)} available, {len(missing_datasets)} missing", verbose=self.verbose
        )

        # Load available datasets
        loaded_datasets = []
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

        for dataset_path, individual_cache_dir in available_datasets:
            # Load the processed dataset
            dataset_cache_dir = os.path.join(individual_cache_dir, "processed_dataset")
            if not os.path.exists(dataset_cache_dir):
                rank_0_print(
                    f"   Warning: Processed dataset not found at {dataset_cache_dir}, skipping...", verbose=self.verbose
                )
                continue

            dataset = Dataset.load_from_disk(dataset_cache_dir, keep_in_memory=True)
            loaded_datasets.append(dataset)

            # Load index mappings
            mappings_file = os.path.join(individual_cache_dir, "index_mappings.json")
            if os.path.exists(mappings_file):
                with open(mappings_file) as f:
                    indices = json.load(f)

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

            if self.verbose:
                rank_0_print(f"  ✅ Loaded {len(dataset)} trajectories from {dataset_path}", verbose=self.verbose)
            offset += len(dataset)

        if not loaded_datasets:
            raise RuntimeError("No datasets could be loaded from the cache")

        # Concatenate datasets if multiple
        if len(loaded_datasets) == 1:
            self.dataset = loaded_datasets[0]
        else:
            self.dataset = concatenate_datasets(loaded_datasets)

        # Store the combined index mappings
        self._combined_indices = combined_indices
        self._cached_ids = self.dataset["id"]
        self._cached_is_robot = self.dataset["is_robot"]

        dataset_type = "training" if is_training else "evaluation"
        rank_0_print(
            f"✅ Loaded {len(self.dataset)} total trajectories from preprocessed {dataset_type} datasets",
            verbose=self.verbose,
        )
        rank_0_print(
            f"  📊 Available datasets: {len(available_datasets)}/{len(missing_datasets) + len(available_datasets)}",
            verbose=self.verbose,
        )
        rank_0_print(f"  📊 Missing datasets: {len(missing_datasets)}", verbose=self.verbose)
