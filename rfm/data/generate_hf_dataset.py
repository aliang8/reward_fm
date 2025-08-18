#!/usr/bin/env python3
"""
Main dataset converter that can convert any dataset to HuggingFace format for RFM model training.
This is a generic converter that works with any dataset-specific loader.
"""

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hide INFO/WARN/ERROR; only FATAL remains
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from pathlib import Path
from datasets import Dataset, Audio
import datasets
from tqdm import tqdm
from dataclasses import dataclass, field
from pyrallis import wrap
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from functools import partial
from rfm.data.helpers import (
    load_sentence_transformer_model,
    create_output_directory,
    flatten_task_data,
    create_hf_trajectory,
)
from rfm.data.dataset_types import Trajectory

# make sure these come after importing torch. otherwise something breaks...
try:
    import absl.logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
try:
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_trajectory_subdir_path(trajectory_idx: int, files_per_subdir: int = 1000) -> str:
    """
    Generate subdirectory path for a trajectory to avoid too many files per directory.

    Args:
        trajectory_idx: Index of the trajectory
        files_per_subdir: Maximum files per subdirectory (default: 1000)

    Returns:
        str: Subdirectory name like 'batch_0000'
    """
    subdir_index = trajectory_idx // files_per_subdir
    return f"batch_{subdir_index:04d}"


# Global dataset features definition
BASE_FEATURES = {
    "id": datasets.Value("string"),
    "task": datasets.Value("string"),
    "lang_vector": datasets.Sequence(datasets.Value("float32")),
    "data_source": datasets.Value("string"),
    "frames": None,  # Will be set based on use_video parameter
    "is_robot": datasets.Value("bool"),
    "quality_label": datasets.Value("string"),
    "preference_group_id": datasets.Value("string"),
    "preference_rank": datasets.Value("int32"),
}


@dataclass
class DatasetConfig:
    """Config for dataset settings"""

    dataset_path: str = field(default="", metadata={"help": "Path to the dataset"})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Name of the dataset (defaults to dataset_type)"}
    )


@dataclass
class OutputConfig:
    """Config for output settings"""

    output_dir: str = field(default="rfm_dataset", metadata={"help": "Output directory for the dataset"})
    max_trajectories: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of trajectories to process (None for all)"}
    )
    max_frames: int = field(
        default=64, metadata={"help": "Maximum number of frames per trajectory (-1 for no downsampling)"}
    )
    use_video: bool = field(default=True, metadata={"help": "Use MP4 videos instead of individual frame images"})
    shortest_edge_size: int = field(default=240, metadata={"help": "Shortest edge size for video resizing"})
    center_crop: Optional[bool] = field(
        default=False,
        metadata={"help": "Center crop the video to the target size. Defaults to False, which means no cropping."},
    )
    fps: int = field(default=10, metadata={"help": "Frames per second for video creation"})
    num_workers: int = field(
        default=-1, metadata={"help": "Number of parallel workers for processing (-1 for auto, 0 for sequential)"}
    )


@dataclass
class HubConfig:
    """Config for HuggingFace Hub settings"""

    push_to_hub: bool = field(default=False, metadata={"help": "Push dataset to HuggingFace Hub"})
    hub_repo_id: Optional[str] = field(default=None, metadata={"help": "HuggingFace Hub repository ID"})
    hub_token: Optional[str] = field(
        default=None, metadata={"help": "HuggingFace Hub token (or set HF_TOKEN environment variable)"}
    )


@dataclass
class GenerateConfig:
    """Main configuration for dataset generation"""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    hub: HubConfig = field(default_factory=HubConfig)


def process_single_trajectory(args):
    """
    Worker function to process a single trajectory.

    Args:
        args: Tuple containing (trajectory_idx, trajectory, lang_vector, hf_creator_fn, output_dir, dataset_name, max_frames, use_video, fps)

    Returns:
        Dict: Processed trajectory data or None if failed
    """
    trajectory_idx, trajectory, lang_vector, hf_creator_fn, output_dir, dataset_name, max_frames, use_video, fps = args

    try:
        # Create output directory for this trajectory with subdirectory structure
        subdir_name = get_trajectory_subdir_path(trajectory_idx)
        full_video_path = os.path.join(
            output_dir, dataset_name.lower(), subdir_name, f"trajectory_{trajectory_idx:04d}.mp4"
        )
        relative_video_path = os.path.join(dataset_name.lower(), subdir_name, f"trajectory_{trajectory_idx:04d}.mp4")
        os.makedirs(os.path.dirname(full_video_path), exist_ok=True)

        # Process trajectory (lang_vector is already computed)
        processed_trajectory = hf_creator_fn(
            traj_dict=trajectory,
            video_path=full_video_path,
            lang_vector=lang_vector,  # Pre-computed language vector
            max_frames=max_frames,
            dataset_name=dataset_name,
            use_video=use_video,
            fps=fps,
        )

        if processed_trajectory is None:
            return None

        # Replace the full path with relative path in the processed trajectory
        if processed_trajectory and "frames" in processed_trajectory:
            processed_trajectory["frames"] = relative_video_path

        return processed_trajectory

    except Exception as e:
        print(f"❌ Error processing trajectory {trajectory_idx}: {e}")
        return None


def convert_dataset_to_hf_format(
    trajectories: List[Dict],
    hf_creator_fn: Callable[[Dict, str, str, int, Any, int, str], Trajectory],
    output_dir: str = "rfm_dataset",
    dataset_name: str = "",
    max_trajectories: int = None,
    max_frames: int = -1,
    use_video: bool = True,
    fps: int = 10,
    num_workers: int = -1,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
    hub_token: Optional[str] = None,
) -> Dataset:
    """Convert a list of trajectories to HuggingFace format."""

    print(f"Converting {dataset_name} dataset to HuggingFace format...")

    # Create output directory
    create_output_directory(output_dir)

    # Validate input
    if not trajectories:
        raise ValueError(f"No trajectories provided for {dataset_name} dataset.")

    print(f"Processing {len(trajectories)} trajectories")

    # Limit trajectories if specified
    if max_trajectories is not None:
        trajectories = trajectories[:max_trajectories]

    # Determine number of workers
    if num_workers == -1:
        num_workers = min(cpu_count(), len(trajectories))
    elif num_workers == 0:
        num_workers = 1  # Sequential processing

    print(f"Using {num_workers} worker(s) for parallel processing")

    # Pre-compute language embeddings to avoid loading sentence transformer in each worker
    print("Pre-computing language embeddings...")
    lang_model = load_sentence_transformer_model()

    lang_vectors = []
    unique_tasks = {}  # Cache for identical task descriptions

    for trajectory in tqdm(trajectories, desc="Computing language embeddings"):
        task_description = trajectory["task"]

        # Use cache to avoid recomputing identical task descriptions
        if task_description not in unique_tasks:
            unique_tasks[task_description] = lang_model.encode(task_description)

        lang_vectors.append(unique_tasks[task_description])

    print(f"Computed embeddings for {len(unique_tasks)} unique task descriptions")

    # Process trajectories
    all_entries = []

    if num_workers == 1:
        # Sequential processing (using pre-computed embeddings)
        for trajectory_idx, (trajectory, lang_vector) in enumerate(
            tqdm(zip(trajectories, lang_vectors), desc="Processing trajectories")
        ):
            # Create output directory for this trajectory with subdirectory structure
            subdir_name = get_trajectory_subdir_path(trajectory_idx)
            trajectory_dir = os.path.join(
                output_dir, dataset_name.lower(), subdir_name, f"trajectory_{trajectory_idx:04d}.mp4"
            )
            os.makedirs(os.path.dirname(trajectory_dir), exist_ok=True)

            processed_trajectory = hf_creator_fn(
                traj_dict=trajectory,
                video_path=trajectory_dir,
                lang_vector=lang_vector,  # Pre-computed language vector
                max_frames=max_frames,
                dataset_name=dataset_name,
                use_video=use_video,
                fps=fps,
            )
            if processed_trajectory is None:
                continue
            all_entries.append(processed_trajectory)
    else:
        # Parallel processing
        print(f"Preparing {len(trajectories)} trajectories for parallel processing...")

        # Prepare arguments for worker processes
        worker_args = []
        for trajectory_idx, (trajectory, lang_vector) in enumerate(zip(trajectories, lang_vectors)):
            args = (
                trajectory_idx,
                trajectory,
                lang_vector,  # Pre-computed language vector
                hf_creator_fn,
                output_dir,
                dataset_name,
                max_frames,
                use_video,
                fps,
            )
            worker_args.append(args)

        # Use spawn to avoid CUDA context issues from forking after TF import
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        # Process trajectories in parallel
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(process_single_trajectory, worker_args),
                    total=len(worker_args),
                    desc="Processing trajectories",
                )
            )

        # Filter out failed trajectories (None results)
        all_entries = [result for result in results if result is not None]

        if len(all_entries) < len(trajectories):
            failed_count = len(trajectories) - len(all_entries)
            print(f"⚠️  {failed_count} trajectories failed to process and were skipped")

    # Create HuggingFace dataset with proper features
    print(f"Creating HuggingFace dataset with {len(all_entries)} entries...")

    # Convert list of entries to dictionary format for from_dict()
    data_dict = {
        "id": [entry["id"] for entry in all_entries],
        "task": [entry["task"] for entry in all_entries],
        "lang_vector": [entry["lang_vector"] for entry in all_entries],
        "data_source": [entry["data_source"] for entry in all_entries],
        "frames": [entry["frames"] for entry in all_entries],
        "is_robot": [entry["is_robot"] for entry in all_entries],
        "quality_label": [entry.get("quality_label") for entry in all_entries],
        "preference_group_id": [entry.get("preference_group_id") for entry in all_entries],
        "preference_rank": [entry.get("preference_rank") for entry in all_entries],
    }

    # Set frames feature based on video mode
    features_dict = BASE_FEATURES.copy()
    if use_video:
        features_dict["frames"] = datasets.Value("string")  # Video file paths as strings
    else:
        features_dict["frames"] = datasets.Sequence(datasets.Image())

    features = datasets.Features(features_dict)

    dataset = Dataset.from_dict(data_dict, features=features)

    print(f"{dataset_name} HuggingFace dataset created successfully!")
    print(f"Total entries: {len(all_entries)}")

    # Push to HuggingFace Hub if requested
    if push_to_hub and hub_repo_id:
        print(f"\nPushing dataset to HuggingFace Hub: {hub_repo_id}")
        try:
            # Push the dataset to the hub with dataset name as config name
            dataset.push_to_hub(
                hub_repo_id,
                config_name=dataset_name.lower(),  # Use dataset name as config name
                token=hub_token,
                private=False,
                commit_message=f"Add {dataset_name} dataset for RFM training",
            )
            print(f"✅ Successfully pushed dataset to: https://huggingface.co/datasets/{hub_repo_id}")
            print(f"📁 Dataset available as config: {dataset_name.lower()}")

            # Also push the video files folder to the hub
            print(f"\nPushing video files to HuggingFace Hub...")
            from huggingface_hub import HfApi

            api = HfApi(token=hub_token)

            # Upload the entire output directory (which contains all the video files)
            api.upload_large_folder(
                folder_path=output_dir,
                repo_id=hub_repo_id,
                repo_type="dataset",
                # commit_message=f"Add video files for {dataset_name} dataset"
            )
            print(f"✅ Successfully pushed video files to: https://huggingface.co/datasets/{hub_repo_id}")

        except Exception as e:
            print(f"❌ Error pushing to hub: {e}")
            print("Dataset was created locally but failed to push to hub")
    elif push_to_hub and not hub_repo_id:
        print("❌ push_to_hub=True but no hub_repo_id provided")
    else:
        # Only save locally if not pushing to hub (to avoid redundant Arrow files)
        dataset_path = os.path.join(output_dir, dataset_name.lower())
        dataset.save_to_disk(dataset_path)
        print(f"Dataset saved locally to: {dataset_path}")

    return dataset


@wrap()
def main(cfg: GenerateConfig):
    """Main function to convert any dataset to HuggingFace format."""

    # Get hub token from environment if not provided
    if cfg.hub.hub_token is None:
        cfg.hub.hub_token = os.getenv("HF_TOKEN")

    # Only require HF_USERNAME if pushing to hub
    if cfg.hub.push_to_hub:
        username = os.getenv("HF_USERNAME")
        if not username:
            raise ValueError(
                "HF_USERNAME is not set. Please export it to push to the Hub, or set hub.push_to_hub=false."
            )
        if cfg.hub.hub_repo_id:
            cfg.hub.hub_repo_id = username + "/" + cfg.hub.hub_repo_id

    # Import the appropriate dataset loader and trajectory creator
    if "libero" in cfg.dataset.dataset_name:
        from rfm.data.dataset_loaders.libero_loader import load_libero_dataset

        # Load the trajectories using the loader
        task_data = load_libero_dataset(cfg.dataset.dataset_path)
        trajectories = flatten_task_data(task_data)
    elif "agibotworld" in (cfg.dataset.dataset_name or "").lower():
        # Stream + convert directly inside the AgiBotWorld loader
        from rfm.data.dataset_loaders.agibotworld_loader import (
            convert_agibotworld_streaming_to_hf,
        )

        dataset = convert_agibotworld_streaming_to_hf(
            dataset_name=cfg.dataset.dataset_path,
            output_dir=cfg.output.output_dir,
            dataset_label=cfg.dataset.dataset_name or "agibotworld",
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )
        # Handle pushing/saving consistently
        if cfg.hub.push_to_hub and cfg.hub.hub_repo_id:
            print(f"\nPushing dataset to HuggingFace Hub: {cfg.hub.hub_repo_id}")
            try:
                # Push the arrow table
                dataset.push_to_hub(
                    cfg.hub.hub_repo_id,
                    config_name=(cfg.dataset.dataset_name or "agibotworld").lower(),
                    token=cfg.hub.hub_token,
                    private=False,
                    commit_message=f"Add {cfg.dataset.dataset_name} dataset for RFM training",
                )
                print(f"✅ Successfully pushed dataset to: https://huggingface.co/datasets/{cfg.hub.hub_repo_id}")

                # Push the large video folder(s)
                print(f"\nPushing video files to HuggingFace Hub...")
                from huggingface_hub import HfApi

                api = HfApi(token=cfg.hub.hub_token)
                api.upload_large_folder(
                    folder_path=cfg.output.output_dir,
                    repo_id=cfg.hub.hub_repo_id,
                    repo_type="dataset",
                )
                print(f"✅ Successfully pushed video files to: https://huggingface.co/datasets/{cfg.hub.hub_repo_id}")
            except Exception as e:
                print(f"❌ Error pushing to hub: {e}")
                print("Dataset was created locally but failed to push videos and/or metadata to hub")
        else:
            dataset_path = os.path.join(cfg.output.output_dir, (cfg.dataset.dataset_name or "agibotworld").lower())
            dataset.save_to_disk(dataset_path)
            print(f"Dataset saved locally to: {dataset_path}")
        print("Dataset conversion complete!")
        return

    elif "egodex" in cfg.dataset.dataset_name.lower():
        from rfm.data.dataset_loaders.egodex_loader import load_egodex_dataset

        # Load the trajectories using the loader with max_trajectories limit
        print(f"Loading EgoDex dataset from: {cfg.dataset.dataset_path}")
        task_data = load_egodex_dataset(
            cfg.dataset.dataset_path,
            cfg.output.max_trajectories,
        )
        trajectories = flatten_task_data(task_data)
    elif cfg.dataset.dataset_name.lower().startswith("oxe_"):
        # Treat OXE like AgiBotWorld: create videos and HF entries directly in the loader
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from rfm.data.dataset_loaders.oxe_loader import convert_oxe_dataset_to_hf

        print(f"Converting OXE dataset directly to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_oxe_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
        )

        # Handle pushing/saving consistently
        if cfg.hub.push_to_hub and cfg.hub.hub_repo_id:
            print(f"\nPushing dataset to HuggingFace Hub: {cfg.hub.hub_repo_id}")
            try:
                dataset.push_to_hub(
                    cfg.hub.hub_repo_id,
                    config_name=(cfg.dataset.dataset_name).lower(),
                    token=cfg.hub.hub_token,
                    private=False,
                    commit_message=f"Add {cfg.dataset.dataset_name} dataset for RFM training",
                )
                print(
                    f"✅ Successfully pushed dataset {cfg.dataset.dataset_name} to: https://huggingface.co/datasets/{cfg.hub.hub_repo_id}"
                )

                # Push the large video folder(s)
                print(f"\nPushing video files to HuggingFace Hub...")
                from huggingface_hub import HfApi

                api = HfApi(token=cfg.hub.hub_token)
                api.upload_folder(
                    folder_path=cfg.output.output_dir,
                    repo_id=cfg.hub.hub_repo_id,
                    repo_type="dataset",
                )
                print(
                    f"✅ Successfully pushed video files for {cfg.dataset.dataset_name} to: https://huggingface.co/datasets/{cfg.hub.hub_repo_id}"
                )
            except Exception as e:
                print(f"❌ Error pushing to hub: {e}")
                print("Dataset was created locally but failed to push videos and/or metadata to hub")
        else:
            dataset_path = os.path.join(cfg.output.output_dir, (cfg.dataset.dataset_name).lower())
            dataset.save_to_disk(dataset_path)
            print(f"Dataset saved locally to: {dataset_path}")
        print("Dataset conversion complete!")
        return
    elif "robofail" in cfg.dataset.dataset_name.lower():
        from rfm.data.dataset_loaders.robofail_loader import load_robofail_dataset

        # Load the trajectories using the loader with max_trajectories limit
        print(f"Loading RoboFail dataset from: {cfg.dataset.dataset_path}")
        task_data = load_robofail_dataset(
            cfg.dataset.dataset_path,
            cfg.output.max_trajectories,
        )
        trajectories = flatten_task_data(task_data)
    else:
        raise ValueError(f"Unknown dataset type: {cfg.dataset.dataset_name}")

    # Convert dataset (non-streaming datasets)
    convert_dataset_to_hf_format(
        trajectories=trajectories,
        hf_creator_fn=partial(
            create_hf_trajectory,
            dataset_name=cfg.dataset.dataset_name,
            use_video=cfg.output.use_video,
            fps=cfg.output.fps,
            shortest_edge_size=cfg.output.shortest_edge_size,
            center_crop=cfg.output.center_crop,
            hub_repo_id=cfg.hub.hub_repo_id,
        ),
        output_dir=cfg.output.output_dir,
        dataset_name=cfg.dataset.dataset_name,
        max_trajectories=cfg.output.max_trajectories,
        max_frames=cfg.output.max_frames,
        use_video=cfg.output.use_video,
        fps=cfg.output.fps,
        num_workers=cfg.output.num_workers,
        push_to_hub=cfg.hub.push_to_hub,
        hub_repo_id=cfg.hub.hub_repo_id,
        hub_token=cfg.hub.hub_token,
    )

    print("Dataset conversion complete!")


if __name__ == "__main__":
    main()
