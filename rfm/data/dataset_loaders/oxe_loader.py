import os
import itertools
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
from multiprocessing import cpu_count
from rfm.data.dataset_helpers.oxe_helper import OXE_DATASET_CONFIGS
from datasets import Dataset
from rfm.data.helpers import (
    create_trajectory_video_optimized,
    load_sentence_transformer_model,
    create_hf_trajectory,
    generate_unique_id,
)

# Disable GPUs for TensorFlow in this loader to avoid CUDA context issues in workers
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import tensorflow_datasets as tfds
import tensorflow as tf

OXE_VALID_DATASETS = [
    "austin_buds_dataset_converted_externally_to_rlds",
    "austin_sirius_dataset_converted_externally_to_rlds",
    "bc_z",
    "berkeley_cable_routing",
    "berkeley_fanuc_manipulation",
    "bridge_v2",
    "dlr_edan_shared_control_converted_externally_to_rlds",
    "droid",
    "fmb",
    "fractal20220817_data",
    "furniture_bench_dataset_converted_externally_to_rlds",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "jaco_play",
    "language_table",
    "stanford_hydra_dataset_converted_externally_to_rlds",
    "taco_play",
    "toto",
    "ucsd_kitchen_dataset_converted_externally_to_rlds",
    "utaustin_mutex",
    "viola",
]
POSSIBLE_LANG_INSTRUCTION_KEYS = [  # valid keys for language instruction in OXE
    "natural_language_instruction",
    "language_instruction",
    "instruction",
]
MAX_LANGTABLE_EPISODES = (
    50_000  # for language table, we only want to label the first 50k episodes b/c it's way too many
)
possible_valid_keys = ["primary", "secondary", "tertiary"]


def _stable_shard_for_index(index: int, shard_modulus: int = 1000) -> str:
    try:
        idx = int(index)
    except Exception:
        idx = abs(hash(str(index)))
    shard_index = idx // shard_modulus
    return f"shard_{shard_index:04d}"


def _build_oxe_video_paths(
    output_dir: str,
    dataset_label: str,
    episode_idx: int,
    view_key: str,
) -> Tuple[str, str]:
    shard_dir = _stable_shard_for_index(episode_idx)
    episode_dir = os.path.join(output_dir, dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}")
    os.makedirs(episode_dir, exist_ok=True)
    filename = f"clip@{view_key}.mp4"
    full_path = os.path.join(episode_dir, filename)
    rel_path = os.path.join(dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}", filename)
    return full_path, rel_path


def _process_single_oxe_episode(args):
    """Worker function to process a single OXE episode.

    This function must be defined at module level to be picklable for multiprocessing.
    """
    episode, ep_idx, task, lang_vec, output_dir, dataset_name, max_frames, fps, valid_img_keys = args

    episode_entries = []

    # Episode is already converted to numpy format
    steps_np = episode["steps"]

    for img_key in valid_img_keys:
        # Check first frame for all-black to prune
        if img_key not in steps_np[0]["observation"]:
            continue
        if np.all(steps_np[0]["observation"][img_key] == 0):
            continue

        frames = [s["observation"][img_key] for s in steps_np if img_key in s["observation"]]
        if not frames:
            continue

        full_path, rel_path = _build_oxe_video_paths(
            output_dir=output_dir,
            dataset_label=dataset_name,
            episode_idx=ep_idx,
            view_key=img_key,
        )

        traj_dict = {
            "id": generate_unique_id(),
            "frames": np.stack(frames) if isinstance(frames[0], np.ndarray) else frames,
            "task": task,
            "is_robot": True,
            "quality_label": "successful",
            "preference_group_id": None,
            "preference_rank": None,
        }

        entry = create_hf_trajectory(
            traj_dict=traj_dict,
            video_path=full_path,
            lang_vector=lang_vec,
            max_frames=max_frames,
            dataset_name=dataset_name,
            use_video=True,
            fps=fps,
        )
        if entry:
            entry["frames"] = rel_path
            episode_entries.append(entry)

    return episode_entries


def convert_oxe_dataset_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: Optional[int] = None,
    max_frames: int = 64,
    fps: int = 10,
    num_workers: int = -1,
) -> Dataset:
    """Convert a single OXE TFDS dataset to HF format by writing videos directly.

    Args:
        dataset_path: Root path containing TFDS builder directories
        dataset_name: Name prefixed with 'oxe_', e.g., 'oxe_language_table'
        output_dir: Where to write video files and dataset
        max_trajectories: Limit number of produced trajectories (None for all)
        max_frames: Max frames per video
        fps: Video fps

    Returns:
        datasets.Dataset with entries containing relative video paths.
    """

    # Normalize name and basic checks
    if dataset_name is None:
        raise ValueError("dataset_name is required")

    base_ds_name = dataset_name.replace("oxe_", "")

    if base_ds_name.endswith("_eval"):
        base_ds_name = base_ds_name[:-5]
        EVAL_MODE = True
        # use eval/val/test
    else:
        EVAL_MODE = False
    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    # Find builder directory/version
    versions = os.listdir(f"{root}/{base_ds_name}")
    if len(versions) == 0:
        raise ValueError(f"No versions found for {base_ds_name} in {root}")

    builder = None
    for version in versions:
        if "incomplete" in version:
            continue
        builder = tfds.builder_from_directory(f"{root}/{base_ds_name}/{version}")
        break
    if builder is None:
        raise ValueError(f"No valid builder found for {base_ds_name} in {root}")

    if EVAL_MODE:
        ds_all_dict = builder.as_dataset()
        splits = list(ds_all_dict.keys())
        splits.remove("train")
        if len(splits) == 0:
            raise ValueError(f"No valid EVAL dataset found for {base_ds_name} in {root}")
        elif len(splits) == 1:
            dataset = builder.as_dataset(split=splits[0], shuffle_files=False)
        else:
            raise ValueError(f"Multiple EVAL splits found for {base_ds_name} in {root}: {splits}")
        print(f"Loaded EVAL dataset for {base_ds_name} in {root}")
        # splits = ["val", "test"]
        # for split in splits:
        #    try:
        #        dataset = builder.as_dataset(split=split, shuffle_files=False)
        #        break
        #    except Exception as e:
        #        print(f"Error loading {split} split: {e}")
        #        dataset = None
        #        continue
        # if dataset is None:
        #    raise ValueError(f"No valid {EVAL_MODE} dataset found for {base_ds_name} in {root}")
    else:
        dataset = builder.as_dataset(split="train", shuffle_files=False)

    # Determine valid image observation keys
    img_key_to_name = OXE_DATASET_CONFIGS[base_ds_name]["image_obs_keys"]
    img_key_to_name = {k: v for k, v in img_key_to_name.items() if k != "wrist"}
    valid_img_keys = list(img_key_to_name.values())

    # Determine number of workers
    if num_workers == -1:
        num_workers = max(1, min(cpu_count(), 8))
    elif num_workers == 0:
        num_workers = 1

    # Language model and cache
    lang_model = load_sentence_transformer_model()
    lang_cache: Dict[str, Any] = {}

    entries: List[Dict[str, Any]] = []
    produced = 0
    max_limit = float("inf") if (max_trajectories is None or max_trajectories == -1) else int(max_trajectories)

    if "language_table" in base_ds_name:
        max_limit = MAX_LANGTABLE_EPISODES

        # Process episodes in batches to avoid OOM
    batch_size = 64  # Process episodes in smaller batches
    entries = []
    produced = 0

    print(f"Processing episodes in batches of {batch_size} with {num_workers} workers...")

    # Process episodes in batches to manage memory
    episode_batch = []
    episode_info_batch = []

    for ep_idx, episode in enumerate(tqdm(dataset, desc=f"Processing {base_ds_name} episodes")):
        if ep_idx >= max_limit:
            break

        # Materialize first step for language and sanity checks
        try:
            first_step = next(iter(tfds.as_numpy(episode["steps"])))
        except StopIteration:
            continue

        # Extract task/instruction
        task: Optional[str] = None
        for key in POSSIBLE_LANG_INSTRUCTION_KEYS:
            if key in first_step.get("observation", {}):
                if base_ds_name == "language_table":
                    t = first_step["observation"][key]
                    task = bytes(t[np.where(t != 0)].tolist()).decode("utf-8")
                else:
                    task = first_step["observation"][key].decode()
                break
            elif key in first_step:
                task = first_step[key].decode()
                break
        if not task:
            continue

        # Precompute embedding
        if task not in lang_cache:
            lang_cache[task] = lang_model.encode(task)
        lang_vec = lang_cache[task]

        # Convert TensorFlow objects to numpy for pickling
        try:
            # Convert episode to numpy format for multiprocessing
            episode_np = tfds.as_numpy(episode)

            # Additional safety: ensure all nested objects are numpy
            def ensure_numpy(obj):
                if hasattr(obj, "numpy"):
                    try:
                        return obj.numpy()
                    except:
                        # If numpy() fails, try to convert to list/array
                        if hasattr(obj, "__array__"):
                            return np.array(obj)
                        else:
                            return str(obj)  # Fallback to string representation
                elif isinstance(obj, dict):
                    return {k: ensure_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return type(obj)(ensure_numpy(item) for item in obj)
                else:
                    return obj

            episode_np = ensure_numpy(episode_np)

            # Test if the converted episode is picklable
            try:
                import pickle

                pickle.dumps(episode_np)
                episode_batch.append(episode_np)
                episode_info_batch.append((ep_idx, task, lang_vec))
            except Exception as pickle_error:
                print(f"Warning: Episode {ep_idx} is not picklable after conversion: {pickle_error}")
                continue
        except Exception as e:
            print(f"Warning: Failed to convert episode {ep_idx} to numpy: {e}")
            continue

        # Process batch when it's full or we've reached the limit
        if len(episode_batch) >= batch_size or ep_idx + 1 >= max_limit:
            print(f"Processing batch of {len(episode_batch)} episodes...")

            if num_workers == 1:
                # Sequential processing
                for args in zip(
                    episode_batch,
                    [info[0] for info in episode_info_batch],
                    [info[1] for info in episode_info_batch],
                    [info[2] for info in episode_info_batch],
                    [output_dir] * len(episode_batch),
                    [dataset_name] * len(episode_batch),
                    [max_frames] * len(episode_batch),
                    [fps] * len(episode_batch),
                    [valid_img_keys] * len(episode_batch),
                ):
                    episode_entries = _process_single_oxe_episode(args)
                    entries.extend(episode_entries)
                    produced += len(episode_entries)
            else:
                # Parallel processing
                from multiprocessing import Pool

                # Prepare arguments for workers
                worker_args = list(
                    zip(
                        episode_batch,
                        [info[0] for info in episode_info_batch],
                        [info[1] for info in episode_info_batch],
                        [info[2] for info in episode_info_batch],
                        [output_dir] * len(episode_batch),
                        [dataset_name] * len(episode_batch),
                        [max_frames] * len(episode_batch),
                        [fps] * len(episode_batch),
                        [valid_img_keys] * len(episode_batch),
                    )
                )

                with Pool(processes=num_workers) as pool:
                    results = list(
                        tqdm(
                            pool.imap_unordered(_process_single_oxe_episode, worker_args),
                            total=len(worker_args),
                            desc=f"Processing batch (workers={num_workers})",
                        )
                    )

                # Collect all results
                for episode_entries in results:
                    entries.extend(episode_entries)
                    produced += len(episode_entries)

            # Clear batch for next iteration
            episode_batch = []
            episode_info_batch = []

            # Check if we've reached the limit
            if produced >= max_limit:
                break

        # For language_table, cap the number of episodes considered
        if base_ds_name == "language_table" and ep_idx + 1 >= MAX_LANGTABLE_EPISODES:
            break

    if not entries:
        return Dataset.from_dict(
            {
                "id": [],
                "task": [],
                "lang_vector": [],
                "data_source": [],
                "frames": [],
                "is_robot": [],
                "quality_label": [],
                "preference_group_id": [],
                "preference_rank": [],
            }
        )

    return Dataset.from_list(entries)
