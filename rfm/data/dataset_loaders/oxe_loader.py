import os
import itertools
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
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

# TODO: double check toto since everything is just "pour". Maybe "pour into the cup"? for all?


class OXEFrameLoader:
    """Serializable frame loader that re-opens TFDS and extracts a specific episode.

    Stores only serializable identifiers so it can be pickled for multiprocessing.
    """

    def __init__(self, builder_dir: str, dataset_name: str, image_key: str, episode_index: int):
        self.builder_dir = builder_dir
        self.dataset_name = dataset_name
        self.image_key = image_key
        self.episode_index = int(episode_index)

    def __call__(self) -> np.ndarray:
        """Re-open TFDS from builder_dir and extract frames for the episode index."""
        # Ensure TF runs CPU-only in worker processes to avoid CUDA context issues
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        tf.config.set_visible_devices([], "GPU")
        builder = tfds.builder_from_directory(self.builder_dir)
        # Use deterministic ordering to ensure index alignment with the metadata pass
        dataset = builder.as_dataset(split=f"train[{self.episode_index}:{self.episode_index + 1}]", shuffle_files=False)

        try:
            target_episode = next(iter(dataset))
        except StopIteration:
            return None

        images = []
        for step in target_episode["steps"]:
            if self.image_key in step["observation"]:
                images.append(step["observation"][self.image_key].numpy())

        return np.stack(images) if len(images) > 0 else None


def load_oxe_dataset(dataset_path: str, max_trajectories: int = -1, dataset_name: str = None) -> Dict[str, List[Dict]]:
    """Load OXE dataset and organize by task, without a separate iterator class."""

    # pop the oxe_ prefix from the dataset name
    dataset_name = dataset_name.replace("oxe_", "")

    if dataset_name is None:
        raise ValueError("Dataset name is required")
    datasets_to_iterate = [dataset_name]

    if max_trajectories == -1:
        max_traj_per_dataset = float("inf")
    else:
        max_traj_per_dataset = max(max_trajectories // len(datasets_to_iterate), 1)
    print(f"max_trajectories per task for OXE is: {max_traj_per_dataset}")
    print(f"Loading OXE dataset from: {dataset_path}")
    print("=" * 100)
    print("LOADING OXE DATASET")
    print("=" * 100)

    dataset_path = Path(os.path.expanduser(dataset_path))
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    total_trajs = 0
    task_data: Dict[str, List[Dict]] = {}
    for dataset_name in datasets_to_iterate:
        print(f"Loading {dataset_name}")
        # make the builder locally to avoid calls to google
        # check which version is available
        versions = os.listdir(f"{dataset_path}/{dataset_name}")
        if len(versions) == 0:
            raise ValueError(f"No versions found for {dataset_name} in {dataset_path}")
        else:
            for version in versions:
                if "incomplete" in version:
                    continue
                else:
                    builder = tfds.builder_from_directory(f"{dataset_path}/{dataset_name}/{version}")
                    # Disable shuffling to keep a stable episode order across processes
                    dataset = builder.as_dataset(split="train", shuffle_files=False)
                    break
        if dataset is None:
            raise ValueError(f"No dataset found for {dataset_name} in {dataset_path}")
        img_key_to_name = OXE_DATASET_CONFIGS[dataset_name]["image_obs_keys"]
        img_key_to_name = {
            k: v for k, v in img_key_to_name.items() if k != "wrist"
        }  # remove wrist since it's not a great view
        valid_img_keys = list(img_key_to_name.values())
        # load each possible valid key as a separate traj and make sure that if they're all black images don't include.
        # skip data loading if no lang
        valid_samples = 0
        builder_dir = os.path.join(str(dataset_path), dataset_name, version)
        for ep_idx, episode in enumerate(tqdm(dataset, desc=f"Processing {dataset_name} episodes")):
            # Safely materialize the first step via tfds.as_numpy to avoid TF variant conversion issues
            try:
                first_step = next(iter(tfds.as_numpy(episode["steps"])))
            except StopIteration:
                continue
            task = None
            for key in POSSIBLE_LANG_INSTRUCTION_KEYS:
                if key in first_step["observation"]:
                    if dataset_name == "language_table":
                        task = first_step["observation"][key]
                        task = bytes(task[np.where(task != 0)].tolist()).decode("utf-8")
                    else:
                        task = first_step["observation"][key].decode()
                    break
                elif key in first_step:
                    task = first_step[key].decode()
                    break
            if task is None or task == "":
                continue

            # create a trajectory for each image key in case trajectory has multiple valid viewpoints
            for img_name_in_step in valid_img_keys:
                if img_name_in_step in first_step["observation"]:
                    # if all black then skip
                    if np.all(first_step["observation"][img_name_in_step] == 0):
                        continue
                    frame_loader = OXEFrameLoader(
                        builder_dir=builder_dir,
                        dataset_name=dataset_name,
                        image_key=img_name_in_step,
                        episode_index=ep_idx,
                    )
                    valid_samples += 1
                    trajectory = {
                        "frames": frame_loader,
                        "actions": None,  # too annoying to add for OXE right now since there's lots of per-dataset processing
                        "is_robot": True,
                        "task": task,
                        "quality_label": "successful",
                        "preference_group_id": None,
                        "preference_rank": None,
                    }

                    task_data.setdefault(task, []).append(trajectory)
            if dataset_name == "language_table":
                if valid_samples >= MAX_LANGTABLE_EPISODES:
                    break
            if valid_samples >= max_traj_per_dataset:
                break
        print(f"Loaded {valid_samples} trajectories for {dataset_name}")
        total_trajs += valid_samples
    return total_trajs, task_data


# --------------------------------------------
# New: Direct OXE -> HF converter that writes videos and returns HF entries
# --------------------------------------------

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


def convert_oxe_dataset_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: Optional[int] = None,
    max_frames: int = 64,
    fps: int = 10,
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

    dataset = builder.as_dataset(split="train", shuffle_files=False)

    # Determine valid image observation keys
    img_key_to_name = OXE_DATASET_CONFIGS[base_ds_name]["image_obs_keys"]
    img_key_to_name = {k: v for k, v in img_key_to_name.items() if k != "wrist"}
    valid_img_keys = list(img_key_to_name.values())

    # Language model and cache
    lang_model = load_sentence_transformer_model()
    lang_cache: Dict[str, Any] = {}

    entries: List[Dict[str, Any]] = []
    produced = 0
    max_limit = float("inf") if (max_trajectories is None or max_trajectories == -1) else int(max_trajectories)

    if "language_table" in base_ds_name:
        max_limit = MAX_LANGTABLE_EPISODES

    for ep_idx, episode in enumerate(tqdm(dataset, desc=f"Converting {base_ds_name} episodes")):
        if produced >= max_limit:
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

        # Build image sequences for each view
        # Iterate all steps once to avoid re-materializing for each view
        steps_np = list(tfds.as_numpy(episode["steps"]))

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
                entries.append(entry)
                produced += 1
                if produced >= max_limit:
                    break

        # For language_table, cap the number of episodes considered
        if base_ds_name == "language_table" and ep_idx + 1 >= MAX_LANGTABLE_EPISODES:
            break

    if not entries:
        return Dataset.from_dict({
            "id": [],
            "task": [],
            "lang_vector": [],
            "data_source": [],
            "frames": [],
            "is_robot": [],
            "quality_label": [],
            "preference_group_id": [],
            "preference_rank": [],
        })

    return Dataset.from_list(entries)
