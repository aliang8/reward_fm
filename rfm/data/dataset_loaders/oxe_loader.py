import tensorflow as tf
from tqdm import tqdm
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from rfm.data.dataset_helpers.oxe_helper import OXE_DATASET_CONFIGS
import os
import itertools
import tensorflow_datasets as tfds
import numpy as np
from typing import Dict, List
from pathlib import Path
from rfm.data.helpers import generate_unique_id

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
        builder = tfds.builder_from_directory(self.builder_dir)
        dataset = builder.as_dataset(split="train")

        target_episode = None

        # Faster: Use itertools.islice to jump directly to the desired episode index
        target_episode = next(itertools.islice(dataset, self.episode_index, self.episode_index + 1), None)

        if target_episode is None:
            return None

        images = []
        for step in tfds.as_numpy(target_episode["steps"]):
            if self.image_key in step["observation"]:
                images.append(step["observation"][self.image_key])

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
                    dataset = builder.as_dataset(split="train")
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
            if task is None:
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
