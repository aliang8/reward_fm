from rfm.data.dataset_helpers.oxe_helper import OXE_DATASET_CONFIGS
import os
import tensorflow_datasets as tfds
import numpy as np
from typing import Dict, List
from pathlib import Path
from rfm.data.helpers import generate_unique_id

OXE_VALID_DATASETS = sorted([
    "austin_buds_dataset_converted_externally_to_rlds",
    "dlr_edan_shared_control_converted_externally_to_rlds",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "toto",
    "austin_sirius_dataset_converted_externally_to_rlds",
    "droid",
    "jaco_play",
    "ucsd_kitchen_dataset_converted_externally_to_rlds",
    "berkeley_cable_routing",
    "fmb",
    "language_table",
    "utaustin_mutex",
    "berkeley_fanuc_manipulation",
    "fractal20220817_data",
    "stanford_hydra_dataset_converted_externally_to_rlds",
    "viola",
    "bridge_v2",
    "furniture_bench_dataset_converted_externally_to_rlds",
    "taco_play",
])
POSSIBLE_LANG_INSTRUCTION_KEYS = [  # valid keys for language instruction in OXE
    "natural_language_instruction",
    "language_instruction",
    "instruction",
]
possible_valid_keys = ["primary", "secondary", "tertiary"]

# TODO: double check toto since everything is just "pour". Maybe "pour into the cup"? for all?


class OXEFrameLoader:
    """Pickle-able frame loader for OXE videos."""

    def __init__(self, episode, image_key: str, dataset_name: str):
        self.episode = episode
        self.image_key = image_key
        self.dataset_name = dataset_name

    def __call__(self) -> np.ndarray:
        """Load frames from the MP4 file when called."""
        images = []
        for step in self.episode["steps"].as_numpy_iterator():
            # extract video
            images.append(step["observation"][self.image_key])
        return images


def load_oxe_dataset(dataset_path: str, max_trajectories: int = -1) -> Dict[str, List[Dict]]:
    """Load OXE dataset and organize by task, without a separate iterator class."""
    print(f"max_trajectories per task for OXE is: {max_trajectories}")
    print(f"Loading OXE dataset from: {dataset_path}")
    print("=" * 100)
    print("LOADING OXE DATASET")
    print("=" * 100)

    dataset_path = Path(os.path.expanduser(dataset_path))
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    task_data: Dict[str, List[Dict]] = {}
    for dataset_name in OXE_VALID_DATASETS:
        dataset = tfds.load(dataset_name, data_dir=dataset_path, split="train")
        img_key_to_name = OXE_DATASET_CONFIGS[dataset_name]["image_obs_keys"]
        # load each possible valid key as a separate traj and make sure that if they're all black images don't include.
        # skip data loading if no lang
        valid_samples = 0
        for episode in dataset:
            if valid_samples >= max_trajectories:
                break
            first_step = next(episode["steps"].as_numpy_iterator())
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
            for _, img_name_in_step in img_key_to_name.items():
                if img_name_in_step in first_step["observation"]:
                    # if all black then skip
                    if np.all(first_step["observation"][img_name_in_step] == 0):
                        continue
                    # iterate to get the actions
                    actions = []
                    for step in episode["steps"]:
                        actions.append(step["action"])
                    actions = np.array(actions)
                    frame_loader = OXEFrameLoader(episode, img_name_in_step, dataset_name)
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
    return task_data
