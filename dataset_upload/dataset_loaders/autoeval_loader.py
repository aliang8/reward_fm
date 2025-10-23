import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from dataset_upload.helpers import generate_unique_id


class AutoEvalFrameLoader:
    """Pickle-able loader that decodes image frames from a pickled episode file on demand.

    Expects each pickle to contain a sequence of steps with observations under
    f.obs['image_primary'] (RGB arrays or encodable).
    """

    def __init__(self, pickle_path: str) -> None:
        self.pickle_path = pickle_path

    def __call__(self) -> np.ndarray:
        with open(self.pickle_path, "rb") as f:
            ep = pickle.load(f)
        frames = np.asarray(ep.obs["image_primary"], dtype=np.uint8)
        return frames


def _make_traj(pickle_path: Path, task: str, success: bool) -> dict[str, Any]:
    traj: dict[str, Any] = {}
    traj["id"] = generate_unique_id()
    traj["task"] = task
    traj["frames"] = AutoEvalFrameLoader(str(pickle_path))
    traj["is_robot"] = True
    traj["quality_label"] = "successful" if success else "failure"
    traj["partial_success"] = 1 if success else 0
    traj["data_source"] = "autoeval"
    traj["preference_group_id"] = None
    traj["preference_rank"] = None
    return traj


def load_autoeval_dataset(dataset_path: str) -> dict[str, list[dict]]:
    """Load AutoEval pickled episodes and return paired trajectories per task.

    Assumes structure like:
      <dataset_path>/eval_data/<episode_folder>/*.pkl

    We treat each subfolder under eval_data as a group, and within it we look
    for pairs of success/failure pickles for the same episode id when possible.
    We also print counts of successes and failures and keep only paired ones.
    """

    root = Path(os.path.expanduser(dataset_path))
    eval_root = root / "eval_data"
    if not eval_root.exists():
        raise FileNotFoundError(f"AutoEval eval_data folder not found: {eval_root}")

    pkl_files = list(eval_root.glob("*.pkl"))

    success_count = 0
    total_count = 0

    task_data: dict[str, list[dict]] = defaultdict(list)

    for pkl in pkl_files:


        # If success flag is stored inside pickle instead of filename, detect here
        def read_success_flag(path: Path) -> bool | None:
            with open(path, "rb") as f:
                ep = pickle.load(f)
            # Try step-level success or episode attribute
            if ep.success is not None:
                return bool(ep.success)
            return None

        with open(pkl, "rb") as f:
            ep = pickle.load(f)
            success = ep.success
            task = ep.language_command
        if success is None or task is None:
            continue

        success_count += int(success)
        total_count += 1
        task_data[task].append(_make_traj(pkl, task, success))

    print(f"AutoEval: successes={success_count}, total={total_count}")
    return task_data


