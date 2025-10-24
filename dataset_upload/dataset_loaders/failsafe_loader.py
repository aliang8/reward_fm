import glob
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dataset_upload.helpers import generate_unique_id

TASK_TO_INSTRUCTION = {
    "FailPickCube-v1": "Pick up the red cube",
    "FailPushCube-v1": "Push and move a cube to a goal region in front of it",
    "FailStackCube-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling",
}


class FailSafeFrameListLoader:
    """Pickle-able loader that reads a list of image paths on demand.

    Returns np.ndarray (T, H, W, 3) uint8.
    """

    def __init__(self, image_paths: list[str]) -> None:
        self.image_paths = image_paths

    def __call__(self) -> np.ndarray:
        frames: list[np.ndarray] = []
        for p in self.image_paths:
            img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frames.append(img_rgb)
        if not frames:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)
        frames_np = np.asarray(frames, dtype=np.uint8)
        return frames_np


def _sorted_pngs(dir_path: Path) -> list[str]:
    files = [str(p) for p in dir_path.glob("*.png")]
    def _key(s: str) -> tuple:
        name = os.path.splitext(os.path.basename(s))[0]
        try:
            return (int(name),)
        except Exception:
            return (name,)
    files.sort(key=_key)
    return files


def _make_traj(image_paths: list[str], task: str, instruction: str, is_success: bool, sub_task: str | None = None) -> dict[str, Any]:
    traj: dict[str, Any] = {}
    traj["id"] = generate_unique_id()
    # Combine main instruction with optional sub_task for clarity
    if sub_task:
        traj["task"] = f"{instruction} | Sub-task: {sub_task}"
    else:
        traj["task"] = instruction
    traj["frames"] = FailSafeFrameListLoader(image_paths)
    traj["is_robot"] = True
    traj["quality_label"] = "successful" if is_success else "failure"
    traj["partial_success"] = 1 if is_success else 0
    traj["data_source"] = "failsafe"
    traj["preference_group_id"] = None
    traj["preference_rank"] = None
    return traj


def _gather_full_episodes(task_dir: Path, view: str, instruction: str) -> list[dict]:
    episodes: list[dict] = []
    # Seeds are numbered directories directly under task_dir
    for seed_dir in sorted([p for p in task_dir.iterdir() if p.is_dir()]):
        # Ground truth (success)
        gt_view_dir = seed_dir / "Ground_Truth" / view
        if gt_view_dir.exists():
            imgs = _sorted_pngs(gt_view_dir)
            if imgs:
                episodes.append(_make_traj(imgs, task_dir.name, instruction, is_success=True))

        # Failures: any subfolder except Ground_Truth
        for attempt_dir in sorted([p for p in seed_dir.iterdir() if p.is_dir() and p.name != "Ground_Truth"]):
            view_dir = attempt_dir / view
            if view_dir.exists():
                imgs = _sorted_pngs(view_dir)
                if imgs:
                    episodes.append(_make_traj(imgs, task_dir.name, instruction, is_success=False))
    return episodes


def _gather_sub_episodes_from_json(dataset_root: Path, view: str) -> list[dict]:
    episodes: list[dict] = []
    # JSON files like vla_data_FailPickCube-v1.json, vla_data_GT_PickCube-v1.json etc.
    json_dir = dataset_root / "json_files"
    if not json_dir.exists():
        json_dir = dataset_root  # fallback if jsons are at root

    json_files = glob.glob(str(json_dir / "vla_data_*.json"))
    for jf in sorted(json_files):
        try:
            with open(jf, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            task_key = entry.get("task")
            instruction = entry.get("instruction") or TASK_TO_INSTRUCTION.get(task_key, task_key or "")
            sub_task = entry.get("sub_task")
            failure_type = entry.get("failure_type", "None")
            # Image list is relative to dataset root
            imgs_rel = entry.get("image", [])
            if not imgs_rel:
                continue
            # Filter by desired view: ensure each path contains "/<view>/"
            if view:
                imgs_rel = [p for p in imgs_rel if f"/{view}/" in p or f"\\{view}\\" in p]
            image_paths = [str((dataset_root / p).resolve()) for p in imgs_rel]
            is_success = (failure_type is None) or (str(failure_type).lower() == "none")
            episodes.append(_make_traj(image_paths, task_key or "failsafe", instruction, is_success=is_success, sub_task=sub_task))
    return episodes


def load_failsafe_dataset(dataset_path: str, include_sub_trajectories: bool = True) -> dict[str, list[dict]]:
    """Load FailSafe dataset from local folders and JSON sub-trajectory annotations.

    Args:
        dataset_path: Root directory containing FailPickCube-v1/ FailPushCube-v1/ FailStackCube-v1/ and jsons
        include_sub_trajectories: Whether to parse vla_data_*.json and include sub-task mini-trajectories

    Returns:
        Mapping: instruction string -> list of trajectory dicts
    """
    views = ["front", "side", "wrist"]
    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"FailSafe dataset path not found: {root}")

    task_dirs = [
        p for p in [root / "FailPickCube-v1", root / "FailPushCube-v1", root / "FailStackCube-v1"]
        if p.exists()
    ]

    task_data: dict[str, list[dict]] = {}

    # Full episodes
    for tdir in task_dirs:
        instruction = TASK_TO_INSTRUCTION.get(tdir.name, tdir.name)
        episodes = _gather_full_episodes(tdir, view=view, instruction=instruction)
        if episodes:
            task_data.setdefault(instruction, []).extend(episodes)

    # Sub-trajectory episodes from JSON
    if include_sub_trajectories:
        for view in views:
            sub_episodes = _gather_sub_episodes_from_json(root, view=view)
            for traj in sub_episodes:
                task = traj["task"]
                task_data.setdefault(task, []).append(traj)

    return task_data


