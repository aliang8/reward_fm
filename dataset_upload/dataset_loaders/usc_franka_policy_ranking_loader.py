#!/usr/bin/env python3
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from dataset_upload.helpers import generate_unique_id

TASK_DESCRIPTION_MAP = {
    "plate_in_sink": "Put the yellow plate in the sink",
    "fold_towel": "Fold the towel in half",
    "red_cube_on_gray_cube": "Stack the red cube on the gray cube",
    "put_banana_on_red_plate": "Put the banana on the red plate",
}

QUALITY_LABEL_MAP = {
    "success": "successful",
    "successful": "successful",
    "subopt": "suboptimal",
    "suboptimal": "suboptimal",
    "fail": "failure",
    "failure": "failure",
}

CAMERA_PATTERN = "camera_south_color*.png"


class USCFrankaFrameLoader:
    """Lazy loader that stitches RGB frames from PNG files."""

    def __init__(self, image_paths: Iterable[str]) -> None:
        image_paths = list(image_paths)
        if not image_paths:
            raise ValueError("USCFrankaFrameLoader requires at least one image path")
        self.image_paths = sorted(image_paths)

    def _load_frame(self, path: str) -> np.ndarray:
        with Image.open(path) as img:
            frame = np.array(img.convert("RGB"), dtype=np.uint8)
        if frame.ndim != 3:
            raise ValueError(f"Expected 3D RGB frame, got shape {frame.shape} for {path}")
        return frame

    def __call__(self) -> np.ndarray:
        frames = [self._load_frame(path) for path in self.image_paths]
        if not frames:
            raise ValueError("No RGB frames were loaded for this trajectory")
        return np.stack(frames, axis=0)


def _default_task_description(task_key: str) -> str:
    if task_key in TASK_DESCRIPTION_MAP:
        return TASK_DESCRIPTION_MAP[task_key]
    return task_key.replace("_", " ").capitalize()


def _parse_folder_metadata(folder_name: str) -> tuple[str, str, str]:
    parts = folder_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected folder name format: {folder_name}")
    attempt_id = parts[-1]
    optimality_key = parts[-2].lower()
    task_key = "_".join(parts[:-2])
    return task_key, optimality_key, attempt_id


def load_usc_franka_policy_ranking_dataset(
    dataset_path: str, max_trajectories: int | None = None
) -> dict[str, list[dict]]:
    """Load USC Franka policy ranking trajectories grouped by language task."""

    root = Path(dataset_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    folders = sorted([path for path in root.iterdir() if path.is_dir()])
    if not folders:
        raise ValueError(f"No task folders found in {root}")

    limit = None if max_trajectories is None or max_trajectories < 0 else int(max_trajectories)
    task_data: dict[str, list[dict]] = defaultdict(list)
    total = 0

    for folder in folders:
        if limit is not None and total >= limit:
            break

        task_key, optimality_key, attempt_id = _parse_folder_metadata(folder.name)
        if optimality_key not in QUALITY_LABEL_MAP:
            raise ValueError(f"Unknown optimality label '{optimality_key}' in folder {folder.name}")

        images_dir = folder / "images"
        image_paths = sorted(str(p) for p in images_dir.glob(CAMERA_PATTERN))
        if not image_paths:
            print(f"⚠️  Skipping {folder} (no matching {CAMERA_PATTERN} files found)")
            continue

        frame_loader = USCFrankaFrameLoader(image_paths)
        task_description = _default_task_description(task_key)

        trajectory = {
            "id": generate_unique_id(),
            "task": task_description,
            "frames": frame_loader,
            "is_robot": True,
            "quality_label": QUALITY_LABEL_MAP[optimality_key],
            "data_source": "usc_franka_policy_ranking",
        }
        import ipdb; ipdb.set_trace()

        task_data[task_description].append(trajectory)
        total += 1

    print(f"Loaded {total} trajectories from {len(task_data)} tasks in USC Franka Policy Ranking dataset")
    return task_data



