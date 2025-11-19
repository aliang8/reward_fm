#!/usr/bin/env python3
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import cv2

from dataset_upload.helpers import generate_unique_id

TASK_DESCRIPTION_MAP = {
    "Add_pepper_to_the_green_bowl": "Add pepper to the green bowl",
    "Collect_the_fork_to_the_yellow_box": "Collect the fork to the yellow box",
    "Press_the_button": "Press the button",
    "Put_the_bread_in_the_oven": "Put the bread in the oven",
    "Put_the_fruit_in_the_yellow_plate": "Put the apple on the yellow plate",
    "Put_the_marker_into_the_pen_cup": "Put the marker into the pen cup",
    "Put_the_red_bowl_on_the_blue_plate": "Put the red bowl on the blue plate",
    "Put_the_red_cup_on_the_purple_coaster": "Put the red cup on the purple coaster",
    "Put_the_rubber_to_the_blue_pencil_box": "Put the eraser in the blue pencil box",
    "Stack_the_green_block_on_the_red_block": "Stack the green block on the red block",
}

QUALITY_LABEL_MAP = {
    "succ": "successful",
    "success": "successful",
    "successful": "successful",
    "subopt": "suboptimal",
    "suboptimal": "suboptimal",
    "fail": "failure",
    "failure": "failure",
}


class KochArmUTDallasFrameLoader:
    """Lazy loader that extracts RGB frames from MP4 video files."""

    def __init__(self, video_path: str) -> None:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        self.video_path = video_path

    def __call__(self) -> np.ndarray:
        """Load all frames from the MP4 video file."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        frames: list[np.ndarray] = []
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        if not frames:
            raise ValueError(f"No frames found in video: {self.video_path}")

        return np.stack(frames, axis=0).astype(np.uint8)


def _default_task_description(task_key: str) -> str:
    """Convert task key to a readable description."""
    if task_key in TASK_DESCRIPTION_MAP:
        return TASK_DESCRIPTION_MAP[task_key]
    return task_key.replace("_", " ").capitalize()


def _parse_video_metadata(video_filename: str) -> tuple[str, str, str]:
    """Parse video filename to extract task, optimality, and demo_idx.
    
    Expected format: {task}_{optimality}_{demo_idx}.mp4
    Example: pick_blue_cup_success_1.mp4
    """
    # Remove .mp4 extension
    name_without_ext = video_filename.replace(".mp4", "")
    parts = name_without_ext.split("_")
    
    if len(parts) < 3:
        raise ValueError(f"Unexpected video filename format: {video_filename}")
    
    demo_idx = parts[-1]
    optimality_key = parts[-2].lower()
    task_key = "_".join(parts[:-2])
    
    return task_key, optimality_key, demo_idx


def load_koch_arm_ut_dallas_dataset(
    dataset_path: str, max_trajectories: int | None = None
) -> dict[str, list[dict]]:
    """Load Koch Arm UT Dallas trajectories grouped by language task."""

    root = Path(dataset_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    if (root / "koch_arm_ut_dallas").exists():
        root = root / "koch_arm_ut_dallas"

    # Find all MP4 files
    video_files = sorted([p for p in root.glob("*.mp4")])
    if not video_files:
        raise ValueError(f"No MP4 files found in {root}")

    limit = None if max_trajectories is None or max_trajectories < 0 else int(max_trajectories)
    task_data: dict[str, list[dict]] = defaultdict(list)
    total = 0

    for video_path in video_files:
        if limit is not None and total >= limit:
            break

        try:
            task_key, optimality_key, demo_idx = _parse_video_metadata(video_path.name)
        except ValueError as e:
            print(f"⚠️  Skipping {video_path.name}: {e}")
            continue

        if optimality_key not in QUALITY_LABEL_MAP:
            print(f"⚠️  Skipping {video_path.name}: Unknown optimality label '{optimality_key}'")
            continue

        frame_loader = KochArmUTDallasFrameLoader(str(video_path))
        task_description = _default_task_description(task_key)

        trajectory = {
            "id": generate_unique_id(),
            "task": task_description,
            "frames": frame_loader,
            "is_robot": True,
            "quality_label": QUALITY_LABEL_MAP[optimality_key],
            "data_source": "koch_arm_ut_dallas",
        }

        task_data[task_description].append(trajectory)
        total += 1

    print(f"Loaded {total} trajectories from {len(task_data)} tasks in Koch Arm UT Dallas dataset")
    return task_data

