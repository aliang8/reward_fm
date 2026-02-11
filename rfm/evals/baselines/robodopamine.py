#!/usr/bin/env python3
"""Robo-Dopamine (GRM) baseline for progress prediction.

Reference: https://github.com/FlagOpen/Robo-Dopamine
Model: https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-3B
Setup: Clone Robo-Dopamine and set ROBODOPAMINE_PATH=/path/to/Robo-Dopamine
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from rfm.utils.logger import get_logger
from rfm.evals.baselines.rbd_inference import GRMInference

logger = get_logger()


class RoboDopamine:
    """Robo-Dopamine GRM baseline. Uses single-view frames for all three camera inputs."""

    def __init__(
        self,
        model_path: str = "tanhuajie2001/Robo-Dopamine-GRM-3B",
        frame_interval: int = 1,
        batch_size: int = 1,
        eval_mode: str = "incremental",
    ):
        self.model_path = model_path
        self.frame_interval = frame_interval
        self.batch_size = batch_size
        self.eval_mode = eval_mode
        self._grm = GRMInference(model_path=model_path, max_image_num=8)

    def compute_progress(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: Optional[str] = None,
    ) -> np.ndarray:
        if frames_array is None or frames_array.size == 0:
            return np.array([], dtype=np.float64)

        num_frames = frames_array.shape[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir) / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for i in range(num_frames):
                frame = frames_array[i]
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(frames_dir / f"frame_{i:06d}.png"),
                    frame_bgr,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
                )

            out_root = Path(tmpdir) / "out"
            out_root.mkdir(parents=True, exist_ok=True)
            # run_pipeline accepts either: (1) a dir of .png files (sorted by name), or (2) a video path (.mp4).
            # We pass a directory of frame_000000.png, frame_000001.png, ...
            # frame_interval: step between sampled frames (0, interval, 2*interval, ...; last frame always included).
            # So frame_interval=1 uses every frame; frame_interval=5 uses every 5th frame.
            run_root = self._grm.run_pipeline(
                cam_high_path=str(frames_dir),
                cam_left_path=None,
                cam_right_path=None,
                out_root=str(out_root),
                task=task_description,
                frame_interval=self.frame_interval,
                batch_size=self.batch_size,
                goal_image=None,
                eval_mode=self.eval_mode,
                visualize=False,
            )

            pred_path = Path(run_root) / "pred_vllm.json"
            with open(pred_path, "r", encoding="utf-8") as f:
                results = json.load(f)

        progress_list = [0.0]
        for item in results:
            p = item.get("progress", 0.0)
            if isinstance(p, str) and p == "Error":
                p = progress_list[-1] if progress_list else 0.0
            progress_list.append(float(p))

        progress_arr = np.clip(np.array(progress_list, dtype=np.float64), 0.0, 1.0)
        if len(progress_arr) < num_frames:
            progress_arr = np.pad(
                progress_arr,
                (0, num_frames - len(progress_arr)),
                mode="edge",
            )
        elif len(progress_arr) > num_frames:
            progress_arr = progress_arr[:num_frames]

        return progress_arr
