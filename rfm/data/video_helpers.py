#!/usr/bin/env python3
"""
Video helper utilities for robust loading of videos with codec handling and
backend fallbacks. Centralized here to be reused across dataset loaders.
"""

import os
import tempfile
import shutil
from typing import Optional, List

import numpy as np
import cv2
import subprocess


def _ffprobe_codec_name(path: str) -> Optional[str]:
    """Return codec_name for the first video stream using ffprobe, or None on failure."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nk=1:nw=1",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        codec_name = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else None
        return codec_name
    except Exception:
        return None


def _reencode_to_h264(input_path: str) -> Optional[str]:
    """Re-encode input video to H.264 yuv420p if ffmpeg is available. Returns output path or None."""
    if shutil.which("ffmpeg") is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
        output_path = tmp_out.name
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                output_path,
            ],
            check=True,
        )
        return output_path
    except Exception:
        try:
            os.unlink(output_path)
        except Exception:
            pass
        return None


def _open_with_best_backend(path: str) -> Optional[cv2.VideoCapture]:
    """Try multiple OpenCV backends and return an opened capture or None."""
    backends: List[int] = [getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY), cv2.CAP_ANY]
    for backend in backends:
        try:
            cap_try = cv2.VideoCapture(path, backend)
            if cap_try.isOpened():
                # Validate we can read at least one frame
                ret, test_frame = cap_try.read()
                if ret and test_frame is not None:
                    cap_try.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return cap_try
            cap_try.release()
        except Exception:
            try:
                cap_try.release()
            except Exception:
                pass
    return None


def load_video_frames(video_input) -> np.ndarray:
    """Load video frames (RGB uint8) from a file path (str/Path) or video bytes.

    - For byte inputs, detect AV1 and re-encode to H.264 for compatibility
    - Uses OpenCV with FFMPEG backend when available
    - Returns numpy array of shape (T, H, W, 3) in RGB order
    """
    temp_files_to_cleanup: List[str] = []
    cap: Optional[cv2.VideoCapture] = None

    if isinstance(video_input, (str, os.PathLike)):
        video_path = str(video_input)
        cap = _open_with_best_backend(video_path)
    else:
        # Save bytes to temp file first
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(video_input)
            temp_input_path = temp_file.name
        temp_files_to_cleanup.append(temp_input_path)

        # If the input is AV1, transcode to H.264 for compatibility
        codec_name = _ffprobe_codec_name(temp_input_path)
        decodable_path = temp_input_path
        if codec_name == "av1":
            h264_path = _reencode_to_h264(temp_input_path)
            if h264_path is not None:
                temp_files_to_cleanup.append(h264_path)
                decodable_path = h264_path
        cap = _open_with_best_backend(decodable_path)

    try:
        frames: List[np.ndarray] = []
        if cap is None or not cap.isOpened():
            raise ValueError("Could not open video file with available backends. If the source is AV1, install AV1 support or enable ffmpeg re-encode.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB for consistency
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)

        cap.release()

        # Clean up temp file(s) if we created any
        for path in temp_files_to_cleanup:
            try:
                os.unlink(path)
            except Exception:
                pass

        if len(frames) == 0:
            raise ValueError("No frames could be extracted from video")

        return np.array(frames)

    except Exception as e:
        if cap is not None:
            cap.release()
        # Clean up temp files in case of error
        for path in temp_files_to_cleanup:
            try:
                os.unlink(path)
            except Exception:
                pass
        raise e

