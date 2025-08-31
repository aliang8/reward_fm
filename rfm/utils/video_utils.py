import os
import cv2
import numpy as np
from typing import Any, List, Optional, Tuple
import io
import base64
from PIL import Image


def extract_frames_from_video(video_path: str, fps: int = 1) -> np.ndarray:
    """
    Extract frames from video file at specified FPS.

    Args:
        video_path: Path to the .mp4 file
        fps: Frames per second to extract (default: 1)

    Returns:
        numpy array of shape (num_frames, H, W, C) with extracted frames
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate frame interval for target FPS
    frame_interval = max(1, int(video_fps / fps))

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        frame_count += 1

    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return np.array(frames)


def _ensure_numpy_frames(frames: Any, frames_shape: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Ensure frames are a numpy array of shape (T, H, W, C).

    Accepts bytes (with shape), numpy array, list of numpy frames, or single frame.
    """
    if frames is None:
        return np.empty((0,))

    # Bytes -> numpy using provided shape
    if isinstance(frames, (bytes, bytearray)):
        if frames_shape is None:
            # Fallback: interpret as uint8 flat array (cannot reshape reliably)
            arr = np.frombuffer(frames, dtype=np.uint8)
            return arr
        if isinstance(frames_shape, list):
            frames_shape = tuple(frames_shape)
        try:
            return np.frombuffer(frames, dtype=np.uint8).reshape(frames_shape)
        except Exception:
            return np.frombuffer(frames, dtype=np.uint8)

    # Already a numpy array
    if isinstance(frames, np.ndarray):
        return frames

    # List of numpy arrays
    if isinstance(frames, list) and all(isinstance(f, np.ndarray) for f in frames):
        return np.stack(frames, axis=0)

    # Unsupported (e.g., file paths) – return as empty; upstream should handle
    return np.empty((0,))


def decode_frames_b64(frames_b64: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for s in frames_b64:
        buf = io.BytesIO(base64.b64decode(s))
        img = Image.open(buf).convert("RGB")
        images.append(img)
    return images


def frames_to_base64_images(frames: Any, frames_shape: Optional[Tuple[int, int, int, int]] = None) -> List[str]:
    """Convert frames to a list of base64-encoded JPEG strings.

    Frames can be ndarray (T,H,W,C), bytes + shape, list of ndarray, or a single frame.
    """
    arr = _ensure_numpy_frames(frames, frames_shape)
    if arr.size == 0:
        return []

    # Normalize to (T, H, W, C)
    if arr.ndim == 3:  # single frame (H,W,C)
        arr = arr[None, ...]
    elif arr.ndim != 4:
        # Unknown shape: cannot encode reliably
        return []

    encoded: List[str] = []
    for i in range(arr.shape[0]):
        frame = arr[i]
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return encoded
