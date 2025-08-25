import os
import cv2
import numpy as np


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