import base64
import io
import os
import random
from typing import Any, Optional

import cv2
import numpy as np
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
    int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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


def _ensure_numpy_frames(frames: Any, frames_shape: tuple[int, int, int, int] | None = None) -> np.ndarray:
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


def decode_frames_b64(frames_b64: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for s in frames_b64:
        buf = io.BytesIO(base64.b64decode(s))
        img = Image.open(buf).convert("RGB")
        images.append(img)
    return images


def frames_to_base64_images(frames: Any, frames_shape: tuple[int, int, int, int] | None = None) -> list[str]:
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

    encoded: list[str] = []
    for i in range(arr.shape[0]):
        frame = arr[i]
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return encoded


def add_text_overlay(frame: np.ndarray, text: str, position: tuple[int, int] = (10, 10), 
                     font_scale: float = 0.5, color: tuple[int, int, int] = (255, 255, 255),
                     thickness: int = 1, bg_color: Optional[tuple[int, int, int]] = None) -> np.ndarray:
    """
    Add text overlay to a frame.
    
    Args:
        frame: Frame in (H, W, C) format, uint8, RGB
        text: Text to add
        position: (x, y) position of text (bottom-left corner of text)
        font_scale: Font scale
        color: Text color (RGB format, will be converted to BGR for cv2)
        thickness: Text thickness
        bg_color: Optional background color (RGB format, will be converted to BGR for cv2)
    
    Returns:
        Frame with text overlay (H, W, C), RGB
    """
    frame_with_text = frame.copy()
    
    # Ensure frame has 3 channels
    if frame_with_text.ndim != 3 or frame_with_text.shape[2] != 3:
        raise ValueError(f"Expected frame with shape (H, W, 3), got {frame_with_text.shape}")
    
    # Convert RGB to BGR for cv2 (cv2 uses BGR)
    frame_bgr = cv2.cvtColor(frame_with_text, cv2.COLOR_RGB2BGR)
    
    # Convert colors from RGB to BGR for cv2
    color_bgr = (color[2], color[1], color[0])
    bg_color_bgr = (bg_color[2], bg_color[1], bg_color[0]) if bg_color is not None else None
    
    # Get text size for background box
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Draw background box if specified
    if bg_color_bgr is not None:
        cv2.rectangle(
            frame_bgr,
            (position[0] - 5, position[1] - text_height - baseline - 5),
            (position[0] + text_width + 5, position[1] + 5),
            bg_color_bgr,
            -1
        )
    
    # Draw text (position is bottom-left corner)
    cv2.putText(
        frame_bgr,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color_bgr,
        thickness,
        cv2.LINE_AA
    )
    
    # Convert back to RGB
    frame_with_text = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    return frame_with_text


def create_video_grid_with_progress(
    video_frames_list: list[Optional[np.ndarray]], 
    trajectory_progress_data: Optional[list[dict]] = None,
    grid_size: tuple[int, int] = (3, 3),
    max_videos: int = 9,
    progress_key_pred: str = "progress_pred",
    progress_key_target: str = "target_progress"
) -> Optional[np.ndarray]:
    """
    Create a grid of videos with progress information overlaid on each video.
    
    Args:
        video_frames_list: List of videos, each in (T, C, H, W) format or None
        trajectory_progress_data: Optional list of dicts with progress information, one per video.
                                  Each dict should have progress_key_pred and progress_key_target keys.
                                  If None, no progress overlay will be added.
        grid_size: Tuple of (rows, cols) for the grid
        max_videos: Maximum number of videos to sample
        progress_key_pred: Key for predicted progress in trajectory_progress_data
        progress_key_target: Key for target progress in trajectory_progress_data
    
    Returns:
        Grid video in (T, C, H, W) format, or None if insufficient valid videos
    """
    # Filter out None videos
    valid_videos = [(idx, v) for idx, v in enumerate(video_frames_list) if v is not None]
    
    grid_cells = grid_size[0] * grid_size[1]
    if len(valid_videos) == 0:
        return None
    
    # Sample available videos (up to grid_cells)
    num_to_sample = min(grid_cells, len(valid_videos))
    sampled_videos = random.sample(valid_videos, num_to_sample)
    
    # Get corresponding progress data if available (assume alignment)
    if trajectory_progress_data is not None:
        valid_items = [
            (v_idx, v, trajectory_progress_data[v_idx])
            for v_idx, v in sampled_videos
        ]
    else:
        valid_items = [(v_idx, v, None) for v_idx, v in sampled_videos]
    
    # Find maximum time dimension across valid videos
    max_time = max(v.shape[0] for _, v, _ in valid_items) if valid_items else 1
    
    # Resize and normalize videos to same size for grid
    processed_videos = []
    target_h, target_w = 128, 128  # Target size for each cell in grid
    
    for _, video, progress_data in valid_items:
        # Pad video to max_time by repeating the last frame
        video_len = video.shape[0]
        if video_len < max_time:
            # Get last frame and repeat it
            last_frame = video[-1:]  # (1, C, H, W)
            num_padding = max_time - video_len
            padding = np.repeat(last_frame, num_padding, axis=0)  # (num_padding, C, H, W)
            video = np.concatenate([video, padding], axis=0)  # (max_time, C, H, W)
        
        # Convert (T, C, H, W) to (T, H, W, C) for processing
        video = video.transpose(0, 2, 3, 1)
        
        # Get progress information
        progress_pred = None
        progress_target = None
        if progress_data is not None:
            progress_pred = progress_data.get(progress_key_pred, None)
            progress_target = progress_data.get(progress_key_target, None)
        
        # Resize each frame and add progress overlay
        resized_frames = []
        for t, frame in enumerate(video):
            # Ensure uint8 format
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            frame_resized = cv2.resize(frame, (target_w, target_h))
            
            # Add progress text overlay in bottom right corner
            if progress_pred is not None and progress_target is not None:
                # Get progress value (use last value if it's a sequence)
                if isinstance(progress_pred, (list, np.ndarray)):
                    pred_val = float(progress_pred[-1]) if len(progress_pred) > 0 else 0.0
                else:
                    pred_val = float(progress_pred)
                
                if isinstance(progress_target, (list, np.ndarray)):
                    target_val = float(progress_target[-1]) if len(progress_target) > 0 else 0.0
                else:
                    target_val = float(progress_target)
                
                # Format text
                progress_text = f"P:{pred_val:.2f} T:{target_val:.2f}"
                
                # Calculate position (bottom right, with padding)
                text_x = target_w - 110
                text_y = target_h - 10
                
                # Add text with background
                frame_resized = add_text_overlay(
                    frame_resized,
                    progress_text,
                    position=(text_x, text_y),
                    font_scale=0.4,
                    color=(255, 255, 255),  # White text
                    thickness=1,
                    bg_color=(0, 0, 0)  # Black background
                )
            
            resized_frames.append(frame_resized)
        
        # Convert back to (T, C, H, W)
        video_resized = np.array(resized_frames).transpose(0, 3, 1, 2)
        processed_videos.append(video_resized)
    
    # Fill remaining grid cells with black videos
    num_black_videos = grid_cells - len(processed_videos)
    for _ in range(num_black_videos):
        # Create black video: (T, C, H, W) with all zeros (black)
        black_video = np.zeros((max_time, 3, target_h, target_w), dtype=np.uint8)
        processed_videos.append(black_video)
    
    # Arrange videos in grid
    grid_frames = []
    for t in range(max_time):
        grid_rows = []
        for row in range(grid_size[0]):
            row_frames = []
            for col in range(grid_size[1]):
                vid_idx = row * grid_size[1] + col
                frame = processed_videos[vid_idx][t]  # (C, H, W)
                # Convert to (H, W, C) for concatenation
                frame = frame.transpose(1, 2, 0)
                row_frames.append(frame)
            # Concatenate horizontally
            row_concat = np.concatenate(row_frames, axis=1)  # (H, total_W, C)
            grid_rows.append(row_concat)
        # Concatenate vertically
        grid_frame = np.concatenate(grid_rows, axis=0)  # (total_H, total_W, C)
        # Convert back to (C, H, W)
        grid_frame = grid_frame.transpose(2, 0, 1)
        grid_frames.append(grid_frame)
    
    # Stack frames: (T, C, H, W)
    grid_video = np.stack(grid_frames, axis=0)
    return grid_video


def create_frame_pair_with_progress(
    eval_result: dict,
    target_h: int = 224,
    target_w: int = 224
) -> Optional[np.ndarray]:
    """
    Create a side-by-side pair of frames (first and last) with progress annotations.
    
    Args:
        eval_result: Evaluation result dict with video_path, progress_pred, target_progress
        target_h: Target height for frames
        target_w: Target width for frames
    
    Returns:
        Single frame with two frames side-by-side in (C, H, 2*W) format, or None if unavailable
    """
    from rfm.data.datasets.helpers import load_frames_from_npz
    
    video_path = eval_result.get("video_path")
    if video_path is None:
        return None
    
    try:
        frames = load_frames_from_npz(video_path)
        frames = frames.transpose(0, 3, 1, 2)  # (T, C, H, W)
        
        if frames.shape[0] < 2:
            return None
        
        # Get first and last frames
        frame1 = frames[0]  # (C, H, W)
        frame2 = frames[-1]  # (C, H, W)
        
        # Get progress values (use first and last from sequences)
        progress_pred = eval_result.get("progress_pred")
        target_progress = eval_result.get("target_progress")
        
        # Extract progress values for first and last frames
        if isinstance(progress_pred, (list, np.ndarray)):
            pred_val_first = float(progress_pred[0]) if len(progress_pred) > 0 else 0.0
            pred_val_last = float(progress_pred[-1]) if len(progress_pred) > 0 else 0.0
        else:
            pred_val_first = float(progress_pred) if progress_pred is not None else 0.0
            pred_val_last = pred_val_first
        
        if isinstance(target_progress, (list, np.ndarray)):
            target_val_first = float(target_progress[0]) if len(target_progress) > 0 else 0.0
            target_val_last = float(target_progress[-1]) if len(target_progress) > 0 else 0.0
        else:
            target_val_first = float(target_progress) if target_progress is not None else 0.0
            target_val_last = target_val_first
        
        # Process first frame
        frame1_hwc = frame1.transpose(1, 2, 0)  # (H, W, C)
        if frame1_hwc.dtype != np.uint8:
            frame1_hwc = np.clip(frame1_hwc * 255, 0, 255).astype(np.uint8)
        frame1_resized = cv2.resize(frame1_hwc, (target_w, target_h))
        
        # Add progress annotation to first frame
        progress_text1 = f"P:{pred_val_first:.2f} T:{target_val_first:.2f}"
        text_x = target_w - 110
        text_y = target_h - 10
        frame1_resized = add_text_overlay(
            frame1_resized,
            progress_text1,
            position=(text_x, text_y),
            font_scale=0.4,
            color=(255, 255, 255),
            thickness=1,
            bg_color=(0, 0, 0)
        )
        
        # Process last frame
        frame2_hwc = frame2.transpose(1, 2, 0)  # (H, W, C)
        if frame2_hwc.dtype != np.uint8:
            frame2_hwc = np.clip(frame2_hwc * 255, 0, 255).astype(np.uint8)
        frame2_resized = cv2.resize(frame2_hwc, (target_w, target_h))
        
        # Add progress annotation to last frame
        progress_text2 = f"P:{pred_val_last:.2f} T:{target_val_last:.2f}"
        frame2_resized = add_text_overlay(
            frame2_resized,
            progress_text2,
            position=(text_x, text_y),
            font_scale=0.4,
            color=(255, 255, 255),
            thickness=1,
            bg_color=(0, 0, 0)
        )
        
        # Concatenate horizontally
        combined_frame = np.concatenate([frame1_resized, frame2_resized], axis=1)  # (H, 2*W, C)
        
        # Convert back to (C, H, W) format (no time dimension)
        combined_frame_chw = combined_frame.transpose(2, 0, 1)  # (C, H, 2*W)
        
        return combined_frame_chw
    except Exception:
        return None


def create_policy_ranking_grid(
    eval_results: list[dict],
    grid_size: tuple[int, int] = (2, 2),
    max_samples: int = 4,
    border_width: int = 2
) -> Optional[np.ndarray]:
    """
    Create a grid of frame pairs with progress annotations from policy_ranking eval results.
    
    Args:
        eval_results: List of evaluation results with video_path, progress_pred, target_progress
        grid_size: Tuple of (rows, cols) for the grid
        max_samples: Maximum number of samples to use
        border_width: Width of border between pairs in pixels
    
    Returns:
        Grid of frame pairs in (C, total_H, total_W) format, or None if unavailable
    """
    # Filter results with valid video_paths
    valid_results = [r for r in eval_results if r.get("video_path") is not None]
    
    if len(valid_results) == 0:
        return None
    
    grid_cells = grid_size[0] * grid_size[1]
    num_to_sample = min(grid_cells, max_samples, len(valid_results))
    
    # Sample random results
    sampled_results = random.sample(valid_results, num_to_sample)
    
    # Create frame pairs for each sampled result
    frame_pairs = []
    target_h, target_w = 224, 224
    
    for result in sampled_results:
        frame_pair = create_frame_pair_with_progress(result, target_h, target_w)
        if frame_pair is not None:
            frame_pairs.append(frame_pair)
    
    if len(frame_pairs) == 0:
        return None
    
    # Fill remaining cells with black frames
    num_black = grid_cells - len(frame_pairs)
    for _ in range(num_black):
        black_pair = np.zeros((3, target_h, 2 * target_w), dtype=np.uint8)
        frame_pairs.append(black_pair)
    
    # Add borders to each pair and arrange in grid
    border_color = np.array([128, 128, 128], dtype=np.uint8)  # Gray border
    
    grid_rows = []
    for row in range(grid_size[0]):
        row_pairs = []
        for col in range(grid_size[1]):
            idx = row * grid_size[1] + col
            pair = frame_pairs[idx]  # (C, H, 2*W)
            # Convert to (H, 2*W, C) for processing
            pair_hwc = pair.transpose(1, 2, 0)
            
            # Add left border (except for first column)
            if col > 0:
                left_border = np.tile(border_color, (target_h, border_width, 1))
                pair_hwc = np.concatenate([left_border, pair_hwc], axis=1)
            
            row_pairs.append(pair_hwc)
        
        # Concatenate horizontally
        row_concat = np.concatenate(row_pairs, axis=1)  # (H, total_W, C)
        
        # Add horizontal border below (except for last row)
        if row < grid_size[0] - 1:
            h_border = np.tile(border_color, (border_width, row_concat.shape[1], 1))
            row_concat = np.concatenate([row_concat, h_border], axis=0)
        
        grid_rows.append(row_concat)
    
    # Concatenate vertically
    grid_frame = np.concatenate(grid_rows, axis=0)  # (total_H, total_W, C)
    
    # Convert back to (C, H, W) and add time dimension for video format (single frame video)
    grid_frame_chw = grid_frame.transpose(2, 0, 1)  # (C, total_H, total_W)
    grid_video = np.expand_dims(grid_frame_chw, axis=0)  # (1, C, total_H, total_W)
    
    return grid_video
