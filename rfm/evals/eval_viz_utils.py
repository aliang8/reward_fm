#!/usr/bin/env python3
"""
Utility functions for visualization in RFM evaluations.
"""

from typing import Optional
import os
import logging
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import decord

logger = logging.getLogger(__name__)


def create_combined_progress_success_plot(
    progress_pred: np.ndarray,
    num_frames: int,
    success_binary: Optional[np.ndarray] = None,
    success_probs: Optional[np.ndarray] = None,
    success_labels: Optional[np.ndarray] = None,
    is_discrete_mode: bool = False,
    num_bins: int = 10,
    title: Optional[str] = None,
    loss: Optional[float] = None,
    pearson: Optional[float] = None,
) -> plt.Figure:
    """Create a combined plot with progress, success binary, and success probabilities.

    This function creates a unified plot with 1 subplot (progress only) or 3 subplots
    (progress, success binary, success probs), similar to the one used in compile_results.py.

    Args:
        progress_pred: Progress predictions array
        num_frames: Number of frames
        success_binary: Optional binary success predictions
        success_probs: Optional success probability predictions
        success_labels: Optional ground truth success labels
        is_discrete_mode: Whether progress is in discrete mode
        num_bins: Number of bins for discrete mode
        title: Optional title for the plot (if None, auto-generated from loss/pearson)
        loss: Optional loss value to display in title
        pearson: Optional pearson correlation to display in title

    Returns:
        matplotlib Figure object
    """
    # Determine if we should show success plots
    has_success_binary = success_binary is not None and len(success_binary) == len(progress_pred)

    if has_success_binary:
        # Three subplots: progress, success (binary), success_probs
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.5))
        ax = axs[0]  # Progress subplot
        ax2 = axs[1]  # Success subplot (binary)
        ax3 = axs[2]  # Success probs subplot
    else:
        # Single subplot: progress only
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax2 = None
        ax3 = None

    # Plot progress
    ax.plot(progress_pred, linewidth=2)
    ax.set_ylabel("Progress")

    # Build title
    if title is None:
        title_parts = ["Progress"]
        if loss is not None:
            title_parts.append(f"Loss: {loss:.3f}")
        if pearson is not None:
            title_parts.append(f"Pearson: {pearson:.2f}")
        title = ", ".join(title_parts)
    ax.set_title(title)

    # Set y-limits and ticks
    ax.set_ylim(0, num_bins - 1 if is_discrete_mode else 1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if is_discrete_mode:
        if num_bins > 5:
            y_ticks = list(range(0, num_bins, 2))
        else:
            y_ticks = list(range(0, num_bins))
    else:
        y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(y_ticks)

    # Setup success binary subplot
    if ax2 is not None:
        ax2.step(range(len(success_binary)), success_binary, where="post", linewidth=2, label="Predicted", color="blue")
        # Add ground truth success labels as green line if available
        if success_labels is not None and len(success_labels) == len(success_binary):
            ax2.step(
                range(len(success_labels)),
                success_labels,
                where="post",
                linewidth=2,
                label="Ground Truth",
                color="green",
            )
        ax2.set_ylabel("Success (Binary)")
        ax2.set_ylim(-0.05, 1.05)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.set_yticks([0, 1])
        ax2.legend()

    # Setup success probs subplot if available
    if ax3 is not None and success_probs is not None:
        ax3.plot(range(len(success_probs)), success_probs, linewidth=2, label="Success Prob", color="purple")
        # Add ground truth success labels as green line if available
        if success_labels is not None and len(success_labels) == len(success_probs):
            ax3.step(
                range(len(success_labels)),
                success_labels,
                where="post",
                linewidth=2,
                label="Ground Truth",
                color="green",
                linestyle="--",
            )
        ax3.set_ylabel("Success Probability")
        ax3.set_ylim(-0.05, 1.05)
        ax3.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax3.legend()

    plt.tight_layout()
    return fig


def extract_frames(video_path: str, fps: float = 1.0) -> np.ndarray:
    """Extract frames from video file as numpy array (T, H, W, C).

    Supports both local file paths and URLs (e.g., HuggingFace Hub URLs).
    Uses the provided ``fps`` to control how densely frames are sampled from
    the underlying video; there is no additional hard cap on the number of frames.

    Args:
        video_path: Path to video file or URL
        fps: Frames per second to extract (default: 1.0)

    Returns:
        numpy array of shape (T, H, W, C) containing extracted frames, or None if error
    """
    if video_path is None:
        return None

    if isinstance(video_path, tuple):
        video_path = video_path[0]

    # Check if it's a URL or local file
    is_url = video_path.startswith(("http://", "https://"))
    is_local_file = os.path.exists(video_path) if not is_url else False

    if not is_url and not is_local_file:
        logger.warning(f"Video path does not exist: {video_path}")
        return None

    try:
        # decord.VideoReader can handle both local files and URLs
        vr = decord.VideoReader(video_path, num_threads=1)
        total_frames = len(vr)

        # Determine native FPS; fall back to a reasonable default if unavailable
        try:
            native_fps = float(vr.get_avg_fps())
        except Exception:
            native_fps = 1.0

        # If user-specified fps is invalid or None, default to native fps
        if fps is None or fps <= 0:
            fps = native_fps

        # Compute how many frames we want based on desired fps
        # num_frames ≈ total_duration * fps = total_frames * (fps / native_fps)
        if native_fps > 0:
            desired_frames = int(round(total_frames * (fps / native_fps)))
        else:
            desired_frames = total_frames

        # Clamp to [1, total_frames]
        desired_frames = max(1, min(desired_frames, total_frames))

        # Evenly sample indices to match the desired number of frames
        if desired_frames == total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, desired_frames, dtype=int).tolist()

        frames_array = vr.get_batch(frame_indices).asnumpy()  # Shape: (T, H, W, C)
        del vr
        return frames_array
    except Exception as e:
        logger.error(f"Error extracting frames from {video_path}: {e}")
        return None


def create_comparison_plot(frames_a: list, frames_b: list, prediction_type: str) -> str:
    """Create side-by-side comparison plot of two videos."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 16

    fig, axes = plt.subplots(2, min(8, max(len(frames_a), len(frames_b))), figsize=(16, 4))

    if len(axes.shape) == 1:
        axes = axes.reshape(2, -1)

    # Sample frames to display
    num_display = min(8, max(len(frames_a), len(frames_b)))
    indices_a = np.linspace(0, len(frames_a) - 1, num_display, dtype=int) if len(frames_a) > 1 else [0]
    indices_b = np.linspace(0, len(frames_b) - 1, num_display, dtype=int) if len(frames_b) > 1 else [0]

    # Display frames from video A (top row)
    for idx, frame_idx in enumerate(indices_a):
        if frame_idx < len(frames_a):
            axes[0, idx].imshow(frames_a[frame_idx])
            axes[0, idx].axis("off")
            axes[0, idx].set_title(f"Frame {frame_idx}", fontsize=12)

    # Display frames from video B (bottom row)
    for idx, frame_idx in enumerate(indices_b):
        if frame_idx < len(frames_b):
            axes[1, idx].imshow(frames_b[frame_idx])
            axes[1, idx].axis("off")
            axes[1, idx].set_title(f"Frame {frame_idx}", fontsize=12)

    # Add row labels
    fig.text(0.02, 0.75, "Video A", rotation=90, fontsize=18, fontweight="bold", va="center")
    fig.text(0.02, 0.25, "Video B", rotation=90, fontsize=18, fontweight="bold", va="center")

    title = f"{prediction_type.capitalize()} Comparison: Video A vs Video B"
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.98)

    plt.tight_layout()

    # Save to temporary file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp_file.name, dpi=150, bbox_inches="tight")
    plt.close()

    return tmp_file.name
