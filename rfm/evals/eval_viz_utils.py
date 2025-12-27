#!/usr/bin/env python3
"""
Utility functions for visualization in RFM evaluations.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


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
        if (
            success_labels is not None
            and len(success_labels) == len(success_binary)
        ):
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
        ax3.plot(
            range(len(success_probs)), success_probs, linewidth=2, label="Success Prob", color="purple"
        )
        # Add ground truth success labels as green line if available
        if (
            success_labels is not None
            and len(success_labels) == len(success_probs)
        ):
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

