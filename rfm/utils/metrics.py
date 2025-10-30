#!/usr/bin/env python3
"""
Utility functions for computing evaluation metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F


def compute_spearman_correlation(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Spearman correlation between prediction and target tensors.

    Args:
        pred: Prediction tensor
        target: Target tensor

    Returns:
        Spearman correlation coefficient
    """
    # Convert to numpy for scipy.stats.spearmanr
    try:
        from scipy.stats import spearmanr

        # NumPy doesn't support bf16/half; cast to float32 before moving to CPU
        pred_f32 = pred.detach().to(dtype=torch.float32)
        target_f32 = target.detach().to(dtype=torch.float32)

        pred_np = pred_f32.cpu().numpy()
        target_np = target_f32.cpu().numpy()

        # Handle 1D arrays
        if pred_np.ndim == 1 and target_np.ndim == 1:
            correlation, _ = spearmanr(pred_np, target_np)
            return torch.tensor(correlation, device=pred.device, dtype=torch.float32)

        # Handle 2D arrays (batch, sequence)
        elif pred_np.ndim == 2 and target_np.ndim == 2:
            correlations = []
            for p, t in zip(pred_np, target_np, strict=False):
                if len(p) > 1 and len(t) > 1:  # Need at least 2 points for correlation
                    corr, _ = spearmanr(p, t)
                    if not np.isnan(corr):
                        correlations.append(corr)

            if correlations:
                return torch.tensor(np.mean(correlations), device=pred.device, dtype=torch.float32)
            else:
                return torch.tensor(0.0, device=pred.device, dtype=torch.float32)

        else:
            raise ValueError(f"Unsupported tensor dimensions: pred={pred_np.ndim}, target={target_np.ndim}")

    except ImportError:
        # Fallback to manual implementation if scipy is not available
        return manual_spearman_correlation(pred, target)


def manual_spearman_correlation(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Manual implementation of Spearman correlation as fallback.

    Args:
        pred: Prediction tensor
        target: Target tensor

    Returns:
        Spearman correlation coefficient
    """
    # Handle 1D tensors
    if pred.ndim == 1 and target.ndim == 1:
        if len(pred) < 2:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Convert to ranks
        pred_ranks = torch.argsort(torch.argsort(pred))
        target_ranks = torch.argsort(torch.argsort(target))

        # Compute correlation using rank formula
        n = len(pred)
        sum_d2 = torch.sum((pred_ranks - target_ranks) ** 2)
        spearman = 1 - (6 * sum_d2) / (n * (n**2 - 1))

        return spearman

    # Handle 2D tensors (batch, sequence)
    elif pred.ndim == 2 and target.ndim == 2:
        correlations = []
        for p, t in zip(pred, target, strict=False):
            if len(p) > 1 and len(t) > 1:
                corr = manual_spearman_correlation(p, t)
                if not torch.isnan(corr):
                    correlations.append(corr)

        if correlations:
            return torch.stack(correlations).mean()
        else:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    else:
        raise ValueError(f"Unsupported tensor dimensions: pred={pred.ndim}, target={target.ndim}")


def compute_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Area Under the Curve (AUC) for binary classification.

    Args:
        scores: Model prediction scores/logits
        labels: Binary labels (0 or 1)

    Returns:
        AUC score
    """
    try:
        from sklearn.metrics import roc_auc_score

        scores_np = scores.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        # Handle edge cases
        if len(np.unique(labels_np)) < 2:
            return 0.5  # Default AUC for single class

        auc = roc_auc_score(labels_np, scores_np)
        return auc

    except ImportError:
        # Fallback implementation if sklearn is not available
        return manual_auc(scores, labels)


def manual_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Manual implementation of AUC as fallback.

    Args:
        scores: Model prediction scores/logits
        labels: Binary labels (0 or 1)

    Returns:
        AUC score
    """
    # Sort by scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_indices]

    # Count positive and negative samples
    n_pos = torch.sum(labels == 1).item()
    n_neg = torch.sum(labels == 0).item()

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Default AUC for single class

    # Calculate true positive rate and false positive rate
    tp = 0
    fp = 0
    prev_score = float("inf")

    area = 0.0

    for i, label in enumerate(sorted_labels):
        if scores[sorted_indices[i]] != prev_score:
            # Calculate area under the curve
            area += trapezoid_area(fp / n_neg, tp / n_pos, fp / n_neg, tp / n_pos)
            prev_score = scores[sorted_indices[i]]

        if label == 1:
            tp += 1
        else:
            fp += 1

    # Add final area
    area += trapezoid_area(fp / n_neg, tp / n_pos, 1.0, 1.0)

    return area


def trapezoid_area(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate area of trapezoid using trapezoid rule.

    Args:
        x1, y1: First point coordinates
        x2, y2: Second point coordinates

    Returns:
        Area of trapezoid
    """
    return (x2 - x1) * (y1 + y2) / 2


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute accuracy for binary classification.

    Args:
        predictions: Binary predictions (0 or 1)
        targets: Binary targets (0 or 1)

    Returns:
        Accuracy score
    """
    return (predictions == targets).float().mean().item()


def compute_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Mean Squared Error.

    Args:
        predictions: Prediction tensor
        targets: Target tensor

    Returns:
        MSE value
    """
    return F.mse_loss(predictions, targets).item()


def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        predictions: Prediction tensor
        targets: Target tensor

    Returns:
        MAE value
    """
    return F.l1_loss(predictions, targets).item()
