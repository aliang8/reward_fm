from typing import List, Dict, Any
import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_pearson(y_true: List[float], y_pred: List[float]) -> float:
    """Compute Pearson correlation, robust to constant inputs; returns np.nan if undefined."""
    import numpy as _np
    from scipy.stats import pearsonr as _pearsonr

    a = _np.asarray(y_true, dtype=float)
    b = _np.asarray(y_pred, dtype=float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return _np.nan
    # If either vector is constant, pearsonr returns nan; keep that behavior
    try:
        corr, _ = _pearsonr(a, b)
    except Exception:
        corr = _np.nan
    return corr


def compute_spearman(y_true: List[float], y_pred: List[float]) -> float:
    """Compute Spearman correlation, robust to constant inputs; returns np.nan if undefined."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return np.nan
    try:
        corr, _ = spearmanr(a, b)
    except Exception:
        corr = np.nan
    return corr


def compute_preference_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute preference accuracy over a list of result dicts.
    Expects keys 'predicted_preference' and 'preference_label' per sample.
    Returns dict with accuracy, correct, total, and skipped counts.
    """
    correct = 0
    total = 0
    skipped = 0
    for r in results:
        pred = r.get("predicted_preference")
        label = r.get("preference_label")
        if pred is None or label is None:
            skipped += 1
            continue
        if pred == label:
            correct += 1
        total += 1
    acc = (correct / total) if total > 0 else None
    return {
        "preference_accuracy": acc,
        "num_correct": correct,
        "num_total": total,
        "num_skipped": skipped,
    }
