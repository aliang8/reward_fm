#!/usr/bin/env python3
"""
Compute confusion matrix metric: trace(A) - sum(off-diagonal)

This metric measures how well the model distinguishes between tasks.
Higher values indicate better task discrimination (diagonal dominates).

Usage:
    python rfm/evals/compute_confusion_metric.py path/to/confusion_matrix.npy
    python rfm/evals/compute_confusion_metric.py path/to/results.json
"""

import argparse
import json
import numpy as np
from pathlib import Path


def compute_trace_minus_offdiag(confusion_matrix: np.ndarray) -> dict:
    """Compute trace(A) - sum(off-diagonal) metric.
    
    Args:
        confusion_matrix: Square confusion matrix (N x N)
        
    Returns:
        Dictionary with computed metrics
    """
    if confusion_matrix.ndim != 2 or confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {confusion_matrix.shape}")
    
    n = confusion_matrix.shape[0]
    
    # Compute trace (sum of diagonal)
    trace = np.trace(confusion_matrix)
    
    # Compute sum of all elements
    total_sum = np.sum(confusion_matrix)
    
    # Sum of off-diagonal = total - trace
    off_diag_sum = total_sum - trace
    
    # Main metric: trace - off-diagonal
    trace_minus_offdiag = trace - off_diag_sum
    
    # Also compute normalized version (divide by matrix size for comparability)
    # This gives average diagonal value - average off-diagonal value
    avg_diagonal = trace / n
    avg_off_diag = off_diag_sum / (n * n - n) if n > 1 else 0.0
    normalized_metric = avg_diagonal - avg_off_diag
    
    return {
        "trace": float(trace),
        "off_diagonal_sum": float(off_diag_sum),
        "trace_minus_offdiag": float(trace_minus_offdiag),
        "avg_diagonal": float(avg_diagonal),
        "avg_off_diagonal": float(avg_off_diag),
        "normalized_trace_minus_offdiag": float(normalized_metric),
        "matrix_size": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute confusion matrix metrics")
    parser.add_argument("input_file", type=str, help="Path to .npy or .json file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file (optional)")
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    
    if input_path.suffix == ".npy":
        # Load numpy array directly
        confusion_matrix = np.load(input_path)
    elif input_path.suffix == ".json":
        # Load from JSON (assuming it contains a 'confusion_matrix' key or is the matrix itself)
        with open(input_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            confusion_matrix = np.array(data)
        elif isinstance(data, dict) and "confusion_matrix" in data:
            confusion_matrix = np.array(data["confusion_matrix"])
        else:
            raise ValueError("JSON file must contain a list (matrix) or dict with 'confusion_matrix' key")
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}. Use .npy or .json")
    
    # Compute metrics
    metrics = compute_trace_minus_offdiag(confusion_matrix)
    
    # Print results
    print(f"\nConfusion Matrix Metrics for: {input_path}")
    print("=" * 50)
    print(f"Matrix size: {metrics['matrix_size']} x {metrics['matrix_size']}")
    print(f"Trace (sum of diagonal): {metrics['trace']:.4f}")
    print(f"Off-diagonal sum: {metrics['off_diagonal_sum']:.4f}")
    print(f"Trace - Off-diagonal: {metrics['trace_minus_offdiag']:.4f}")
    print("-" * 50)
    print(f"Avg diagonal value: {metrics['avg_diagonal']:.4f}")
    print(f"Avg off-diagonal value: {metrics['avg_off_diagonal']:.4f}")
    print(f"Normalized (avg_diag - avg_offdiag): {metrics['normalized_trace_minus_offdiag']:.4f}")
    
    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics to: {args.output}")
    
    return metrics


if __name__ == "__main__":
    main()

