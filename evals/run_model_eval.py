#!/usr/bin/env python3
"""
Client script to iteratively generate evaluation batches and send them to an
evaluation server (e.g., localhost:8000). The server is expected to return a
dictionary with keys:
  - predictions: [] # list of predictions for each sample in the batch. (1 if chosen preferred, else 0, -1 means no preference)
  - reward_chosen: [] # list of list of per-frame rewards for the chosen trajectory
  - reward_rejected: [] # list of list of per-frame rewards for the rejected trajectory

Usage:
  # Run evaluation with default config:
  python evals/run_model_eval.py

  # Run evaluation with custom config:
  python evals/run_model_eval.py --config rfm/configs/eval_config.yaml

  # After evaluation, compute metrics and create visualizations:
  python evals/compile_results.py --results_path logs/rfm_eval/dataset_name/timestamp/results.json
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import json
import os
from datetime import datetime
from pathlib import Path

from rfm.configs.eval_configs import EvaluationConfig
from rfm.utils.setup_utils import setup_eval_dataset
from evals.eval_utils import build_batch_payload, post_batch


def _evaluate_samples(server_url: str, samples: List[Any]) -> Dict[str, Any]:
    """Send samples to evaluation server and return raw response."""
    payload = build_batch_payload(samples)
    resp = post_batch(server_url, payload)
    return resp


def _save_result_as_json(samples: List[Any], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Save detailed results for each sample in the batch."""

    # Extract data from response
    predictions = response["predictions"]
    prediction_probs = response["prediction_probs"]
    preference_labels = response["preference_labels"]
    progress_pred_chosen = response["progress_pred_chosen"]
    progress_pred_rejected = response["progress_pred_rejected"]

    batch_results = []

    for i, sample in enumerate(samples):
        result_entry = {
            "chosen_id": sample.chosen_id,
            "rejected_id": sample.rejected_id,
            "chosen_task": sample.chosen_task,
            "rejected_task": sample.rejected_task,
            "data_gen_strategy": sample.data_gen_strategy,
            "num_frames_rewound": sample.num_frames_rewound,
            "preference_label": int(preference_labels[i]),
            "predicted_preference": int(predictions[i]),
            "predicted_preference_prob": prediction_probs[i],
            "progress_pred_chosen": progress_pred_chosen[i],
            "progress_pred_rejected": progress_pred_rejected[i],
            "target_progress_chosen": sample.target_progress_chosen,
            "target_progress_rejected": sample.target_progress_rejected,
            "chosen_quality_label": sample.chosen_quality_label,
            "rejected_quality_label": sample.rejected_quality_label,
        }

        # save additional infos for logging metrics
        if sample.metadata and sample.metadata.get("chosen_bin_idx") is not None:
            result_entry["bin_idx_chosen"] = sample.metadata["chosen_bin_idx"]
        if sample.metadata and sample.metadata.get("rejected_bin_idx") is not None:
            result_entry["bin_idx_rejected"] = sample.metadata["rejected_bin_idx"]

        if sample.data_gen_strategy == "video_binned":
            result_entry["video_path"] = sample.video_path
            result_entry["chosen_start_end"] = sample.chosen_start_end
            result_entry["rejected_start_end"] = sample.rejected_start_end
            result_entry["fps"] = sample.fps

        batch_results.append(result_entry)

    return batch_results


def iter_eval_batches(
    eval_cfg: EvaluationConfig,
    server_url: str,
    num_batches: int = 10,
    batch_size: int = 4,
) -> List[Dict[str, Any]]:
    """Run evaluation batches and return detailed results."""
    # Create eval data generator and dataset-like iterator
    dataset = setup_eval_dataset(eval_cfg)

    # Determine actual number of batches
    dataset_size = len(dataset)
    if num_batches == -1:
        # Go through the full dataset
        actual_num_batches = (dataset_size + batch_size - 1) // batch_size  # Ceiling division
        print(f"\nProcessing FULL DATASET: {dataset_size} samples in {actual_num_batches} batches of size {batch_size}")
    else:
        actual_num_batches = num_batches
        print(f"\nProcessing {actual_num_batches} batches of size {batch_size} (dataset size: {dataset_size})")

    all_detailed_results: List[Dict[str, Any]] = []
    idx = 0

    for batch_idx in range(actual_num_batches):
        # Check if we've reached the end of the dataset
        if idx >= dataset_size:
            print(f"\nReached end of dataset after {batch_idx} batches")
            break

        # Assemble a batch of Sample objects (Preference or Similarity)
        batch_samples = []
        for j in range(batch_size):
            if idx + j < dataset_size:
                batch_samples.append(dataset[idx + j])
            else:
                break  # Don't go beyond dataset size

        if not batch_samples:
            break  # No more samples

        # Evaluate this batch
        batch_result = _evaluate_samples(server_url, batch_samples)

        # Process detailed results for this batch
        batch_results = _save_result_as_json(batch_samples, batch_result)
        all_detailed_results.extend(batch_results)
        print(f"Processed batch {batch_idx + 1}/{actual_num_batches} ({len(batch_results)} samples)")

        # Update index
        idx += len(batch_samples)

        # Print progress
        print(f"Progress: {idx}/{dataset_size} samples ({idx / dataset_size * 100:.1f}%)")

    print(f"\nTotal samples processed: {len(all_detailed_results)}")
    return all_detailed_results


def main():
    """Main evaluation function using simple argparse for config loading."""
    import argparse
    import yaml
    from rfm.configs.experiment_configs import DataConfig

    parser = argparse.ArgumentParser(description="Evaluate RFM model")
    parser.add_argument(
        "--config", type=str, default="rfm/configs/eval_config.yaml", help="Path to evaluation configuration file"
    )
    args = parser.parse_args()

    # Load evaluation config manually
    print(f"Loading evaluation config from: {args.config}")
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    cfg = EvaluationConfig(**config_dict)
    cfg.data = DataConfig(**config_dict["data"])
    print(f"Evaluation config: {cfg}")

    # Run evaluation and get results
    all_results = iter_eval_batches(
        eval_cfg=cfg,
        server_url=f"http://localhost:{cfg.server_port}",
        num_batches=cfg.num_batches,
        batch_size=cfg.batch_size,
    )

    # Create results directory structure
    dataset_name = f"{cfg.data.eval_datasets[0].replace('/', '_')}_{cfg.data.eval_subsets[0]}"
    model_name = cfg.model_path.replace("/", "_")
    eval_log_dir = Path(cfg.log_dir) / model_name / dataset_name / cfg.data.dataset_type
    os.makedirs(eval_log_dir, exist_ok=True)

    # Save results to JSON
    results_file = eval_log_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nEvaluation complete!")
    print(f"Results saved to: {results_file}")
    print(f"Total samples processed: {len(all_results)}")


if __name__ == "__main__":
    main()
