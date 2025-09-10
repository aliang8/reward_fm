#!/usr/bin/env python3
"""
Client script to iteratively generate evaluation batches and send them to an
evaluation server (e.g., localhost:8000). The server is expected to return a
dictionary with keys:
  - outputs_preference: dictionary containing preference predictions
  - outputs_progress: dictionary containing progress predictions for both trajectories

Usage:
  # Run evaluation with default config:
  python evals/run_model_eval.py

  # Run evaluation with custom config:
  python evals/run_model_eval.py --config rfm/configs/eval_config.yaml

  # After evaluation, compute metrics and create visualizations:
  python evals/compile_results.py --results_path logs/rfm_eval/dataset_name/timestamp/dataset_type_preference.json
"""

from __future__ import annotations

from typing import Any, Dict, List
import json
import os
import asyncio
import aiohttp
from datetime import datetime
from pathlib import Path
import time

from rfm.configs.eval_configs import EvaluationConfig
from rfm.utils.setup_utils import setup_eval_dataset
from evals.eval_utils import post_batch, post_batch_npy, post_batch_npy_async, build_payload

from rfm.utils.logging import _timer, timer
from rfm.data.dataset_types import SampleType


def _save_result_as_json(
    samples: List[SampleType], response: Dict[str, Any]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Save detailed results for each sample in the batch.

    Returns:
        tuple: (preference_results, progress_results)
    """

    # Extract preference and progress outputs from response
    if "outputs_preference" not in response or "outputs_progress" not in response:
        raise ValueError("Server response missing required 'outputs_preference' or 'outputs_progress' keys")

    preference_response = response["outputs_preference"]
    progress_response = response["outputs_progress"]

    # Extract preference data
    if preference_response:
        predictions = preference_response.get("predictions", [])
        prediction_probs = preference_response.get("prediction_probs", [])
        preference_labels = preference_response.get("preference_labels", [])

    # Extract progress data
    if progress_response:
        progress_pred_A = progress_response.get("progress_pred_A", [])

    batch_results = []

    for i, sample in enumerate(samples):
        # Create preference result entry
        if preference_response:
            chosen_meta = {
                "id": sample.chosen_trajectory.id,
                "data_source": sample.chosen_trajectory.data_source,
                "data_gen_strategy": sample.chosen_trajectory.data_gen_strategy,
                "target_progress": sample.chosen_trajectory.target_progress,
                "metadata": sample.chosen_trajectory.metadata,
            }
            rejected_meta = {
                "id": sample.rejected_trajectory.id,
                "data_source": sample.rejected_trajectory.data_source,
                "data_gen_strategy": sample.rejected_trajectory.data_gen_strategy,
                "target_progress": sample.rejected_trajectory.target_progress,
                "metadata": sample.rejected_trajectory.metadata,
            }

            preference_entry = {
                "preference_label": int(preference_labels[i]),
                "predicted_preference": int(predictions[i]),
                "predicted_preference_prob": prediction_probs[i] if prediction_probs else None,
                "chosen_meta": chosen_meta,
                "rejected_meta": rejected_meta,
            }
        else:
            preference_entry = None

        # Create progress result entry
        if progress_response:
            progress_entry = {
                "id": sample.trajectory.id,
                "data_source": sample.trajectory.data_source,
                "data_gen_strategy": sample.trajectory.data_gen_strategy,
                "target_progress": sample.trajectory.target_progress,
                "metadata": sample.trajectory.metadata,
                "progress_pred_A": progress_pred_A[i] if i < len(progress_pred_A) else [],
            }
        else:
            progress_entry = None

        batch_results.append((preference_entry, progress_entry))

    # Separate preference and progress results
    preference_results = [entry[0] for entry in batch_results]
    progress_results = [entry[1] for entry in batch_results]

    return preference_results, progress_results


async def iter_eval_batches_async(
    eval_cfg: EvaluationConfig,
    server_url: str,
    num_batches: int = 10,
    batch_size: int = 4,
    max_concurrent_requests: int = 4,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run evaluation batches asynchronously with concurrent requests."""
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

    all_preference_results: List[Dict[str, Any]] = []
    all_progress_results: List[Dict[str, Any]] = []
    idx = 0

    # Use aiohttp session for concurrent requests
    async with aiohttp.ClientSession() as session:
        while idx < dataset_size and len(all_preference_results) < actual_num_batches * batch_size:
            # Create up to max_concurrent_requests batches
            batch_tasks = []
            batch_samples_list = []

            for _ in range(max_concurrent_requests):
                if idx >= dataset_size:
                    break

                # Assemble a batch of Sample objects
                batch_samples = []
                for j in range(batch_size):
                    if idx + j < dataset_size:
                        batch_samples.append(dataset[idx + j])
                    else:
                        break

                if not batch_samples:
                    break

                # Build payload and create async task
                files, sample_data = build_payload(batch_samples)
                with timer("time/evaluate_batch", verbose=True):
                    # Create the coroutine (don't await it yet)
                    task = post_batch_npy_async(session, server_url, files, sample_data)

                batch_tasks.append(task)
                batch_samples_list.append(batch_samples)
                idx += len(batch_samples)

            if not batch_tasks:
                break

            # Execute all batches concurrently
            start_time = time.time()
            batch_results_list = await asyncio.gather(*batch_tasks, return_exceptions=True)
            round_time = time.time() - start_time

            # Process results and handle any errors
            for i, (batch_result, batch_samples) in enumerate(zip(batch_results_list, batch_samples_list)):
                # Process detailed results for this batch
                preference_results, progress_results = _save_result_as_json(batch_samples, batch_result)
                all_preference_results.extend(preference_results)
                all_progress_results.extend(progress_results)

            # Print progress
            print(f"Completed {len(batch_tasks)} batches in {round_time:.2f}s")
            print(
                f"Progress: {min(idx, dataset_size)}/{dataset_size} samples ({min(idx, dataset_size) / dataset_size * 100:.1f}%)"
            )

    print(f"\nTotal samples processed: {len(all_preference_results)}")
    return all_preference_results, all_progress_results


def iter_eval_batches_sync(
    eval_cfg: EvaluationConfig,
    server_url: str,
    num_batches: int = 10,
    batch_size: int = 4,
    post_function: callable = post_batch,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run evaluation batches synchronously (original implementation)."""
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

    all_preference_results: List[Dict[str, Any]] = []
    all_progress_results: List[Dict[str, Any]] = []
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
        files, sample_data = build_payload(batch_samples)
        batch_result = post_batch_npy(server_url, files, sample_data)

        # Process detailed results for this batch
        preference_results, progress_results = _save_result_as_json(batch_samples, batch_result)
        all_preference_results.extend(preference_results)
        all_progress_results.extend(progress_results)
        print(f"Processed batch {batch_idx + 1}/{actual_num_batches} ({len(preference_results)} samples)")

        # Update index
        idx += len(batch_samples)

        # Print progress
        print(f"Progress: {idx}/{dataset_size} samples ({idx / dataset_size * 100:.1f}%)")

    print(f"\nTotal samples processed: {len(all_preference_results)}")
    return all_preference_results, all_progress_results


def main():
    """Main evaluation function using simple argparse for config loading."""
    import argparse
    import yaml
    from rfm.configs.experiment_configs import DataConfig

    parser = argparse.ArgumentParser(description="Evaluate RFM model")
    parser.add_argument(
        "--config", type=str, default="rfm/configs/eval_config.yaml", help="Path to evaluation configuration file"
    )
    parser.add_argument(
        "--use-async", action="store_true", help="Use async concurrent evaluation (recommended for multi-GPU server)"
    )
    parser.add_argument("--max_concurrent", type=int, default=8, help="Maximum concurrent requests (default: 4)")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config with dot-path assignments, e.g., --set data.max_frames=8 --set model.base_model_id='Qwen/...'.",
    )

    args = parser.parse_args()

    # Load evaluation config manually
    print(f"Loading evaluation config from: {args.config}")
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    cfg = EvaluationConfig(**config_dict)
    cfg.data = DataConfig(**config_dict["data"])

    # Apply overrides from --set key=value (dot-path)
    import ast

    for assignment in args.set:
        if "=" not in assignment:
            continue
        key, value_str = assignment.split("=", 1)
        try:
            value = ast.literal_eval(value_str)
        except Exception:
            value = value_str
        target = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            target = getattr(target, p)
        setattr(target, parts[-1], value)

    print(f"Evaluation config: {cfg}")

    # Run evaluation and get results
    if args.use_async:
        print(f"Using ASYNC evaluation with max {args.max_concurrent} concurrent requests")
        preference_results, progress_results = asyncio.run(
            iter_eval_batches_async(
                eval_cfg=cfg,
                server_url=f"http://localhost:{cfg.server_port}",
                num_batches=cfg.num_batches,
                batch_size=cfg.batch_size,
                max_concurrent_requests=args.max_concurrent,
            )
        )
    else:
        print("Using SYNC evaluation (single request at a time)")
        preference_results, progress_results = iter_eval_batches_sync(
            eval_cfg=cfg,
            server_url=f"http://localhost:{cfg.server_port}",
            num_batches=cfg.num_batches,
            batch_size=cfg.batch_size,
        )

    # filter out None entries
    preference_results = [result for result in preference_results if result is not None]
    progress_results = [result for result in progress_results if result is not None]

    # Create results directory structure
    dataset_name = f"{cfg.data.eval_datasets[0].replace('/', '_')}_{cfg.data.eval_subsets[0]}"
    model_name = cfg.model_path.replace("/", "_") if cfg.model_path else f"base_model_{cfg.data.model_type}"
    eval_log_dir = Path(cfg.log_dir) / model_name / dataset_name
    os.makedirs(eval_log_dir, exist_ok=True)

    # Save preference results to JSON
    preference_file = eval_log_dir / f"{cfg.data.dataset_type}_preference.json"
    if len(preference_results) > 0:
        with open(preference_file, "w") as f:
            json.dump(preference_results, f, indent=2)
        print(f"Preference results saved to: {preference_file}")

    # Save progress results to JSON
    progress_file = eval_log_dir / f"{cfg.data.dataset_type}_progress.json"
    if len(progress_results) > 0:
        with open(progress_file, "w") as f:
            json.dump(progress_results, f, indent=2)
        print(f"Progress results saved to: {progress_file}")

    print(f"\nEvaluation complete!")
    print(f"Total samples processed: {len(preference_results)}")


if __name__ == "__main__":
    main()
