#!/usr/bin/env python3
"""
Client script to iteratively generate evaluation batches and send them to an
evaluation server (e.g., localhost:8000). The server is expected to return a
dictionary with keys:
  - eval_loss
  - eval_accuracy
  - eval_reward_diff
  - eval_avg_reward_chosen
  - eval_avg_reward_rejected
  - demo_reward_alignment (per-frame Spearman correlation)

Usage:
  uv run python evals/run_model_eval.py --config_path=rfm/configs/config.yaml \
      --server_url=http://localhost:8000 --num_batches=10 --batch_size=4
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from tqdm import tqdm

from rfm.configs.experiment_configs import ExperimentConfig
from setup_utils import (
    setup_eval_data_generator,
    setup_batch_collator,
    setup_model_and_processor,
)
from evals.eval_utils import (
    load_experiment_config_from_yaml,
    build_batch_payload,
    post_batch,
)

KEY_TO_MEANING = {
    "eval_loss": "Loss",
    "eval_accuracy": "Accuracy of Predicting the Correct Preference",
    "eval_reward_diff": "Reward Difference between Chosen and Rejected",
    "eval_avg_reward_chosen": "Average Reward Assigned to (Chosen)",
    "eval_avg_reward_rejected": "Average Reward (Rejected)",
    "demo_reward_alignment": "Spearman Correlation between Predicted Progress and Ground Truth Progress (per-frame ordering)",
}


def iter_eval_batches(
    cfg: ExperimentConfig,
    server_url: str,
    num_batches: int = 10,
    batch_size: int = 4,
) -> List[Dict[str, Any]]:
    # Create eval data generator and dataset-like iterator
    data_generator = setup_eval_data_generator(cfg)
    dataset = type("_Dataset", (), {
        "__len__": lambda self: cfg.data.eval_subset_size if cfg.data.eval_subset_size and cfg.data.eval_subset_size > 0 else 10_000_000,
        "__getitem__": lambda self, idx: data_generator._create_preference_sample() if idx % 2 == 0 else data_generator._create_similarity_sample(),
    })()

    results: List[Dict[str, Any]] = []
    idx = 0
    for _ in tqdm(range(num_batches), desc="Evaluating batches"):
        # Assemble a batch of Sample objects (Preference or Similarity)
        samples = [dataset[idx + j] for j in range(batch_size)]
        idx += batch_size

        # Convert to wire payload (base64 frames)
        payload = build_batch_payload(samples)

        # POST to server
        resp = post_batch(server_url, payload)
        results.append(resp)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="rfm/configs/config.yaml")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000")
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    cfg = load_experiment_config_from_yaml(args.config_path)

    results = iter_eval_batches(
        cfg=cfg,
        server_url=args.server_url,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
    )

    # Print an aggregated summary
    keys = [
        "eval_loss",
        "eval_accuracy",
        "eval_reward_diff",
        "eval_avg_reward_chosen",
        "eval_avg_reward_rejected",
        "demo_reward_alignment",
    ]
    agg = {k: 0.0 for k in keys}
    # keeps track of which keys are always None
    none_keys = {k: [] for k in keys}
    n = max(1, len(results))
    for r in results:
        for k in keys:
            if k in r and r[k] is None:
                none_keys[k].append(1)
            elif k in r and isinstance(r[k], (int, float)):
                none_keys[k].append(0)
                agg[k] += float(r[k])
    for k in keys:
        agg[k] /= n
    for k in none_keys:
        if all(none_keys[k]):
            print(f"WARNING: {k} is always None, removing")
            agg.pop(k)
            continue

    print("\nEvaluation summary (averaged across batches):")
    for k, v in agg.items():
        print(f"  {k}: {v:.6f}")
        print(f"explanation:    {KEY_TO_MEANING[k]}")


if __name__ == "__main__":
    main()


