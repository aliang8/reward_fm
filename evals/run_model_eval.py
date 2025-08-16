#!/usr/bin/env python3
"""
Client script to iteratively generate evaluation batches and send them to an
evaluation server (e.g., localhost:8000). The server is expected to return a
dictionary with keys:
  - predictions: [] # list of predictions for each sample in the batch. (1 if chosen preferred, else 0, -1 means no preference)
  - reward_chosen: [] # list of list of per-frame rewards for the chosen trajectory
  - reward_rejected: [] # list of list of per-frame rewards for the rejected trajectory

Usage:
  uv run python evals/run_model_eval.py --config_path=rfm/configs/config.yaml \
      --server_url=http://localhost:8000 --num_batches=10 --batch_size=4
  
  # Override config values:
  uv run python evals/run_model_eval.py --config_path=rfm/configs/config.yaml \
      --set data.max_frames=16 --set data.eval_subset_size=1000
"""

from __future__ import annotations

import argparse
import ast
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.stats import spearmanr

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
    "predictions": "Preference predictions for each sample in the batch (1 if chosen preferred, else 0, -1 means no preference)",
    "reward_chosen": "List of list of per-frame rewards for the chosen trajectory",
    "reward_rejected": "List of list of per-frame rewards for the rejected trajectory",
}


def _compute_metrics_from_response(samples: List[Any], resp: Dict[str, Any]) -> Dict[str, Any]:
    preds: List[int] = resp.get("predictions", []) or []
    rewards_chosen: List[List[float]] = resp.get("reward_chosen", []) or []
    rewards_rejected: List[List[float]] = resp.get("reward_rejected", []) or []

    n = max(1, len(preds))
    eval_accuracy = float(sum(1 for p in preds if int(p) == 1) / n)

    flat_chosen = [x for seq in rewards_chosen for x in (seq or [])]
    flat_rejected = [x for seq in rewards_rejected for x in (seq or [])]
    eval_avg_reward_chosen = float(np.mean(flat_chosen)) if flat_chosen else None
    eval_avg_reward_rejected = float(np.mean(flat_rejected)) if flat_rejected else None
    eval_reward_diff = (
        float(eval_avg_reward_chosen - eval_avg_reward_rejected)
        if (eval_avg_reward_chosen is not None and eval_avg_reward_rejected is not None)
        else None
    )

    # Progress alignment using chosen trajectory's per-frame rewards as predictions
    rhos: List[float] = []
    for i, s in enumerate(samples):
        try:
            targets = getattr(s, "target_progress_A", None)
            preds_prog = rewards_chosen[i] if i < len(rewards_chosen) else []
            if not targets or not preds_prog:
                continue
            # Heuristic: downsample targets to roughly match model stride (e.g., Qwen predicts ~1 per 2 frames)
            targets_ds = targets[::2]
            L = min(len(targets_ds), len(preds_prog))
            if L < 2:
                continue
            rho = spearmanr(preds_prog[:L], targets_ds[:L]).correlation
            if rho is not None and not np.isnan(rho):
                rhos.append(float(rho))
        except Exception:
            continue
    demo_reward_alignment = float(np.mean(rhos)) if rhos else None

    return {
        "eval_accuracy": eval_accuracy,
        "eval_reward_diff": eval_reward_diff,
        "eval_avg_reward_chosen": eval_avg_reward_chosen,
        "eval_avg_reward_rejected": eval_avg_reward_rejected,
        "demo_reward_alignment": demo_reward_alignment,
    }


def _evaluate_samples(server_url: str, samples: List[Any]) -> Dict[str, Any]:
    payload = build_batch_payload(samples)
    resp = post_batch(server_url, payload)
    return _compute_metrics_from_response(samples, resp)


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
        #"__getitem__": lambda self, idx: data_generator._create_preference_sample() if idx % 2 == 0 else data_generator._create_similarity_sample(),
        "__getitem__": lambda self, idx: data_generator._create_preference_sample(),
    })()

    results: List[Dict[str, Any]] = []
    idx = 0
    for _ in tqdm(range(num_batches), desc="Evaluating batches"):
        # Assemble a batch of Sample objects (Preference or Similarity)
        samples = [dataset[idx + j] for j in range(batch_size)]
        idx += batch_size
        results.append(_evaluate_samples(server_url, samples))
    return results


def iter_eval_all_preferences(
    cfg: ExperimentConfig,
    server_url: str,
    batch_size: int = 4,
) -> List[Dict[str, Any]]:
    """Iterate deterministically over the evaluation dataset by pairing each
    optimal trajectory with a negative (suboptimal from same task if available,
    otherwise a rewind), and evaluate all such pairs.

    This covers the entire set of optimal trajectories once, avoiding a fixed
    num_batches limit.
    """
    from rfm.data.batch_collator import PreferenceSample

    data_generator = setup_eval_data_generator(cfg)

    optimal = list(data_generator.get_optimal_trajectories())
    suboptimal = list(data_generator.get_suboptimal_trajectories())

    # Map task -> suboptimal list for quick lookup
    subs_by_task: Dict[str, List[dict]] = {}
    for traj in suboptimal:
        subs_by_task.setdefault(traj["task"], []).append(traj)

    def deserialize(frames, frames_shape):
        # Use data_generator's helper to deserialize bytes → np.ndarray
        if isinstance(frames_shape, list):
            frames_shape = tuple(frames_shape)
        if isinstance(frames, (bytes, bytearray)):
            return data_generator._deserialize_frames(frames, shape=frames_shape)
        return frames

    def calc_progress(traj: dict, frames: Optional[Any] = None):
        return data_generator._calculate_target_progress(traj, frames=frames)

    # Build all PreferenceSample items
    samples: List[PreferenceSample] = []
    for opt in optimal:
        task = opt["task"]
        neg: Optional[dict] = None
        # Prefer suboptimal from same task
        candidates = [t for t in subs_by_task.get(task, []) if t["id"] != opt["id"]]
        if candidates:
            neg = candidates[0]
        else:
            # Fallback: rewind-generated negative
            neg = data_generator._create_rewind_trajectory(opt)

        # Deserialize frames once for both
        opt_shape = opt.get("frames_shape")
        rej_shape = neg.get("frames_shape")
        opt_frames = deserialize(opt["frames"], opt_shape)
        rej_frames = deserialize(neg["frames"], rej_shape)

        # Target progress
        tp_A = calc_progress(opt, opt_frames)
        tp_B = calc_progress(neg, rej_frames)

        # Normalize shapes to tuples
        if isinstance(opt_shape, list):
            opt_shape = tuple(opt_shape)
        if isinstance(rej_shape, list):
            rej_shape = tuple(rej_shape)

        samples.append(
            PreferenceSample(
                id=str(opt["id"]),
                task=str(opt["task"]),
                lang_vector=opt.get("lang_vector"),
                data_source=str(opt.get("data_source", "")),
                frames=opt["frames"],
                frames_shape=opt_shape,
                quality_label=str(opt.get("quality_label", "successful")),
                is_robot=bool(opt.get("is_robot", True)),
                metadata=opt.get("metadata"),
                chosen_frames=opt_frames,
                rejected_frames=rej_frames,
                chosen_frames_shape=opt_shape,
                rejected_frames_shape=rej_shape,
                preferred_trajectory="chosen",
                chosen_id=str(opt["id"]),
                rejected_id=str(neg["id"]),
                rejected_task=str(neg["task"]),
                rejected_lang_vector=neg.get("lang_vector"),
                rejected_data_source=str(neg.get("data_source", "")),
                rejected_quality_label=str(neg.get("quality_label", "")),
                rejected_is_robot=bool(neg.get("is_robot", True)),
                target_progress_A=tp_A,
                target_progress_B=tp_B,
            )
        )

    # Send in batches
    results: List[Dict[str, Any]] = []
    for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating (all prefs)"):
        batch = samples[i : i + batch_size]
        results.append(_evaluate_samples(server_url, batch))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="rfm/configs/config.yaml")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000")
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--iterate_all_preferences",
        action="store_true",
        help="Evaluate one preference pair per optimal trajectory to cover the dataset.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config with dot-path assignments, e.g., --set data.max_frames=8 --set model.base_model_id='Qwen/...'.",
    )
    args = parser.parse_args()

    cfg = load_experiment_config_from_yaml(args.config_path)

    # Apply overrides from --set key=value (dot-path)
    for assignment in args.set:
        if "=" not in assignment:
            print(f"Warning: Invalid --set argument '{assignment}', skipping. Use format: key=value")
            continue
        key, value_str = assignment.split("=", 1)
        try:
            value = ast.literal_eval(value_str)
        except Exception:
            value = value_str
        target = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if not hasattr(target, p):
                print(f"Warning: Config path '{key}' is invalid, skipping override")
                break
            target = getattr(target, p)
        else:
            setattr(target, parts[-1], value)
            print(f"Applied config override: {key} = {value}")

    if args.iterate_all_preferences:
        results = iter_eval_all_preferences(cfg=cfg, server_url=args.server_url, batch_size=args.batch_size)
    else:
        results = iter_eval_batches(
            cfg=cfg,
            server_url=args.server_url,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
        )

    # Print an aggregated summary
    keys = [
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


