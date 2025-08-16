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
from typing import Any, Dict, List, Optional, Union
import numpy as np
from scipy.stats import spearmanr

from tqdm import tqdm

from rfm.configs.experiment_configs import ExperimentConfig
from rfm.utils.setup_utils import (
    setup_eval_data_generator,
    setup_batch_collator,
    setup_model_and_processor,
)
from evals.eval_utils import (
    load_experiment_config_from_yaml,
    build_batch_payload,
    post_batch,
)
from rfm.data.batch_collator import PreferenceSample, SimilaritySample, PairedVideoSample

KEY_TO_MEANING = {
    "eval_loss": "Loss",
    "eval_accuracy": "Accuracy of Predicting the Correct Preference",
    "eval_reward_diff": "Reward Difference between Chosen and Rejected",
    "eval_avg_reward_chosen": "Average Reward Assigned to (Chosen)",
    "eval_avg_reward_rejected": "Average Reward (Rejected)",
    "demo_reward_alignment": "Spearman Correlation between Predicted Progress and Ground Truth Progress (per-frame ordering)",
}


def _compute_metrics_from_response(
    response: Dict[str, Any], 
    samples: List[Union[PreferenceSample, SimilaritySample, PairedVideoSample]]
) -> Dict[str, Any]:
    """Compute metrics from model response and samples."""
    
    # Extract predictions and rewards
    preds = response.get("predictions", [])
    rewards_chosen = response.get("rewards_chosen", [])
    rewards_rejected = response.get("rewards_rejected", [])
    progress_predictions = response.get("progress_predictions", [])
    
    if not preds or not rewards_chosen or not rewards_rejected:
        return {}
    
    # Base metrics computation (compute once for efficiency)
    metrics = {}
    
    # Preference accuracy
    if any(s.sample_type == "preference" for s in samples):
        correct_preds = sum(1 for p in preds if p == 1) # this is because the first trajectory is the chosen trajectory
        metrics["eval_accuracy"] = correct_preds / len(preds)
    
    # Reward metrics
    if rewards_chosen and rewards_rejected:
        metrics["eval_reward_diff"] = np.mean(rewards_chosen) - np.mean(rewards_rejected)
        metrics["eval_avg_reward_chosen"] = np.mean(rewards_chosen)
        metrics["eval_avg_reward_rejected"] = np.mean(rewards_rejected)
    
    # Progress alignment (for similarity samples)
    if progress_predictions and any(s.sample_type == "similarity" for s in samples):
        progress_targets = [s.target_progress for s in samples if s.sample_type == "similarity"]
        if len(progress_predictions) == len(progress_targets):
            try:
                correlation, _ = spearmanr(progress_predictions, progress_targets)
                metrics["demo_reward_alignment"] = correlation if not np.isnan(correlation) else None
            except:
                metrics["demo_reward_alignment"] = None
    
    # Granular metrics computation
    granular_metrics = {}
    
    # Helper function to compute metrics for a subset of indices
    def get_metrics_for_subset(indices: List[int], metric_name: str) -> Optional[float]:
        """Compute a specific metric for a subset of samples."""
        if not indices:
            return None
        
        subset_preds = [preds[i] for i in indices if i < len(preds)]
        subset_rewards_chosen = [rewards_chosen[i] for i in indices if i < len(rewards_chosen)]
        subset_rewards_rejected = [rewards_rejected[i] for i in indices if i < len(rewards_rejected)]
        
        if not subset_preds:
            return None
        
        if metric_name.startswith("accuracy_"):
            correct = sum(1 for p in subset_preds if p == "chosen")
            return correct / len(subset_preds)
        elif metric_name.startswith("avg_reward_chosen_"):
            return np.mean(subset_rewards_chosen) if subset_rewards_chosen else None
        elif metric_name.startswith("avg_reward_rejected_"):
            return np.mean(subset_rewards_rejected) if subset_rewards_rejected else None
        elif metric_name.startswith("reward_diff_"):
            if subset_rewards_chosen and subset_rewards_rejected:
                return np.mean(subset_rewards_chosen) - np.mean(subset_rewards_rejected)
            return None
        elif metric_name.startswith("progress_alignment_"):
            subset_progress_preds = [progress_predictions[i] for i in indices if i < len(progress_predictions)]
            subset_progress_targets = [s.target_progress for i, s in enumerate(samples) if i in indices and s.sample_type == "similarity"]
            if len(subset_progress_preds) == len(subset_progress_targets) and subset_progress_preds:
                try:
                    correlation, _ = spearmanr(subset_progress_preds, subset_progress_targets)
                    return correlation if not np.isnan(correlation) else None
                except:
                    return None
            return None
        return None
    
    # Sample type analysis
    sample_type_indices = {}
    for i, sample in enumerate(samples):
        sample_type = sample.sample_type
        if sample_type not in sample_type_indices:
            sample_type_indices[sample_type] = []
        sample_type_indices[sample_type].append(i)
    
    # Add sample type counts
    for sample_type, indices in sample_type_indices.items():
        granular_metrics[f"count_{sample_type}"] = len(indices)
    
    # Sample type metrics
    for sample_type, indices in sample_type_indices.items():
        granular_metrics[f"accuracy_{sample_type}"] = get_metrics_for_subset(indices, "accuracy_")
        granular_metrics[f"avg_reward_chosen_{sample_type}"] = get_metrics_for_subset(indices, "avg_reward_chosen_")
        granular_metrics[f"avg_reward_rejected_{sample_type}"] = get_metrics_for_subset(indices, "avg_reward_rejected_")
        granular_metrics[f"reward_diff_{sample_type}"] = get_metrics_for_subset(indices, "reward_diff_")
        if any(s.sample_type == "similarity" for s in [samples[i] for i in indices]):
            granular_metrics[f"progress_alignment_{sample_type}"] = get_metrics_for_subset(indices, "progress_alignment_")
    
    # Rewound frame analysis (for paired video samples)
    rewound_indices = [i for i, s in enumerate(samples) if hasattr(s, 'num_frames_rewound') and s.num_frames_rewound is not None]
    if rewound_indices:
        rewound_frame_counts = {}
        for i in rewound_indices:
            num_frames = samples[i].num_frames_rewound
            if num_frames not in rewound_frame_counts:
                rewound_frame_counts[num_frames] = []
            rewound_frame_counts[num_frames].append(i)
        
        # Add rewound frame counts
        for num_frames, indices in rewound_frame_counts.items():
            granular_metrics[f"count_rewound_{num_frames}"] = len(indices)
        
        # Rewound frame metrics
        for num_frames, indices in rewound_frame_counts.items():
            granular_metrics[f"accuracy_rewound_{num_frames}"] = get_metrics_for_subset(indices, "accuracy_")
            granular_metrics[f"avg_reward_chosen_rewound_{num_frames}"] = get_metrics_for_subset(indices, "avg_reward_chosen_")
            granular_metrics[f"avg_reward_rejected_rewound_{num_frames}"] = get_metrics_for_subset(indices, "avg_reward_rejected_")
            granular_metrics[f"reward_diff_rewound_{num_frames}"] = get_metrics_for_subset(indices, "reward_diff_")
            if any(s.sample_type == "similarity" for s in [samples[i] for i in indices]):
                granular_metrics[f"progress_alignment_rewound_{num_frames}"] = get_metrics_for_subset(indices, "progress_alignment_")
    
    # Rejected quality analysis (for paired video samples)
    rejected_quality_indices = {}
    for i, sample in enumerate(samples):
        if hasattr(sample, 'rejected_quality_label') and sample.rejected_quality_label:
            quality = sample.rejected_quality_label
            if quality not in rejected_quality_indices:
                rejected_quality_indices[quality] = []
            rejected_quality_indices[quality].append(i)
    
    # Add rejected quality counts
    for quality, indices in rejected_quality_indices.items():
        granular_metrics[f"count_rejected_{quality}"] = len(indices)
    
    # Rejected quality metrics
    for quality, indices in rejected_quality_indices.items():
        granular_metrics[f"accuracy_rejected_{quality}"] = get_metrics_for_subset(indices, "accuracy_")
        granular_metrics[f"avg_reward_chosen_rejected_{quality}"] = get_metrics_for_subset(indices, "avg_reward_chosen_")
        granular_metrics[f"avg_reward_rejected_rejected_{quality}"] = get_metrics_for_subset(indices, "avg_reward_rejected_")
        granular_metrics[f"reward_diff_rejected_{quality}"] = get_metrics_for_subset(indices, "reward_diff_")
        if any(s.sample_type == "similarity" for s in [samples[i] for i in indices]):
            granular_metrics[f"progress_alignment_rejected_{quality}"] = get_metrics_for_subset(indices, "progress_alignment_")
    
    return {
        **metrics,  # Include base metrics
        **granular_metrics  # Include all granular metrics
    }


def _evaluate_samples(server_url: str, samples: List[Any]) -> Dict[str, Any]:
    payload = build_batch_payload(samples)
    resp = post_batch(server_url, payload)
    return _compute_metrics_from_response(resp, samples)


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
    for batch_idx in tqdm(range(num_batches), desc="Evaluating batches"):
        # Assemble a batch of Sample objects (Preference or Similarity)
        samples = [dataset[idx + j] for j in range(batch_size)]
        idx += batch_size
        
        # Evaluate this batch
        batch_result = _evaluate_samples(server_url, samples)
        results.append(batch_result)
        
        # Print batch results immediately
        print(f"\n" + "="*80)
        print(f"📦 BATCH {batch_idx + 1} RESULTS (Processed)")
        print("="*80)
        
        # Extract main metrics for this batch
        keys = [
            "eval_accuracy",
            "eval_reward_diff", 
            "eval_avg_reward_chosen",
            "eval_avg_reward_rejected",
            "demo_reward_alignment",
        ]
        batch_main_metrics = {k: batch_result.get(k) for k in keys if k in batch_result}
        
        # Extract granular metrics for this batch
        granular_keys = [k for k in batch_result.keys() 
                        if k.startswith(("accuracy_", "avg_reward_chosen_", "avg_reward_rejected_", "reward_diff_", "progress_alignment_", "count_"))]
        batch_granular = {k: batch_result.get(k) for k in granular_keys if k in batch_result}
        
        # Print batch summary
        generate_metrics_summary(
            metrics=batch_main_metrics,
            granular_metrics=batch_granular,
            title=f"BATCH {batch_idx + 1} RESULTS",
            level="batch"
        )
        
        # Add separator between batches (except for the last one)
        if batch_idx < num_batches - 1:
            print("\n" + "─" * 80)
            print("🔄 NEXT BATCH")
            print("─" * 80)
    
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

    # Build flattened lists of indices from preprocessed maps
    optimal_indices: List[int] = []
    for inds in data_generator.optimal_by_task.values():
        optimal_indices.extend(inds)

    subs_by_task_indices: Dict[str, List[int]] = data_generator.suboptimal_by_task

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
    for opt_idx in optimal_indices:
        opt = data_generator.dataset[opt_idx]
        task = opt["task"]

        # Prefer suboptimal from same task; choose any index != opt_idx
        neg_idx: Optional[int] = None
        cand_indices = [idx for idx in subs_by_task_indices.get(task, []) if idx != opt_idx]

        if cand_indices:
            neg_idx = cand_indices[0]
            neg = data_generator.dataset[neg_idx]
            # Load frames from npz
            rej_frames = data_generator._load_frames_from_npz(neg["frames"])
        else:
            # Fallback: rewind-generated negative dict with numpy frames
            neg = data_generator._create_rewind_trajectory(opt)
            rej_frames = neg["frames"]

        # Load optimal frames from npz
        opt_frames = data_generator._get_trajectory_frames(opt_idx)

        # Target progress
        tp_A = data_generator._calculate_target_progress(opt, opt_frames)
        tp_B = data_generator._calculate_target_progress(neg, rej_frames)

        # Shapes
        opt_shape = tuple(opt_frames.shape) if hasattr(opt_frames, "shape") else None
        rej_shape = tuple(rej_frames.shape) if hasattr(rej_frames, "shape") else None
        
        # Determine sample type and num_frames_rewound
        sample_type = "suboptimal_same_task" if neg_idx is not None else "rewind_same_task_fallback"
        num_frames_rewound = None
        if neg_idx is None:  # This is a rewound trajectory
            num_frames_rewound = neg.get('metadata', {}).get('num_frames_rewound')

        samples.append(
            PreferenceSample(
                id=str(opt.get("id")),
                task=str(opt.get("task")),
                lang_vector=opt.get("lang_vector"),
                data_source=str(opt.get("data_source", "")),
                frames=opt.get("frames"),
                frames_shape=opt.get("frames_shape"),
                quality_label=str(opt.get("quality_label", "successful")),
                is_robot=bool(opt.get("is_robot", True)),
                metadata=opt.get("metadata"),
                chosen_frames=opt_frames,
                rejected_frames=rej_frames,
                chosen_frames_shape=opt_shape,
                rejected_frames_shape=rej_shape,
                preferred_trajectory="chosen",
                chosen_id=str(opt.get("id")),
                rejected_id=str(neg.get("id")) if isinstance(neg, dict) else str(neg.get("id")),
                rejected_task=str(neg.get("task")) if isinstance(neg, dict) else str(neg.get("task")),
                rejected_lang_vector=neg.get("lang_vector") if isinstance(neg, dict) else neg.get("lang_vector"),
                rejected_data_source=str(neg.get("data_source", "")) if isinstance(neg, dict) else str(neg.get("data_source", "")),
                rejected_quality_label=str(neg.get("quality_label", "")) if isinstance(neg, dict) else str(neg.get("quality_label", "")),
                rejected_is_robot=bool(neg.get("is_robot", True)) if isinstance(neg, dict) else bool(neg.get("is_robot", True)),
                target_progress_A=tp_A,
                target_progress_B=tp_B,
                sample_type=sample_type,
                num_frames_rewound=num_frames_rewound
            )
        )

    # Send in batches
    results: List[Dict[str, Any]] = []
    total_batches = (len(samples) + batch_size - 1) // batch_size  # Ceiling division
    for batch_idx, i in enumerate(tqdm(range(0, len(samples), batch_size), desc="Evaluating (all prefs)")):
        batch = samples[i : i + batch_size]
        batch_result = _evaluate_samples(server_url, batch)
        results.append(batch_result)
        
        # Print batch results immediately
        print(f"\n" + "="*80)
        print(f"📦 BATCH {batch_idx + 1}/{total_batches} RESULTS (All Preferences)")
        print("="*80)
        
        # Extract main metrics for this batch
        keys = [
            "eval_accuracy",
            "eval_reward_diff", 
            "eval_avg_reward_chosen",
            "eval_avg_reward_rejected",
            "demo_reward_alignment",
        ]
        batch_main_metrics = {k: batch_result.get(k) for k in keys if k in batch_result}
        
        # Extract granular metrics for this batch
        granular_keys = [k for k in batch_result.keys() 
                        if k.startswith(("accuracy_", "avg_reward_chosen_", "avg_reward_rejected_", "reward_diff_", "progress_alignment_", "count_"))]
        batch_granular = {k: batch_result.get(k) for k in granular_keys if k in batch_result}
        
        # Print batch summary
        generate_metrics_summary(
            metrics=batch_main_metrics,
            granular_metrics=batch_granular,
            title=f"BATCH {batch_idx + 1}/{total_batches} RESULTS",
            level="batch"
        )
        
        # Add separator between batches (except for the last one)
        if batch_idx < total_batches - 1:
            print("\n" + "─" * 80)
            print("🔄 NEXT BATCH")
            print("─" * 80)
    
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
    
    # Add granular metrics keys
    granular_keys = []
    for r in results:
        for k in r.keys():
            if k.startswith(("accuracy_", "avg_reward_chosen_", "avg_reward_rejected_", "reward_diff_", "progress_alignment_", "count_")):
                if k not in granular_keys:
                    granular_keys.append(k)
    
    # Sort granular keys for consistent display
    granular_keys.sort()
    
    agg = {k: 0.0 for k in keys + granular_keys}
    # keeps track of which keys are always None
    none_keys = {k: [] for k in keys + granular_keys}
    # keeps track of how many batches actually have each metric
    metric_counts = {k: 0 for k in keys + granular_keys}
    n = max(1, len(results))
    for r in results:
        for k in keys + granular_keys:
            if k in r and r[k] is None:
                none_keys[k].append(1)
            elif k in r and isinstance(r[k], (int, float)):
                none_keys[k].append(0)
                agg[k] += float(r[k])
                metric_counts[k] += 1
    
    # Only average metrics that actually appeared in some batches
    for k in keys + granular_keys:
        if metric_counts[k] > 0:
            agg[k] /= metric_counts[k]
        else:
            agg[k] = None  # Set to None if metric never appeared
    
    for k in none_keys:
        if all(none_keys[k]):
            print(f"WARNING: {k} is always None, removing")
            agg.pop(k)
            continue

    # Print evaluation summary (averaged across batches)
    # Create granular metrics dictionary from aggregated results
    granular_metrics = {k: agg[k] for k in granular_keys if k in agg}
    
    generate_metrics_summary(
        metrics=agg,
        granular_metrics=granular_metrics,
        title="EVALUATION SUMMARY (Averaged Across Batches)",
        level="final"
    )


def generate_metrics_summary(metrics: dict, granular_metrics: dict, title: str, level: str = "batch"):
    """Generate a nicely formatted metrics summary print.
    
    Args:
        metrics: Dictionary of main metrics
        granular_metrics: Dictionary of granular metrics
        title: Title for the summary section
        level: Either "batch" or "final" to indicate the level of detail
    """
    print(f"\n" + "="*80)
    print(f"📊 {title}")
    print("="*80)
    
    # Print main metrics
    if metrics:
        print(f"\n📈 Main Metrics")
        print("-" * 12)
        for k in sorted(metrics.keys()):
            if metrics[k] is not None:
                value = metrics[k]
                # Format the metric name for better readability
                display_name = k.replace("_", " ").title()
                
                # Add emojis for different metric types
                if k.startswith("accuracy"):
                    metric_icon = "🎯"
                elif k.startswith("reward_diff"):
                    metric_icon = "📊"
                elif k.startswith("avg_reward_chosen"):
                    metric_icon = "✅"
                elif k.startswith("avg_reward_rejected"):
                    metric_icon = "❌"
                elif k.startswith("progress_alignment"):
                    metric_icon = "📈"
                else:
                    metric_icon = "📋"
                
                print(f"  {metric_icon} {display_name}: {value:.6f}")
                
                # Add explanation based on metric type
                if k.startswith("accuracy"):
                    print(f"     💡 Accuracy of Predicting the Correct Preference ({level} level)")
                elif k.startswith("avg_reward_chosen"):
                    print(f"     💡 Average Reward Assigned to (Chosen) ({level} level)")
                elif k.startswith("avg_reward_rejected"):
                    print(f"     💡 Average Reward (Rejected) ({level} level)")
                elif k.startswith("reward_diff"):
                    print(f"     💡 Reward Difference between Chosen and Rejected ({level} level)")
                elif k.startswith("progress_alignment"):
                    print(f"     💡 Spearman Correlation between Predicted Progress and Ground Truth Progress ({level} level)")
                else:
                    print(f"     💡 {k}")
            else:
                # Handle None values
                display_name = k.replace("_", " ").title()
                print(f"  ❌ {display_name}: Not Available")
                print(f"     💡 This metric was not computed for any samples in this {level}")
    
    # Print granular metrics if available
    if granular_metrics:
        print(f"\n" + "="*80)
        print(f"🔍 GRANULAR METRICS ANALYSIS ({level.upper()} LEVEL)")
        print("="*80)
        
        # Separate count metrics from performance metrics
        count_metrics = {k: v for k, v in granular_metrics.items() if k.startswith("count_")}
        performance_metrics = {k: v for k, v in granular_metrics.items() if not k.startswith("count_")}
        
        # Print count metrics first
        if count_metrics:
            total_samples = sum(count_metrics.values())
            print(f"\n📊 Sample Counts (Total: {total_samples})")
            print("-" * 15)
            for k in sorted(count_metrics.keys()):
                if count_metrics[k] is not None:
                    value = count_metrics[k]
                    # Format the metric name for better readability
                    display_name = k.replace("count_", "").replace("_", " ").title()
                    
                    # Add emojis for different count types
                    if "rewound" in k:
                        metric_icon = "⏪"
                    elif "rejected" in k:
                        metric_icon = "❌"
                    else:
                        metric_icon = "📊"
                    
                    print(f"  {metric_icon} {display_name}: {value}")
                    print(f"     💡 Number of samples in this category")
        
        # Group performance metrics by type for better organization
        metric_groups = {
            "🎯 Sample Type Analysis": [],
            "⏪ Rewound Frame Analysis": [],
            "❌ Rejected Quality Analysis": []
        }
        
        for k in performance_metrics.keys():
            if k.startswith("accuracy_") and not k.startswith(("accuracy_rewound_", "accuracy_rejected_")):
                metric_groups["🎯 Sample Type Analysis"].append(k)
            elif k.startswith("accuracy_rewound_"):
                metric_groups["⏪ Rewound Frame Analysis"].append(k)
            elif k.startswith("accuracy_rejected_"):
                metric_groups["❌ Rejected Quality Analysis"].append(k)
            elif k.startswith(("avg_reward_chosen_", "avg_reward_rejected_", "reward_diff_", "progress_alignment_")):
                # Add to appropriate group based on the metric type
                if any(k.startswith(prefix) for prefix in ["avg_reward_chosen_", "avg_reward_rejected_", "reward_diff_", "progress_alignment_"]):
                    if "rewound_" in k:
                        metric_groups["⏪ Rewound Frame Analysis"].append(k)
                    elif "rejected_" in k:
                        metric_groups["❌ Rejected Quality Analysis"].append(k)
                    else:
                        metric_groups["🎯 Sample Type Analysis"].append(k)
        
        # Print each group with nice formatting
        for group_name, group_metrics in metric_groups.items():
            if group_metrics:
                print(f"\n{group_name}")
                print("-" * len(group_name))
                
                # Sort metrics within each group for consistent display
                sorted_metrics = sorted(group_metrics)
                
                for k in sorted_metrics:
                    if k in performance_metrics and performance_metrics[k] is not None:
                        value = performance_metrics[k]
                        # Format the metric name for better readability
                        display_name = k.replace("_", " ").title()
                        
                        # Add emojis for different metric types
                        if k.startswith("accuracy_"):
                            metric_icon = "🎯"
                        elif k.startswith("avg_reward_chosen_"):
                            metric_icon = "✅"
                        elif k.startswith("avg_reward_rejected_"):
                            metric_icon = "❌"
                        elif k.startswith("reward_diff_"):
                            metric_icon = "📊"
                        elif k.startswith("progress_alignment_"):
                            metric_icon = "📈"
                        else:
                            metric_icon = "📋"
                        
                        print(f"  {metric_icon} {display_name}: {value:.6f}")
                        
                        # Add explanation for granular metrics
                        if k.startswith("accuracy_"):
                            print(f"     💡 Accuracy of Predicting the Correct Preference (granular breakdown)")
                        elif k.startswith("avg_reward_chosen_"):
                            print(f"     💡 Average Reward Assigned to (Chosen) (granular breakdown)")
                        elif k.startswith("avg_reward_rejected_"):
                            print(f"     💡 Average Reward (Rejected) (granular breakdown)")
                        elif k.startswith("reward_diff_"):
                            print(f"     💡 Reward Difference between Chosen and Rejected (granular breakdown)")
                        elif k.startswith("progress_alignment_"):
                            print(f"     💡 Spearman Correlation between Predicted Progress and Ground Truth Progress (granular breakdown)")
                        else:
                            print(f"     💡 {k}")
                    elif k in performance_metrics and performance_metrics[k] is None:
                        # Handle None values in granular metrics
                        display_name = k.replace("_", " ").title()
                        print(f"  ❌ {display_name}: Not Available")
                        print(f"     💡 This metric was not computed for any samples in this category")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()


