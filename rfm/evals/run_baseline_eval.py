#!/usr/bin/env python3
"""
Script to run baseline evaluations (GVL, RL-VLM-F, VLAC) on datasets.

Usage:
    # Run RL-VLM-F preference evaluation
    uv run python rfm/evals/run_baseline_eval.py \
        baseline_type=rlvlmf \
        vlm_provider=gemini \
        custom_eval.eval_types=[quality_preference] \
        custom_eval.quality_preference=[aliangdw_metaworld_metaworld_eval]
    
    # Run GVL progress evaluation
    uv run python rfm/evals/run_baseline_eval.py \
        baseline_type=gvl \
        custom_eval.eval_types=[reward_alignment] \
        custom_eval.reward_alignment=[aliangdw_metaworld_metaworld_eval]
    
    # Run VLAC progress evaluation
    uv run python rfm/evals/run_baseline_eval.py \
        baseline_type=vlac \
        vlac_model_path=/path/to/vlac/model \
        custom_eval.eval_types=[reward_alignment] \
        custom_eval.reward_alignment=[aliangdw_metaworld_metaworld_eval]
"""

import copy
import json
import os
from typing import Optional, Dict, Any, List
from datetime import datetime

import numpy as np
from PIL import Image
from hydra import main as hydra_main
from omegaconf import DictConfig
from tqdm import tqdm

from rfm.configs.eval_configs import BaselineEvalConfig
from rfm.configs.experiment_configs import DataConfig, CustomEvaluationConfig
from rfm.utils.setup_utils import setup_custom_eval_dataset, resolve_dataset_keys
from rfm.utils.distributed import is_rank_0
from rfm.utils.logger import get_logger
from rfm.utils.config_utils import display_config, convert_hydra_to_dataclass
from rfm.data.dataset_types import PreferenceSample, ProgressSample
from rfm.evals.baselines.rlvlmf import RLVLMF
from rfm.evals.baselines.gvl import GeminiVideoAnalyzerHDF5
from rfm.evals.baselines.vlac import VLAC

logger = get_logger()


def process_preference_sample(
    sample: PreferenceSample,
    baseline: RLVLMF
) -> Dict[str, Any]:
    """Process a single preference sample with baseline."""
    chosen_traj = sample.chosen_trajectory
    rejected_traj = sample.rejected_trajectory
    
    # Convert frames to PIL Images
    chosen_images = []
    if chosen_traj.frames is not None:
        if isinstance(chosen_traj.frames, np.ndarray):
            for frame in chosen_traj.frames:
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                chosen_images.append(Image.fromarray(frame))
        elif isinstance(chosen_traj.frames, list):
            # Assume paths or already images
            for frame in chosen_traj.frames:
                if isinstance(frame, str):
                    chosen_images.append(Image.open(frame))
                else:
                    chosen_images.append(frame)
    
    rejected_images = []
    if rejected_traj.frames is not None:
        if isinstance(rejected_traj.frames, np.ndarray):
            for frame in rejected_traj.frames:
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                rejected_images.append(Image.fromarray(frame))
        elif isinstance(rejected_traj.frames, list):
            for frame in rejected_traj.frames:
                if isinstance(frame, str):
                    rejected_images.append(Image.open(frame))
                else:
                    rejected_images.append(frame)
    
    task = chosen_traj.task or rejected_traj.task or ""
    
    # Compute preference
    result = baseline.compute_preference(chosen_images, rejected_images, task)
    
    # Build result dict similar to trainer format
    return {
        "prediction_prob": result["prediction_prob"],
        "is_correct": result.get("is_correct", None),  # May be None if no ground truth
        "task": task,
        "data_source": chosen_traj.data_source or rejected_traj.data_source,
        "chosen_data_gen_strategy": chosen_traj.data_gen_strategy,
        "rejected_data_gen_strategy": rejected_traj.data_gen_strategy,
        "vlm_preference": result["vlm_preference"],
    }


def process_progress_sample(
    sample: ProgressSample,
    analyzer: Optional[GeminiVideoAnalyzerHDF5] = None,
    vlac_model: Optional[VLAC] = None
) -> Dict[str, Any]:
    """Process a single progress sample with baseline."""
    traj = sample.trajectory
    
    # Get frames array
    frames_array = None
    if traj.frames is not None:
        if isinstance(traj.frames, np.ndarray):
            frames_array = traj.frames
            if frames_array.dtype != np.uint8:
                frames_array = np.clip(frames_array, 0, 255).astype(np.uint8)
        elif isinstance(traj.frames, list):
            # Convert list of images/paths to array
            frame_list = []
            for frame in traj.frames:
                if isinstance(frame, str):
                    img = np.array(Image.open(frame))
                elif isinstance(frame, Image.Image):
                    img = np.array(frame)
                else:
                    img = np.array(frame)
                frame_list.append(img)
            frames_array = np.stack(frame_list)
    
    if frames_array is None or frames_array.size == 0:
        logger.warning("No frames found in trajectory")
        return None
    
    # Compute progress
    if vlac_model is not None:
        # Use VLAC model
        task = traj.task or ""
        progress_pred = vlac_model.compute_progress(frames_array, task_description=task)
    elif analyzer is not None:
        # Use GVL analyzer
        progress_pred = analyzer.compute_progress(frames_array)
    else:
        raise ValueError("Either analyzer or vlac_model must be provided")
    
    # Convert to numpy array and normalize to [0, 1] if needed
    progress_array = np.array([p / 100.0 if p is not None else 0.0 for p in progress_pred])
    
    # Get target progress if available
    target_progress = traj.target_progress
    if target_progress is not None:
        target_array = np.array(target_progress)
    else:
        target_array = None
    
    # Build result dict similar to trainer format
    result = {
        "progress_pred": progress_array.tolist(),
        "task": traj.task,
        "data_source": traj.data_source,
        "data_gen_strategy": traj.data_gen_strategy,
    }
    
    if target_array is not None:
        result["target_progress"] = target_array.tolist()
        # Compute MSE
        if len(progress_array) == len(target_array):
            mse = float(np.mean((progress_array - target_array) ** 2))
            result["mse"] = mse
    
    if traj.quality_label is not None:
        result["quality_label"] = traj.quality_label
    
    return result


def run_baseline_evaluation(
    cfg: BaselineEvalConfig,
    base_data_cfg: DataConfig
) -> Dict[str, Any]:
    """Run baseline evaluation on datasets."""
    
    # Initialize baseline
    if cfg.baseline_type == "rlvlmf":
        baseline = RLVLMF(
            vlm_provider=cfg.vlm_provider,
            temperature=cfg.temperature
        )
        analyzer = None
        vlac_model = None
    elif cfg.baseline_type == "gvl":
        api_key = cfg.gvl_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set for GVL baseline")
        
        # Create a dummy analyzer - task will be set per sample
        analyzer = GeminiVideoAnalyzerHDF5(
            api_key=api_key,
            task_description="",  # Will be set per sample
            max_frames=cfg.gvl_max_frames,
            offset=cfg.gvl_offset
        )
        baseline = None
        vlac_model = None
    elif cfg.baseline_type == "vlac":
        if not cfg.vlac_model_path:
            raise ValueError("vlac_model_path is required for VLAC baseline")
        
        vlac_model = VLAC(
            model_path=cfg.vlac_model_path,
            device=cfg.vlac_device,
            model_type=cfg.vlac_model_type,
            temperature=cfg.vlac_temperature,
            batch_num=cfg.vlac_batch_num,
            skip=cfg.vlac_skip,
            frame_skip=cfg.vlac_frame_skip
        )
        baseline = None
        analyzer = None
    else:
        raise ValueError(f"Unknown baseline_type: {cfg.baseline_type}. Must be 'rlvlmf', 'gvl', or 'vlac'")
    
    all_metrics = {}
    
    # Process each evaluation type
    for eval_type in cfg.custom_eval.eval_types:
        logger.info(f"=" * 80)
        logger.info(f"Running {eval_type} evaluation with {cfg.baseline_type} baseline")
        logger.info(f"=" * 80)
        
        # Get datasets for this eval type
        eval_datasets = getattr(cfg.custom_eval, eval_type, [])
        if not eval_datasets:
            logger.warning(f"No datasets specified for {eval_type}, skipping")
            continue
        
        # Resolve dataset keys
        resolved_datasets = resolve_dataset_keys(eval_datasets, split="eval")
        logger.info(f"Resolved datasets for {eval_type}: {resolved_datasets}")
        
        eval_type_metrics = {}
        
        for dataset_name in resolved_datasets:
            logger.info(f"Processing dataset: {dataset_name}")
            
            # Create data config for this dataset (similar to trainer)
            eval_data_cfg = copy.deepcopy(base_data_cfg)
            eval_data_cfg.dataset_type = "rfm"
            eval_data_cfg.eval_datasets = [dataset_name]
            
            # Setup dataset
            sampler_kwargs = {
                "random_seed": cfg.custom_eval.custom_eval_random_seed,
            }
            
            if eval_type == "reward_alignment":
                sampler_kwargs["max_trajectories"] = cfg.custom_eval.reward_alignment_max_trajectories
            elif eval_type == "policy_ranking":
                sampler_kwargs["num_examples_per_quality_pr"] = cfg.custom_eval.num_examples_per_quality_pr
                sampler_kwargs["max_tasks"] = cfg.custom_eval.policy_ranking_max_tasks
            elif "quality_preference" in eval_type:
                sampler_kwargs["comparisons_per_task"] = cfg.custom_eval.comparisons_per_task
            
            dataset = setup_custom_eval_dataset(
                cfg=eval_data_cfg,
                sampler_type=eval_type,
                is_eval=True,
                verbose=True,
                sampler_kwargs=sampler_kwargs
            )
            
            # Process samples
            eval_results = []
            for i, sample in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}")):
                try:
                    if cfg.baseline_type == "rlvlmf" and isinstance(sample, PreferenceSample):
                        result = process_preference_sample(sample, baseline)
                        if result:
                            eval_results.append(result)
                    elif cfg.baseline_type in ["gvl", "vlac"] and isinstance(sample, ProgressSample):
                        if cfg.baseline_type == "gvl":
                            # Update analyzer task for this sample
                            analyzer.task_description = sample.trajectory.task or ""
                            result = process_progress_sample(sample, analyzer=analyzer)
                        else:  # vlac
                            result = process_progress_sample(sample, vlac_model=vlac_model)
                        if result:
                            eval_results.append(result)
                    else:
                        logger.warning(f"Sample type mismatch: baseline={cfg.baseline_type}, sample={type(sample)}")
                except Exception as e:
                    logger.error(f"Error processing sample {i}: {e}")
                    continue
            
            logger.info(f"Processed {len(eval_results)} samples from {dataset_name}")
            
            # Compute metrics (simplified - would need to import metric computation from trainer)
            # For now, just save results
            if cfg.output_dir:
                results_file = os.path.join(
                    cfg.output_dir,
                    f"{eval_type}_{dataset_name}_results.json"
                )
                with open(results_file, "w") as f:
                    json.dump(eval_results, f, indent=2)
                logger.info(f"Saved results to {results_file}")
            
            # Store basic metrics
            if eval_results:
                if cfg.baseline_type == "rlvlmf":
                    # Preference metrics
                    correct = [r.get("is_correct") for r in eval_results if r.get("is_correct") is not None]
                    if correct:
                        accuracy = sum(correct) / len(correct)
                        eval_type_metrics[f"{dataset_name}/accuracy"] = accuracy
                elif cfg.baseline_type in ["gvl", "vlac"]:
                    # Progress metrics
                    mse_values = [r.get("mse") for r in eval_results if r.get("mse") is not None]
                    if mse_values:
                        avg_mse = np.mean(mse_values)
                        eval_type_metrics[f"{dataset_name}/mse"] = float(avg_mse)
        
        all_metrics[eval_type] = eval_type_metrics
    
    return all_metrics


@hydra_main(version_base=None, config_path="rfm/configs", config_name="baseline_eval_config")
def main(cfg: DictConfig):
    """Main entry point for baseline evaluation."""
    # Convert Hydra config to dataclass
    baseline_cfg = convert_hydra_to_dataclass(cfg, BaselineEvalConfig)
    
    # Display config
    display_config(baseline_cfg)
    
    # Validate baseline type
    if baseline_cfg.baseline_type not in ["gvl", "vlac", "rlvlmf"]:
        raise ValueError(f"baseline_type must be 'gvl', 'vlac', or 'rlvlmf', got {baseline_cfg.baseline_type}")
    
    # Setup output directory
    if baseline_cfg.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        baseline_cfg.output_dir = os.path.join(
            "./baseline_eval_output",
            f"{baseline_cfg.baseline_type}_{timestamp}"
        )
    
    os.makedirs(baseline_cfg.output_dir, exist_ok=True)
    logger.info(f"Output directory: {baseline_cfg.output_dir}")
    
    # Create data config (needed for dataset setup)
    # We need to load a base config to get data_root and other settings
    # For now, use minimal config - datasets will be set per eval type
    # In practice, you might want to load from an existing config file
    data_cfg = DataConfig(
        train_datasets=[],
        eval_datasets=[],  # Will be set per eval type
        data_root=None,  # Should be set from environment or config
    )
    
    # Run evaluation
    metrics = run_baseline_evaluation(baseline_cfg, data_cfg)
    
    # Save metrics
    if metrics and is_rank_0():
        metrics_file = os.path.join(baseline_cfg.output_dir, "baseline_metrics.json")
        metrics_serializable = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                metrics_serializable[k] = {
                    k2: float(v2) if isinstance(v2, (int, float, np.number)) else v2
                    for k2, v2 in v.items()
                }
            else:
                metrics_serializable[k] = v
        
        with open(metrics_file, "w") as f:
            json.dump(metrics_serializable, f, indent=2)
        logger.info(f"Saved metrics to: {metrics_file}")
    
    logger.info("\nBaseline evaluation complete!")
    return metrics


if __name__ == "__main__":
    main()

