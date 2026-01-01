#!/usr/bin/env python3
"""
Script to run baseline evaluations (GVL, RL-VLM-F, VLAC) on datasets.

Usage:
    # Run RL-VLM-F preference evaluation
    uv run python rfm/evals/run_baseline_eval.py \
        reward_model=rlvlmf \
        vlm_provider=gemini \
        custom_eval.eval_types=[quality_preference] \
        custom_eval.quality_preference=[aliangdw_metaworld_metaworld_eval]
    
    # Run GVL progress evaluation
    uv run python rfm/evals/run_baseline_eval.py \
        reward_model=gvl \
        custom_eval.eval_types=[reward_alignment] \
        custom_eval.reward_alignment=[aliangdw_metaworld_metaworld_eval]
    
    # Run VLAC progress evaluation (requires separate dependency set due to trl conflict)
    PYTHONPATH=.venv-vlac/bin/python rfm/evals/run_baseline_eval.py \
        reward_model=vlac \
        vlac_model_path=InternRobotics/VLAC \
        custom_eval.eval_types=[reward_alignment] \
        custom_eval.reward_alignment=[aliangdw_metaworld_metaworld_eval]

"""

import copy
import json
import os
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
from hydra import main as hydra_main
from omegaconf import DictConfig
from tqdm import tqdm

from rfm.configs.eval_configs import BaselineEvalConfig
from rfm.configs.experiment_configs import DataConfig, CustomEvaluationConfig
from rfm.utils.setup_utils import setup_custom_eval_dataset
from rfm.data.datasets.base import resolve_dataset_keys
from rfm.utils.distributed import is_rank_0
from rfm.utils.logger import get_logger
from rfm.utils.config_utils import display_config, convert_hydra_to_dataclass
from rfm.data.dataset_types import PreferenceSample, ProgressSample, SimilaritySample
from rfm.data.collators.utils import convert_frames_to_pil_images, frames_to_numpy_array
from rfm.evals.baselines.rlvlmf import RLVLMF
from rfm.evals.baselines.gvl import GVL
from rfm.evals.baselines.vlac import VLAC
from rfm.evals.baselines.rfm_model import RFMModel
from rfm.data.dataset_types import SampleType
from rfm.evals.compile_results import compute_eval_metrics

logger = get_logger()


def _create_plot_with_video_gif(
    fig: plt.Figure,
    video_frames: Optional[np.ndarray],
    output_path: str,
    plot_width: int = 800,
    video_height: int = 224,
    fps: int = 2,
) -> None:
    """Create a GIF combining a static plot with animated video frames side by side.

    Args:
        fig: Matplotlib figure to include as static plot
        video_frames: Video frames array of shape [T, C, H, W] or [T, H, W, C]
        output_path: Path to save the GIF
        plot_width: Width of the plot in pixels
        video_height: Height of the video frames in pixels
        fps: Frames per second for the GIF
    """
    if video_frames is None or video_frames.size == 0:
        # If no video, just save the plot as PNG
        fig.savefig(output_path.replace(".gif", ".png"), dpi=150, bbox_inches="tight")
        return

    # Convert matplotlib figure to PIL Image
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plot_img = Image.open(buf)
    plot_img = plot_img.convert("RGB")

    # Resize plot to desired width while maintaining aspect ratio
    plot_aspect = plot_img.height / plot_img.width
    plot_height = int(plot_width * plot_aspect)
    plot_img = plot_img.resize((plot_width, plot_height), Image.Resampling.LANCZOS)

    # Process video frames
    # video_frames is [T, C, H, W] - need to convert to [T, H, W, C] for PIL
    if video_frames.ndim == 4:
        if video_frames.shape[1] == 3 or video_frames.shape[1] == 1:  # [T, C, H, W]
            video_frames = video_frames.transpose(0, 2, 3, 1)  # [T, H, W, C]
        # Now it's [T, H, W, C]

    # Resize video frames to match video_height
    num_frames = video_frames.shape[0]
    frame_height, frame_width = video_frames.shape[1], video_frames.shape[2]
    video_aspect = frame_height / frame_width
    video_width = int(video_height / video_aspect)

    # Create combined frames
    combined_frames = []
    for t in range(num_frames):
        frame = video_frames[t]

        # Ensure uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Convert to PIL Image
        if frame.shape[2] == 1:  # Grayscale
            frame_pil = Image.fromarray(frame[:, :, 0], mode="L").convert("RGB")
        else:
            frame_pil = Image.fromarray(frame, mode="RGB")

        # Resize video frame
        frame_pil = frame_pil.resize((video_width, video_height), Image.Resampling.LANCZOS)

        # Combine plot and video side by side
        # Use the maximum height and pad if needed
        max_height = max(plot_height, video_height)
        combined = Image.new("RGB", (plot_width + video_width, max_height), color="white")

        # Paste plot on the left
        plot_y = (max_height - plot_height) // 2
        combined.paste(plot_img, (0, plot_y))

        # Paste video frame on the right
        video_y = (max_height - video_height) // 2
        combined.paste(frame_pil, (plot_width, video_y))

        # Convert to numpy array for imageio
        combined_frames.append(np.array(combined))

    # Save as GIF
    imageio.mimwrite(output_path, combined_frames, fps=fps, loop=0)
    plt.close(fig)  # Close figure to free memory


def _make_json_serializable(obj: Any) -> Any:
    """Convert numpy types and other non-serializable objects to JSON-serializable types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif obj is None:
        return None
    else:
        return obj


def process_preference_sample(sample: PreferenceSample, model: RLVLMF) -> Dict[str, Any]:
    """Process a single preference sample with baseline."""
    chosen_traj = sample.chosen_trajectory
    rejected_traj = sample.rejected_trajectory

    # Convert frames to PIL Images
    chosen_images = convert_frames_to_pil_images(chosen_traj.frames)
    rejected_images = convert_frames_to_pil_images(rejected_traj.frames)

    assert chosen_traj.task == rejected_traj.task, "Chosen and rejected trajectories must have the same task"

    # Compute preference
    result = model.compute_preference(
        chosen_images=chosen_images,
        rejected_images=rejected_images,
        task_description=chosen_traj.task,
    )

    chosen_metadata = {
        "quality_label": chosen_traj.quality_label,
        "data_source": chosen_traj.data_source,
        "task": chosen_traj.task,
        "id": chosen_traj.id,
        "video_path": chosen_traj.frames if isinstance(chosen_traj.frames, str) else None,
    }
    if chosen_traj.partial_success is not None:
        chosen_metadata["partial_success"] = chosen_traj.partial_success

    rejected_metadata = {
        "quality_label": rejected_traj.quality_label,
        "data_source": rejected_traj.data_source,
        "task": rejected_traj.task,
        "id": rejected_traj.id,
        "video_path": rejected_traj.frames if isinstance(rejected_traj.frames, str) else None,
    }
    if rejected_traj.partial_success is not None:
        rejected_metadata["partial_success"] = rejected_traj.partial_success

    prediction_prob = result.get("prediction_prob")
    is_correct = result.get("is_correct")
    preference_pred = result.get("preference_pred")

    return {
        "preference_pred": float(preference_pred)
        if preference_pred is not None
        else (float(prediction_prob) if prediction_prob is not None else None),
        "preference_labels": 1.0,  # Always 1.0 because chosen trajectory is always preferred by construction
        "is_correct": bool(is_correct) if is_correct is not None else None,
        "task": chosen_traj.task,
        "data_source": chosen_traj.data_source or rejected_traj.data_source,
        "chosen_data_gen_strategy": chosen_traj.data_gen_strategy,
        "rejected_data_gen_strategy": rejected_traj.data_gen_strategy,
        "metadata": {
            "chosen_metadata": chosen_metadata,
            "rejected_metadata": rejected_metadata,
        },
    }


def process_progress_sample(
    sample: ProgressSample,
    model: Union[GVL, VLAC],
) -> Dict[str, Any]:
    """Process a single progress sample with baseline."""
    traj = sample.trajectory

    # Get frames array
    frames_array = frames_to_numpy_array(traj.frames)

    if frames_array is None or frames_array.size == 0:
        logger.warning("No frames found in trajectory")
        return None

    progress_pred = model.compute_progress(frames_array, task_description=traj.task)

    # Convert to numpy array and normalize to [0, 1] if needed
    progress_array = np.array([p if p is not None else 0.0 for p in progress_pred])
    # Note: GVL/VLAC already return normalized [0, 1] values, so no division by 100 needed

    # Build metadata dict - get video_path and frame_step from trajectory metadata
    metadata = {}
    if traj.id is not None:
        metadata["id"] = traj.id
    if traj.metadata is not None:
        metadata["video_path"] = traj.metadata.get("video_path")
        frame_step = traj.metadata.get("frame_step")
        if frame_step is not None:
            metadata["frame_step"] = frame_step

    # Build result dict
    result = {
        "progress_pred": progress_array.tolist(),
        "task": traj.task,
        "data_source": traj.data_source,
        "data_gen_strategy": traj.data_gen_strategy,
        "metadata": metadata,
        "id": traj.id,
        "video_path": metadata.get("video_path"),
        "partial_success": traj.partial_success,
        "target_progress": traj.target_progress,
        "quality_label": traj.quality_label,
    }

    return result


def process_batched_rfm_samples(
    dataset,
    model: RFMModel,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """Process RFM/ReWiND samples using batched computation with minibatching.

    Args:
        dataset: Dataset object that supports indexing (e.g., CustomEvalDataset)
        model: RFMModel instance
        batch_size: Batch size for processing samples

    Returns:
        List of result dictionaries in the same format as process_progress_sample/process_preference_sample
    """
    dataset_len = len(dataset)

    # Group indices by sample type (iterate once, only storing indices)
    progress_indices = []
    preference_indices = []
    similarity_indices = []

    for i in range(dataset_len):
        sample = dataset[i]
        if isinstance(sample, ProgressSample):
            progress_indices.append(i)
        elif isinstance(sample, PreferenceSample):
            preference_indices.append(i)
        elif isinstance(sample, SimilaritySample):
            similarity_indices.append(i)
        else:
            logger.warning(f"Unknown sample type: {type(sample)}")

    results = []

    # Process progress samples in minibatches
    if progress_indices:
        for batch_start in tqdm(range(0, len(progress_indices), batch_size), desc="Processing progress batches"):
            batch_indices = progress_indices[batch_start : batch_start + batch_size]
            batch = [dataset[i] for i in batch_indices]
            progress_preds = model.compute_batched_progress(batch)
            for sample, progress_pred in zip(batch, progress_preds):
                traj = sample.trajectory

                # Build metadata dict - get video_path and frame_step from trajectory metadata
                metadata = {}
                if traj.id is not None:
                    metadata["id"] = traj.id
                if traj.metadata is not None:
                    metadata["video_path"] = traj.metadata.get("video_path")
                    frame_step = traj.metadata.get("frame_step")
                    if frame_step is not None:
                        metadata["frame_step"] = frame_step

                # Build result dict
                result = {
                    "progress_pred": progress_pred,
                    "task": traj.task,
                    "data_source": traj.data_source,
                    "data_gen_strategy": traj.data_gen_strategy,
                    "metadata": metadata,
                    "id": traj.id,
                    "video_path": metadata.get("video_path"),
                    "partial_success": traj.partial_success,
                    "target_progress": traj.target_progress,
                    "quality_label": traj.quality_label,
                }
                results.append(result)

    # Process preference samples in minibatches
    if preference_indices:
        for batch_start in tqdm(range(0, len(preference_indices), batch_size), desc="Processing preference batches"):
            batch_indices = preference_indices[batch_start : batch_start + batch_size]
            batch = [dataset[i] for i in batch_indices]
            preference_results = model.compute_batched_preference(batch)
            for sample, result in zip(batch, preference_results):
                chosen_traj = sample.chosen_trajectory
                rejected_traj = sample.rejected_trajectory

                chosen_metadata = {
                    "quality_label": chosen_traj.quality_label,
                    "data_source": chosen_traj.data_source,
                    "task": chosen_traj.task,
                    "id": chosen_traj.id,
                    "video_path": chosen_traj.frames if isinstance(chosen_traj.frames, str) else None,
                }
                if chosen_traj.partial_success is not None:
                    chosen_metadata["partial_success"] = chosen_traj.partial_success

                rejected_metadata = {
                    "quality_label": rejected_traj.quality_label,
                    "data_source": rejected_traj.data_source,
                    "task": rejected_traj.task,
                    "id": rejected_traj.id,
                    "video_path": rejected_traj.frames if isinstance(rejected_traj.frames, str) else None,
                }
                if rejected_traj.partial_success is not None:
                    rejected_metadata["partial_success"] = rejected_traj.partial_success

                prediction_prob = result.get("prediction_prob")
                is_correct = result.get("is_correct")
                preference_pred = result.get("preference_pred")

                formatted_result = {
                    "preference_pred": float(preference_pred)
                    if preference_pred is not None
                    else (float(prediction_prob) if prediction_prob is not None else None),
                    "preference_labels": 1.0,  # Always 1.0 because chosen trajectory is always preferred by construction
                    "is_correct": bool(is_correct) if is_correct is not None else None,
                    "task": chosen_traj.task,
                    "data_source": chosen_traj.data_source or rejected_traj.data_source,
                    "chosen_data_gen_strategy": chosen_traj.data_gen_strategy,
                    "rejected_data_gen_strategy": rejected_traj.data_gen_strategy,
                    "metadata": {
                        "chosen_metadata": chosen_metadata,
                        "rejected_metadata": rejected_metadata,
                    },
                }
                results.append(formatted_result)

    # Process similarity samples in minibatches (if needed in the future)
    if similarity_indices:
        for batch_start in tqdm(range(0, len(similarity_indices), batch_size), desc="Processing similarity batches"):
            batch_indices = similarity_indices[batch_start : batch_start + batch_size]
            batch = [dataset[i] for i in batch_indices]
            similarity_results = model.compute_batched_similarity(batch)
            # For now, similarity samples are not used in baseline evaluation
            # but we process them for completeness
            for sample, result in zip(batch, similarity_results):
                # Format similarity result if needed
                results.append(result)

    return results


def run_baseline_evaluation(cfg: BaselineEvalConfig, base_data_cfg: DataConfig) -> Dict[str, Any]:
    """Run baseline evaluation on datasets."""

    # Initialize model
    if cfg.reward_model == "rlvlmf":
        model = RLVLMF(vlm_provider=cfg.vlm_provider, temperature=cfg.temperature)
    elif cfg.reward_model == "gvl":
        # API key is read from GEMINI_API_KEY environment variable
        model = GVL(max_frames=cfg.gvl_max_frames, offset=cfg.gvl_offset)
    elif cfg.reward_model == "vlac":
        if not cfg.vlac_model_path:
            raise ValueError("vlac_model_path is required for VLAC baseline")

        model = VLAC(
            model_path=cfg.vlac_model_path,
            device=cfg.vlac_device,
            model_type=cfg.vlac_model_type,
            temperature=cfg.vlac_temperature,
            batch_num=cfg.vlac_batch_num,
            skip=cfg.vlac_skip,
            frame_skip=cfg.vlac_frame_skip,
            use_images=cfg.vlac_use_images,
        )
    elif cfg.reward_model in ["rfm", "rewind"]:
        if not cfg.rfm_checkpoint_path:
            raise ValueError("rfm_checkpoint_path is required for RFM/ReWiND reward model")

        model = RFMModel(checkpoint_path=cfg.rfm_checkpoint_path)
    else:
        raise ValueError(
            f"Unknown reward_model: {cfg.reward_model}. Must be 'rlvlmf', 'gvl', 'vlac', 'rfm', or 'rewind'"
        )

    all_metrics = {}

    # Process each evaluation type
    for eval_type in cfg.custom_eval.eval_types:
        logger.info(f"=" * 80)
        logger.info(f"Running {eval_type} evaluation with {cfg.reward_model} reward model")
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
                sampler_kwargs["use_frame_steps"] = cfg.custom_eval.use_frame_steps
            elif eval_type == "policy_ranking":
                sampler_kwargs["num_examples_per_quality_pr"] = cfg.custom_eval.num_examples_per_quality_pr
                sampler_kwargs["num_partial_successes"] = cfg.custom_eval.num_partial_successes
                sampler_kwargs["max_tasks"] = cfg.custom_eval.policy_ranking_max_tasks
                sampler_kwargs["use_frame_steps"] = cfg.custom_eval.use_frame_steps
            elif "quality_preference" in eval_type:
                sampler_kwargs["comparisons_per_task"] = cfg.custom_eval.comparisons_per_task
                sampler_kwargs["max_comparisons"] = cfg.custom_eval.max_comparisons

            dataset = setup_custom_eval_dataset(
                cfg=eval_data_cfg, sampler_type=eval_type, is_eval=True, verbose=True, sampler_kwargs=sampler_kwargs
            )

            # Process samples
            eval_results = []

            if cfg.reward_model in ["rfm", "rewind"]:
                # For RFM/ReWiND, process dataset using indices to avoid materializing entire dataset
                logger.info(f"Processing {len(dataset)} samples in batches for RFM/ReWiND")

                try:
                    batch_results = process_batched_rfm_samples(dataset, model, batch_size=cfg.rfm_batch_size)
                    eval_results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    raise
            else:
                # For other models, process samples one at a time
                for i, sample in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}")):
                    try:
                        if cfg.reward_model == "rlvlmf" and isinstance(sample, PreferenceSample):
                            result = process_preference_sample(sample, model)
                            if result:
                                eval_results.append(result)
                        elif cfg.reward_model in ["gvl", "vlac"] and isinstance(sample, ProgressSample):
                            result = process_progress_sample(sample, model)
                            if result:
                                eval_results.append(result)
                        else:
                            logger.warning(
                                f"Sample type mismatch: reward_model={cfg.reward_model}, sample={type(sample)}"
                            )
                    except Exception as e:
                        logger.error(f"Error processing sample {i}: {e}")
                        continue

            logger.info(f"Processed {len(eval_results)} samples from {dataset_name}")

            # Save results to JSON
            if cfg.output_dir:
                results_file = os.path.join(cfg.output_dir, f"{eval_type}_{dataset_name}_results.json")
                with open(results_file, "w") as f:
                    json.dump(eval_results, f, indent=2)
                logger.info(f"Saved results to {results_file}")

            # Compute metrics using the same functions as the trainer
            if eval_results:
                # Determine data_source from first result
                data_source = eval_results[0].get("data_source") if eval_results else None

                if eval_type == "quality_preference":
                    # Quality preference evaluation for rlvlmf, rfm, rewind
                    if cfg.reward_model not in ["rlvlmf", "rfm", "rewind"]:
                        raise ValueError(
                            f"quality_preference evaluation only supported for rlvlmf, rfm, rewind, got {cfg.reward_model}"
                        )

                    eval_metrics_result = compute_eval_metrics(
                        eval_type="quality_preference",
                        results=eval_results,
                        progress_pred_type="absolute",  # Not used for preference
                        is_discrete_mode=False,  # Not used for preference
                        num_bins=None,  # Not used for preference
                        data_source=data_source,
                    )
                    if isinstance(eval_metrics_result, tuple):
                        metrics_dict, task_groups, task_details = eval_metrics_result
                        # Save task_groups and task_details if available
                        if cfg.output_dir:
                            task_groups_file = os.path.join(
                                cfg.output_dir, f"{eval_type}_{dataset_name}_task_groups.json"
                            )
                            task_details_file = os.path.join(
                                cfg.output_dir, f"{eval_type}_{dataset_name}_task_details.json"
                            )
                            with open(task_groups_file, "w") as f:
                                json.dump(_make_json_serializable(task_groups), f, indent=2)
                            with open(task_details_file, "w") as f:
                                json.dump(_make_json_serializable(task_details), f, indent=2)
                            logger.info(f"Saved task_groups to {task_groups_file}")
                            logger.info(f"Saved task_details to {task_details_file}")
                    else:
                        metrics_dict = eval_metrics_result

                    # Extract metrics from the returned dict
                    for key, value in metrics_dict.items():
                        if isinstance(value, (int, float)):
                            eval_type_metrics[f"{dataset_name}/{key}"] = float(value)

                else:
                    # Progress evaluation (reward_alignment, policy_ranking) for gvl, vlac, rfm, rewind
                    if cfg.reward_model not in ["gvl", "vlac", "rfm", "rewind"]:
                        raise ValueError(
                            f"Progress evaluation only supported for gvl, vlac, rfm, rewind, got {cfg.reward_model}"
                        )

                    eval_metrics_result = compute_eval_metrics(
                        eval_type=eval_type,
                        results=eval_results,
                        progress_pred_type="absolute_wrt_total_frames",  # Baselines use absolute progress
                        is_discrete_mode=False,  # Baselines output continuous values
                        num_bins=None,
                        data_source=data_source,
                    )

                    if isinstance(eval_metrics_result, tuple):
                        if eval_type == "reward_alignment":
                            metrics_dict, plots, video_frames_list, _ = eval_metrics_result
                            # Save plots with videos as GIFs if available
                            if plots and cfg.output_dir:
                                plots_dir = os.path.join(cfg.output_dir, f"{eval_type}_{dataset_name}_plots")
                                os.makedirs(plots_dir, exist_ok=True)
                                for i, fig in enumerate(plots):
                                    video_frames = video_frames_list[i] if i < len(video_frames_list) else None
                                    gif_path = os.path.join(plots_dir, f"trajectory_{i:04d}.gif")
                                    _create_plot_with_video_gif(fig, video_frames, gif_path)
                                logger.info(f"Saved {len(plots)} plot+video GIFs to {plots_dir}")
                        elif eval_type == "policy_ranking":
                            metrics_dict, task_groups, task_details = eval_metrics_result
                            # Save task_groups and task_details if available
                            if cfg.output_dir:
                                task_groups_file = os.path.join(
                                    cfg.output_dir, f"{eval_type}_{dataset_name}_task_groups.json"
                                )
                                task_details_file = os.path.join(
                                    cfg.output_dir, f"{eval_type}_{dataset_name}_task_details.json"
                                )
                                with open(task_groups_file, "w") as f:
                                    json.dump(_make_json_serializable(task_groups), f, indent=2)
                                with open(task_details_file, "w") as f:
                                    json.dump(_make_json_serializable(task_details), f, indent=2)
                                logger.info(f"Saved task_groups to {task_groups_file}")
                                logger.info(f"Saved task_details to {task_details_file}")
                        else:
                            metrics_dict = eval_metrics_result[0] if len(eval_metrics_result) > 0 else {}
                    else:
                        metrics_dict = eval_metrics_result

                    # Extract metrics from the returned dict
                    for key, value in metrics_dict.items():
                        if isinstance(value, (int, float)):
                            eval_type_metrics[f"{dataset_name}/{key}"] = float(value)

        all_metrics[eval_type] = eval_type_metrics

    return all_metrics


@hydra_main(version_base=None, config_path="../configs", config_name="baseline_eval_config")
def main(cfg: DictConfig):
    """Main entry point for baseline evaluation."""
    # Convert Hydra config to dataclass
    baseline_cfg = convert_hydra_to_dataclass(cfg, BaselineEvalConfig)

    # Display config
    display_config(baseline_cfg)

    # Validate reward model
    if baseline_cfg.reward_model not in ["gvl", "vlac", "rlvlmf", "rfm", "rewind"]:
        raise ValueError(
            f"reward_model must be 'gvl', 'vlac', 'rlvlmf', 'rfm', or 'rewind', got {baseline_cfg.reward_model}"
        )

    # Setup output directory
    if baseline_cfg.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        baseline_cfg.output_dir = os.path.join("./baseline_eval_output", f"{baseline_cfg.reward_model}_{timestamp}")

    os.makedirs(baseline_cfg.output_dir, exist_ok=True)
    logger.info(f"Output directory: {baseline_cfg.output_dir}")

    # Create data config with default settings
    # Datasets will be set per eval type during processing
    data_cfg = DataConfig(
        max_frames=baseline_cfg.gvl_max_frames,
    )

    display_config(data_cfg)

    # Run evaluation
    metrics = run_baseline_evaluation(baseline_cfg, data_cfg)

    # Save metrics
    if metrics and is_rank_0():
        metrics_file = os.path.join(baseline_cfg.output_dir, "baseline_metrics.json")
        metrics_serializable = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                metrics_serializable[k] = {
                    k2: float(v2) if isinstance(v2, (int, float, np.number)) else v2 for k2, v2 in v.items()
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
