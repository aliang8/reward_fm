#!/usr/bin/env python3
"""
Relabel rewards in offline HDF5 datasets using common reward relabeling functions.

This script:
- Loads a reward model or connects to an eval server
- Iterates through episodes in H5 files
- Extracts frames and language instructions
- Calls common relabeling functions to compute per-timestep rewards
- Writes new H5 files with updated 'rewards' datasets

Assumes datasets follow a robomimic-style layout:
    /data/{demo}/actions
    /data/{demo}/obs/{key}
    /data/{demo}/rewards
    /data/{demo}/dones
    language annotation under /data/{demo}/language_instruction or /data/{demo}/obs/language

Example usage (eval server):
    uv run python scripts/relabel_h5_rewards.py \
        --h5_paths rfm_offlineRL_1_succ.h5 \
        --reward-model-path aliangdw/qwen4b_pref_prog_succ_8_frames_all_part2 \
        --sentence-encoder sentence-transformers/all-MiniLM-L6-v2 \
        --batch_size 32

Example usage (RoboReward baseline):
    uv run python scripts/relabel_h5_rewards.py \
        --h5_paths /scr/shared/reward_fm/play_datasets/play_dataset_test_subtrajs_gripper.h5 \
        --reward-model roboreward \
        --reward-model-path teetone/RoboReward-4B    
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

# Disable HDF5 file locking for GPFS compatibility (must be set before importing h5py)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import common relabeling functions
from convert_lerobot_common import (
    relabel_rewards_with_model,
    relabel_rewards_with_eval_server,
    relabel_rewards_with_baseline_eval_server,
    load_reward_model_for_relabeling,
    load_baseline_model_for_relabeling,
    print_dataset_info,
    HAS_BASELINE_MODELS,
)

# Import frame subsampling helper
try:
    from rfm.data.datasets.helpers import linspace_subsample_frames
    HAS_SUBSAMPLE_HELPER = True
except ImportError:
    HAS_SUBSAMPLE_HELPER = False
    logger.warning("linspace_subsample_frames not available. Frame subsampling will be disabled.")


def open_h5_with_retry(path: str, mode: str = "r", max_retries: int = 5, base_delay: float = 2.0):
    """
    Open an HDF5 file with retry logic for GPFS file locking issues.
    
    Args:
        path: Path to the H5 file
        mode: File mode ('r', 'w', 'a', etc.)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (will be exponentially increased)
    
    Returns:
        h5py.File object
    
    Raises:
        Exception: If all retries fail
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            # Try with locking disabled at the file level too
            return h5py.File(path, mode, locking=False)
        except (BlockingIOError, OSError) as e:
            # Catch both BlockingIOError and OSError (GPFS can throw either)
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"GPFS file locking issue opening {path} (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Failed to open {path} after {max_retries} attempts: {e}")
    raise last_error


def build_output_path_for_input(
    input_path: str,
    output_path: Optional[str],
    language_instruction: Optional[str] = None,
    image_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Resolve the output file path for a given input H5.
    - If output_path is None: place next to input, with _relabeled suffix.
    - If output_path is an existing directory: write inside it with _relabeled suffix.
    - If a single input is used and output_path is a file path (endswith .h5): use that path.
    - If language_instruction, image_key, or model are provided, they will be incorporated into the filename (sanitized).
    """
    import re

    def _sanitize(s: str, max_len: int = 50) -> str:
        s = s.replace("/", "_")
        s = re.sub(r'[^\w\s\-.]', '', s)
        s = re.sub(r'[-\s.]+', '_', s).strip('_')
        return s[:max_len] if s else ""

    suffixes = []
    if model:
        suffixes.append(f"model_{_sanitize(model, 50)}")
    if language_instruction:
        suffixes.append(_sanitize(language_instruction, 50))
    if image_key:
        suffixes.append(f"img_{_sanitize(image_key, 40)}")
    suffix = "_" + "_".join(suffixes) if suffixes else ""

    if output_path is None:
        return input_path.replace(".h5", f"_relabeled{suffix}.h5")
    op = Path(output_path)
    if op.is_dir() or (output_path.endswith(os.sep) and not op.exists()):
        op = op if op.is_dir() else op
        return str((op / (Path(input_path).stem + f"_relabeled{suffix}.h5")).with_suffix(".h5"))
    if output_path.lower().endswith(".h5"):
        return output_path
    return str((op / (Path(input_path).stem + f"_relabeled{suffix}.h5")).with_suffix(".h5"))


def compute_env_reward(episode_len: int, sparse: bool) -> np.ndarray:
    """
    Compute environment reward for an episode.
    
    Args:
        episode_len: Number of timesteps in the episode
        sparse: If True, returns -1 for all timesteps except last (0).
                If False, returns all zeros.
    
    Returns:
        Array of env rewards with shape (episode_len,)
    """
    if sparse:
        env_reward = np.full(episode_len, -1.0, dtype=np.float32)
        env_reward[-1] = 0.0  # Last frame gets 0
    else:
        env_reward = np.zeros(episode_len, dtype=np.float32)
    return env_reward


def copy_group_with_relabel(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    demo_name: str,
    rewards_map: Dict[str, np.ndarray],
    root_group: h5py.Group,
    sparse_env_reward: bool = False,
    success_preds_map: Optional[Dict[str, np.ndarray]] = None,
):
    """
    Recursively copy src_group to dst_group, replacing 'rewards' dataset when encountered.
    Also creates 'progress_reward', 'env_reward', and optionally 'success_preds' datasets.
    """
    for key in src_group.keys():
        obj = src_group[key]
        if isinstance(obj, h5py.Group):
            sub_group = dst_group.create_group(key)
            copy_group_with_relabel(obj, sub_group, demo_name, rewards_map, root_group, sparse_env_reward, success_preds_map)
        else:
            if key == "rewards":
                # Get episode length
                episode_len = len(root_group["actions"])
                
                # Get progress reward (relabeled rewards)
                if demo_name in rewards_map and rewards_map[demo_name] is not None:
                    progress_reward = rewards_map[demo_name]
                else:
                    # Fallback to zeros
                    progress_reward = np.zeros(episode_len, dtype=np.float32)
                
                # Compute env reward
                env_reward = compute_env_reward(episode_len, sparse_env_reward)
                
                # Combined rewards = env_reward + progress_reward
                combined_rewards = env_reward + progress_reward
                
                # Write all three reward fields
                dst_group.create_dataset("progress_reward", data=progress_reward, compression="gzip")
                dst_group.create_dataset("env_reward", data=env_reward, compression="gzip")
                dst_group.create_dataset("rewards", data=combined_rewards, compression="gzip")
                
                # Write success_preds if available
                if success_preds_map is not None and demo_name in success_preds_map and success_preds_map[demo_name] is not None:
                    dst_group.create_dataset("success_preds", data=success_preds_map[demo_name], compression="gzip")
            else:
                # Copy dataset with compression if it's large
                if obj.size > 1000:
                    dst_group.create_dataset(key, data=obj, compression="gzip")
                else:
                    dst_group.create_dataset(key, data=obj)


def extract_frames_from_episode(demo_group: h5py.Group, image_keys: List[str]) -> Optional[np.ndarray]:
    """
    Extract frames from an episode, trying different image keys.
    
    Returns:
        Array of frames in HWC uint8 format, or None if no images found
    """
    obs_group = demo_group.get("obs")
    if obs_group is None:
        return None
    
    # Try to find image data
    for key in image_keys:
        if key in obs_group:
            images = obs_group[key][:]
            # Ensure HWC format and uint8
            if images.ndim == 4:  # (T, H, W, C) or (T, C, H, W)
                if images.shape[1] == 3 and images.shape[3] != 3:
                    # Assume CHW format, convert to HWC
                    images = np.transpose(images, (0, 2, 3, 1))
                # Convert to uint8 if needed
                if images.dtype != np.uint8:
                    if images.max() <= 1.0:
                        images = (images * 255).clip(0, 255).astype(np.uint8)
                    else:
                        images = images.clip(0, 255).astype(np.uint8)
                return images
    return None


def get_language_instruction(demo_group: h5py.Group) -> str:
    """
    Extract language instruction from an episode.
    Tries language_instruction first, then obs/language embedding.
    """
    # Try language_instruction field
    if "language_instruction" in demo_group:
        instruction = demo_group["language_instruction"][()]
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")
        return instruction
    
    # Fallback: try to get from metadata or use default
    if "language" in demo_group.attrs:
        return str(demo_group.attrs["language"])
    
    # Use demo name as fallback
    return f"Episode {demo_group.name.split('/')[-1]}"


def relabel_episode_with_baseline(
    demo_group: h5py.Group,
    demo_name: str,
    image_keys: List[str],
    baseline_model,
    language_instruction: str,
    eval_server_url: Optional[str] = None,
    batch_size: int = 32,
    max_frames: int = 16,
    test_batched: bool = False,
    use_batched: Optional[bool] = None,
    use_frame_steps: bool = False,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Relabel rewards for a single episode using a baseline model (GVL, VLAC, RoboReward, RFMModel).
    
    Returns:
        Tuple of (rewards array, success_preds array or None). Returns (None, None) if relabeling failed.
    """
    # Extract frames
    frames = extract_frames_from_episode(demo_group, image_keys)

    if frames is None:
        logger.warning(f"Episode {demo_name}: No images found, skipping relabeling")
        return None, None
    
    logger.debug(f"Episode {demo_name}: Extracted {len(frames)} frames with shape {frames.shape}")
    
    try:
        logger.info(f"Episode {demo_name}: Starting reward relabeling with baseline model ({len(frames)} timesteps, use_frame_steps={use_frame_steps})")
        
        if eval_server_url is not None:
            # Use baseline eval server
            logger.debug(f"Episode {demo_name}: Using baseline eval server at {eval_server_url}")
            all_frames = [frames[i] for i in range(len(frames))]
            progress_predictions, _ = relabel_rewards_with_baseline_eval_server(
                all_frames=all_frames,
                language_instruction=language_instruction,
                episode_id=demo_name,
                eval_server_url=eval_server_url,
                batch_size=batch_size,
                image_key=image_keys[0] if image_keys else "image",
                max_frames=max_frames,
            )
            progress_array = np.array(progress_predictions, dtype=np.float32)
        elif not use_frame_steps:
            # Whole trajectory mode: feed entire trajectory once, get N-dim progress vector
            logger.debug(f"Episode {demo_name}: Using whole trajectory mode (use_frame_steps=False)")
            
            # Subsample frames if needed
            frames_to_use = frames
            if max_frames > 0 and HAS_SUBSAMPLE_HELPER and len(frames) > max_frames:
                frames_to_use, _ = linspace_subsample_frames(frames, num_frames=max_frames)
                logger.debug(f"Episode {demo_name}: Subsampled {len(frames)} frames to {len(frames_to_use)} frames")
            
            # Call compute_progress once on the whole trajectory
            progress_pred = baseline_model.compute_progress(frames_to_use, task_description=language_instruction)
            
            # Convert to numpy array
            if isinstance(progress_pred, list):
                progress_values = np.array([float(v) if v is not None else 0.0 for v in progress_pred], dtype=np.float32)
            elif isinstance(progress_pred, np.ndarray):
                progress_values = progress_pred.astype(np.float32)
            else:
                progress_values = np.zeros(len(frames_to_use), dtype=np.float32)
            
            # If we subsampled, interpolate back to original trajectory length
            if len(progress_values) != len(frames):
                # Linear interpolation to match original trajectory length
                original_indices = np.linspace(0, len(progress_values) - 1, len(frames))
                progress_array = np.interp(original_indices, np.arange(len(progress_values)), progress_values).astype(np.float32)
                logger.debug(f"Episode {demo_name}: Interpolated {len(progress_values)} progress values to {len(frames)} timesteps")
            else:
                progress_array = progress_values
        else:
            # Frame-step mode: Use direct baseline model with linspace-subsampled subsequences
            # Instead of building N subsequences (one per timestep), we:
            # 1. Pick max_frames indices using linspace
            # 2. Build max_frames subsequences: [0:idx1], [0:idx2], ..., [0:N]
            # 3. Call compute_progress_batched once
            # 4. Interpolate back to full trajectory length
            logger.debug(f"Episode {demo_name}: Using frame-step mode with linspace subsampling (use_frame_steps=True)")
            
            # num_frames = len(frames)
            # num_subsequences = 8
            
            # # Get linspace indices for subsequence endpoints
            # # E.g., for 100 frames and 8 subsequences: [0, 14, 28, 42, 57, 71, 85, 99]
            # subsequence_end_indices = np.linspace(0, num_frames - 1, num_subsequences, dtype=int)
            # logger.debug(f"Episode {demo_name}: Using {num_subsequences} subsequences at indices: {subsequence_end_indices.tolist()}")
            subsequence_end_indices = list(range(0, len(frames)))
            
            # Build subsequences
            batch_subsequences_list = []
            task_descriptions_list = []
            
            for end_idx in subsequence_end_indices:
                # Build subsequence from start to end_idx (inclusive)
                frames_subsequence = frames[: end_idx + 1]  # Shape: (end_idx+1, H, W, C)
                
                # Subsample frames within each subsequence if it exceeds max_frames
                if max_frames > 0 and HAS_SUBSAMPLE_HELPER and len(frames_subsequence) > max_frames:
                    frames_subsequence, _ = linspace_subsample_frames(frames_subsequence, num_frames=max_frames)
                
                batch_subsequences_list.append(frames_subsequence)
                task_descriptions_list.append(language_instruction)
            
            logger.info(f"Episode {demo_name}: Built {len(batch_subsequences_list)} subsequences for batched processing")
            
            # Call compute_progress_batched in chunks of batch_size
            chunk_size = batch_size
            sampled_progress = []
            start_time = time.time()
            for chunk_start in tqdm(range(0, len(batch_subsequences_list), chunk_size), desc=f"batching progress"):
                chunk_end = min(chunk_start + chunk_size, len(batch_subsequences_list))
                sub_list = batch_subsequences_list[chunk_start:chunk_end]
                task_list = task_descriptions_list[chunk_start:chunk_end]
                batch_progress_results = baseline_model.compute_progress_batched(sub_list, task_list)
                for result_list in batch_progress_results:
                    if isinstance(result_list, list) and len(result_list) > 0:
                        sampled_progress.append(float(result_list[-1]) if result_list[-1] is not None else 0.0)
                    elif isinstance(result_list, np.ndarray) and len(result_list) > 0:
                        sampled_progress.append(float(result_list[-1]) if result_list[-1] is not None else 0.0)
                    else:
                        sampled_progress.append(0.0)
            elapsed_time = time.time() - start_time
            logger.info(
                f"Episode {demo_name}: Batched calls completed in {elapsed_time:.3f}s "
                f"({elapsed_time/len(batch_subsequences_list)*1000:.2f}ms per subsequence, chunk_size={chunk_size})"
            )
            
            progress_array = np.array(sampled_progress, dtype=np.float32)
            logger.debug(f"Episode {demo_name}: Got {len(sampled_progress)} progress values")
        
        # Normalize to [0, 1] if needed (some models return [0, 100] or [1, 5])
        if progress_array.max() > 1.0:
            if progress_array.max() <= 5.0:
                # Likely discrete scores (e.g., RoboReward 1-5), normalize to [0, 1]
                progress_array = (progress_array - 1.0) / 4.0
            elif progress_array.max() <= 100.0:
                # Likely percentage [0, 100], normalize to [0, 1]
                progress_array = progress_array / 100.0
        
        rewards = progress_array
        logger.info(
            f"Episode {demo_name}: Relabeling complete - rewards range: [{rewards.min():.4f}, {rewards.max():.4f}], "
            f"mean: {rewards.mean():.4f}, std: {rewards.std():.4f}, shape: {rewards.shape}", 
        )
        # Baseline models don't provide success predictions
        return rewards, None
    except Exception as e:
        logger.error(f"Episode {demo_name}: Error relabeling rewards: {e}", exc_info=True)
        return None, None


def relabel_episode(
    demo_group: h5py.Group,
    demo_name: str,
    image_keys: List[str],
    reward_model,
    exp_config,
    batch_collator,
    tokenizer,
    sentence_encoder: SentenceTransformer,
    eval_server_url: Optional[str],
    batch_size: int,
    max_frames: int,
    reward_model_type: str = "rfm",
    baseline_model=None,
    baseline_eval_server_url: Optional[str] = None,
    language_instruction: Optional[str] = None,
    test_batched: bool = False,
    use_batched: Optional[bool] = None,
    use_frame_steps: bool = False,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Relabel rewards for a single episode.
    
    Returns:
        Tuple of (rewards array, success_preds array or None). Returns (None, None) if relabeling failed.
    """
    # Extract frames
    frames = extract_frames_from_episode(demo_group, image_keys)
    if frames is None:
        logger.warning(f"Episode {demo_name}: No images found, skipping relabeling")
        return None, None
    
    logger.debug(f"Episode {demo_name}: Extracted {len(frames)} frames with shape {frames.shape}")
    
    # Use provided language instruction if available, otherwise extract from episode
    if language_instruction is None:
        language_instruction = get_language_instruction(demo_group)
    logger.debug(f"Episode {demo_name}: Language instruction: {language_instruction[:100]}...")
    
    # Use baseline model if specified (returns tuple already)
    if reward_model_type in ["gvl", "vlac", "roboreward"]:
        if baseline_model is not None or baseline_eval_server_url is not None:
            return relabel_episode_with_baseline(
                demo_group=demo_group,
                demo_name=demo_name,
                image_keys=image_keys,
                baseline_model=baseline_model,
                language_instruction=language_instruction,
                eval_server_url=baseline_eval_server_url,
                batch_size=batch_size,
                max_frames=max_frames,
                test_batched=test_batched,
                use_batched=use_batched,
                use_frame_steps=use_frame_steps,
            )
    
    # Convert frames to list format expected by RFM relabeling functions
    all_frames = [frames[i] for i in range(len(frames))]
    
    try:
        logger.info(f"Episode {demo_name}: Starting reward relabeling ({len(all_frames)} timesteps)")
        if eval_server_url is not None:
            # Use eval server
            logger.debug(f"Episode {demo_name}: Using eval server at {eval_server_url}")
            progress_predictions, success_probs = relabel_rewards_with_eval_server(
                all_frames=all_frames,
                language_instruction=language_instruction,
                episode_id=demo_name,
                eval_server_url=eval_server_url,
                sentence_model=sentence_encoder,
                batch_size=batch_size,
                image_key=image_keys[0] if image_keys else "image",
                max_frames=max_frames,
            )
        else:
            # Use direct model
            logger.debug(f"Episode {demo_name}: Using direct model")
            progress_predictions, success_probs = relabel_rewards_with_model(
                all_frames=all_frames,
                language_instruction=language_instruction,
                episode_id=demo_name,
                reward_model=reward_model,
                exp_config=exp_config,
                batch_collator=batch_collator,
                tokenizer=tokenizer,
                sentence_model=sentence_encoder,
                batch_size=batch_size,
                image_key=image_keys[0] if image_keys else "image",
            )
        
        # Use progress predictions as rewards
        rewards = np.array(progress_predictions, dtype=np.float32)
        
        # Convert success_probs to numpy array if available
        success_preds = None
        if success_probs is not None:
            success_preds = np.array(success_probs, dtype=np.float32)
            logger.info(
                f"Episode {demo_name}: Relabeling complete - rewards range: [{rewards.min():.4f}, {rewards.max():.4f}], "
                f"mean: {rewards.mean():.4f}, success_preds range: [{success_preds.min():.4f}, {success_preds.max():.4f}]"
            )
        else:
            logger.info(
                f"Episode {demo_name}: Relabeling complete - rewards range: [{rewards.min():.4f}, {rewards.max():.4f}], "
                f"mean: {rewards.mean():.4f}, std: {rewards.std():.4f}"
            )
        return rewards, success_preds
    except Exception as e:
        logger.error(f"Episode {demo_name}: Error relabeling rewards: {e}", exc_info=True)
        return None, None


def find_image_keys(obs_group: h5py.Group) -> List[str]:
    """
    Find image observation keys in the obs group.
    """
    image_keys = []
    for key in obs_group.keys():
        data = obs_group[key]
        if isinstance(data, h5py.Dataset):
            # Check if it looks like image data (3D or 4D with last dim likely being channels)
            shape = data.shape
            if len(shape) >= 3:
                # Could be (H, W, C) or (T, H, W, C) or (T, C, H, W)
                if any(kw in key.lower() for kw in ["image", "rgb", "camera", "cam", "pixel"]):
                    image_keys.append(key)
    return image_keys


def main():
    parser = argparse.ArgumentParser(description="Relabel rewards in offline H5 datasets")
    parser.add_argument("--h5_paths", type=str, nargs="+", required=True, help="One or more input H5 paths")
    parser.add_argument(
        "--reward-model",
        type=str,
        default="rfm",
        choices=["rfm", "rewind", "gvl", "vlac", "roboreward"],
        help="Type of reward model to use: 'rfm' or 'rewind' for RFM models, 'gvl', 'vlac', or 'roboreward' for baseline models (default: rfm)",
    )
    parser.add_argument(
        "--reward-model-path",
        type=str,
        default=None,
        help="HuggingFace model ID or local checkpoint path. Required for RFM/ReWiND/VLAC/RoboReward models.",
    )
    parser.add_argument(
        "--eval-server-url",
        type=str,
        default=None,
        help="URL of eval server for reward relabeling (e.g., http://localhost:8001). Works with RFM/ReWiND models.",
    )
    parser.add_argument(
        "--baseline-eval-server-url",
        type=str,
        default=None,
        help="URL of baseline eval server for reward relabeling (e.g., http://localhost:8001). Works with GVL/VLAC/RoboReward models.",
    )
    parser.add_argument(
        "--sentence-encoder",
        type=str,
        default=None,
        help="SentenceTransformer model name for encoding language instructions. Required for RFM/ReWiND models, optional for baseline models.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for reward inference")
    parser.add_argument("--output_path", type=str, default=None, help="Output file path (single) or output directory")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=32,
        help="Maximum number of frames per sample. For RoboReward, frames exceeding this will be subsampled using linspace (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for reward model (default: cuda if available, else cpu)",
    )
    parser.add_argument("--info", action="store_true", help="Print info about output datasets after relabeling")
    parser.add_argument(
        "--image-key",
        type=str,
        default=None,
        help="Specific observation key to use for relabeling (e.g., 'image', 'image_top'). If not specified, auto-detects from available image keys.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--language-instruction",
        type=str,
        default=None,
        help="Language instruction to use for all episodes. If provided, will update the H5 file with this instruction and use it for relabeling instead of extracting from the dataset.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: only process the first 5 episodes for testing.",
    )
    parser.add_argument(
        "--test-batched",
        action="store_true",
        help="Test mode: compare compute_progress vs compute_progress_batched predictions (only for baseline models).",
    )
    parser.add_argument(
        "--use-batched",
        action="store_true",
        default=None,
        help="Force use of compute_progress_batched if available (only for baseline models). By default, batched mode is used automatically when available.",
    )
    parser.add_argument(
        "--no-batched",
        action="store_true",
        help="Disable batched mode and use individual compute_progress calls (only for baseline models).",
    )
    parser.add_argument(
        "--use-frame-steps",
        action="store_true",
        default=False,
        help="If set, compute rewards using linspace-subsampled subsequences: picks max_frames indices via linspace, "
             "builds subsequences [0:idx1], [0:idx2], ..., [0:N], calls compute_progress_batched once, then interpolates "
             "back to full trajectory length. Much faster than per-timestep processing. "
             "If not set (default), compute rewards for the whole trajectory at once.",
    )
    parser.add_argument(
        "--rewards-only",
        action="store_true",
        default=False,
        help="If set, create a minimal H5 file containing only rewards and language instructions (no observations, actions, etc.). "
             "Useful for saving disk space when you only need the relabeled rewards.",
    )
    parser.add_argument(
        "--sparse-env-reward",
        action="store_true",
        default=False,
        help="If set, env_reward will be 0 for the last frame and -1 for all other timesteps. "
             "If not set, env_reward will be all zeros. The final 'rewards' field = env_reward + progress_reward.",
    )
    args = parser.parse_args()
    
    # Handle conflicting flags
    if args.use_batched and args.no_batched:
        parser.error("Cannot specify both --use-batched and --no-batched")
    if args.test_batched and (args.use_batched or args.no_batched):
        parser.error("--test-batched cannot be used with --use-batched or --no-batched")

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Validate arguments
    if args.reward_model in ["rfm", "rewind"]:
        if args.reward_model_path is None and args.eval_server_url is None:
            parser.error(f"For {args.reward_model} models, either --reward-model-path or --eval-server-url must be provided")
        if args.reward_model_path is not None and args.eval_server_url is not None:
            parser.error("Cannot provide both --reward-model-path and --eval-server-url")
        if args.sentence_encoder is None:
            parser.error(f"For {args.reward_model} models, --sentence-encoder is required")
        if args.baseline_eval_server_url is not None:
            parser.error("--baseline-eval-server-url is only for baseline models (gvl, vlac, roboreward)")
    elif args.reward_model in ["gvl", "vlac", "roboreward"]:
        if args.reward_model in ["vlac", "roboreward"] and args.reward_model_path is None and args.baseline_eval_server_url is None:
            parser.error(f"For {args.reward_model} model, either --reward-model-path or --baseline-eval-server-url must be provided")
        if args.eval_server_url is not None:
            parser.error(f"--eval-server-url is only for RFM/ReWiND models. Use --baseline-eval-server-url for {args.reward_model}")
        if args.reward_model_path is not None and args.baseline_eval_server_url is not None:
            parser.error("Cannot provide both --reward-model-path and --baseline-eval-server-url")
        if not HAS_BASELINE_MODELS and args.baseline_eval_server_url is None:
            parser.error(f"Baseline models not available. Install reward_fm package to use {args.reward_model}, or use --baseline-eval-server-url")
    else:
        parser.error(f"Unknown reward model type: {args.reward_model}")

    # Log startup information
    logger.info("="*60)
    logger.info("Reward Relabeling Script")
    logger.info("="*60)
    logger.info(f"Input files: {len(args.h5_paths)} file(s)")
    for i, path in enumerate(args.h5_paths, 1):
        logger.info(f"  {i}. {path}")
    logger.info(f"Reward model type: {args.reward_model}")
    if args.reward_model_path:
        logger.info(f"Reward model path: {args.reward_model_path}")
    if args.eval_server_url:
        logger.info(f"Eval server: {args.eval_server_url}")
    if args.baseline_eval_server_url:
        logger.info(f"Baseline eval server: {args.baseline_eval_server_url}")
    if args.sentence_encoder:
        logger.info(f"Sentence encoder: {args.sentence_encoder}")
    if args.language_instruction:
        logger.info(f"Language instruction: {args.language_instruction}")
    if args.debug:
        logger.info("DEBUG MODE: Only processing first 5 episodes")
    if args.test_batched:
        logger.info("TEST MODE: Will compare compute_progress vs compute_progress_batched")
    if args.use_batched:
        logger.info("BATCHED MODE: Will use compute_progress_batched if available")
    if args.no_batched:
        logger.info("INDIVIDUAL MODE: Will use individual compute_progress calls")
    if args.use_frame_steps:
        logger.info(f"FRAME-STEP MODE: Computing rewards using {args.max_frames} linspace-subsampled subsequences, then interpolating")
    else:
        logger.info("WHOLE-TRAJECTORY MODE: Computing rewards for entire trajectory at once (faster)")
    if args.rewards_only:
        logger.info("REWARDS-ONLY MODE: Output will contain only rewards and language instructions")
    if args.sparse_env_reward:
        logger.info("SPARSE ENV REWARD: env_reward = -1 for all timesteps except last (0)")
    else:
        logger.info("ZERO ENV REWARD: env_reward = 0 for all timesteps")
    logger.info("Output fields: progress_reward (relabeled), env_reward, rewards = env_reward + progress_reward")
    logger.info(f"Batch size: {args.batch_size}")
    if args.image_key:
        logger.info(f"Image key: {args.image_key}")
    logger.info(f"Max frames: {args.max_frames}")
    logger.info("="*60)

    # Normalize input paths
    h5_paths: List[str] = [str(Path(p)) for p in args.h5_paths]
    for p in h5_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"H5 path not found: {p}")

    # Prepare output destination (create dir if needed in multi-input case)
    if args.output_path is not None:
        op = Path(args.output_path)
        # Create directory if multiple inputs or output ends with os.sep
        if len(h5_paths) > 1 or (str(op).endswith(os.sep) and not op.exists()):
            op.mkdir(parents=True, exist_ok=True)

    # Load reward model or baseline model
    reward_model = None
    exp_config = None
    batch_collator = None
    tokenizer = None
    baseline_model = None
    sentence_encoder = None
    
    if args.reward_model in ["rfm", "rewind"]:
        # Load RFM/ReWiND model
        logger.info("Loading RFM/ReWiND reward model or connecting to eval server...")
        reward_model, exp_config, batch_collator, tokenizer = load_reward_model_for_relabeling(
            reward_model_path=args.reward_model_path,
            eval_server_url=args.eval_server_url,
            device=args.device,
        )
        logger.info("Reward model/eval server ready")
        
        # Load sentence encoder
        logger.info(f"Loading sentence encoder: {args.sentence_encoder}")
        sentence_encoder = SentenceTransformer(args.sentence_encoder)
        logger.info("Sentence encoder loaded")
    elif args.reward_model in ["gvl", "vlac", "roboreward"]:
        # Load baseline model or connect to baseline eval server
        logger.info(f"Loading {args.reward_model} baseline model or connecting to eval server...")
        baseline_model, _, _, _ = load_baseline_model_for_relabeling(
            reward_model_type=args.reward_model,
            reward_model_path=args.reward_model_path,
            baseline_eval_server_url=args.baseline_eval_server_url,
            device=args.device,
            max_frames=args.max_frames,
        )
        logger.info(f"{args.reward_model} baseline model/eval server ready")
        
        # Load sentence encoder if provided (optional for baseline models)
        if args.sentence_encoder:
            logger.info(f"Loading sentence encoder: {args.sentence_encoder}")
            sentence_encoder = SentenceTransformer(args.sentence_encoder)
            logger.info("Sentence encoder loaded")

    # Process each H5 file
    total_stats = {"total": 0, "success": 0, "failed": 0}
    for file_idx, input_h5 in enumerate(h5_paths, 1):
        model_for_path = args.reward_model_path or args.reward_model
        output_h5 = build_output_path_for_input(
            input_h5, args.output_path,
            language_instruction=args.language_instruction,
            image_key=args.image_key,
            model=model_for_path,
        )
        Path(output_h5).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing file {file_idx}/{len(h5_paths)}: {input_h5}")
        logger.info(f"Output: {output_h5}")
        logger.info(f"{'='*60}")

        rewards_map: Dict[str, np.ndarray] = {}
        success_preds_map: Dict[str, np.ndarray] = {}
        stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}

        with open_h5_with_retry(input_h5, "r") as infile:
            if "data" not in infile:
                logger.warning(f"No 'data' group found in {input_h5}, skipping")
                continue

            in_data = infile["data"]
            demo_names = list(in_data.keys())
            
            # Apply debug mode limit if enabled
            if args.debug:
                demo_names = demo_names[:5]
                logger.info(f"DEBUG MODE: Limiting to first 5 episodes (out of {len(in_data.keys())} total)")
            
            stats["total"] = len(demo_names)
            logger.info(f"Found {len(demo_names)} episodes in dataset")

            # Find image keys from first episode
            image_keys = []
            if len(demo_names) > 0:
                first_demo = in_data[demo_names[0]]
                obs_group = first_demo.get("obs")
                if obs_group is not None:
                    image_keys = find_image_keys(obs_group)
                    logger.info(f"Image keys found: {image_keys}")
                else:
                    logger.warning("No 'obs' group found in first episode")
            
            # Use specified image key if provided, otherwise use auto-detected keys
            if args.image_key is not None:
                if args.image_key not in image_keys:
                    logger.warning(
                        f"Specified image key '{args.image_key}' not found in auto-detected keys: {image_keys}"
                    )
                    if obs_group is not None:
                        logger.info(f"Available keys in first episode: {list(obs_group.keys())}")
                    # Try to use it anyway in case it exists in other episodes
                    image_keys = [args.image_key]
                    logger.info(f"Will attempt to use specified key: {args.image_key}")
                else:
                    image_keys = [args.image_key]
                    logger.info(f"Using specified image key: {args.image_key}")
            elif len(image_keys) == 0:
                logger.warning("No image keys found via auto-detection")
                if obs_group is not None:
                    logger.info(f"Available keys in first episode: {list(obs_group.keys())}")

            # Process each episode
            logger.info(f"Starting reward relabeling for {len(demo_names)} episodes...")
            for demo_name in tqdm(demo_names, desc="  Relabeling episodes"):
                demo_group = in_data[demo_name]
                
                # Use provided language instruction if available, otherwise extract from episode
                language_instruction = args.language_instruction if args.language_instruction else get_language_instruction(demo_group)
                
                # Determine use_batched setting
                use_batched = None
                if args.use_batched:
                    use_batched = True
                elif args.no_batched:
                    use_batched = False
                # Otherwise, None means auto-detect (default behavior)
                
                rewards, success_preds = relabel_episode(
                    demo_group=demo_group,
                    demo_name=demo_name,
                    image_keys=image_keys,
                    reward_model=reward_model,
                    exp_config=exp_config,
                    batch_collator=batch_collator,
                    tokenizer=tokenizer,
                    sentence_encoder=sentence_encoder,
                    eval_server_url=args.eval_server_url,
                    batch_size=args.batch_size,
                    max_frames=args.max_frames,
                    reward_model_type=args.reward_model,
                    baseline_model=baseline_model,
                    baseline_eval_server_url=args.baseline_eval_server_url,
                    language_instruction=language_instruction,
                    test_batched=args.test_batched,
                    use_batched=use_batched,
                    use_frame_steps=args.use_frame_steps,
                )
                if rewards is not None:
                    rewards_map[demo_name] = rewards
                    if success_preds is not None:
                        success_preds_map[demo_name] = success_preds
                    stats["success"] += 1
                else:
                    stats["failed"] += 1

            # Update total statistics
            total_stats["total"] += stats["total"]
            total_stats["success"] += stats["success"]
            total_stats["failed"] += stats["failed"]
            
            # Log summary statistics for this file
            logger.info(f"\nRelabeling summary for {input_h5}:")
            logger.info(f"  Total episodes: {stats['total']}")
            logger.info(f"  Successfully relabeled: {stats['success']}")
            logger.info(f"  Failed/Skipped: {stats['failed']}")
            if success_preds_map:
                logger.info(f"  Episodes with success_preds: {len(success_preds_map)}")
            if stats["total"] > 0:
                success_rate = 100.0 * stats["success"] / stats["total"]
                logger.info(f"  Success rate: {success_rate:.1f}%")

            # Write relabeled H5 file (while input file is still open to avoid GPFS locking issues)
            logger.info(f"Writing relabeled file: {output_h5}")
            logger.debug(f"Rewards computed for {len(rewards_map)}/{len(demo_names)} episodes")
            
            with open_h5_with_retry(output_h5, "w") as outfile:
                # Copy file-level attributes
                logger.debug("Copying file-level attributes...")
                for attr_name, attr_val in infile.attrs.items():
                    outfile.attrs[attr_name] = attr_val

                # Update metadata
                outfile.attrs["rewards_relabeled"] = True
                outfile.attrs["reward_model_type"] = args.reward_model
                outfile.attrs["sparse_env_reward"] = args.sparse_env_reward
                if args.rewards_only:
                    outfile.attrs["rewards_only"] = True
                if args.reward_model_path is not None:
                    outfile.attrs["reward_model_path"] = args.reward_model_path
                    logger.debug(f"Stored reward_model_path: {args.reward_model_path}")
                if args.eval_server_url is not None:
                    outfile.attrs["eval_server_url"] = args.eval_server_url
                    logger.debug(f"Stored eval_server_url: {args.eval_server_url}")
                if args.baseline_eval_server_url is not None:
                    outfile.attrs["baseline_eval_server_url"] = args.baseline_eval_server_url
                    logger.debug(f"Stored baseline_eval_server_url: {args.baseline_eval_server_url}")

                # Create /data group
                out_data = outfile.create_group("data")

                if args.rewards_only:
                    # Minimal mode: only write rewards and language instructions
                    logger.debug(f"Writing {len(demo_names)} episodes (rewards-only mode)...")
                    for demo_name in tqdm(demo_names, desc="  Writing episodes", leave=False):
                        dg_in = in_data[demo_name]
                        dg_out = out_data.create_group(demo_name)
                        
                        # Get episode length
                        episode_len = len(dg_in["actions"]) if "actions" in dg_in else len(rewards_map.get(demo_name, []))
                        
                        # Get progress reward (relabeled rewards)
                        if demo_name in rewards_map and rewards_map[demo_name] is not None:
                            progress_reward = rewards_map[demo_name]
                        else:
                            progress_reward = np.zeros(episode_len, dtype=np.float32)
                        
                        # # Compute env reward
                        # env_reward = compute_env_reward(episode_len, args.sparse_env_reward)
                        
                        # # Combined rewards = env_reward + progress_reward
                        # combined_rewards = env_reward + progress_reward
                        
                        # # Write all three reward fields
                        # dg_out.create_dataset("progress_reward", data=progress_reward, compression="gzip")
                        # dg_out.create_dataset("env_reward", data=env_reward, compression="gzip")
                        dg_out.create_dataset("rewards", data=progress_reward, compression="gzip")
                        
                        # Write success_preds if available
                        if demo_name in success_preds_map and success_preds_map[demo_name] is not None:
                            dg_out.create_dataset("success_preds", data=success_preds_map[demo_name], compression="gzip")
                        
                        # Write language instruction
                        if args.language_instruction is not None:
                            if isinstance(args.language_instruction, str):
                                dg_out.create_dataset("language_instruction", data=args.language_instruction.encode("utf-8"))
                            else:
                                dg_out.create_dataset("language_instruction", data=args.language_instruction)
                        elif "language_instruction" in dg_in:
                            # Copy existing language instruction
                            dg_out.create_dataset("language_instruction", data=dg_in["language_instruction"][()])
                        
                        # Copy essential attributes (episode index, etc.)
                        for attr_name, attr_val in dg_in.attrs.items():
                            dg_out.attrs[attr_name] = attr_val
                        
                        # Store episode length for reference
                        if "actions" in dg_in:
                            dg_out.attrs["episode_length"] = len(dg_in["actions"])
                    
                    logger.debug("Finished writing all episodes (rewards-only)")
                else:
                    # Full mode: copy everything and replace rewards
                    logger.debug(f"Copying {len(demo_names)} episodes to output file...")
                    for demo_name in tqdm(demo_names, desc="  Writing episodes", leave=False):
                        dg_in = in_data[demo_name]
                        dg_out = out_data.create_group(demo_name)
                        copy_group_with_relabel(dg_in, dg_out, demo_name, rewards_map, dg_in, args.sparse_env_reward, success_preds_map)
                        # Copy demo attributes
                        for attr_name, attr_val in dg_in.attrs.items():
                            dg_out.attrs[attr_name] = attr_val

                        # Update language instruction if provided
                        if args.language_instruction is not None:
                            # Delete existing language_instruction if it exists
                            if "language_instruction" in dg_out:
                                del dg_out["language_instruction"]
                            # Store as bytes if it's a string
                            if isinstance(args.language_instruction, str):
                                dg_out.create_dataset("language_instruction", data=args.language_instruction.encode("utf-8"))
                            else:
                                dg_out.create_dataset("language_instruction", data=args.language_instruction)
                            logger.debug(f"Updated language_instruction for {demo_name}")
                    logger.debug("Finished writing all episodes")

        logger.info(f"✅ Successfully saved relabeled dataset: {output_h5}")
        if args.info:
            print_dataset_info(output_h5)
    
    # Final summary across all files
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total files processed: {len(h5_paths)}")
    logger.info(f"Total episodes: {total_stats['total']}")
    logger.info(f"Successfully relabeled: {total_stats['success']}")
    logger.info(f"Failed/Skipped: {total_stats['failed']}")
    if total_stats["total"] > 0:
        overall_success_rate = 100.0 * total_stats["success"] / total_stats["total"]
        logger.info(f"Overall success rate: {overall_success_rate:.1f}%")
    logger.info(f"{'='*60}")
    logger.info("All files processed successfully!")


if __name__ == "__main__":
    main()
