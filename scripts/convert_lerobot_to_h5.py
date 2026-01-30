#!/usr/bin/env python3
"""
Convert LeRobot dataset(s) to HDF5 format compatible with H5ReplayBuffer.

Pipeline:
    1. Load LeRobot dataset
    2. Extract all frames using DataLoader (parallel, fast)
    3. Split frames into episodes
    4. Process episodes (normalize, resize images)
    5. Write to HDF5

Usage:
    uv run python scripts/convert_lerobot_to_h5.py \
        --repo-ids HenryZhang/rfm_5_demos1768605063.996896 \
        --output so101_dataset.h5 \
        --image-height 480 \
        --image-width 640 \
        --normalize \
        --info
"""

import argparse
import h5py
import numpy as np
import torch
from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from sentence_transformers import SentenceTransformer

from convert_lerobot_common import (
    normalize_obs_key,
    compute_normalization_stats,
    print_dataset_info,
    compute_text_embeddings,
)

# Optional cv2 for image resizing
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def center_crop(image, size):
    """Center crop an image or a batch of images to the specified size.

    Supports both 3D (H, W, C) and 4D (N, H, W, C) arrays.
    """
    if image is None or size is None:
        return image

    if image.ndim == 3:
        h, w = image.shape[:2]
        crop = min(size, h, w)
        x = (w - crop) // 2
        y = (h - crop) // 2
        return image[y : y + crop, x : x + crop, :]
    elif image.ndim == 4:
        h, w = image.shape[1:3]
        crop = min(size, h, w)
        x = (w - crop) // 2
        y = (h - crop) // 2
        return image[:, y : y + crop, x : x + crop, :]
    else:
        return image
        
# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Frame:
    """Single frame of data extracted from dataset."""
    index: int
    action: np.ndarray | None = None
    observations: dict[str, np.ndarray] = field(default_factory=dict)
    reward: float = 0.0
    done: bool = False
    valid: bool = True


@dataclass
class Episode:
    """Processed episode ready for HDF5 writing."""
    demo_id: str
    episode_index: int
    language_instruction: str
    actions: np.ndarray
    observations: dict[str, np.ndarray]
    next_observations: dict[str, np.ndarray]
    rewards: np.ndarray | None
    dones: np.ndarray | None
    num_samples: int


@dataclass
class ConversionConfig:
    """Configuration for dataset conversion."""
    include_images: bool = True
    include_rewards: bool = True
    include_dones: bool = True
    image_size: int | None = None
    image_height: int | None = None
    image_width: int | None = None
    normalize: bool = False
    action_norm_mode: str = "minmax"
    extraction_batch_size: int = 32
    extraction_num_workers: int = 8


# =============================================================================
# Episode Extraction (DataLoader per episode for memory efficiency)
# =============================================================================

class EpisodeFrameDataset(Dataset):
    """Dataset for extracting frames from a single episode range."""
    
    def __init__(
        self,
        dataset: LeRobotDataset,
        frame_indices: list[int],
        obs_keys: set[str],
        action_key: str,
        include_rewards: bool = True,
        include_dones: bool = True,
    ):
        self.dataset = dataset
        self.frame_indices = frame_indices
        self.obs_keys = obs_keys
        self.action_key = action_key
        self.include_rewards = include_rewards
        self.include_dones = include_dones
    
    def __len__(self) -> int:
        return len(self.frame_indices)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Extract single frame by local index."""
        frame_idx = self.frame_indices[idx]
        try:
            item = self.dataset[frame_idx]
        except Exception as e:
            return {"local_idx": idx, "frame_idx": frame_idx, "valid": False, "error": str(e)}
        
        result = {"local_idx": idx, "frame_idx": frame_idx, "valid": True}
        
        # Action
        if self.action_key in item:
            action = item[self.action_key]
            result["action"] = action.cpu().numpy() if torch.is_tensor(action) else action
        
        # Observations
        result["obs"] = {}
        for key in self.obs_keys:
            if key in item:
                val = item[key]
                result["obs"][key] = val.cpu().numpy() if torch.is_tensor(val) else val
        
        # Reward
        if self.include_rewards and "reward" in item:
            reward = item["reward"]
            result["reward"] = float(reward.cpu().numpy() if torch.is_tensor(reward) else reward)
        
        # Done
        if self.include_dones and "done" in item:
            done = item["done"]
            result["done"] = bool(done.cpu().numpy() if torch.is_tensor(done) else done)
        
        return result


def extract_episode_frames(
    dataset: LeRobotDataset,
    from_idx: int,
    to_idx: int,
    obs_keys: set[str],
    action_key: str,
    config: ConversionConfig,
) -> list[Frame] | None:
    """
    Extract frames for a single episode using DataLoader.
    
    Returns:
        List of Frame objects in order, or None if extraction fails.
    """
    frame_indices = list(range(from_idx, to_idx))
    if len(frame_indices) == 0:
        return None
    
    ep_dataset = EpisodeFrameDataset(
        dataset=dataset,
        frame_indices=frame_indices,
        obs_keys=obs_keys,
        action_key=action_key,
        include_rewards=config.include_rewards,
        include_dones=config.include_dones,
    )
    
    # Use DataLoader for parallel extraction within this episode
    loader = DataLoader(
        ep_dataset,
        batch_size=config.extraction_batch_size,
        shuffle=False,
        num_workers=config.extraction_num_workers,
        pin_memory=False,
        collate_fn=lambda batch: batch,
    )
    
    # Collect frames (may arrive out of order due to parallel loading)
    frames_dict: dict[int, Frame] = {}
    
    for batch in loader:
        for item in batch:
            local_idx = item["local_idx"]
            if not item.get("valid", False):
                frames_dict[local_idx] = Frame(index=item["frame_idx"], valid=False)
                continue
            
            frames_dict[local_idx] = Frame(
                index=item["frame_idx"],
                action=item.get("action"),
                observations=item.get("obs", {}),
                reward=item.get("reward", 0.0),
                done=item.get("done", False),
                valid=True,
            )
    
    # Return frames in order
    return [frames_dict[i] for i in range(len(frame_indices)) if i in frames_dict]


def get_episode_metadata(dataset: LeRobotDataset) -> tuple[list[dict], dict[int, str]]:
    """Extract episode boundaries and task mapping from dataset metadata."""
    episodes = []
    task_map = {}
    
    # Build task index to task name mapping
    if hasattr(dataset.meta, "tasks") and dataset.meta.tasks is not None:
        tasks = dataset.meta.tasks
        if hasattr(tasks, 'iterrows'):  # pandas DataFrame
            for task_name, row in tasks.iterrows():
                task_idx = row.get("task_index", task_name)
                task_map[task_idx] = task_name
        else:  # HuggingFace Dataset
            for i in range(len(tasks)):
                task_row = tasks[i]
                task_idx = task_row.get("task_index", i)
                task_name = task_row.get("task", f"task_{i}")
                task_map[task_idx] = task_name
    
    # Extract episode boundaries
    if hasattr(dataset.meta, "episodes") and dataset.meta.episodes is not None:
        for i in range(len(dataset.meta.episodes)):
            row = dataset.meta.episodes[i]
            episodes.append({
                "index": i,
                "from_idx": int(row["dataset_from_index"]),
                "to_idx": int(row["dataset_to_index"]),
                "tasks": row.get("tasks"),
            })
    
    return episodes, task_map


def frames_to_episode_data(
    frames: list[Frame],
    ep_idx: int,
    language_instruction: str,
    obs_keys: set[str],
    config: ConversionConfig,
) -> dict | None:
    """Convert list of frames to episode data dict."""
    actions = []
    obs_dict = {key: [] for key in obs_keys}
    rewards = []
    dones = []
    
    for i, frame in enumerate(frames):
        if not frame.valid:
            continue
        
        if frame.action is not None:
            actions.append(frame.action)
        
        for key in obs_keys:
            if key in frame.observations:
                obs_dict[key].append(frame.observations[key])
        
        if config.include_rewards:
            rewards.append(frame.reward)
        
        if config.include_dones:
            is_last = (i == len(frames) - 1)
            dones.append(frame.done or is_last)
    
    if len(actions) == 0:
        return None
    
    # Ensure last done is True
    if config.include_dones and len(dones) > 0:
        dones[-1] = True
    
    return {
        "ep_idx": ep_idx,
        "language_instruction": language_instruction,
        "actions": actions,
        "obs": obs_dict,
        "rewards": rewards,
        "dones": dones,
    }


# =============================================================================
# Episode Processing (Step 3: Normalize, resize, format)
# =============================================================================

def resize_images(images: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize batch of images to target dimensions."""
    if not HAS_CV2:
        raise ImportError("cv2 required for image resizing. Install: pip install opencv-python")
    
    if images.ndim == 3:
        return cv2.resize(images, (width, height))
    elif images.ndim == 4:
        resized = np.zeros((images.shape[0], height, width, images.shape[3]), dtype=images.dtype)
        for i in range(images.shape[0]):
            resized[i] = cv2.resize(images[i], (width, height))
        return resized
    else:
        raise ValueError(f"Expected 3D or 4D array, got {images.shape}")


def process_episode(
    episode_data: dict,
    repo_id: str,
    image_keys: list[str],
    config: ConversionConfig,
    norm_stats: dict | None,
) -> Episode | None:
    """
    Process raw episode data: normalize actions/states, resize images, create next_obs.
    """
    actions_list = episode_data["actions"]
    obs_dict = episode_data["obs"]
    
    if len(actions_list) == 0:
        return None
    
    # Stack and normalize actions
    actions = np.stack(actions_list).astype(np.float32)
    
    if norm_stats:
        if norm_stats.get("action_norm_mode") == "minmax" and "action_min" in norm_stats:
            actions = 2 * (actions - norm_stats["action_min"]) / norm_stats["action_range"] - 1
            actions = np.clip(actions, -0.999, 0.999)
        elif "action_mean" in norm_stats:
            actions = (actions - norm_stats["action_mean"]) / norm_stats["action_std"]
    
    # Process observations
    processed_obs = {}
    processed_next_obs = {}
    
    for key, obs_list in obs_dict.items():
        if len(obs_list) == 0:
            continue
        
        normalized_key = normalize_obs_key(key)
        is_image = key in image_keys
        
        if is_image and not config.include_images:
            continue
        
        try:
            # Stack observations
            stacked = np.stack(obs_list) if isinstance(obs_list[0], np.ndarray) else np.array(obs_list)
            
            if is_image:
                # Convert CHW -> HWC if needed
                if stacked.ndim == 4 and stacked.shape[1] == 3:
                    stacked = np.transpose(stacked, (0, 2, 3, 1))
                
                # Convert float [0,1] -> uint8 [0,255]
                if stacked.dtype in (np.float32, np.float64):
                    if stacked.max() <= 1.0:
                        stacked = (stacked * 255).clip(0, 255).astype(np.uint8)
                    else:
                        stacked = stacked.clip(0, 255).astype(np.uint8)
                
                # Resize images
                if config.image_height and config.image_width:
                    stacked = resize_images(stacked, config.image_height, config.image_width)
                elif config.image_size:
                    stacked = center_crop(stacked, config.image_size)
            else:
                # Low-dimensional data
                stacked = stacked.astype(np.float32)
                
                # Normalize state if stats available
                if normalized_key == "state" and norm_stats and "state_mean" in norm_stats:
                    stacked = (stacked - norm_stats["state_mean"]) / norm_stats["state_std"]
            
            processed_obs[normalized_key] = stacked
            
            # Create next_obs (shift by 1, repeat last)
            if len(stacked) > 1:
                processed_next_obs[normalized_key] = np.concatenate([stacked[1:], stacked[-1:]], axis=0)
        
        except Exception:
            continue
    
    # Rewards and dones
    rewards = None
    if config.include_rewards and episode_data["rewards"]:
        rewards = np.array(episode_data["rewards"], dtype=np.float32)
    
    dones = None
    if config.include_dones and episode_data["dones"]:
        dones = np.array(episode_data["dones"], dtype=bool)
        if len(dones) > 0:
            dones[-1] = True
    
    return Episode(
        demo_id=f"{repo_id.replace('/', '_')}_ep{episode_data['ep_idx']:06d}",
        episode_index=episode_data["ep_idx"],
        language_instruction=episode_data["language_instruction"],
        actions=actions,
        observations=processed_obs,
        next_observations=processed_next_obs,
        rewards=rewards,
        dones=dones,
        num_samples=len(actions_list),
    )


# =============================================================================
# HDF5 Writing (Step 4: Write to file)
# =============================================================================

def write_episode_to_h5(
    episode: Episode,
    data_group: h5py.Group,
    sentence_encoder: SentenceTransformer | None,
):
    """Write single episode to HDF5 group."""
    demo_group = data_group.create_group(episode.demo_id)
    
    # Language instruction
    demo_group.create_dataset(
        "language_instruction",
        data=episode.language_instruction,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    
    # Observations
    obs_group = demo_group.create_group("obs")
    
    # Language embedding
    if sentence_encoder is not None:
        embedding = compute_text_embeddings(
            episode.language_instruction,
            sentence_encoder,
            use_autocast=True,
            show_progress_bar=False,
        )
        if torch.is_tensor(embedding):
            embedding = embedding.cpu().numpy()
        obs_group.create_dataset("language", data=embedding, compression="gzip")
    
    # Write observations
    for key, data in episode.observations.items():
        obs_group.create_dataset(key, data=data, compression="gzip")
    
    # Next observations
    next_obs_group = demo_group.create_group("next_obs")
    for key, data in episode.next_observations.items():
        next_obs_group.create_dataset(key, data=data, compression="gzip")
    
    # Actions
    demo_group.create_dataset("actions", data=episode.actions, compression="gzip")
    
    # Rewards
    if episode.rewards is not None:
        demo_group.create_dataset("rewards", data=episode.rewards, compression="gzip")
    
    # Dones
    if episode.dones is not None:
        demo_group.create_dataset("dones", data=episode.dones, compression="gzip")
    
    # Metadata
    demo_group.attrs["episode_index"] = episode.episode_index
    demo_group.attrs["num_samples"] = episode.num_samples
    demo_group.attrs["language_instruction"] = episode.language_instruction


# =============================================================================
# Main Conversion Pipeline
# =============================================================================

def get_dataset_keys(dataset: LeRobotDataset) -> tuple[set[str], str, list[str], list[str]]:
    """Extract observation keys, action key, and categorize into image/low-dim."""
    obs_keys = set()
    action_key = "actions"
    camera_keys = dataset.meta.camera_keys if hasattr(dataset.meta, "camera_keys") else []
    
    if hasattr(dataset, "features"):
        for key in dataset.features.keys():
            if key in ["episode_index", "task_index", "timestamp", "index"]:
                continue
            if key in ("actions", "action"):
                action_key = key
            else:
                obs_keys.add(key)
    
    # Categorize keys
    image_keys = []
    low_dim_keys = []
    for key in sorted(obs_keys):
        if key in camera_keys or any(kw in key.lower() for kw in ["image", "rgb", "camera", "cam"]):
            image_keys.append(key)
        else:
            low_dim_keys.append(key)
    
    return obs_keys, action_key, image_keys, low_dim_keys


def convert_single_dataset(
    repo_id: str,
    data_group: h5py.Group,
    root: str | Path | None,
    config: ConversionConfig,
    norm_stats: dict | None,
    sentence_encoder: SentenceTransformer | None,
    demo_counter_start: int = 0,
) -> int:
    """Convert a single LeRobot dataset to HDF5, processing one episode at a time."""
    print(f"\nLoading dataset: {repo_id}")
    
    # Load dataset
    try:
        dataset = LeRobotDataset(repo_id, root=root, download_videos=True, video_backend="pyav")
    except Exception as e:
        print(f"  Warning: {e}, trying revision='main'")
        try:
            dataset = LeRobotDataset(repo_id, root=root, download_videos=True, video_backend="pyav", revision="main")
        except Exception as e2:
            print(f"  Error loading dataset: {e2}")
            return demo_counter_start
    
    print(f"  Frames: {len(dataset)}, Episodes: {dataset.num_episodes}")
    
    # Check for episode metadata
    if not (hasattr(dataset, "meta") and hasattr(dataset.meta, "episodes") and dataset.meta.episodes is not None):
        print(f"  Warning: No episode metadata, skipping")
        return demo_counter_start
    
    # Get keys
    obs_keys, action_key, image_keys, low_dim_keys = get_dataset_keys(dataset)
    print(f"  Image keys: {image_keys}")
    print(f"  Low-dim keys: {low_dim_keys}")
    print(f"  Action key: {action_key}")
    
    # Get episode metadata
    episode_metadata, task_map = get_episode_metadata(dataset)
    print(f"  Processing {len(episode_metadata)} episodes (one at a time)...")
    
    demo_counter = demo_counter_start
    failed_episodes = 0
    
    for ep_meta in tqdm(episode_metadata, desc="  Episodes"):
        ep_idx = ep_meta["index"]
        from_idx = ep_meta["from_idx"]
        to_idx = ep_meta["to_idx"]
        
        if to_idx <= from_idx:
            continue
        
        # Get language instruction
        language_instruction = None
        if ep_meta.get("tasks"):
            tasks = ep_meta["tasks"]
            if isinstance(tasks, list) and len(tasks) > 0:
                language_instruction = task_map.get(tasks[0], str(tasks[0]))
        if language_instruction is None:
            language_instruction = f"{repo_id} Episode {ep_idx}"
        
        try:
            # Extract frames for this episode using DataLoader
            frames = extract_episode_frames(
                dataset, from_idx, to_idx, obs_keys, action_key, config
            )
            
            if frames is None or len(frames) == 0:
                failed_episodes += 1
                continue
            
            # Convert frames to episode data
            ep_data = frames_to_episode_data(
                frames, ep_idx, language_instruction, obs_keys, config
            )
            
            if ep_data is None:
                failed_episodes += 1
                continue
            
            # Process episode (normalize, resize, etc.)
            episode = process_episode(ep_data, repo_id, image_keys, config, norm_stats)
            
            if episode is None:
                failed_episodes += 1
                continue
            
            # Write to HDF5
            write_episode_to_h5(episode, data_group, sentence_encoder)
            demo_counter += 1
            
        except Exception as e:
            print(f"    Warning: Episode {ep_idx} failed: {e}")
            failed_episodes += 1
            continue
    
    if failed_episodes > 0:
        print(f"  Warning: {failed_episodes} episodes failed to process")
    
    return demo_counter


def convert_datasets(
    repo_ids: list[str],
    output_path: str,
    root: str | Path | None = None,
    config: ConversionConfig | None = None,
    sentence_encoder: SentenceTransformer | None = None,
):
    """
    Main conversion function: convert LeRobot dataset(s) to HDF5.
    """
    if config is None:
        config = ConversionConfig()
    
    print(f"Converting LeRobot dataset(s): {repo_ids}")
    print(f"Output: {output_path}")
    
    if config.image_height and config.image_width:
        print(f"Image dimensions: {config.image_height}x{config.image_width}")
    elif config.image_size:
        print(f"Image size (square): {config.image_size}x{config.image_size}")
    
    # Compute normalization stats if needed
    norm_stats = None
    if config.normalize:
        norm_stats = compute_normalization_stats(repo_ids, root, action_norm_mode=config.action_norm_mode)
        
        if config.action_norm_mode == "minmax" and "action_min" not in norm_stats:
            print("Warning: No action stats found, skipping normalization")
            norm_stats = None
        elif config.action_norm_mode == "zscore" and "action_mean" not in norm_stats:
            print("Warning: No action stats found, skipping normalization")
            norm_stats = None
    
    # Remove existing output file
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()
    
    # Convert
    with h5py.File(output_path, "w") as f:
        data_group = f.create_group("data")
        demo_counter = 0
        
        for repo_id in repo_ids:
            demo_counter = convert_single_dataset(
                repo_id=repo_id,
                data_group=data_group,
                root=root,
                config=config,
                norm_stats=norm_stats,
                sentence_encoder=sentence_encoder,
                demo_counter_start=demo_counter,
            )
        
        # Write normalization stats
        if norm_stats:
            norm_group = f.create_group("normalization")
            for key, value in norm_stats.items():
                if isinstance(value, str):
                    norm_group.attrs[key] = value
                else:
                    norm_group.create_dataset(key, data=value)
            f.attrs["normalized"] = True
        else:
            f.attrs["normalized"] = False
        
        # Global metadata
        f.attrs["total_demos"] = demo_counter
        f.attrs["format_version"] = "1.0"
        f.attrs["includes_images"] = config.include_images
        f.attrs["repo_ids"] = ",".join(repo_ids)
        
        if config.image_height and config.image_width:
            f.attrs["image_height"] = config.image_height
            f.attrs["image_width"] = config.image_width
        elif config.image_size:
            f.attrs["image_size"] = config.image_size
    
    print(f"\nConversion complete!")
    print(f"Total demos: {demo_counter}")
    print(f"Output: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot dataset(s) to HDF5 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required
    parser.add_argument("--repo-ids", type=str, nargs="+", required=True, help="LeRobot dataset repo_id(s)")
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 file path")
    
    # Dataset options
    parser.add_argument("--root", type=str, default=None, help="Root directory for LeRobot datasets")
    
    # Image options
    parser.add_argument("--no-images", action="store_true", help="Exclude images")
    parser.add_argument("--image-size", type=int, help="Square crop size (legacy)")
    parser.add_argument("--image-height", type=int, help="Target image height")
    parser.add_argument("--image-width", type=int, help="Target image width")
    
    # Content options
    parser.add_argument("--no-rewards", action="store_true", help="Exclude rewards")
    parser.add_argument("--no-dones", action="store_true", help="Exclude dones")
    
    # Normalization
    parser.add_argument("--normalize", action="store_true", help="Normalize actions and states")
    parser.add_argument("--action-norm-mode", choices=["minmax", "zscore"], default="minmax")
    
    # Language encoding
    parser.add_argument("--sentence-encoder", type=str, help="SentenceTransformer model name")
    
    # Performance
    parser.add_argument("--batch-size", type=int, default=256, help="Extraction batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    
    # Output
    parser.add_argument("--info", action="store_true", help="Print dataset info after conversion")
    
    args = parser.parse_args()
    
    # Validate
    if (args.image_height is None) != (args.image_width is None):
        parser.error("--image-height and --image-width must be used together")
    
    # Build config
    config = ConversionConfig(
        include_images=not args.no_images,
        include_rewards=not args.no_rewards,
        include_dones=not args.no_dones,
        image_size=args.image_size,
        image_height=args.image_height,
        image_width=args.image_width,
        normalize=args.normalize,
        action_norm_mode=args.action_norm_mode,
        extraction_batch_size=args.batch_size,
        extraction_num_workers=args.num_workers,
    )
    
    # Load sentence encoder
    sentence_encoder = None
    if args.sentence_encoder:
        print(f"Loading sentence encoder: {args.sentence_encoder}")
        sentence_encoder = SentenceTransformer(args.sentence_encoder)
    
    # Convert
    convert_datasets(
        repo_ids=args.repo_ids,
        output_path=args.output,
        root=args.root,
        config=config,
        sentence_encoder=sentence_encoder,
    )
    
    # Print info
    if args.info:
        print_dataset_info(args.output)


if __name__ == "__main__":
    main()
