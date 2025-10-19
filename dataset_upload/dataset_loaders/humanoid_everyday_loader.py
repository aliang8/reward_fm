import os
import gc
import glob
import zipfile
import tempfile
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset

from dataset_upload.helpers import (
    create_hf_trajectory,
    generate_unique_id,
    load_sentence_transformer_model,
)
from tqdm import tqdm

# Disable GPUs for TensorFlow in this loader to avoid CUDA context issues in workers
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _stable_shard_for_index(index: int, shard_modulus: int = 1000) -> str:
    """Generate stable shard directory name for trajectory indexing."""
    try:
        idx = int(index)
    except Exception:
        idx = abs(hash(str(index)))
    shard_index = idx // shard_modulus
    return f"shard_{shard_index:04d}"


def _build_humanoid_video_paths(
    output_dir: str,
    dataset_label: str,
    episode_idx: int,
    task_name: str,
) -> tuple[str, str]:
    """Build video paths for humanoid everyday dataset."""
    shard_dir = _stable_shard_for_index(episode_idx)
    episode_dir = os.path.join(output_dir, dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}")
    os.makedirs(episode_dir, exist_ok=True)
    filename = f"clip@{task_name}.mp4"
    full_path = os.path.join(episode_dir, filename)
    rel_path = os.path.join(dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}", filename)
    return full_path, rel_path


def _extract_task_name_from_zip(zip_path: str) -> str:
    """Extract task name from zip file path."""
    # Get the basename without extension
    task_name = os.path.basename(zip_path).replace('.zip', '')
    return task_name


def _process_single_humanoid_episode(args):
    """Process a single episode from humanoid everyday dataset."""
    episode_data, ep_idx, task, lang_vec, output_dir, dataset_name, max_frames, fps = args
    
    episode_entries = []
    
    try:
        # Extract frames from episode data
        frames = []
        for step_data in episode_data:
            if 'image' in step_data:
                # Convert numpy array to uint8 if needed
                img = step_data['image']
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                frames.append(img)
        
        if not frames:
            return episode_entries
        
        # Skip episodes that are too long
        if len(frames) > 1000:
            print(f"Skipping episode {ep_idx} because it's too long, length is {len(frames)}")
            del frames
            return episode_entries
        
        full_path, rel_path = _build_humanoid_video_paths(
            output_dir=output_dir,
            dataset_label=dataset_name,
            episode_idx=ep_idx,
            task_name=task.replace(' ', '_'),
        )
        
        # Create trajectory dictionary
        traj_dict = {
            "id": generate_unique_id(),
            "frames": frames,
            "task": task,
            "is_robot": True,
            "quality_label": "successful",
            "preference_group_id": None,
            "preference_rank": None,
        }
        
        try:
            entry = create_hf_trajectory(
                traj_dict=traj_dict,
                video_path=full_path,
                lang_vector=lang_vec,
                max_frames=max_frames,
                dataset_name=dataset_name,
                use_video=True,
                fps=fps,
            )
        except Exception as e:
            print(f"Warning: Failed to create HF trajectory for ep {ep_idx}: {e}")
            return episode_entries
            
        if entry:
            entry["frames"] = rel_path
            episode_entries.append(entry)
            
    except Exception as e:
        print(f"Warning: Failed to process episode {ep_idx}: {e}")
        return episode_entries
    
    return episode_entries


def _load_single_humanoid_episode(zip_path: str, episode_idx: int):
    """Load a single episode from humanoid everyday dataset zip file."""
    try:
        # Import humanoid_everyday dataloader
        from humanoid_everyday import Dataloader

        # check if the non zip path exists
        try:
            ds = Dataloader(zip_path.replace(".zip", ""))
        except Exception as e:
            # Load dataset from zip file
            print(f"failed to load from folder: {e}, trying zip extraction method")
            ds = Dataloader(zip_path)
        
        # Get the specific episode
        episode = ds[episode_idx]
        
        # Convert episode to list of step dictionaries
        episode_data = []
        for step in episode:
            episode_data.append(step)
        
        return episode_data
        
    except ImportError:
        print(f"Warning: humanoid_everyday package not found. Please install it with: pip install humanoid_everyday")
        return None
    except Exception as e:
        print(f"Warning: Failed to load episode {episode_idx} from {zip_path}: {e}")
        return None


def _get_humanoid_zip_episode_count(zip_path: str):
    """Get the number of episodes in a humanoid everyday dataset zip file."""
    try:
        # Import humanoid_everyday dataloader
        from humanoid_everyday import Dataloader
        
        # Load dataset from zip file
        ds = Dataloader(zip_path)
        
        return len(ds)
        
    except ImportError:
        print(f"Warning: humanoid_everyday package not found. Please install it with: pip install humanoid_everyday")
        return 0
    except Exception as e:
        print(f"Warning: Failed to get episode count from {zip_path}: {e}")
        return 0


def convert_humanoid_everyday_dataset_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
    num_workers: int = -1,
) -> Dataset:
    """Convert Humanoid Everyday datasets to HF format by writing videos directly.

    Args:
        dataset_path: Root path that contains zip files with humanoid everyday datasets.
        dataset_name: Name to tag the resulting dataset (e.g., 'humanoid_everyday').
        output_dir: Where to write video files and dataset.
        max_trajectories: Limit number of produced trajectories (None/-1 for all).
        max_frames: Max frames per video.
        fps: Video fps.
        num_workers: Number of workers for parallel processing.
    """
    
    # Normalize and checks
    if dataset_name is None:
        raise ValueError("dataset_name is required")

    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Find all zip files in the dataset path
    zip_files = glob.glob(os.path.join(root, "**/*.zip"), recursive=True)
    if not zip_files:
        raise FileNotFoundError(f"No .zip files found in {dataset_path}")

    print(f"Found {len(zip_files)} zip files to process")

    # Determine workers
    if num_workers == -1:
        num_workers = min(cpu_count(), 8)
    elif num_workers == 0:
        num_workers = 1

    # Language model and cache
    lang_model = load_sentence_transformer_model()
    lang_cache: dict[str, Any] = {}

    # Process all zip files
    all_entries: list[dict[str, Any]] = []
    produced = 0
    max_limit = float("inf") if (max_trajectories is None or max_trajectories == -1) else int(max_trajectories)
    
    for zip_file in tqdm(zip_files, desc="Processing zip files"):
        if produced >= max_limit:
            break
            
        print(f"Processing zip file: {zip_file}")
        
        # Extract task name from zip file
        task_name = _extract_task_name_from_zip(zip_file)
        
        # Precompute embedding for this task
        if task_name not in lang_cache:
            lang_cache[task_name] = lang_model.encode(task_name)
        lang_vec = lang_cache[task_name]
        
        # Get episode count for this zip file
        episode_count = _get_humanoid_zip_episode_count(zip_file)
        if episode_count == 0:
            print(f"No episodes found in {zip_file}")
            continue
            
        print(f"Found {episode_count} episodes in {zip_file}")
        
        # Process episodes one at a time to save memory
        for ep_idx in range(episode_count):
            if produced >= max_limit:
                break
                
            # Load single episode
            episode_data = _load_single_humanoid_episode(zip_file, ep_idx)
            if episode_data is None:
                print(f"Failed to load episode {ep_idx} from {zip_file}")
                continue
                
            # Process single episode
            episode_entries = _process_single_humanoid_episode((
                episode_data, ep_idx, task_name, lang_vec, output_dir, dataset_name, max_frames, fps
            ))
            
            all_entries.extend(episode_entries)
            produced += len(episode_entries)
            
            # Clean up episode data to free memory
            del episode_data
            gc.collect()
            
            if produced >= max_limit:
                break

    if not all_entries:
        return Dataset.from_dict({
            "id": [],
            "task": [],
            "lang_vector": [],
            "data_source": [],
            "frames": [],
            "is_robot": [],
            "quality_label": [],
            "preference_group_id": [],
            "preference_rank": [],
        })
    
    return Dataset.from_list(all_entries)
