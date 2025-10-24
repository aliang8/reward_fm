"""
AgiBotWorld dataset loader for the generic dataset converter for RFM model training.
This module contains AgiBotWorld-specific logic for loading and processing data using
HuggingFace streaming to efficiently handle large datasets.
"""

import os
import json
import h5py
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets
import datasets as hfds
from tqdm import tqdm
from helpers import (
    load_sentence_transformer_model,
    create_hf_trajectory,
)
from multiprocessing import Pool, cpu_count
from functools import partial
from rfm.data.helpers import generate_unique_id
from rfm.data.video_helpers import load_video_frames
import math
import time

# Episode/task helpers built earlier
try:
    from rfm.data.data_scripts.agibot import get_episode_record
except Exception:
    # Fallback for direct execution context
    from ..data_scripts.agibot.agibot_helper import get_episode_record  # type: ignore


# ------------------------------
# Small utilities
# ------------------------------

CAMERA_KEYS = {
    "head_color",
    "head_left_fisheye_color",
    "head_right_fisheye_color",
    "head_center_fisheye_color",
}


def _stable_shard_for_episode(episode_id: str, shard_modulus: int = 1000) -> str:
    """Return a stable top-level shard name based on episode id.

    Keeps at most ~shard_modulus episode directories per shard.
    """

    try:
        idx = int(episode_id)
    except Exception:
        idx = abs(hash(episode_id))
    shard_index = idx // shard_modulus
    return f"shard_{shard_index:04d}"


def _parse_episode_and_camera(key: str) -> Tuple[str, Optional[str]]:
    """Parse __key__ like '678985/videos/head_color' -> ('678985', 'head_color')."""
    parts = key.split("/")
    if len(parts) < 3:
        return parts[0], None
    return parts[0], parts[2]


def _build_video_paths(
    output_dir: str,
    dataset_name: str,
    episode_id: str,
    subtask_idx: int,
    camera: str,
) -> Tuple[str, str]:
    """Return (full_path, relative_path) using a two-level shard + per-episode layout.

    Layout:
      <output>/<dataset>/<shard_X>/<episode_id>/clip_<k>@<camera>.mp4
    This avoids >1k files per directory while keeping resume-friendly structure.
    """
    shard_dir = _stable_shard_for_episode(episode_id)
    episode_dir = os.path.join(output_dir, dataset_name.lower(), shard_dir, f"episode_{episode_id}")
    os.makedirs(episode_dir, exist_ok=True)
    filename = f"clip_{subtask_idx}@{camera}.mp4"
    full_path = os.path.join(episode_dir, filename)
    rel_path = os.path.join(dataset_name.lower(), shard_dir, f"episode_{episode_id}", filename)
    return full_path, rel_path


def _should_skip_shard(
    output_dir: str,
    dataset_name: str,
    shard_dir: str,
    threshold: Optional[int],
    cache: Dict[str, bool],
) -> bool:
    """Return True if the shard should be skipped due to existing episode folder count.

    Counts immediate child directories named like 'episode_*' under
    <output>/<dataset>/<shard_dir> once per shard and caches the decision.
    A threshold <= 0 or None disables skipping.
    """
    if threshold is None or threshold <= 0:
        return False
    if shard_dir in cache:
        return cache[shard_dir]
    shard_path = os.path.join(output_dir, dataset_name.lower(), shard_dir)
    total = 0
    try:
        with os.scandir(shard_path) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                name = entry.name
                if name.startswith("episode_"):
                    total += 1
                    if total > threshold:
                        break
    except Exception:
        # If any issue reading the directory, do not skip
        total = 0

    print(f"shard_dir: {shard_dir}, total: {total}, threshold: {threshold}")
    skip = total > threshold
    cache[shard_dir] = skip
    return skip


def _hf_shard_all_seen_shards_full(
    ds,
    dataset_name: str,
    output_dir: str,
    dataset_label: str,
    threshold: Optional[int],
    max_samples: int = 1000,
) -> Tuple[bool, int, set]:
    """Peek up to max_samples items from an HF shard to decide if it's safe to skip entirely.

    We only read lightweight metadata (`__key__`), map each sample to its stable
    output shard via episode id, and check whether ALL encountered stable shards
    already exceed the folder-count threshold. If we find any sample mapping to
    a shard that is NOT full, we return False immediately (do not skip).
    """
    if threshold is None or threshold <= 0:
        return False, 0, set()
    shard_skip_cache: Dict[str, bool] = {}
    encountered_stable_shards: set = set()
    it = iter(ds)
    count = 0
    while count < max_samples:
        try:
            s = next(it)
        except StopIteration:
            break
        except Exception:
            # On transient read error, err on side of not skipping
            return False, len(encountered_stable_shards), encountered_stable_shards
        key = s.get("__key__", "")
        episode_id, _camera = _parse_episode_and_camera(key)
        if not episode_id:
            count += 1
            continue
        stable_shard = _stable_shard_for_episode(episode_id)
        encountered_stable_shards.add(stable_shard)
        if not _should_skip_shard(
            output_dir=output_dir,
            dataset_name=dataset_label,
            shard_dir=stable_shard,
            threshold=threshold,
            cache=shard_skip_cache,
        ):
            return False, len(encountered_stable_shards), encountered_stable_shards
        count += 1
    # If we encountered nothing, don't skip; otherwise skip
    return (len(encountered_stable_shards) > 0), len(encountered_stable_shards), encountered_stable_shards

def _collect_unique_texts_for_batch(records: List[Tuple[str, dict]]) -> List[str]:
    """Collect unique instruction texts from a list of (episode_id, record) pairs."""
    texts: List[str] = []
    seen: set = set()
    for _episode_id, rec in records:
        # Full trajectory instruction
        full_text = rec.get("task_name") or rec.get("task_description") or ""
        if full_text and full_text not in seen:
            seen.add(full_text)
            texts.append(full_text)

        # Subtasks
        actions = rec.get("label_info", {}).get("action_config", [])
        for a in actions:
            t = (a or {}).get("action_text", "").strip()
            if t and t not in seen:
                seen.add(t)
                texts.append(t)
    return texts


def _encode_texts(texts: List[str], model) -> Dict[str, Any]:
    """Encode a list of texts to vectors using a preloaded model, with caching."""
    if not texts:
        return {}
    vectors = model.encode(texts)
    return {t: v for t, v in zip(texts, vectors)}


def _frames_for_subrange(frames: np.ndarray, start: int, end: int) -> np.ndarray:
    """Return frames[start:end] with guardrails; [start, end) semantics."""
    start = max(int(start), 0)
    end = min(int(end), len(frames))
    if start >= end:
        return np.empty((0,), dtype=object)
    return frames[start:end]


def _process_single_stream_sample(
    sample: Dict[str, Any],
    embeddings: Dict[str, Any],
    output_dir: str,
    dataset_name: str,
    max_frames: int,
    fps: int,
) -> List[Dict]:
    """Process one streaming sample: returns zero or more HF entries.

    This function does not load any language model; it expects embeddings for
    the relevant instruction texts to be provided.
    """

    result_entries: List[Dict] = []

    # Extract key and keep only camera samples we care about
    key = sample.get("__key__") or ""
    episode_id, camera = _parse_episode_and_camera(key)
    if not camera or camera not in CAMERA_KEYS:
        return result_entries

    # Load associated episode record for task/subtasks
    try:
        _json_path, rec = get_episode_record(episode_id)
    except Exception:
        return result_entries

    # Lazily decode frames only if we must write any missing videos.
    # IMPORTANT: Accessing sample["mp4"] may trigger a remote fetch; avoid unless needed.
    frames_cache: Optional[np.ndarray] = None

    def get_frames() -> np.ndarray:
        nonlocal frames_cache
        if frames_cache is None:
            try:
                video_bytes_local = sample.get("mp4")
            except Exception:
                return np.empty((0,), dtype=object)
            if not video_bytes_local:
                return np.empty((0,), dtype=object)
            try:
                frames_cache = load_video_frames(video_bytes_local)
            except Exception:
                return np.empty((0,), dtype=object)
        return frames_cache

    # Build entries: full + subtasks
    # Full trajectory
    full_text = rec.get("task_name") or rec.get("task_description") or ""
    if full_text:
        subtask_idx = 0
        full_out_path, rel_path = _build_video_paths(output_dir, dataset_name, episode_id, subtask_idx, camera)
        # Always route through create_hf_trajectory which internally creates the video only if missing.
        # Provide frames as a callable so frames are only loaded if writing is needed.

        lang_vec = embeddings.get(full_text)
        if lang_vec is None:
            # As a fallback, keep empty vector to avoid crashing
            lang_vec = np.zeros((384,), dtype=np.float32)

        traj_dict = {
            "id": generate_unique_id(),
            "frames": (lambda: get_frames()),
            "task": full_text,
            "is_robot": True,
            "quality_label": "successful",
            "preference_group_id": None,
            "preference_rank": None,
        }
        entry = create_hf_trajectory(
            traj_dict=traj_dict,
            video_path=full_out_path,
            lang_vector=lang_vec,
            max_frames=max_frames,
            dataset_name=dataset_name,
            use_video=True,
            fps=fps,
        )
        if entry:
            entry["frames"] = rel_path
            result_entries.append(entry)

    # Subtasks
    actions = rec.get("label_info", {}).get("action_config", [])
    for i, a in enumerate(actions, start=1):
        if not isinstance(a, dict):
            continue
        text = (a.get("action_text") or "").strip()
        if not text:
            continue
        start = a.get("start_frame", 0)
        # Use a large sentinel end bound; actual frames length will be applied lazily when needed
        end = a.get("end_frame", 10**9)
        out_path, rel_path = _build_video_paths(output_dir, dataset_name, episode_id, i, camera)

        lang_vec = embeddings.get(text)
        if lang_vec is None:
            lang_vec = np.zeros((384,), dtype=np.float32)

        traj_dict = {
            "id": generate_unique_id(),
            # Provide subrange lazily; only computed if the subclip video is missing
            "frames": (lambda s=start, e=end: _frames_for_subrange(get_frames(), s, e)),
            "task": text,
            "is_robot": True,
            "quality_label": "successful",
            "preference_group_id": None,
            "preference_rank": None,
        }
        entry = create_hf_trajectory(
            traj_dict=traj_dict,
            video_path=out_path,
            lang_vector=lang_vec,
            max_frames=max_frames,
            dataset_name=dataset_name,
            use_video=True,
            fps=fps,
        )
        if entry:
            entry["frames"] = rel_path
            result_entries.append(entry)

    return result_entries


def _convert_agibotworld_shard_worker(args) -> Dataset:
    """Worker wrapper to process a single dataset shard.

    Args tuple: (
        dataset_name, output_dir, dataset_label, max_trajectories,
        max_frames, fps, num_workers, dataset_num_shards, shard_index,
        skip_shard_videos_threshold
    )
    """
    (
        dataset_name,
        output_dir,
        dataset_label,
        max_trajectories,
        max_frames,
        fps,
        num_workers,
        dataset_num_shards,
        shard_index,
        skip_shard_videos_threshold,
    ) = args

    return convert_agibotworld_streaming_to_hf(
        dataset_name=dataset_name,
        output_dir=output_dir,
        dataset_label=dataset_label,
        max_trajectories=max_trajectories,
        max_frames=max_frames,
        fps=fps,
        num_workers=num_workers,
        dataset_num_shards=dataset_num_shards,
        shard_index=shard_index,
        skip_shard_videos_threshold=skip_shard_videos_threshold,
        parallelize_shards=True,
    )


def convert_agibotworld_streaming_to_hf(
    dataset_name: str,
    output_dir: str,
    dataset_label: str = "agibotworld",
    max_trajectories: Optional[int] = None,
    max_frames: int = 64,
    fps: int = 10,
    num_workers: int = -1,
    dataset_num_shards: int = 2500,
    shard_index: Optional[int] = None,
    skip_shard_videos_threshold: Optional[int] = 400,
    parallelize_shards: bool = True,
) -> Dataset:
    """Stream AgiBotWorld, extract camera videos, and write HF entries.

    Returns a datasets.Dataset built from the collected entries. All videos are
    saved to disk under output_dir.

    Sharding:
      - Provide dataset_num_shards > 1 to split HF stream into N shards.
      - If shard_index is provided (0-based), only that shard is processed.
      - If shard_index is None and parallelize_shards is True, shards are
        processed in parallel and then concatenated.
    """

    # Top-level shard parallelization orchestration
    print(f"dataset_num_shards: {dataset_num_shards}, shard_index: {shard_index}, parallelize_shards: {parallelize_shards}")
    if dataset_num_shards > 1 and shard_index is None and parallelize_shards:
        per_shard_max = None
        if max_trajectories is not None:
            per_shard_max = max(1, math.ceil(max_trajectories / dataset_num_shards))

        shard_indices = list(range(dataset_num_shards))
        print(
            f"Launching {dataset_num_shards} shard workers; per_shard_max={per_shard_max}"
        )
        max_procs = max(1, min(cpu_count(), 8))
        shard_pool_procs = min(dataset_num_shards, max_procs)
        per_shard_workers = max(1, max_procs // dataset_num_shards)
        # Early skip via lightweight peek: for each HF shard, sample up to N keys and
        # check whether all encountered stable shards are already full by folder count.
        filtered_indices = shard_indices
 
        with Pool(processes=shard_pool_procs) as pool:
            args_list = [
                (
                    dataset_name,
                    output_dir,
                    dataset_label,
                    per_shard_max,
                    max_frames,
                    fps,
                    per_shard_workers,
                    dataset_num_shards,
                    si,
                    skip_shard_videos_threshold,
                )
                for si in filtered_indices
            ]
            shard_datasets = list(
                tqdm(
                    pool.imap_unordered(_convert_agibotworld_shard_worker, args_list),
                    total=len(args_list),
                    desc=f"Shards (N={dataset_num_shards})",
                    leave=False,
                )
            )

        # Filter empties and combine
        non_empty = [d for d in shard_datasets if d is not None and len(d) > 0]
        if not non_empty:
            return Dataset.from_dict(
                {
                    "id": [],
                    "task": [],
                    "lang_vector": [],
                    "data_source": [],
                    "frames": [],
                    "is_robot": [],
                    "quality_label": [],
                    "preference_group_id": [],
                    "preference_rank": [],
                }
            )
        combined = concatenate_datasets(non_empty)
        if max_trajectories is not None and len(combined) > max_trajectories:
            combined = combined.select(range(max_trajectories))
        return combined

    # Load streaming dataset (single shard or pre-selected shard)
    ds = load_dataset(dataset_name, streaming=True, split="train")
    if dataset_num_shards > 1:
        # If shard_index is not provided and we didn't parallelize above, default to first shard
        effective_shard_index = shard_index if shard_index is not None else 0
        print(
            f"Processing dataset shard {effective_shard_index + 1}/{dataset_num_shards}"
        )
        try:
            ds = ds.shard(dataset_num_shards, effective_shard_index)
        except Exception as e:
            print(f"Warning: failed to apply dataset.shard(...): {e}", "effective_shard_index",
             effective_shard_index, "dataset_num_shards", dataset_num_shards)
        # Note: We avoid any peeking to keep streaming robust across schema variations.
        # Skipping whole shards is disabled here; rely on main loop checks per sample.

    # Do not cast the streaming dataset; iterate keys and check presence of 'mp4' lazily

    # Determine workers
    if num_workers == -1:
        num_workers = max(1, min(cpu_count(), 8))
    elif num_workers == 0:
        num_workers = 1

    # Language model for batch embedding
    lang_model = load_sentence_transformer_model()

    entries: List[Dict] = []
    processed = 0  # number of streaming samples actually flushed/processed
    default_batch_size = 64
    batch_size = default_batch_size if (max_trajectories is None) else min(default_batch_size, max_trajectories)
    batch_samples: List[Dict[str, Any]] = []
    batch_records: List[Tuple[str, dict]] = []

    # Simple live stats
    seen_samples = 0
    skipped_camera = 0
    skipped_no_record = 0
    skipped_no_mp4 = 0
    skipped_due_to_full_shard = 0
    shard_skip_cache: Dict[str, bool] = {}
    decode_fail = 0

    def flush_batch():
        nonlocal entries, processed, batch_samples, batch_records
        if not batch_samples:
            return

        # Collect unique texts and encode once
        unique_texts = _collect_unique_texts_for_batch(batch_records)
        emb_map = _encode_texts(unique_texts, lang_model)

        if num_workers == 1:
            for sample in tqdm(batch_samples, desc="Batch (seq)", leave=False):
                res = _process_single_stream_sample(
                    sample=sample,
                    embeddings=emb_map,
                    output_dir=output_dir,
                    dataset_name=dataset_label,
                    max_frames=max_frames,
                    fps=fps,
                )
                # res is a list; extend and update decode_fail if nothing produced due to decode error
                entries.extend(res)
        else:
            with Pool(processes=num_workers) as pool:
                worker = partial(
                    _process_single_stream_sample,
                    embeddings=emb_map,
                    output_dir=output_dir,
                    dataset_name=dataset_label,
                    max_frames=max_frames,
                    fps=fps,
                )
                for res in tqdm(
                    pool.imap_unordered(worker, batch_samples),
                    total=len(batch_samples),
                    desc=f"Batch (workers={num_workers})",
                    leave=False,
                ):
                    entries.extend(res)

        processed += len(batch_samples)
        batch_samples = []
        batch_records = []

    print(f"Streaming {dataset_name}; workers={num_workers}, batch_size={batch_size}")
    stream_pbar = tqdm(desc="Streaming samples", unit="sample", dynamic_ncols=True)

    ds_iter = iter(ds)
    max_next_retries = 5
    base_sleep_s = 1.0
    while True:
        if max_trajectories is not None and processed >= max_trajectories:
            break

        # Pull next sample with transient-error retries
        attempt = 0
        sample = None
        while True:
            try:
                sample = next(ds_iter)
                break
            except StopIteration:
                sample = None
                break
            except Exception as e:
                if attempt >= max_next_retries:
                    print(f"Giving up after {max_next_retries} next() retries due to: {e}")
                    sample = None
                    break
                sleep_s = base_sleep_s * (2 ** attempt)
                print(f"next() error: {e}; retrying in {sleep_s:.1f}s (attempt {attempt+1}/{max_next_retries})")
                time.sleep(sleep_s)
                attempt += 1

        if sample is None:
            # Either exhausted or persistent failure; flush and stop
            flush_batch()
            break

        key = sample.get("__key__", "")
        episode_id, camera = _parse_episode_and_camera(key)
        seen_samples += 1
        stream_pbar.update(1)
        if not camera or camera not in CAMERA_KEYS:
            skipped_camera += 1
            continue

        # Per-sample skip removed in favor of HF shard-level skipping based on shard_index

        # Ensure episode record exists; gather for embedding planning
        try:
            _json_path, rec = get_episode_record(episode_id)
        except Exception:
            skipped_no_record += 1
            continue

        # Require mp4 content; if absent (e.g., png-only shard), skip early
        if not sample.get("mp4"):
            skipped_no_mp4 += 1
            continue

        batch_samples.append(sample)
        batch_records.append((episode_id, rec))

        if len(batch_samples) >= batch_size:
            flush_batch()

        # If user asked for a very small number, don't wait for another full batch
        if max_trajectories is not None and (processed + len(batch_samples)) >= max_trajectories:
            flush_batch()
            break

    # Final flush
    flush_batch()
    stream_pbar.close()

    # Build HF dataset from entries
    if not entries:
        return Dataset.from_dict(
            {
                "id": [],
                "task": [],
                "lang_vector": [],
                "data_source": [],
                "frames": [],
                "is_robot": [],
                "quality_label": [],
                "preference_group_id": [],
                "preference_rank": [],
            }
        )

    # datasets can infer features; rely on default
    print(
        f"Done. seen={seen_samples}, entries={len(entries)}, "
        f"skipped_camera={skipped_camera}, skipped_no_record={skipped_no_record}, "
        f"skipped_no_mp4={skipped_no_mp4}, skipped_full_shard={skipped_due_to_full_shard}"
    )
    return Dataset.from_list(entries)


def load_agibotworld_dataset(dataset_name_or_path: str, max_trajectories: int = 100) -> Dict[str, List[Dict]]:
    """Load AgiBotWorld dataset using HuggingFace streaming and extract head_color.mp4 files.

    Args:
        dataset_name_or_path: HuggingFace dataset name (e.g. "agibot-world/AgiBotWorld-Alpha")
                             or local path to the dataset

    Returns:
        Dictionary mapping task names to lists of trajectory dictionaries
    """

    print(f"Loading AgiBotWorld dataset from: {dataset_name_or_path}")
    print("=" * 100)
    print("LOADING AGIBOTWORLD DATASET")
    print("=" * 100)

    task_data = {}

    # Check if it's a local path or HuggingFace dataset name
    if os.path.exists(dataset_name_or_path):
        # Local dataset
        task_data = _load_local_agibotworld(dataset_name_or_path, max_trajectories)
    else:
        # HuggingFace dataset - use streaming
        task_data = _load_streaming_agibotworld(dataset_name_or_path, max_trajectories)

    print(
        f"Loaded {sum(len(trajectories) for trajectories in task_data.values())} trajectories from {len(task_data)} tasks"
    )
    return task_data


# NOTE: As the dataset is too large, we did not test this function extensively and it may be out of date.
def _load_local_agibotworld(base_path: str, max_trajectories: int = 100, max_frames: int = 32) -> Dict[str, List[Dict]]:
    """Load AgiBotWorld dataset from local files, starting with task_info JSON files."""
    base_path = Path(base_path)
    task_data = {}

    # Define required directories
    observations_dir = base_path / "observations"
    task_info_dir = base_path / "task_info"
    proprio_stats_dir = base_path / "proprio_stats"

    if not observations_dir.exists():
        raise FileNotFoundError(f"Observations directory not found: {observations_dir}")
    if not task_info_dir.exists():
        raise FileNotFoundError(f"Task info directory not found: {task_info_dir}")

    # Start by iterating over task_info JSON files to get proper task names
    task_info_files = list(task_info_dir.glob("*.json"))

    if not task_info_files:
        raise FileNotFoundError(f"No task info JSON files found in: {task_info_dir}")

    print(f"Found {len(task_info_files)} task info files")

    total_trajectories = 0

    for task_info_file in tqdm(task_info_files, desc="Processing tasks"):
        if total_trajectories >= max_trajectories:
            print(f"Reached max_trajectories limit ({max_trajectories}), stopping...")
            break

        # Extract task ID from filename (e.g., "task_392.json" -> "392")
        task_id = task_info_file.stem.replace("task_", "")

        # Load task information from JSON
        task_info = _load_task_info(task_info_file)

        if not task_info:
            print(f"Skipping task {task_id} - no valid task info")
            continue

        # Extract proper task name from first episode (they should all have the same task)
        if task_info and len(task_info) > 0:
            first_episode = task_info[0]
            task_name = first_episode.get("task_name", f"Task {task_id}")
            task_description = first_episode.get("task_description", f"AgiBotWorld Task {task_id}")
        else:
            task_name = f"Task {task_id}"
            task_description = f"AgiBotWorld Task {task_id}"

        print(f"Processing task {task_id}: '{task_name}'")

        # Get the corresponding task directory
        task_dir = observations_dir / task_id
        if not task_dir.exists():
            print(f"Task directory not found: {task_dir}, skipping...")
            continue

        trajectories = []

        # Process episodes based on the information in task_info JSON
        for episode_info in task_info:
            if total_trajectories >= max_trajectories:
                break

            episode_id = str(episode_info.get("episode_id", ""))
            if not episode_id:
                continue

            # Check if episode directory exists
            episode_dir = task_dir / episode_id
            if not episode_dir.exists():
                print(f"Episode directory not found: {episode_dir}, skipping episode {episode_id}")
                continue

            # Look for head_color.mp4 file
            videos_dir = episode_dir / "videos"
            head_color_video = videos_dir / "head_color.mp4"

            if head_color_video.exists():
                # Load proprioceptive data
                proprio_file = proprio_stats_dir / task_id / episode_id / "proprio_stats.h5"
                actions = _load_actions_from_h5(proprio_file)

                # Process video: resize to 256x256 and downsample frames
                try:
                    processed_frames = load_video_frames(head_color_video)

                    trajectory = {
                        "frames": processed_frames,  # Processed video frames
                        "actions": actions,
                        "is_robot": True,  # AgiBotWorld is robot data
                        "task": task_name,  # Use the descriptive task name from JSON
                        "optimal": "optimal",  # Assume all AgiBotWorld trajectories are optimal
                    }
                except Exception as e:
                    print(f"  ❌ Failed to process video {head_color_video}: {e}")
                    continue

                trajectories.append(trajectory)
                total_trajectories += 1

                print(f"  ✅ Loaded episode {episode_id} ({total_trajectories}/{max_trajectories})")
            else:
                print(f"  ❌ head_color.mp4 not found for episode {episode_id}")

        if trajectories:
            # Use proper task name from JSON instead of generic "task_{id}"
            task_data[task_name] = trajectories
            print(f"Added {len(trajectories)} trajectories for task '{task_name}'")

    print(f"Loaded {total_trajectories} total trajectories from {len(task_data)} tasks")
    return task_data


def _load_streaming_agibotworld(dataset_name: str, max_trajectories: int = 100) -> Dict[str, List[Dict]]:
    """Legacy helper no longer used. Kept for compatibility."""
    raise NotImplementedError("Use convert_agibotworld_streaming_to_hf() for streaming conversion.")


def _load_task_info(task_info_file: Path) -> List[Dict]:
    """Load task information from JSON file."""
    if not task_info_file.exists():
        print(f"Task info file not found: {task_info_file}")
        return []

    try:
        with open(task_info_file, "r") as f:
            task_info = json.load(f)
        return task_info if isinstance(task_info, list) else [task_info]
    except Exception as e:
        print(f"Error loading task info from {task_info_file}: {e}")
        return []


def _load_actions_from_h5(proprio_file: Path) -> np.ndarray:
    """Load actions from proprioceptive H5 file."""
    if not proprio_file.exists():
        print(f"Proprioceptive file not found: {proprio_file}")
        return np.array([])

    try:
        with h5py.File(proprio_file, "r") as f:
            # According to AgiBotWorld docs, actions are stored under /action
            if "action" in f:
                action_group = f["action"]

                # Try to extract joint actions (most common for manipulation)
                if "joint" in action_group and "position" in action_group["joint"]:
                    actions = action_group["joint"]["position"][:]
                elif "end" in action_group and "position" in action_group["end"]:
                    # Use end-effector positions if joint positions not available
                    end_positions = action_group["end"]["position"][:]
                    end_orientations = (
                        action_group["end"]["orientation"][:] if "orientation" in action_group["end"] else None
                    )

                    if end_orientations is not None:
                        # Concatenate position and orientation for full 6-DOF actions
                        # Reshape orientations from [N, 2, 4] to [N, 8] (both arms)
                        end_orientations_flat = end_orientations.reshape(end_orientations.shape[0], -1)
                        # Reshape positions from [N, 2, 3] to [N, 6]
                        end_positions_flat = end_positions.reshape(end_positions.shape[0], -1)
                        actions = np.concatenate([end_positions_flat, end_orientations_flat], axis=1)
                    else:
                        actions = end_positions.reshape(end_positions.shape[0], -1)
                else:
                    print(f"No recognizable action data found in {proprio_file}")
                    return np.array([])

                return actions
            else:
                print(f"No action group found in {proprio_file}")
                return np.array([])

    except Exception as e:
        print(f"Error loading actions from {proprio_file}: {e}")
        return np.array([])
