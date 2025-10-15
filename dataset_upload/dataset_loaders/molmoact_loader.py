import io
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from datasets import Dataset, load_dataset
from PIL import Image
from tqdm import tqdm

from dataset_upload.helpers import (
    create_hf_trajectory,
    generate_unique_id,
    load_sentence_transformer_model,
)


def _stable_shard_for_index(index: int, shard_modulus: int = 1000) -> str:
    try:
        idx = int(index)
    except Exception:
        idx = abs(hash(str(index)))
    shard_index = idx // shard_modulus
    return f"shard_{shard_index:04d}"


def _build_molmo_video_paths(
    output_dir: str,
    dataset_label: str,
    episode_idx: int,
    view_key: str,
) -> tuple[str, str]:
    shard_dir = _stable_shard_for_index(episode_idx)
    episode_dir = os.path.join(output_dir, dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}")
    os.makedirs(episode_dir, exist_ok=True)
    filename = f"clip@{view_key}.mp4"
    full_path = os.path.join(episode_dir, filename)
    rel_path = os.path.join(dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}", filename)
    return full_path, rel_path


def _to_rgb_numpy(img_cell: Any) -> Optional[np.ndarray]:
    """Convert a datasets Image cell (dict with bytes/path, PIL.Image, or np.ndarray) to RGB uint8 ndarray."""
    if img_cell is None:
        return None
    # Already numpy HxWxC
    if isinstance(img_cell, np.ndarray):
        if img_cell.ndim == 3 and img_cell.shape[-1] in (1, 3, 4):
            if img_cell.shape[-1] == 1:
                img_cell = np.repeat(img_cell, 3, axis=-1)
            elif img_cell.shape[-1] == 4:
                img_cell = img_cell[..., :3]
            if img_cell.dtype != np.uint8:
                img_cell = img_cell.astype(np.uint8, copy=False)
            return img_cell
        return None
    # PIL
    if isinstance(img_cell, Image.Image):
        return np.asarray(img_cell.convert("RGB"), dtype=np.uint8)
    # dict with bytes
    if isinstance(img_cell, dict):
        data = img_cell.get("bytes")
        if data is None:
            path = img_cell.get("path")
            if path and os.path.exists(path):
                with Image.open(path) as im:
                    return np.asarray(im.convert("RGB"), dtype=np.uint8)
            return None
        with Image.open(io.BytesIO(data)) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    # Unknown
    return None


def convert_molmoact_dataset_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
) -> Dataset:
    """Stream the MolmoAct LeRobot (parquet) dataset and convert to HF by writing videos directly.

    Assumes dataset_path points to a folder containing the MolmoAct parquet files (can include multiple subfolders).
    Groups rows by `episode_index` and writes videos per view: `first_view`, `second_view`.
    """

    # Discover parquet files
    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"MolmoAct dataset path not found: {root}")

    patterns = ["**/*.parquet", "*.parquet"]
    data_files: list[str] = []
    for pat in patterns:
        data_files.extend([str(p) for p in root.glob(pat)])
    if not data_files:
        raise ValueError(f"No parquet files found under {root}")

    # Stream the dataset
    ds_iter = load_dataset(
        "parquet",
        data_files={"train": data_files},
        split="train",
        streaming=True,
    )

    # Language model (use a simple constant description or task_index if present)
    lang_model = load_sentence_transformer_model()

    current_ep: Optional[int] = None
    frames_by_view: dict[str, list[np.ndarray]] = {}
    entries: list[dict] = []
    produced = 0
    max_limit = float("inf") if (max_trajectories is None or max_trajectories == -1) else int(max_trajectories)

    def flush_episode(ep_idx: int, task_text: str) -> None:
        nonlocal produced, entries
        if not frames_by_view:
            return
        lang_vec = lang_model.encode(task_text)
        for view_key, frames in frames_by_view.items():
            if not frames:
                continue
            # Skip empty/black sequences heuristically
            if isinstance(frames[0], np.ndarray) and np.all(frames[0] == 0):
                continue

            full_path, rel_path = _build_molmo_video_paths(
                output_dir=output_dir,
                dataset_label=dataset_name,
                episode_idx=ep_idx,
                view_key=view_key,
            )

            traj_dict = {
                "id": generate_unique_id(),
                "frames": frames,  # pass list to avoid extra copies
                "task": task_text,
                "is_robot": True,
                "quality_label": "successful",
                "preference_group_id": None,
                "preference_rank": None,
            }

            entry = create_hf_trajectory(
                traj_dict=traj_dict,
                video_path=full_path,
                lang_vector=lang_vec,
                max_frames=max_frames,
                dataset_name=dataset_name,
                use_video=True,
                fps=fps,
            )
            if entry:
                entry["frames"] = rel_path
                entries.append(entry)
                produced += 1

    # Iterate streaming rows and group by episode_index
    print("Streaming MolmoAct rows and grouping by episode_index...")
    for row in tqdm(ds_iter, desc="MolmoAct rows"):
        ep_idx = int(row.get("episode_index", -1))
        if ep_idx < 0:
            # Skip rows without episode index
            continue

        # If new episode starts, flush previous
        if current_ep is None:
            current_ep = ep_idx
            frames_by_view = {"first_view": [], "second_view": []}
        elif ep_idx != current_ep:
            # Use task_index if present; otherwise a constant label
            task_text = (
                f"MolmoAct task {row.get('task_index', 0)}"
                if row.get("task_index") is not None
                else "MolmoAct"
            )
            flush_episode(current_ep, task_text)
            if produced >= max_limit:
                break
            current_ep = ep_idx
            frames_by_view = {"first_view": [], "second_view": []}

        # Append frames for available views
        for view_key in ("first_view", "second_view"):
            cell = row.get(view_key)
            img = _to_rgb_numpy(cell)
            if img is not None:
                frames_by_view[view_key].append(img)

    # Flush last episode
    if current_ep is not None and produced < max_limit:
        task_text = "MolmoAct"
        flush_episode(current_ep, task_text)

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

    return Dataset.from_list(entries)


