#!/usr/bin/env python3
"""
RoboFAC dataset loader for the generic dataset converter for Robometer model training.
Loads MINT-SJTU/RoboFAC-dataset structure: realworld_data/<task>/videos/*.mp4 and simulation_data.
"""

from pathlib import Path

from dataset_upload.helpers import generate_unique_id, load_sentence_transformer_model
from dataset_upload.video_helpers import load_video_frames
from tqdm import tqdm


class RoboFACFrameLoader:
    """Pickle-able loader that reads RoboFAC video files on demand."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def __call__(self):
        """Load frames from video file. Returns np.ndarray (T, H, W, 3) uint8."""
        return load_video_frames(Path(self.file_path))


def _folder_to_task_description(folder_name: str) -> str:
    """Convert folder name to human-readable task (e.g. so100_insert_cylinder_error -> Insert cylinder)."""
    # Remove so100_ prefix and _error suffix if present
    name = folder_name.replace("so100_", "").replace("_error", "")
    return name.replace("_", " ").strip().title()


def _find_mp4_under(path: Path) -> list[Path]:
    """Find all .mp4 files under path (recursive). Handles videos/ or videos/chunk-000/ etc."""
    if not path.exists():
        return []
    return sorted(path.rglob("*.mp4"))


def _simulation_quality_from_folder(folder_name: str) -> str:
    """Map simulation_data subfolder to quality_label (successful / failure)."""
    name_lower = folder_name.lower()
    if "success" in name_lower and "fail" not in name_lower:
        return "successful"
    if "fail" in name_lower or "error" in name_lower:
        return "failure"
    return "failure"  # default for unknown


def _parse_simulation_video_path(video_path: Path, root: Path) -> tuple[str, str, str]:
    """From a video under simulation_data/, derive (task, quality_label, data_source).

    Paths like: simulation_data/success_data/UprightStack-v1/stack_error/<uuid>.mp4
    -> task = UprightStack-v1, quality = successful, data_source = simulation_data/success_data
    """
    try:
        rel = video_path.relative_to(root)
    except ValueError:
        return "Simulation", "failure", "simulation_data"
    parts = rel.parts
    if len(parts) < 3:
        return "Simulation", "failure", "simulation_data"
    # parts[0] = simulation_data, parts[1] = success_data|failure_data|..., parts[2] = task name
    if parts[0] != "simulation_data":
        return "Simulation", "failure", "simulation_data"
    quality_folder = parts[1]
    task_name = parts[2]
    quality_label = _simulation_quality_from_folder(quality_folder)
    # Human-readable task: UprightStack-v1 -> "Upright Stack v1"
    task_desc = task_name.replace("-", " ").replace("_", " ").strip().title()
    data_source = f"simulation_data/{quality_folder}"
    return task_desc, quality_label, data_source


def _discover_robofac_trajectories(
    dataset_path: Path,
    *,
    realworld: bool = True,
    simulation: bool = True,
) -> list[tuple[Path, str, str, str]]:
    """Discover all video files in RoboFAC dataset structure.

    Expected structure (from MINT-SJTU/RoboFAC-dataset):
        realworld_data/
            so100_insert_cylinder_error/
                videos/
                    *.mp4   OR  videos/chunk-000/*.mp4  (recursive)
            ...
        simulation_data/
            success_data/   or  failure_data/
                <TaskName>/   e.g. UprightStack-v1
                    (optional subdirs, e.g. stack_error/)
                        *.mp4

    If realworld_data is not found at dataset_path, also tries dataset_path / "main"
    (some download methods put repo content under a main/ subfolder).

    Returns:
        List of (video_path, task, quality_label, data_source) for each trajectory.
    """
    out: list[tuple[Path, str, str, str]] = []

    # Resolve root: support both /path/to/RoboFAC-dataset and /path/to/RoboFAC-dataset/main
    root = dataset_path
    realworld_path = root / "realworld_data"
    if not realworld_path.exists() and (root / "main").is_dir():
        root = root / "main"
        realworld_path = root / "realworld_data"
    if not realworld_path.exists():
        print(f"Warning: realworld_data not found at {dataset_path} or {dataset_path}/main")
    else:
        if realworld:
            for task_dir in sorted(realworld_path.iterdir()):
                if not task_dir.is_dir() or task_dir.name.startswith("."):
                    continue
                task_name = task_dir.name
                task_desc = _folder_to_task_description(task_name)
                videos = _find_mp4_under(task_dir)
                for vid in videos:
                    out.append((vid, task_desc, "failure", f"realworld_data/{task_name}"))
                if videos:
                    print(f"  realworld_data/{task_name}: {len(videos)} videos")

    if simulation:
        sim_path = root / "simulation_data"
        if sim_path.exists():
            # Discover all mp4s under simulation_data and parse path for task + quality
            videos = _find_mp4_under(sim_path)
            for vid in videos:
                task_desc, quality_label, data_source = _parse_simulation_video_path(vid, root)
                out.append((vid, task_desc, quality_label, data_source))
            if videos:
                print(f"  simulation_data: {len(videos)} videos (task/quality from path)")
        else:
            print("Warning: simulation_data not found")

    return out


def load_robofac_dataset(
    dataset_path: str,
    max_trajectories: int | None = None,
    realworld: bool = True,
    simulation: bool = True,
) -> dict[str, list[dict]]:
    """Load RoboFAC dataset and organize by task.

    Args:
        dataset_path: Path to the RoboFAC dataset root (e.g. .../RoboFAC-dataset or .../RoboFAC-dataset/main).
        max_trajectories: Maximum number of trajectories to load (None for all).
        realworld: Include realworld_data subfolders.
        simulation: Include simulation_data.

    Returns:
        Dictionary mapping task names to lists of trajectory dicts (frames, task, quality_label, etc.).
    """
    print("Loading RoboFAC dataset from:", dataset_path)
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"RoboFAC dataset path not found: {dataset_path}")

    print("Discovering videos (realworld_data/ and simulation_data/)...")
    traj_list = _discover_robofac_trajectories(
        dataset_path, realworld=realworld, simulation=simulation
    )
    if not traj_list:
        raise FileNotFoundError(
            f"No .mp4 videos found under {dataset_path}. "
            "Check that the path points to the RoboFAC-dataset root (containing realworld_data/ and optionally simulation_data/). "
            "If you downloaded with Hugging Face CLI, the root may be under a 'main' subfolder; the loader will try that automatically."
        )
    if max_trajectories is not None and max_trajectories != -1:
        traj_list = traj_list[:max_trajectories]

    print(f"Found {len(traj_list)} trajectory videos total")

    task_data: dict[str, list[dict]] = {}
    for video_path, task_desc, quality_label, data_source in tqdm(
        traj_list, desc="Building RoboFAC trajectories"
    ):
        frame_loader = RoboFACFrameLoader(str(video_path))
        partial = 1.0 if quality_label == "successful" else 0.0
        trajectory = {
            "frames": frame_loader,
            "actions": None,
            "is_robot": True,
            "task": task_desc,
            "quality_label": quality_label,
            "data_source": data_source,
            "partial_success": partial,
            "id": generate_unique_id(),
        }
        task_data.setdefault(task_desc, []).append(trajectory)

    total = sum(len(v) for v in task_data.values())
    print(f"Loaded {total} trajectories from {len(task_data)} tasks")
    return task_data
