"""
Common utilities for LeRobot dataset conversion scripts.
"""

import h5py
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    raise ImportError(
        "lerobot is required. Please install it or ensure it's available in your environment."
    )

from rfm.utils.embedding_utils import compute_text_embeddings

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from rfm.utils.save import load_model_from_hf
    from rfm.evals.eval_utils import raw_dict_to_sample, build_payload
    from rfm.evals.eval_server import compute_batch_outputs
    from rfm.utils.setup_utils import setup_batch_collator
    HAS_REWARD_RELABELING = True
except ImportError:
    HAS_REWARD_RELABELING = False
    print("Warning: Reward relabeling not available. Install reward_fm package for reward relabeling support.")

# Import baseline models (optional - only if reward_fm is available)
try:
    from rfm.evals.baselines.gvl import GVL
    from rfm.evals.baselines.vlac import VLAC
    from rfm.evals.baselines.roboreward import RoboReward
    from rfm.evals.baselines.rfm_model import RFMModel
    HAS_BASELINE_MODELS = True
except ImportError:
    HAS_BASELINE_MODELS = False
    GVL = None
    VLAC = None
    RoboReward = None
    RFMModel = None


def normalize_obs_key(key: str) -> str:
    """
    Normalize observation key names to simplified format.
    
    Examples:
        observation.images.top -> image_top
        observation.images.side -> image_side
        observation.state -> state
        observation.agent_pos -> state
    """
    # Handle observation.images.X -> image_X
    if key.startswith("observation.images."):
        camera_name = key.replace("observation.images.", "")
        return f"image_{camera_name}"
    
    # Handle observation.state or observation.agent_pos -> state
    if key in ["observation.state", "observation.agent_pos"]:
        return "state"
    
    # Handle other observation.X -> X
    if key.startswith("observation."):
        return key.replace("observation.", "")
    
    # Keep as-is if no pattern matches
    return key


class _NormStatsDataset(Dataset):
    """Dataset over frame indices for normalization stats collection."""

    def __init__(self, dataset, indices: list[int], action_key: str, state_keys: list[str]):
        self.dataset = dataset
        self.indices = indices
        self.action_key = action_key
        self.state_keys = state_keys

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        idx = self.indices[i]
        item = self.dataset[idx]
        action = None
        if self.action_key in item:
            a = item[self.action_key]
            action = a.cpu().numpy() if torch.is_tensor(a) else np.asarray(a, dtype=np.float32)
        state = None
        for sk in self.state_keys:
            if sk in item:
                s = item[sk]
                state = s.cpu().numpy() if torch.is_tensor(s) else np.asarray(s, dtype=np.float32)
                break
        return action, state


def _norm_stats_collate(batch: list[tuple]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Collate (action, state) pairs; filter Nones and return flat lists."""
    actions = [b[0] for b in batch if b[0] is not None]
    states = [b[1] for b in batch if b[1] is not None]
    return actions, states


def compute_normalization_stats(
    repo_ids: list[str],
    root: str | Path | None,
    action_norm_mode: str = "minmax",  # "minmax" for [-1,1] range, "zscore" for mean=0,std=1
    batch_size: int = 256,
    num_workers: int = 0,
) -> dict:
    """
    Compute normalization stats for actions and states across all datasets.

    Uses a DataLoader to iterate over frames (optionally with multiple workers).

    Args:
        repo_ids: List of dataset repo IDs
        root: Root directory for datasets
        action_norm_mode: "minmax" normalizes actions to [-1, 1] (required for SquashedGaussian policies)
                         "zscore" normalizes to mean=0, std=1
        batch_size: DataLoader batch size (default: 256)
        num_workers: DataLoader workers; use 0 to avoid pickling issues (default: 0)

    Returns dict with normalization parameters.
    """
    print(f"Computing normalization statistics (action_norm_mode={action_norm_mode})...")
    all_actions: list[np.ndarray] = []
    all_states: list[np.ndarray] = []

    for repo_id in repo_ids:
        print(f"  Scanning {repo_id}...")
        try:
            dataset = LeRobotDataset(repo_id, root=root, download_videos=True, video_backend="pyav")
        except Exception as e:
            print(f"  Warning: Version check failed, trying with revision='main': {e}")
            try:
                dataset = LeRobotDataset(repo_id, root=root, download_videos=True, video_backend="pyav", revision="main")
            except Exception as e2:
                print(f"  Error: Could not load dataset {repo_id}: {e2}")
                continue

        if not (hasattr(dataset, "meta") and hasattr(dataset.meta, "episodes") and dataset.meta.episodes is not None):
            continue

        episodes = dataset.meta.episodes
        print(f"  Found {len(episodes)} episodes")

        action_key = "actions"
        if hasattr(dataset, "features"):
            if "actions" in dataset.features:
                action_key = "actions"
            elif "action" in dataset.features:
                action_key = "action"

        state_keys = ["observation.state", "observation.agent_pos"]

        indices: list[int] = []
        for row in episodes:
            from_idx = int(row["dataset_from_index"])
            to_idx = int(row["dataset_to_index"])
            if to_idx > from_idx:
                indices.extend(range(from_idx, to_idx))

        if not indices:
            continue

        ds = _NormStatsDataset(dataset, indices, action_key, state_keys)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=32,
            collate_fn=_norm_stats_collate,
            pin_memory=False,
        )

        for actions_batch, states_batch in tqdm(loader, desc="  Computing normalization statistics"):
            all_actions.extend(actions_batch)
            all_states.extend(states_batch)

    stats: dict = {"action_norm_mode": action_norm_mode}
    
    if all_actions:
        all_actions = np.stack(all_actions)
        
        if action_norm_mode == "minmax":
            # Min-max normalization to [-1, 1] range (required for tanh-squashed policies)
            stats["action_min"] = np.min(all_actions, axis=0).astype(np.float32)
            stats["action_max"] = np.max(all_actions, axis=0).astype(np.float32)
            # Add small margin to avoid exact -1/1 at boundaries
            action_range = stats["action_max"] - stats["action_min"]
            action_range = np.maximum(action_range, 1e-6)  # Avoid division by zero
            stats["action_range"] = action_range
            print(f"  Actions: min range [{stats['action_min'].min():.2f}, {stats['action_min'].max():.2f}], "
                  f"max range [{stats['action_max'].min():.2f}, {stats['action_max'].max():.2f}]")
        else:
            # Z-score normalization (mean=0, std=1)
            stats["action_mean"] = np.mean(all_actions, axis=0).astype(np.float32)
            stats["action_std"] = np.std(all_actions, axis=0).astype(np.float32)
            stats["action_std"] = np.maximum(stats["action_std"], 1e-6)
            print(f"  Actions: mean range [{stats['action_mean'].min():.2f}, {stats['action_mean'].max():.2f}], "
                  f"std range [{stats['action_std'].min():.2f}, {stats['action_std'].max():.2f}]")
    
    if all_states:
        all_states = np.stack(all_states)
        # Always use z-score for states (they don't need to be bounded)
        stats["state_mean"] = np.mean(all_states, axis=0).astype(np.float32)
        stats["state_std"] = np.std(all_states, axis=0).astype(np.float32)
        stats["state_std"] = np.maximum(stats["state_std"], 1e-6)
        print(f"  States: mean range [{stats['state_mean'].min():.2f}, {stats['state_mean'].max():.2f}], "
              f"std range [{stats['state_std'].min():.2f}, {stats['state_std'].max():.2f}]")
    
    return stats


def relabel_rewards_with_eval_server(
    all_frames: list[np.ndarray],
    language_instruction: str,
    episode_id: str,
    eval_server_url: str,
    sentence_model: SentenceTransformer | None = None,
    batch_size: int = 32,
    image_key: str = "image",
    max_frames: int = 16,
    timeout_s: float = 120.0,
) -> tuple[list[float], list[float]]:
    """
    Relabel rewards for a full trajectory using a remote eval server.
    
    Builds subsequences [0:1], [0:2], [0:3], ... and processes them through the eval server
    to get progress predictions and success probabilities for each timestep.
    
    Args:
        all_frames: List of frames for the full trajectory (each frame is HWC uint8)
        language_instruction: Language instruction string for the episode
        episode_id: Episode ID string
        eval_server_url: URL of the eval server (e.g., "http://localhost:8001")
        sentence_model: Optional sentence transformer for computing text embeddings
        batch_size: Batch size for processing subsequences
        image_key: Image key name (for logging)
        max_frames: Maximum number of frames to use per sample
        timeout_s: Request timeout in seconds
    
    Returns:
        Tuple of (progress_predictions, success_probs) - one value per timestep
    """
    if not HAS_REWARD_RELABELING:
        raise ImportError("Reward relabeling not available. Install reward_fm package.")
    if not HAS_REQUESTS:
        raise ImportError("requests package is required for eval server support. Install with: pip install requests")
    
    # Compute text embedding from language instruction
    text_embedding = None
    if sentence_model is not None:
        text_embedding_tensor = compute_text_embeddings(language_instruction, sentence_model)
        text_embedding = text_embedding_tensor.cpu().numpy()
    else:
        raise ValueError("sentence_model is required for reward relabeling")
    
    # Build subsequences: [0:1], [0:2], [0:3], ...
    all_samples = []
    for i in range(len(all_frames)):
        # Build subsequence from start to current step (0 to i+1)
        frames_subsequence = np.array(all_frames[: i + 1])  # Shape: (i+1, H, W, C)
        
        # Prepare raw data dict for reward model
        raw_data = dict(
            frames=frames_subsequence,
            task=language_instruction,
            id=episode_id,
            metadata=dict(subsequence_length=len(frames_subsequence)),
            text_embedding=text_embedding,
        )
        
        # Convert to sample
        sample = raw_dict_to_sample(
            raw_data=raw_data,
            max_frames=max_frames,
            sample_type="progress",
        )
        all_samples.append(sample)
    
    # Process subsequences in batches
    all_progress = []
    all_success_probs = []
    
    num_batches = (len(all_samples) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(all_samples), batch_size), desc="  Processing batches", leave=False, total=num_batches):
        batch_end = min(batch_start + batch_size, len(all_samples))
        batch_samples = all_samples[batch_start:batch_end]
        
        # Build payload for eval server
        files, sample_data = build_payload(batch_samples)
        
        # Add use_frame_steps=False to data dict since we're already building subsequences
        import json
        data_dict = {f"sample_{i}": json.dumps(sample) for i, sample in enumerate(sample_data)}
        data_dict["use_frame_steps"] = "false"
        
        # Send request to eval server
        try:
            import requests
            url = eval_server_url.rstrip("/") + "/evaluate_batch_npy"
            resp = requests.post(url, files=files, data=data_dict, timeout=timeout_s)
            resp.raise_for_status()
            batch_outputs = resp.json()
            
            # Extract progress predictions from outputs_progress
            outputs_progress = batch_outputs.get("outputs_progress", {})
            if "progress_pred" in outputs_progress:
                progress_pred_raw = outputs_progress["progress_pred"]
                for progress_seq in progress_pred_raw:
                    if isinstance(progress_seq, list) and len(progress_seq) > 0:
                        # Take the last progress value in the sequence
                        all_progress.append(float(progress_seq[-1]))
                    else:
                        all_progress.append(0.0)
            else:
                all_progress.extend([0.0] * len(batch_samples))
            
            # Extract success probabilities if available
            outputs_success = batch_outputs.get("outputs_success", {})
            if outputs_success and "success_probs" in outputs_success:
                success_probs_raw = outputs_success["success_probs"]
                for success_seq in success_probs_raw:
                    if isinstance(success_seq, list) and len(success_seq) > 0:
                        # Take the last success probability in the sequence
                        all_success_probs.append(float(success_seq[-1]))
                    else:
                        all_success_probs.append(0.0)
            else:
                all_success_probs.extend([0.0] * len(batch_samples))
        except Exception as e:
            print(f"    Error calling eval server: {e}")
            # Fallback to zeros on error
            all_progress.extend([0.0] * len(batch_samples))
            all_success_probs.extend([0.0] * len(batch_samples))
    
    return all_progress, all_success_probs


def relabel_rewards_with_baseline_eval_server(
    all_frames: list[np.ndarray],
    language_instruction: str,
    episode_id: str,
    eval_server_url: str,
    batch_size: int = 32,
    image_key: str = "image",
    max_frames: int = 16,
    timeout_s: float = 120.0,
) -> tuple[list[float], list[float]]:
    """
    Relabel rewards for a full trajectory using a baseline eval server.
    
    Builds subsequences [0:1], [0:2], [0:3], ... and processes them through the baseline eval server
    to get progress predictions for each timestep.
    
    Args:
        all_frames: List of frames for the full trajectory (each frame is HWC uint8)
        language_instruction: Language instruction string for the episode
        episode_id: Episode ID string
        eval_server_url: URL of the baseline eval server (e.g., "http://localhost:8001")
        batch_size: Batch size for processing subsequences
        image_key: Image key name (for logging)
        max_frames: Maximum number of frames to use per sample
        timeout_s: Request timeout in seconds
    
    Returns:
        Tuple of (progress_predictions, success_probs) - one value per timestep
        Note: Baseline models typically don't return success_probs, so that list will be empty
    """
    if not HAS_REWARD_RELABELING:
        raise ImportError("Reward relabeling not available. Install reward_fm package.")
    if not HAS_REQUESTS:
        raise ImportError("requests package is required for eval server support. Install with: pip install requests")
    
    # Build subsequences: [0:1], [0:2], [0:3], ...
    all_samples = []
    for i in range(len(all_frames)):
        # Build subsequence from start to current step (0 to i+1)
        frames_subsequence = np.array(all_frames[: i + 1])  # Shape: (i+1, H, W, C)
        
        # Prepare raw data dict for baseline model
        raw_data = dict(
            frames=frames_subsequence,
            task=language_instruction,
            id=episode_id,
            metadata=dict(subsequence_length=len(frames_subsequence)),
        )
        
        # Convert to sample
        sample = raw_dict_to_sample(
            raw_data=raw_data,
            max_frames=max_frames,
            sample_type="progress",
        )
        all_samples.append(sample)
    
    # Process subsequences in batches
    all_progress = []
    
    num_batches = (len(all_samples) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(all_samples), batch_size), desc="  Processing batches", leave=False, total=num_batches):
        batch_end = min(batch_start + batch_size, len(all_samples))
        batch_samples = all_samples[batch_start:batch_end]
        
        # Build payload for eval server
        files, sample_data = build_payload(batch_samples)
        
        # Add use_frame_steps=False to data dict since we're already building subsequences
        import json
        data_dict = {f"sample_{i}": json.dumps(sample) for i, sample in enumerate(sample_data)}
        data_dict["use_frame_steps"] = "false"
        
        # Send request to baseline eval server
        try:
            import requests
            url = eval_server_url.rstrip("/") + "/evaluate_batch_npy"
            resp = requests.post(url, files=files, data=data_dict, timeout=timeout_s)
            resp.raise_for_status()
            batch_outputs = resp.json()
            
            # Extract progress predictions from outputs_progress
            outputs_progress = batch_outputs.get("outputs_progress", {})
            if "progress_pred" in outputs_progress:
                progress_pred_raw = outputs_progress["progress_pred"]
                for progress_seq in progress_pred_raw:
                    if isinstance(progress_seq, list) and len(progress_seq) > 0:
                        # Take the last progress value in the sequence
                        all_progress.append(float(progress_seq[-1]))
                    else:
                        all_progress.append(0.0)
            else:
                all_progress.extend([0.0] * len(batch_samples))
        except Exception as e:
            print(f"    Error calling baseline eval server: {e}")
            # Fallback to zeros on error
            all_progress.extend([0.0] * len(batch_samples))
    
    # Baseline models typically don't return success probabilities
    all_success_probs = [0.0] * len(all_progress)
    
    return all_progress, all_success_probs


def load_baseline_model_for_relabeling(
    reward_model_type: str,
    reward_model_path: str | None = None,
    baseline_eval_server_url: str | None = None,
    device: str | None = None,
    max_frames: int = 16,
):
    """
    Load baseline model for relabeling or configure baseline eval server.
    
    Args:
        reward_model_type: Type of baseline model ("gvl", "vlac", "roboreward")
        reward_model_path: Path to model checkpoint (required for VLAC/RoboReward if not using eval server)
        baseline_eval_server_url: URL of the baseline eval server (e.g., "http://localhost:8001")
        device: Device to use for local model (default: cuda if available, else cpu)
        max_frames: Maximum frames for GVL model
    
    Returns:
        Tuple of (baseline_model, None, None, None) for direct model,
        or (None, None, None, None) if using eval server
    """
    if not HAS_BASELINE_MODELS:
        raise ImportError("Baseline models not available. Install reward_fm package.")
    
    if reward_model_type not in ["gvl", "vlac", "roboreward"]:
        raise ValueError(f"Invalid baseline model type: {reward_model_type}")
    
    # Using baseline eval server - return None for all model components
    if baseline_eval_server_url is not None:
        print(f"Using baseline eval server at {baseline_eval_server_url} for reward relabeling")
        # Test connection
        try:
            if not HAS_REQUESTS:
                raise ImportError("requests package is required for eval server support. Install with: pip install requests")
            import requests
            resp = requests.get(f"{baseline_eval_server_url.rstrip('/')}/health", timeout=5.0)
            resp.raise_for_status()
            print("  Baseline eval server connection successful")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to baseline eval server at {baseline_eval_server_url}: {e}")
        return None, None, None, None
    
    # Using direct baseline model loading
    print(f"Loading {reward_model_type} baseline model...")
    if reward_model_type == "gvl":
        baseline_model = GVL(max_frames=max_frames)
    elif reward_model_type == "vlac":
        if reward_model_path is None:
            raise ValueError("--reward-model-path is required for VLAC")
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        baseline_model = VLAC(model_path=reward_model_path, device=device)
    elif reward_model_type == "roboreward":
        baseline_model = RoboReward(
            model_path=reward_model_path or "teetone/RoboReward-4B",
            use_unsloth=False,
        )
    else:
        raise ValueError(f"Unknown baseline model type: {reward_model_type}")
    
    print(f"  {reward_model_type} baseline model loaded successfully")
    return baseline_model, None, None, None


def relabel_rewards_with_model(
    all_frames: list[np.ndarray],
    language_instruction: str,
    episode_id: str,
    reward_model,
    exp_config,
    batch_collator,
    tokenizer,
    sentence_model: SentenceTransformer | None = None,
    batch_size: int = 32,
    image_key: str = "image",
) -> tuple[list[float], list[float]]:
    """
    Relabel rewards for a full trajectory using the reward model.
    
    Builds subsequences [0:1], [0:2], [0:3], ... and processes them through the reward model
    to get progress predictions and success probabilities for each timestep.
    
    Args:
        all_frames: List of frames for the full trajectory (each frame is HWC uint8)
        language_instruction: Language instruction string for the episode
        episode_id: Episode ID string
        reward_model: Loaded reward model
        exp_config: Experiment configuration
        batch_collator: Batch collator for preparing inputs
        tokenizer: Tokenizer for the reward model
        sentence_model: Optional sentence transformer for computing text embeddings
        batch_size: Batch size for processing subsequences
        image_key: Image key name (for logging)
    
    Returns:
        Tuple of (progress_predictions, success_probs) - one value per timestep
    """
    if not HAS_REWARD_RELABELING:
        raise ImportError("Reward relabeling not available. Install reward_fm package.")
    
    # Compute text embedding from language instruction
    text_embedding = None
    if sentence_model is not None:
        text_embedding_tensor = compute_text_embeddings(language_instruction, sentence_model)
        text_embedding = text_embedding_tensor.cpu().numpy()
    else:
        raise ValueError("sentence_model is required for reward relabeling")
    
    # Get max_frames from config
    max_frames = exp_config.data.max_frames if hasattr(exp_config.data, "max_frames") else 16
    
    # Build subsequences: [0:1], [0:2], [0:3], ...
    all_samples = []
    for i in range(len(all_frames)):
        # Build subsequence from start to current step (0 to i+1)
        frames_subsequence = np.array(all_frames[: i + 1])  # Shape: (i+1, H, W, C)
        
        # Prepare raw data dict for reward model
        raw_data = dict(
            frames=frames_subsequence,
            task=language_instruction,
            id=episode_id,
            metadata=dict(subsequence_length=len(frames_subsequence)),
            text_embedding=text_embedding,
        )
        
        # Convert to sample
        sample = raw_dict_to_sample(
            raw_data=raw_data,
            max_frames=max_frames,
            sample_type="progress",
        )
        all_samples.append(sample)
    
    # Process subsequences in batches
    all_progress = []
    all_success_probs = []
    device = reward_model.device
    
    # Infer is_discrete_mode and num_bins from exp_config
    progress_loss_type = exp_config.loss.progress_loss_type if hasattr(exp_config.loss, "progress_loss_type") else "continuous"
    is_discrete_mode = progress_loss_type.lower() == "discrete"
    if is_discrete_mode:
        num_bins = exp_config.loss.progress_discrete_bins if hasattr(exp_config.loss, "progress_discrete_bins") else 10
    else:
        num_bins = None
    
    num_batches = (len(all_samples) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(all_samples), batch_size), desc="  Processing batches", leave=False, total=num_batches):
        batch_end = min(batch_start + batch_size, len(all_samples))
        batch_samples = all_samples[batch_start:batch_end]
        
        # Collate batch using batch collator
        batch_inputs = batch_collator(batch_samples)
        
        # Extract progress_inputs and move to device
        progress_inputs = batch_inputs["progress_inputs"]
        progress_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in progress_inputs.items()
        }
        
        # Compute outputs
        with torch.inference_mode():
            batch_outputs = compute_batch_outputs(
                model=reward_model,
                tokenizer=tokenizer,
                batch_inputs=progress_inputs,
                sample_type="progress",
                is_discrete_mode=is_discrete_mode,
                num_bins=num_bins,
            )
        
        # Extract progress predictions (one per sample - last value in sequence)
        if "progress_pred" in batch_outputs:
            progress_pred_raw = batch_outputs["progress_pred"]
            for progress_seq in progress_pred_raw:
                if isinstance(progress_seq, list) and len(progress_seq) > 0:
                    # Take the last progress value in the sequence
                    all_progress.append(float(progress_seq[-1]))
                else:
                    all_progress.append(0.0)
        else:
            all_progress.extend([0.0] * len(batch_samples))
        
        # Extract success probabilities if available
        if "outputs_success" in batch_outputs:
            success_probs_raw = batch_outputs["outputs_success"]["success_probs"]
            for success_seq in success_probs_raw:
                if isinstance(success_seq, list) and len(success_seq) > 0:
                    # Take the last success probability in the sequence
                    all_success_probs.append(float(success_seq[-1]))
                else:
                    all_success_probs.append(0.0)
        else:
            all_success_probs.extend([0.0] * len(batch_samples))
    
    return all_progress, all_success_probs


def load_reward_model_for_relabeling(
    reward_model_path: str | None = None,
    eval_server_url: str | None = None,
    device: str | None = None,
):
    """
    Load reward model for relabeling or configure eval server.
    
    Either reward_model_path or eval_server_url must be provided, but not both.
    
    Args:
        reward_model_path: Path to reward model checkpoint (HuggingFace model ID or local path)
        eval_server_url: URL of the eval server (e.g., "http://localhost:8001")
        device: Device to use for local model (default: cuda if available, else cpu)
    
    Returns:
        Tuple of (reward_model, exp_config, batch_collator, tokenizer) for direct model,
        or (None, None, None, None) if using eval server
    """
    if not HAS_REWARD_RELABELING:
        raise ImportError("Reward relabeling requested but reward_fm package is not available.")
    
    if reward_model_path is None and eval_server_url is None:
        raise ValueError("Either reward_model_path or eval_server_url must be provided")
    if reward_model_path is not None and eval_server_url is not None:
        raise ValueError("Cannot provide both reward_model_path and eval_server_url")
    
    # Using eval server - return None for all model components
    if eval_server_url is not None:
        print(f"Using eval server at {eval_server_url} for reward relabeling")
        # Test connection
        try:
            if not HAS_REQUESTS:
                raise ImportError("requests package is required for eval server support. Install with: pip install requests")
            import requests
            resp = requests.get(f"{eval_server_url.rstrip('/')}/health", timeout=5.0)
            resp.raise_for_status()
            print("  Eval server connection successful")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to eval server at {eval_server_url}: {e}")
        return None, None, None, None
    
    # Using direct model loading
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    
    print(f"Loading reward model from {reward_model_path} on {device}...")
    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=reward_model_path,
        device=device_obj,
    )
    reward_model = reward_model.to(device_obj)
    reward_model.eval()
    
    # Ensure use_multi_image is True for reward relabeling
    if not exp_config.data.use_multi_image:
        print("  Setting use_multi_image=True for reward relabeling")
        exp_config.data.use_multi_image = True
    
    # Set up batch collator
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)
    
    print("  Reward model loaded successfully")
    return reward_model, exp_config, batch_collator, tokenizer


def print_dataset_info(file_path: str):
    """Print information about the converted dataset."""
    with h5py.File(file_path, "r") as f:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {file_path}")
        print(f"{'=' * 60}")
        print(f"Format version: {f.attrs.get('format_version', 'unknown')}")
        print(f"Total demos: {f.attrs.get('total_demos', 'unknown')}")
        print(f"Includes images: {f.attrs.get('includes_images', 'unknown')}")
        print(f"Normalized: {f.attrs.get('normalized', False)}")
        if "repo_ids" in f.attrs:
            print(f"Source repo_ids: {f.attrs['repo_ids']}")
        if "image_height" in f.attrs and "image_width" in f.attrs:
            print(f"Image dimensions: {f.attrs['image_height']}x{f.attrs['image_width']} (HxW)")
        elif "image_size" in f.attrs:
            print(f"Image size: {f.attrs['image_size']}x{f.attrs['image_size']} (square)")
        if "rewards_relabeled" in f.attrs and f.attrs["rewards_relabeled"]:
            print(f"Rewards relabeled: True")
            if "reward_model_type" in f.attrs:
                print(f"Reward model type: {f.attrs['reward_model_type']}")
            if "reward_model_path" in f.attrs:
                print(f"Reward model: {f.attrs['reward_model_path']}")
            if "eval_server_url" in f.attrs:
                print(f"Eval server: {f.attrs['eval_server_url']}")
            if "baseline_eval_server_url" in f.attrs:
                print(f"Baseline eval server: {f.attrs['baseline_eval_server_url']}")
        
        # Show normalization stats if present
        if "normalization" in f:
            print(f"\nNormalization stats:")
            norm_group = f["normalization"]
            # Show attributes (like action_norm_mode)
            for attr_key in norm_group.attrs.keys():
                print(f"  {attr_key}: {norm_group.attrs[attr_key]}")
            # Show datasets
            for key in norm_group.keys():
                data = norm_group[key][:]
                print(f"  {key}: shape={data.shape}, range=[{data.min():.4f}, {data.max():.4f}]")
        
        data_group = f["data"]
        demo_ids = list(data_group.keys())
        print(f"\nNumber of demos: {len(demo_ids)}")
        
        # Sample first demo
        if len(demo_ids) > 0:
            first_demo = demo_ids[0]
            print(f"\nSample demo: {first_demo}")
            demo_group = data_group[first_demo]
            
            print(f"  Episode index: {demo_group.attrs.get('episode_index', 'N/A')}")
            print(f"  Num samples: {demo_group.attrs.get('num_samples', 'N/A')}")
            
            if "language_instruction" in demo_group:
                instruction = demo_group["language_instruction"][()]
                if isinstance(instruction, bytes):
                    instruction = instruction.decode("utf-8")
                print(f'  Language instruction: "{instruction}"')
            
            if "actions" in demo_group:
                print(f"  Actions shape: {demo_group['actions'].shape}")
            
            if "obs" in demo_group:
                print(f"  Observations:")
                for key in demo_group["obs"].keys():
                    print(f"    {key}: {demo_group['obs'][key].shape}")
            
            if "next_obs" in demo_group:
                print(f"  Next observations:")
                for key in demo_group["next_obs"].keys():
                    print(f"    {key}: {demo_group['next_obs'][key].shape}")
            
            if "rewards" in demo_group:
                print(f"  Rewards shape: {demo_group['rewards'].shape}")
            if "dones" in demo_group:
                print(f"  Dones shape: {demo_group['dones'].shape}")

