#!/usr/bin/env python3
"""VLAC baseline for progress prediction.

VLAC (Vision-Language-Action-Critic) is a general-purpose pair-wise critic model
designed for real-world robot reinforcement learning and data refinement.
It provides robust evaluation capabilities for task progress prediction and
task completion verification based on images and task descriptions.

Reference: https://github.com/InternRobotics/VLAC
Model: https://huggingface.co/InternRobotics/VLAC

Downloading the model:
    # Option 1: Use Hugging Face CLI
    pip install huggingface_hub
    huggingface-cli download InternRobotics/VLAC
    
    # Option 2: Use Python API (automatic in VLAC.__init__)
    from rfm.evals.baselines.vlac import VLAC
    model = VLAC(model_path="InternRobotics/VLAC")  # Auto-downloads if not found
    
    # Option 3: Use download function directly
    from rfm.evals.baselines.vlac import download_vlac_model
    model_path = download_vlac_model(local_dir="./models/vlac")
"""

import os
import tempfile
from typing import List, Dict, Optional
import numpy as np
import cv2

from evo_vlac import GAC_model
from evo_vlac.utils.video_tool import compress_video

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def download_vlac_model(
    repo_id: str = "InternRobotics/VLAC",
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None
) -> str:
    """
    Download VLAC model from Hugging Face.
    
    Args:
        repo_id: Hugging Face repository ID (default: "InternRobotics/VLAC")
        cache_dir: Optional cache directory for Hugging Face downloads
        local_dir: Optional local directory to download model to (if None, uses cache)
    
    Returns:
        Path to the downloaded model directory
    
    Example:
        >>> model_path = download_vlac_model()
        >>> model_path = download_vlac_model(local_dir="./models/vlac")
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required to download models. Install with: pip install huggingface_hub"
        )
    
    print(f"Downloading VLAC model from {repo_id}...")
    model_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_dir_use_symlinks=False  # Use actual files, not symlinks
    )
    print(f"Model downloaded to: {model_path}")
    return model_path


class VLAC:
    """VLAC baseline for progress prediction using pair-wise comparison.
    
    VLAC uses a pair-wise comparison mechanism to predict task progress
    for each frame in a trajectory. It requires a local model checkpoint
    to be loaded.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        model_type: str = "internvl2",
        temperature: float = 0.5,
        top_k: int = 1,
        batch_num: int = 5,
        skip: int = 5,
        frame_skip: bool = True,
        auto_download: bool = True
    ):
        """
        Initialize VLAC model.
        
        Args:
            model_path: Path to local VLAC model checkpoint, or Hugging Face repo ID 
                       (e.g., "InternRobotics/VLAC"). If repo ID and model doesn't exist 
                       locally, will download automatically if auto_download=True.
            device: Device to run model on (e.g., "cuda:0")
            model_type: Model type (default: "internvl2")
            temperature: Temperature for generation
            top_k: Top-k sampling
            batch_num: Batch number for processing
            skip: Pair-wise step size
            frame_skip: Whether to skip frames for efficiency
            auto_download: If True and model_path is a Hugging Face repo ID, automatically 
                          download the model if not found locally
        """
        # Check if model_path exists locally first
        if not os.path.exists(model_path):
            # If not found locally and looks like a Hugging Face repo ID (contains "/" but not an absolute path)
            # and auto_download is enabled, download from Hugging Face
            is_hf_repo_id = (
                "/" in model_path and 
                not os.path.isabs(model_path) and 
                not model_path.startswith(".") and
                not model_path.startswith("~")
            )
            
            if is_hf_repo_id and auto_download:
                if not HF_AVAILABLE:
                    raise ImportError(
                        "huggingface_hub is required to download models. Install with: pip install huggingface_hub"
                    )
                print(f"Model path '{model_path}' not found locally. Downloading from Hugging Face...")
                model_path = download_vlac_model(repo_id=model_path)
            elif not is_hf_repo_id:
                raise FileNotFoundError(
                    f"Model path '{model_path}' does not exist. "
                    f"To download from Hugging Face, use model_path='InternRobotics/VLAC'"
                )
        
        self.model_path = model_path
        self.device = device
        self.model_type = model_type
        self.temperature = temperature
        self.top_k = top_k
        self.batch_num = batch_num
        self.skip = skip
        self.frame_skip = frame_skip
        
        # Initialize model
        self.critic = GAC_model(tag='critic')
        self.critic.init_model(
            model_path=model_path,
            model_type=model_type,
            device_map=device
        )
        self.critic.temperature = temperature
        self.critic.top_k = top_k
        self.critic.set_config()
        self.critic.set_system_prompt()
    
    def compute_progress(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: Optional[str] = None
    ) -> List[Optional[float]]:
        """
        Compute progress predictions for frames using VLAC baseline.
        
        Args:
            frames_array: (N, H, W, 3) uint8 array from trajectory frames
            task_description: Task description text
            reference_video_path: Optional path to reference video for in-context learning
            
        Returns:
            List of task completion percentages (0-100) for each frame.
            None values indicate frames where prediction failed.
        """
        if frames_array is None or frames_array.size == 0:
            return []
        
        # Create temporary video file from frames
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "trajectory.mp4")
            self._frames_to_video(frames_array, video_path, fps=5.0)
            
            # Compress video (VLAC expects compressed video)
            compressed_video = os.path.join(tmpdir, "compressed.mp4")
            _, output_fps = compress_video(video_path, compressed_video, fps=5.0)
            
            # Run VLAC trajectory critic
            result_path, value_list, critic_list, done_list = self.critic.web_trajectory_critic(
                task_description=task_description,
                main_video_path=compressed_video,
                reference_video_path=reference_video_path,
                batch_num=self.batch_num,
                ref_num=6 if reference_video_path else 0,
                think=False,
                skip=self.skip,
                rich=False,
                reverse_eval=False,
                output_path=tmpdir,
                fps=float(output_fps),
                frame_skip=self.frame_skip,
                done_flag=False,
                in_context_done=False,
                done_threshold=0.9,
                video_output=False
            )
            
            # Extract progress values from value_list
            # value_list contains progress predictions for each frame
            # VLAC returns values in [0, 1] range, we need to convert to [0, 100]
            if value_list and len(value_list) > 0:
                # Convert to list of floats, handling any None values
                progress_list = []
                for val in value_list:
                    if val is not None:
                        if isinstance(val, (int, float)):
                            # VLAC typically returns values in [0, 1] range
                            # Convert to [0, 100] if needed
                            if val <= 1.0:
                                progress_list.append(float(val * 100))
                            else:
                                # Already in 0-100 range
                                progress_list.append(float(val))
                        else:
                            progress_list.append(None)
                    else:
                        progress_list.append(None)
                
                # Ensure we have predictions for all frames
                num_frames = frames_array.shape[0]
                if len(progress_list) < num_frames:
                    # Pad with None
                    progress_list.extend([None] * (num_frames - len(progress_list)))
                elif len(progress_list) > num_frames:
                    # Truncate to match frame count
                    progress_list = progress_list[:num_frames]
                
                return progress_list
            else:
                # Return None for all frames if no predictions
                return [None] * frames_array.shape[0]
    
    def _frames_to_video(self, frames_array: np.ndarray, output_path: str, fps: float = 5.0):
        """Convert frames array to video file."""
        height, width = frames_array.shape[1], frames_array.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames_array:
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()


if __name__ == "__main__":
    """Command-line interface for downloading VLAC model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download VLAC model from Hugging Face")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="InternRobotics/VLAC",
        help="Hugging Face repository ID (default: InternRobotics/VLAC)"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Local directory to download model to (default: uses Hugging Face cache)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for Hugging Face downloads (default: uses default cache)"
    )
    
    args = parser.parse_args()
    
    try:
        model_path = download_vlac_model(
            repo_id=args.repo_id,
            cache_dir=args.cache_dir,
            local_dir=args.local_dir
        )
        print(f"\n✓ Model successfully downloaded to: {model_path}")
        print(f"\nYou can now use it with:")
        print(f"  VLAC(model_path='{model_path}')")
        if args.local_dir:
            print(f"  or")
            print(f"  VLAC(model_path='{args.local_dir}')")
    except Exception as e:
        print(f"Error downloading model: {e}")
        exit(1)

