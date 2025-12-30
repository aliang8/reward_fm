#!/usr/bin/env python3
"""
Script to run RFM evaluation on videos/images.
Supports three execution modes: Gradio, Azure server (direct), or local model.
Supports two evaluation types: progress (single video) and preference (two videos).

Features:
  - Task inference: If --task is not specified, the task name is automatically inferred from the video filename.
    For filenames with commas (e.g., "put_battery_in_red_bowl,fail_left.mp4"), the task is everything before
    the comma. Otherwise, the task is derived by splitting on underscores and removing success/fail/failure suffixes.
  - Batch processing: When --video points to a directory, all video files in that directory are processed sequentially.
    Progress is shown for each video, and a summary of predictions is printed at the end.

Usage Examples:

  # Progress evaluation via Gradio
  uv run python rfm/evals/run_eval.py --mode gradio --eval-type progress \
      --gradio-url https://rewardfm-rewardeval-ui.hf.space \
      --output-dir ./my_results \
      --video video.mp4 --task "Pick up the red block" \
      --server-url http://40.119.56.66:8000 --fps 1.0

  # Progress evaluation via Azure server (direct)
  uv run python rfm/evals/run_eval.py --mode server --eval-type progress \
      --server-url http://40.119.56.66:8000 \
      --output-dir ./my_results \
      --video video.mp4 --task "Pick up the red block" --fps 1.0

  # Progress evaluation via local model
  uv run python rfm/evals/run_eval.py --mode local --eval-type progress \
      --model-path rewardfm/pref_prog_2frames_all \
      --video video.mp4 --task "Pick up the red block" --fps 1.0 

  # Preference evaluation via Gradio
  uv run python rfm/evals/run_eval.py --mode gradio --eval-type preference \
      --gradio-url https://rewardfm-rewardeval-ui.hf.space \
      --video-a video1.mp4 --video-b video2.mp4 --task "Pick up the red block" \
      --server-url http://40.119.56.66:8000

  # Preference evaluation via server
  uv run python rfm/evals/run_eval.py --mode server --eval-type preference \
      --server-url http://40.119.56.66:8000 \
      --video-a video1.mp4 --video-b video2.mp4 --task "Pick up the red block"
"""

import argparse
import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

import numpy as np
import matplotlib
import requests

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import decord

from rfm.data.dataset_types import Trajectory, ProgressSample, PreferenceSample
from rfm.evals.eval_utils import (
    build_payload,
    post_batch_npy,
    find_video_files,
    infer_task_from_video_name,
    setup_output_directory,
)
from rfm.evals.eval_viz_utils import create_combined_progress_success_plot, extract_frames, create_comparison_plot

from gradio_client import Client, file as gradio_file
import torch
from rfm.utils.save import load_model_from_hf
from rfm.utils.setup_utils import setup_batch_collator
from rfm.evals.eval_server import forward_model


def save_info_file(
    save_dir: str,
    filename_base: str,
    video_path: Optional[str] = None,
    video_a_path: Optional[str] = None,
    video_b_path: Optional[str] = None,
    task_text: str = "",
    prediction_type: Optional[str] = None,
    fps: float = 1.0,
    info_text: str = "",
    **kwargs,
) -> str:
    """Save inference info to a text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    info_file = os.path.join(save_dir, f"{filename_base}_info_{timestamp}.txt")

    try:
        with open(info_file, "w") as f:
            if video_path:
                f.write(f"Video: {video_path}\n")
            if video_a_path:
                f.write(f"Video A: {video_a_path}\n")
            if video_b_path:
                f.write(f"Video B: {video_b_path}\n")
            f.write(f"Task: {task_text}\n")
            if prediction_type:
                f.write(f"Prediction Type: {prediction_type}\n")
            f.write(f"FPS: {fps}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            for key, value in kwargs.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n{'=' * 60}\n")
            f.write("INFERENCE RESULTS\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(info_text)
        return info_file
    except Exception as e:
        print(f"Warning: Could not save info file: {e}")
        return None


def run_progress_gradio(
    gradio_url: str,
    video_path: str,
    task_text: str,
    server_url: str,
    fps: float,
    output_dir: str,
) -> Dict[str, Any]:
    """Run progress evaluation via Gradio API."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Connecting to Gradio app at {gradio_url}...")
    client = Client(gradio_url)

    video_path = os.path.abspath(video_path)
    video_file = gradio_file(video_path)

    print(f"Running inference on video: {video_path}")
    result = client.predict(
        video_file,
        task_text,
        server_url,
        fps,
        api_name="/process_single_video",
    )

    prog_succ_plot_path, info_text = result

    # Download and save files locally
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = Path(video_path).stem

    local_prog_succ_plot = os.path.join(output_dir, f"{video_name}_prog_succ_{timestamp}.png")

    if prog_succ_plot_path:
        local_prog_succ_plot = download_gradio_file(client, prog_succ_plot_path, local_prog_succ_plot, gradio_url)

    info_file = save_info_file(
        output_dir, video_name, video_path=video_path, task_text=task_text, fps=fps, info_text=info_text
    )

    return {
        "prog_succ_plot": local_prog_succ_plot,
        "info": info_text,
        "info_file": info_file,
    }


def run_preference_gradio(
    gradio_url: str,
    video_a_path: str,
    video_b_path: str,
    task_text: str,
    server_url: str,
    fps: float,
    output_dir: str,
) -> Dict[str, Any]:
    """Run preference evaluation via Gradio API."""
    if not os.path.exists(video_a_path):
        raise FileNotFoundError(f"Video A file not found: {video_a_path}")
    if not os.path.exists(video_b_path):
        raise FileNotFoundError(f"Video B file not found: {video_b_path}")

    print(f"Connecting to Gradio app at {gradio_url}...")
    client = Client(gradio_url)

    video_a_path = os.path.abspath(video_a_path)
    video_b_path = os.path.abspath(video_b_path)
    video_a_file = gradio_file(video_a_path)
    video_b_file = gradio_file(video_b_path)

    print(f"Running preference inference on videos:")
    result = client.predict(
        video_a_file,
        video_b_file,
        task_text,
        "preference",
        server_url,
        fps,
        api_name="/process_dual_videos",
    )

    result_text, comparison_plot_path = result

    # Download and save files locally
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_a_name = Path(video_a_path).stem
    video_b_name = Path(video_b_path).stem

    local_comparison_plot = os.path.join(output_dir, f"{video_a_name}_vs_{video_b_name}_preference_{timestamp}.png")

    if comparison_plot_path:
        local_comparison_plot = download_gradio_file(client, comparison_plot_path, local_comparison_plot, gradio_url)

    info_file = save_info_file(
        output_dir,
        f"{video_a_name}_vs_{video_b_name}_preference",
        video_a_path=video_a_path,
        video_b_path=video_b_path,
        task_text=task_text,
        prediction_type="preference",
        fps=fps,
        info_text=result_text,
    )

    return {
        "result": result_text,
        "comparison_plot": local_comparison_plot,
        "info_file": info_file,
    }


def get_model_info(server_url: str) -> None:
    """Ping the server to check health and get model information."""
    try:
        # Check health
        health_url = server_url.rstrip("/") + "/health"
        health_response = requests.get(health_url, timeout=5.0)
        health_response.raise_for_status()
        health_data = health_response.json()

        available_gpus = health_data.get("available_gpus", 0)
        total_gpus = health_data.get("total_gpus", 0)
        print(f"  ✓ Server health check passed: {available_gpus}/{total_gpus} GPUs available")

        # Get model info
        model_info_url = server_url.rstrip("/") + "/model_info"
        model_info_response = requests.get(model_info_url, timeout=5.0)
        if model_info_response.status_code == 200:
            model_info = model_info_response.json()

            print(f"  Model Information:")
            print(f"    - Model Path: {model_info.get('model_path', 'Unknown')}")
            print(f"    - Number of GPUs: {model_info.get('num_gpus', 'Unknown')}")

            model_arch = model_info.get("model_architecture", {})
            if model_arch and "error" not in model_arch:
                model_class = model_arch.get("model_class", "Unknown")
                total_params = model_arch.get("total_parameters")
                if total_params is not None:
                    print(f"    - Model Class: {model_class}")
                    print(f"    - Total Parameters: {total_params:,}")

                    trainable_params = model_arch.get("trainable_parameters")
                    if trainable_params is not None:
                        print(f"    - Trainable Parameters: {trainable_params:,}")
        else:
            print(f"  ⚠ Could not fetch model info (status: {model_info_response.status_code})")

    except requests.exceptions.RequestException as e:
        print(f"  ⚠ Could not ping server: {e}")


def download_gradio_file(client: Client, remote_path: str, local_path: str, gradio_url: str = None) -> str:
    """Download a file from Gradio server to local path."""
    if remote_path is None:
        return None

    try:
        if os.path.exists(remote_path) and not remote_path.startswith("http"):
            shutil.copy2(remote_path, local_path)
            return local_path

        if hasattr(client, "download"):
            try:
                downloaded = client.download(remote_path, local_path)
                if downloaded and os.path.exists(downloaded):
                    if downloaded != local_path and os.path.exists(downloaded):
                        shutil.copy2(downloaded, local_path)
                    return local_path
            except Exception as e:
                print(f"  Note: client.download() failed, trying alternative: {e}")

        if remote_path.startswith("http"):
            response = requests.get(remote_path, timeout=30)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(response.content)
                return local_path

        if gradio_url:
            base_url = gradio_url.rstrip("/")
        else:
            base_url = str(client.src) if hasattr(client, "src") else str(client)
            base_url = base_url.rstrip("/")

        file_path = remote_path if remote_path.startswith("/") else "/" + remote_path
        for url_format in [f"{base_url}/file={file_path}", f"{base_url}{file_path}"]:
            try:
                response = requests.get(url_format, timeout=30)
                if response.status_code == 200:
                    with open(local_path, "wb") as f:
                        f.write(response.content)
                    return local_path
            except:
                continue

        print(f"Warning: Could not download {remote_path}, using original path")
        return remote_path
    except Exception as e:
        print(f"Warning: Could not download {remote_path}: {e}")
        return remote_path


def run_progress_server(
    server_url: str,
    video_path: str,
    task_text: str,
    fps: float,
    output_dir: str,
) -> Dict[str, Any]:
    """Run progress evaluation directly with Azure server."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Connecting to eval server at {server_url}...")
    get_model_info(server_url)

    # Extract frames using helper function
    frames_array = extract_frames(video_path, fps=fps)
    if frames_array is None or frames_array.size == 0:
        raise ValueError("Could not extract frames from video.")

    if frames_array.dtype != np.uint8:
        frames_array = np.clip(frames_array, 0, 255).astype(np.uint8)

    num_frames = frames_array.shape[0]
    frames_shape = frames_array.shape

    # Create target progress and success labels
    target_progress = np.linspace(0.0, 1.0, num=num_frames).tolist()
    success_label = [1.0 if prog > 0.5 else 0.0 for prog in target_progress]

    # Create Trajectory and ProgressSample
    trajectory = Trajectory(
        task=task_text,
        frames=frames_array,
        frames_shape=frames_shape,
        target_progress=target_progress,
        success_label=success_label,
        metadata={"source": "eval_script"},
    )

    progress_sample = ProgressSample(
        trajectory=trajectory,
        data_gen_strategy="demo",
    )

    # Build payload and send to server using helper function
    files, sample_data = build_payload([progress_sample])
    response = post_batch_npy(server_url, files, sample_data, timeout_s=120.0)

    # Process response
    outputs_progress = response.get("outputs_progress", {})
    progress_pred = outputs_progress.get("progress_pred", [])
    outputs_success = response.get("outputs_success", {})
    success_probs = outputs_success.get("success_probs", []) if outputs_success else None

    # Extract predictions
    if progress_pred and len(progress_pred) > 0:
        progress_array = np.array(progress_pred[0])
    else:
        progress_array = np.array([])

    if success_probs and len(success_probs) > 0:
        success_array = np.array(success_probs[0])
    else:
        success_array = None

    # Create combined plot using shared helper function
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = Path(video_path).stem

    # Convert success_array to binary if available
    success_binary = None
    if success_array is not None:
        success_binary = (success_array > 0.5).astype(float)

    # Create combined plot
    fig = create_combined_progress_success_plot(
        progress_pred=progress_array if len(progress_array) > 0 else np.array([0.0]),
        num_frames=num_frames,
        success_binary=success_binary,
        success_probs=success_array,
        success_labels=None,  # No ground truth labels available
        is_discrete_mode=False,
        num_bins=10,
        title=f"Progress & Success - {video_name}",
    )

    # Save the combined plot
    prog_succ_plot_path = os.path.join(output_dir, f"{video_name}_prog_succ_{timestamp}.png")
    fig.savefig(prog_succ_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Create info text
    info_text = f"**Frames processed:** {num_frames}\n"
    if len(progress_array) > 0:
        info_text += f"**Final progress:** {progress_array[-1]:.3f}\n"
    if success_array is not None and len(success_array) > 0:
        info_text += f"**Final success probability:** {success_array[-1]:.3f}\n"

    info_file = save_info_file(
        output_dir, video_name, video_path=video_path, task_text=task_text, fps=fps, info_text=info_text
    )

    return {
        "prog_succ_plot": prog_succ_plot_path,
        "info": info_text,
        "info_file": info_file,
    }


def run_preference_server(
    server_url: str,
    video_a_path: str,
    video_b_path: str,
    task_text: str,
    fps: float,
    output_dir: str,
) -> Dict[str, Any]:
    """Run preference evaluation directly with Azure server."""
    if not os.path.exists(video_a_path):
        raise FileNotFoundError(f"Video A file not found: {video_a_path}")
    if not os.path.exists(video_b_path):
        raise FileNotFoundError(f"Video B file not found: {video_b_path}")

    print(f"Connecting to eval server at {server_url}...")
    get_model_info(server_url)

    # Extract frames using helper function
    frames_array_a = extract_frames(video_a_path, fps=fps)
    frames_array_b = extract_frames(video_b_path, fps=fps)

    if frames_array_a is None or frames_array_a.size == 0:
        raise ValueError("Could not extract frames from video A.")
    if frames_array_b is None or frames_array_b.size == 0:
        raise ValueError("Could not extract frames from video B.")

    if frames_array_a.dtype != np.uint8:
        frames_array_a = np.clip(frames_array_a, 0, 255).astype(np.uint8)
    if frames_array_b.dtype != np.uint8:
        frames_array_b = np.clip(frames_array_b, 0, 255).astype(np.uint8)

    num_frames_a = frames_array_a.shape[0]
    num_frames_b = frames_array_b.shape[0]

    # Create trajectories
    target_progress_a = np.linspace(0.0, 1.0, num=num_frames_a).tolist()
    target_progress_b = np.linspace(0.0, 1.0, num=num_frames_b).tolist()
    success_label_a = [1.0 if prog > 0.5 else 0.0 for prog in target_progress_a]
    success_label_b = [1.0 if prog > 0.5 else 0.0 for prog in target_progress_b]

    trajectory_a = Trajectory(
        task=task_text,
        frames=frames_array_a,
        frames_shape=frames_array_a.shape,
        target_progress=target_progress_a,
        success_label=success_label_a,
        metadata={"source": "eval_script", "trajectory": "A"},
    )

    trajectory_b = Trajectory(
        task=task_text,
        frames=frames_array_b,
        frames_shape=frames_array_b.shape,
        target_progress=target_progress_b,
        success_label=success_label_b,
        metadata={"source": "eval_script", "trajectory": "B"},
    )

    # Create preference sample
    sample = PreferenceSample(
        chosen_trajectory=trajectory_a,
        rejected_trajectory=trajectory_b,
        data_gen_strategy="demo",
    )

    # Build payload and send to server
    files, sample_data = build_payload([sample])

    response = post_batch_npy(server_url, files, sample_data, timeout_s=120.0)

    # Process response
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_a_name = Path(video_a_path).stem
    video_b_name = Path(video_b_path).stem

    outputs_preference = response.get("outputs_preference", {})
    prediction_probs = outputs_preference.get("prediction_probs", [])

    result_text = f"**Preference Prediction:**\n"
    if prediction_probs and len(prediction_probs) > 0:
        prob = prediction_probs[0]
        result_text += f"- Probability (A preferred): {prob:.3f}\n"
        result_text += f"- Interpretation: {'Video A is preferred' if prob > 0.5 else 'Video B is preferred'}\n"
    else:
        result_text += "Could not extract preference prediction from server response.\n"

    # Create comparison plot using helper function
    frames_a_list = [Image.fromarray(frame) for frame in frames_array_a]
    frames_b_list = [Image.fromarray(frame) for frame in frames_array_b]
    comparison_plot_file = create_comparison_plot(frames_a_list, frames_b_list, "preference")

    comparison_plot_path = os.path.join(output_dir, f"{video_a_name}_vs_{video_b_name}_preference_{timestamp}.png")
    shutil.copy2(comparison_plot_file, comparison_plot_path)
    os.unlink(comparison_plot_file)  # Clean up temp file

    info_file = save_info_file(
        output_dir,
        f"{video_a_name}_vs_{video_b_name}_preference",
        video_a_path=video_a_path,
        video_b_path=video_b_path,
        task_text=task_text,
        prediction_type="preference",
        fps=fps,
        info_text=result_text,
    )

    return {
        "result": result_text,
        "comparison_plot": comparison_plot_path,
        "info_file": info_file,
    }


def load_model_for_inference(model_path: str, output_dir: str, device: str = "cuda") -> Dict[str, Any]:
    """Load model and setup batch collator for inference.

    Returns:
        Dictionary with keys: exp_cfg, tokenizer, processor, model, batch_collator
    """
    print(f"Loading model from: {model_path}...")
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    exp_cfg, tokenizer, processor, model = load_model_from_hf(model_path=model_path, device=device_obj)
    print(f"✅ Model loaded successfully")

    # Setup batch collator for inference
    batch_collator = setup_batch_collator(processor, tokenizer, exp_cfg, is_eval=True)
    model.eval()

    return {
        "exp_cfg": exp_cfg,
        "tokenizer": tokenizer,
        "processor": processor,
        "model": model,
        "batch_collator": batch_collator,
    }


def run_progress_model(
    video_path: str,
    task_text: str,
    fps: float,
    output_dir: str,
    model_components: Dict[str, Any],
) -> Dict[str, Any]:
    """Run progress evaluation using pre-loaded local model.

    Args:
        video_path: Path to video file
        task_text: Task description
        fps: Frames per second
        output_dir: Output directory
        model_components: Dictionary with loaded model components (from load_model_for_inference)
    """
    model = model_components["model"]
    batch_collator = model_components["batch_collator"]
    device = next(model.parameters()).device

    # Extract frames
    frames_array = extract_frames(video_path, fps=fps)
    if frames_array is None or frames_array.size == 0:
        raise ValueError("Could not extract frames from video.")

    if frames_array.dtype != np.uint8:
        frames_array = np.clip(frames_array, 0, 255).astype(np.uint8)

    num_frames = frames_array.shape[0]
    frames_shape = frames_array.shape

    # Create trajectory
    target_progress = np.linspace(0.0, 1.0, num=num_frames).tolist()
    success_label = [1.0 if prog > 0.5 else 0.0 for prog in target_progress]

    trajectory = Trajectory(
        task=task_text,
        frames=frames_array,
        frames_shape=frames_shape,
        target_progress=target_progress,
        success_label=success_label,
        metadata={"source": "eval_script"},
    )

    progress_sample = ProgressSample(
        trajectory=trajectory,
        data_gen_strategy="demo",
    )

    # Prepare batch inputs
    batch_inputs = batch_collator([progress_sample])

    # Move inputs to device
    progress_inputs = batch_inputs["progress_inputs"]
    for key, value in progress_inputs.items():
        if isinstance(value, torch.Tensor):
            progress_inputs[key] = value.to(device)

    # Run forward pass using forward_model
    model_output, _ = forward_model(model, progress_inputs, sample_type="progress")

    # Extract progress predictions (similar to compute_batch_outputs in eval_server.py)
    progress_array = np.array([])
    success_array = None

    progress_logits = model_output.progress_logits
    if progress_logits is not None and isinstance(progress_logits, dict):
        seq_A = progress_logits.get("A")
        if seq_A is not None:
            # Extract first (and only) sample's progress predictions
            if seq_A.shape[0] > 0:
                # Convert to float32 to handle BFloat16 tensors
                progress_pred = seq_A[0].detach().cpu().float().flatten().tolist()
                progress_array = np.array(progress_pred)

    # Extract success predictions if the model has a success head
    success_logits = getattr(model_output, "success_logits", None)
    if success_logits is not None and isinstance(success_logits, dict):
        seq_A_success = success_logits.get("A")
        if seq_A_success is not None:
            # Extract first (and only) sample's success logits
            if seq_A_success.shape[0] > 0:
                # Convert to float32 to handle BFloat16 tensors, then convert logits to probabilities using sigmoid
                success_logits_sample = seq_A_success[0].detach().cpu().float()
                success_probs = torch.sigmoid(success_logits_sample)
                success_array = success_probs.flatten().numpy()

    # Create combined plot using shared helper function
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = Path(video_path).stem

    # Convert success_array to binary if available
    success_binary = None
    if success_array is not None:
        success_binary = (success_array > 0.5).astype(float)

    # Create combined plot
    fig = create_combined_progress_success_plot(
        progress_pred=progress_array if len(progress_array) > 0 else np.array([0.0]),
        num_frames=num_frames,
        success_binary=success_binary,
        success_probs=success_array,
        success_labels=None,  # No ground truth labels available
        is_discrete_mode=False,
        num_bins=10,
        title=f"Progress & Success - {video_name}",
    )

    # Save the combined plot
    prog_succ_plot_path = os.path.join(output_dir, f"{video_name}_prog_succ_{timestamp}.png")
    fig.savefig(prog_succ_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    info_text = f"**Frames processed:** {num_frames}\n"
    if len(progress_array) > 0:
        info_text += f"**Final progress:** {progress_array[-1]:.3f}\n"
    if success_array is not None and len(success_array) > 0:
        info_text += f"**Final success probability:** {success_array[-1]:.3f}\n"

    info_file = save_info_file(
        output_dir, video_name, video_path=video_path, task_text=task_text, fps=fps, info_text=info_text
    )

    return {
        "prog_succ_plot": prog_succ_plot_path,
        "info": info_text,
        "info_file": info_file,
    }


def run_preference_model(
    video_a_path: str,
    video_b_path: str,
    task_text: str,
    fps: float,
    output_dir: str,
    model_components: Dict[str, Any],
) -> Dict[str, Any]:
    """Run preference evaluation using pre-loaded local model.

    Args:
        video_a_path: Path to first video file
        video_b_path: Path to second video file
        task_text: Task description
        fps: Frames per second
        output_dir: Output directory
        model_components: Dictionary with loaded model components (from load_model_for_inference)
    """
    model = model_components["model"]
    batch_collator = model_components["batch_collator"]
    device = next(model.parameters()).device

    # Extract frames from both videos
    frames_array_a = extract_frames(video_a_path, fps=fps)
    frames_array_b = extract_frames(video_b_path, fps=fps)

    if frames_array_a is None or frames_array_a.size == 0:
        raise ValueError("Could not extract frames from video A.")
    if frames_array_b is None or frames_array_b.size == 0:
        raise ValueError("Could not extract frames from video B.")

    if frames_array_a.dtype != np.uint8:
        frames_array_a = np.clip(frames_array_a, 0, 255).astype(np.uint8)
    if frames_array_b.dtype != np.uint8:
        frames_array_b = np.clip(frames_array_b, 0, 255).astype(np.uint8)

    num_frames_a = frames_array_a.shape[0]
    num_frames_b = frames_array_b.shape[0]

    # Create trajectories
    target_progress_a = np.linspace(0.0, 1.0, num=num_frames_a).tolist()
    target_progress_b = np.linspace(0.0, 1.0, num=num_frames_b).tolist()
    success_label_a = [1.0 if prog > 0.5 else 0.0 for prog in target_progress_a]
    success_label_b = [1.0 if prog > 0.5 else 0.0 for prog in target_progress_b]

    trajectory_a = Trajectory(
        task=task_text,
        frames=frames_array_a,
        frames_shape=frames_array_a.shape,
        target_progress=target_progress_a,
        success_label=success_label_a,
        metadata={"source": "eval_script", "trajectory": "A"},
    )

    trajectory_b = Trajectory(
        task=task_text,
        frames=frames_array_b,
        frames_shape=frames_array_b.shape,
        target_progress=target_progress_b,
        success_label=success_label_b,
        metadata={"source": "eval_script", "trajectory": "B"},
    )

    # Create preference sample
    preference_sample = PreferenceSample(
        chosen_trajectory=trajectory_a,
        rejected_trajectory=trajectory_b,
        data_gen_strategy="demo",
    )

    # Prepare batch inputs
    batch_inputs = batch_collator([preference_sample])

    # Move inputs to device
    preference_inputs = batch_inputs["preference_inputs"]
    for key, value in preference_inputs.items():
        if isinstance(value, torch.Tensor):
            preference_inputs[key] = value.to(device)

    # Run forward pass using forward_model
    model_output, _ = forward_model(model, preference_inputs, sample_type="preference")

    # Extract preference predictions (similar to compute_batch_outputs in eval_server.py)
    prediction_prob = None
    if model_output.pref_logits is not None:
        logits = model_output.pref_logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        # Extract first (and only) sample's preference probability
        if probs.shape[0] > 0:
            prediction_prob = float(probs[0].detach().cpu().float().item())

    # Create result text
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_a_name = Path(video_a_path).stem
    video_b_name = Path(video_b_path).stem

    result_text = f"**Preference Prediction:**\n"
    if prediction_prob is not None:
        result_text += f"- Probability (A preferred): {prediction_prob:.3f}\n"
        result_text += (
            f"- Interpretation: {'Video A is preferred' if prediction_prob > 0.5 else 'Video B is preferred'}\n"
        )
    else:
        result_text += "Could not extract preference prediction from model output.\n"

    # Create comparison plot using helper function
    frames_a_list = [Image.fromarray(frame) for frame in frames_array_a]
    frames_b_list = [Image.fromarray(frame) for frame in frames_array_b]
    comparison_plot_file = create_comparison_plot(frames_a_list, frames_b_list, "preference")

    comparison_plot_path = os.path.join(output_dir, f"{video_a_name}_vs_{video_b_name}_preference_{timestamp}.png")
    shutil.copy2(comparison_plot_file, comparison_plot_path)
    os.unlink(comparison_plot_file)  # Clean up temp file

    info_file = save_info_file(
        output_dir,
        f"{video_a_name}_vs_{video_b_name}_preference",
        video_a_path=video_a_path,
        video_b_path=video_b_path,
        task_text=task_text,
        prediction_type="preference",
        fps=fps,
        info_text=result_text,
    )

    return {
        "result": result_text,
        "comparison_plot": comparison_plot_path,
        "info_file": info_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Run RFM evaluation on videos/images")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["gradio", "server", "local"],
        required=True,
        help="Execution mode: gradio (via Gradio app), server (direct Azure server), or local (local model)",
    )
    parser.add_argument(
        "--eval-type",
        type=str,
        choices=["progress", "preference"],
        required=True,
        help="Evaluation type: progress (single video) or preference (two videos)",
    )
    parser.add_argument(
        "--gradio-url",
        type=str,
        help="URL of the Gradio app (required for gradio mode)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        help="Eval server URL (required for gradio and server modes)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model checkpoint (required for local mode)",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file or directory containing videos (required for progress evaluation)",
    )
    parser.add_argument(
        "--video-a",
        type=str,
        help="Path to first video file (required for preference evaluation)",
    )
    parser.add_argument(
        "--video-b",
        type=str,
        help="Path to second video file (required for preference evaluation)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task description. If not specified, will be inferred from video filename.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract from video (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files. If not specified, creates timestamped directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model inference (default: cuda, only for local mode)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "gradio" and not args.gradio_url:
        parser.error("--gradio-url is required for gradio mode")
    if args.mode in ["gradio", "server"] and not args.server_url:
        parser.error("--server-url is required for gradio and server modes")
    if args.mode == "local" and not args.model_path:
        parser.error("--model-path is required for local mode")

    # Validate eval-type specific arguments
    if args.eval_type == "progress":
        if not args.video:
            parser.error("--video is required for progress evaluation")
        if args.video_a or args.video_b:
            parser.error("--video-a and --video-b are not used for progress evaluation")
    elif args.eval_type == "preference":
        if not args.video_a or not args.video_b:
            parser.error("Both --video-a and --video-b are required for preference evaluation")
        if args.video:
            parser.error("--video is not used for preference evaluation")

    # Setup output directory
    video_path_for_dir = args.video if args.eval_type == "progress" else args.video_a
    output_dir = setup_output_directory(args.output_dir, video_path_for_dir)
    print(f"Output directory: {output_dir}")

    # Load model once if using local mode
    model_components = None
    if args.mode == "local":
        model_components = load_model_for_inference(args.model_path, output_dir, args.device)

    if args.eval_type == "progress":
        # Check if input is a directory
        video_path = Path(args.video)
        if video_path.is_dir():
            # Process all videos in directory
            video_files = find_video_files(args.video)
            if not video_files:
                print(f"Warning: No video files found in directory: {args.video}")
                sys.exit(1)

            print(f"Found {len(video_files)} video file(s) in directory: {args.video}")
            print("=" * 60)

            results = []
            predictions = []
            for i, video_file in enumerate(video_files, 1):
                video_name = Path(video_file).name
                print(f"\n[{i}/{len(video_files)}] Processing: {video_name}")
                print("-" * 60)

                # Infer task from video name if not specified
                task_text = args.task if args.task else infer_task_from_video_name(video_file)
                if not args.task:
                    print(f"  Inferred task: {task_text}")

                try:
                    if args.mode == "gradio":
                        result = run_progress_gradio(
                            args.gradio_url, video_file, task_text, args.server_url, args.fps, output_dir
                        )
                    elif args.mode == "server":
                        result = run_progress_server(args.server_url, video_file, task_text, args.fps, output_dir)
                    else:  # local
                        result = run_progress_model(video_file, task_text, args.fps, output_dir, model_components)
                    results.append((video_file, result))

                    # Extract predictions for summary
                    if result:
                        # Try to extract from info text first
                        info_text = result.get("info", "")
                        final_progress = None
                        final_success = None

                        # Parse final progress and success probability from info text
                        for line in info_text.split("\n"):
                            if "Final progress" in line or "**Final progress**" in line:
                                try:
                                    # Handle both "**Final progress:** 0.123" and "Final progress: 0.123"
                                    value_str = line.split(":")[-1].strip().replace("*", "").strip()
                                    final_progress = float(value_str)
                                except:
                                    pass
                            if "Final success probability" in line or "**Final success probability**" in line:
                                try:
                                    value_str = line.split(":")[-1].strip().replace("*", "").strip()
                                    final_success = float(value_str)
                                except:
                                    pass

                        predictions.append({
                            "video": video_name,
                            "task": task_text,
                            "final_progress": final_progress,
                            "final_success_prob": final_success,
                        })

                    print(f"✓ Completed: {video_name}")
                except Exception as e:
                    print(f"✗ Error processing {video_name}: {e}")
                    results.append((video_file, None))
                    predictions.append({
                        "video": video_name,
                        "task": task_text,
                        "final_progress": None,
                        "final_success_prob": None,
                        "error": str(e),
                    })

            print("\n" + "=" * 60)
            print("Batch evaluation completed!")
            print("=" * 60)
            successful = sum(1 for _, r in results if r is not None)
            print(f"Successfully processed: {successful}/{len(video_files)} videos")
            print(f"Outputs saved to: {output_dir}")

            # Print predictions summary
            if predictions:
                print("\n" + "=" * 60)
                print("PREDICTIONS SUMMARY")
                print("=" * 60)
                for pred in predictions:
                    print(f"\nVideo: {pred['video']}")
                    print(f"  Task: {pred['task']}")
                    if pred.get("error"):
                        print(f"  Error: {pred['error']}")
                    else:
                        if pred["final_progress"] is not None:
                            print(f"  Final Progress: {pred['final_progress']:.3f}")
                        if pred["final_success_prob"] is not None:
                            print(f"  Final Success Probability: {pred['final_success_prob']:.3f}")
                print("\n" + "=" * 60)
        else:
            # Single video progress evaluation
            # Infer task from video name if not specified
            task_text = args.task if args.task else infer_task_from_video_name(args.video)
            if not args.task:
                print(f"Inferred task from video name: {task_text}")

            if args.mode == "gradio":
                result = run_progress_gradio(
                    args.gradio_url, args.video, task_text, args.server_url, args.fps, output_dir
                )
            elif args.mode == "server":
                result = run_progress_server(args.server_url, args.video, task_text, args.fps, output_dir)
            else:  # local
                result = run_progress_model(args.video, task_text, args.fps, output_dir, model_components)

            print("\n" + "=" * 60)
            print("Evaluation completed successfully!")
            print("=" * 60)
            print(f"Outputs saved to: {output_dir}")

            # Print predictions
            if result:
                print("\n" + "=" * 60)
                print("PREDICTIONS")
                print("=" * 60)
                info_text = result.get("info", "")
                print(info_text)
                print("=" * 60)
    else:  # preference evaluation
        # Infer task from first video name if not specified
        task_text = args.task if args.task else infer_task_from_video_name(args.video_a)
        if not args.task:
            print(f"Inferred task from video name: {task_text}")

        if args.mode == "gradio":
            result = run_preference_gradio(
                args.gradio_url,
                args.video_a,
                args.video_b,
                task_text,
                args.server_url,
                args.fps,
                output_dir,
            )
        elif args.mode == "server":
            result = run_preference_server(args.server_url, args.video_a, args.video_b, task_text, args.fps, output_dir)
        else:  # local
            result = run_preference_model(args.video_a, args.video_b, task_text, args.fps, output_dir, model_components)

        print("\n" + "=" * 60)
        print("Evaluation completed successfully!")
        print("=" * 60)
        print(f"Outputs saved to: {output_dir}")

        # Print predictions
        if result:
            print("\n" + "=" * 60)
            print("PREDICTIONS")
            print("=" * 60)
            result_text = result.get("result", "")
            print(result_text)
            print("=" * 60)


if __name__ == "__main__":
    main()
