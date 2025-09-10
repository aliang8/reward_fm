#!/usr/bin/env python3
"""
FastAPI server to evaluate RFM model batches with async multi-GPU support.

Endpoint:
  POST /evaluate_batch
Request payload (JSON):

Response payload (JSON), per-sample outputs for client-side aggregation:
  {
    "predictions": List[int],              # 1 if chosen preferred, else 0
    "reward_chosen": List[List[float]],    # per-frame rewards for chosen (maps to progress head A)
    "reward_rejected": List[List[float]]   # per-frame rewards for rejected (maps to progress head B)
  }

e.g.: uv run /home/jessez/reward_fm/evals/server.py --config_path=/home/jessez/reward_fm/rfm/configs/config.yaml --host=0.0.0.0 --port=8000
"""

from __future__ import annotations
import yaml
import base64
import io
from typing import Any, Dict, List, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time
import ast
import re

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
import json

from rfm.utils.setup_utils import setup_model_and_processor
from rfm.configs.eval_configs import EvaluationConfig
from rfm.configs.experiment_configs import ModelConfig
from rfm.data.batch_collator import BatchCollator, PreferenceSample
from rfm.data.vqa_batch_collator import VQABatchCollator
from rfm.data.dataset_types import PreferenceSample, ProgressSample


class AsyncGPUPool:
    """Async GPU pool for handling multiple requests efficiently across multiple GPUs."""

    def __init__(
        self,
        model_config: ModelConfig,
        model_path: str,
        num_gpus: int = None,
        max_workers: int = None,
    ):
        self.model_config = model_config
        self.model_path = model_path
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.max_workers = max_workers or self.num_gpus

        if self.num_gpus == 0:
            raise RuntimeError("No CUDA devices available")

        print(f"Initializing async GPU pool with {self.num_gpus} GPUs and {self.max_workers} workers")

        # Initialize GPU pool
        self.gpu_pool = queue.Queue(maxsize=self.num_gpus)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.gpu_stats = {}

        # Initialize GPUs
        self._initialize_gpus()

        print(f"GPU pool initialized successfully")

    def _initialize_gpus(self):
        """Initialize models on all GPUs."""
        for gpu_id in range(self.num_gpus):
            device = f"cuda:{gpu_id}"
            print(f"Loading model on GPU {gpu_id} ({device})")

            # Load model on specific GPU
            tokenizer, processor, model = setup_model_and_processor(self.model_config, self.model_path)

            model = model.to(device)
            model.eval()

            # Initialize GPU stats
            self.gpu_stats[gpu_id] = {
                "total_requests": 0,
                "total_processing_time": 0.0,
                "last_used": time.time(),
                "status": "ready",
            }

            # Add to pool
            self.gpu_pool.put(
                {"model": model, "processor": processor, "device": device, "gpu_id": gpu_id, "created_at": time.time()}
            )

            print(f"Successfully loaded model on GPU {gpu_id}")

    async def process_batch(self, batch_data: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """Process a batch using an available GPU asynchronously."""
        loop = asyncio.get_event_loop()

        # Get GPU from pool (this will block until one is available)
        gpu_info = await loop.run_in_executor(self.executor, self.gpu_pool.get)

        start_time = time.time()

        # Update GPU stats
        self.gpu_stats[gpu_info["gpu_id"]]["status"] = "processing"
        self.gpu_stats[gpu_info["gpu_id"]]["last_used"] = start_time

        try:
            # Process batch in thread pool
            result = await loop.run_in_executor(self.executor, self._process_batch_sync, gpu_info, batch_data)

            # Update stats
            processing_time = time.time() - start_time
            self.gpu_stats[gpu_info["gpu_id"]]["total_requests"] += 1
            self.gpu_stats[gpu_info["gpu_id"]]["total_processing_time"] += processing_time

            return result

        finally:
            # Always return GPU to pool and update stats
            processing_time = time.time() - start_time
            self.gpu_stats[gpu_info["gpu_id"]]["total_requests"] += 1
            self.gpu_stats[gpu_info["gpu_id"]]["total_processing_time"] += processing_time
            self.gpu_stats[gpu_info["gpu_id"]]["status"] = "ready"
            self.gpu_pool.put(gpu_info)

    def _process_batch_sync(self, gpu_info: Dict, batch_data: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """Synchronous batch processing on specific GPU."""

        # Handle both dict and list inputs
        if isinstance(batch_data, dict):
            # Legacy format - extract samples from dict
            samples = batch_data.get("samples", [])
        else:
            # New format - batch_data is already a list of samples
            samples = batch_data

        if not samples:
            raise ValueError("No samples found in batch data")

        print(f"Processing {len(samples)} samples on GPU {gpu_info['gpu_id']}")

        # Create batch collator with processor from this GPU
        if "qwen" in self.model_config.base_model_id.lower():
            if self.model_config.model_type == "default":
                batch_collator = BatchCollator(
                    processor=gpu_info["processor"],
                    resized_height=128,  # You might want to make this configurable
                    resized_width=128,
                )
            elif self.model_config.model_type == "vqa":
                batch_collator = VQABatchCollator(
                    processor=gpu_info["processor"],
                    resized_height=128,  # You might want to make this configurable
                    resized_width=128,
                    training=False,
                    inference=True,
                )
            else:
                raise ValueError(f"Model type {self.model_config.model_type} not supported")

            input_samples = []
            for sample in samples:
                if sample["sample_type"] == "preference":
                    input_samples.append(PreferenceSample(**sample))
                elif sample["sample_type"] == "progress":
                    input_samples.append(ProgressSample(**sample))
            # This time batch_inputs will be a dictionary with preference_inputs, similarity_inputs, and progress_inputs
            batch_inputs = batch_collator(input_samples)

            # Move inputs to the correct GPU
            device = gpu_info["device"]
            for key, value in batch_inputs["preference_inputs"].items():
                if isinstance(value, torch.Tensor):
                    batch_inputs["preference_inputs"][key] = value.to(device)
            for key, value in batch_inputs["progress_inputs"].items():
                if isinstance(value, torch.Tensor):
                    batch_inputs["progress_inputs"][key] = value.to(device)
        else:
            print(f"Model type {self.model_config.base_model_id} not supported")
            raise ValueError(f"Model type {self.model_config.base_model_id} not supported")

        if batch_inputs["num_preferences"] > 0:
            if self.model_config.model_type == "vqa":
                outputs_preference = compute_batch_outputs_vqa(
                    gpu_info["model"],
                    gpu_info["processor"].tokenizer,
                    batch_inputs["preference_inputs"],
                    mode="preference",
                )
            else:
                # Run inference for preference samples
                outputs_preference = compute_batch_outputs(
                    gpu_info["model"], gpu_info["processor"].tokenizer, batch_inputs["preference_inputs"]
                )
        else:
            outputs_preference = None

        if batch_inputs["num_progress"] > 0:
            if self.model_config.model_type == "vqa":
                outputs_progress = compute_batch_outputs_vqa(
                    gpu_info["model"], gpu_info["processor"].tokenizer, batch_inputs["progress_inputs"], mode="progress"
                )
            else:
                # Run inference for progress samples - only compute progress for trajectory A
                outputs_progress = compute_batch_outputs_progress_only(
                    gpu_info["model"], gpu_info["processor"].tokenizer, batch_inputs["progress_inputs"]
                )
        else:
            outputs_progress = None

        return {
            "outputs_preference": outputs_preference,
            "outputs_progress": outputs_progress,
        }

    def get_pool_status(self):
        """Get status of the GPU pool."""
        return {
            "total_gpus": self.num_gpus,
            "available_gpus": self.gpu_pool.qsize(),
            "max_workers": self.max_workers,
            "gpu_stats": self.gpu_stats,
            "pool_size": self.gpu_pool.maxsize,
        }

    def shutdown(self):
        """Shutdown the GPU pool and executor."""
        print("Shutting down GPU pool...")
        self.executor.shutdown(wait=True)
        print("GPU pool shutdown complete")


def compute_batch_outputs(model, tokenizer, batch_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Compute batch outputs for preference prediction."""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        outputs, progress_logits, _ = model(
            input_ids=batch_inputs["input_ids"].to(device),
            attention_mask=batch_inputs["attention_mask"].to(device),
            pixel_values=batch_inputs.get("pixel_values", None).to(device)
            if batch_inputs.get("pixel_values") is not None
            else None,
            pixel_values_videos=batch_inputs.get("pixel_values_videos", None).to(device)
            if batch_inputs.get("pixel_values_videos") is not None
            else None,
            image_grid_thw=batch_inputs.get("image_grid_thw", None).to(device)
            if batch_inputs.get("image_grid_thw") is not None
            else None,
            video_grid_thw=batch_inputs.get("video_grid_thw", None).to(device)
            if batch_inputs.get("video_grid_thw") is not None
            else None,
            second_per_grid_ts=batch_inputs.get("second_per_grid_ts", None).to(device)
            if batch_inputs.get("second_per_grid_ts") is not None
            else None,
            sample_type="preference",
        )

        logits = outputs.logits.squeeze(-1)  # [B]
        probs = torch.sigmoid(logits)  # [B]
        print(f"predicted preference probabilities: {probs}")
        preds = (probs > 0.5).long()  # [B]

        predictions = preds.detach().cpu().tolist()

        # Extract progress predictions for A and B if available
        # Map back to chosen/rejected based on preference labels
        progress_pred_chosen: List[List[float]] = []
        progress_pred_rejected: List[List[float]] = []

        if isinstance(progress_logits, dict):
            # Get preference labels to determine which trajectory (A or B) corresponds to chosen/rejected
            preference_labels = batch_inputs["preference_labels"].cpu().tolist()

            for i, (label, seq_A, seq_B) in enumerate(
                zip(preference_labels, progress_logits.get("A", []), progress_logits.get("B", []))
            ):
                if label == 1.0:
                    # First trajectory (A) is chosen, second trajectory (B) is rejected
                    chosen_seq = seq_A
                    rejected_seq = seq_B
                else:
                    # First trajectory (A) is rejected, second trajectory (B) is chosen
                    chosen_seq = seq_B
                    rejected_seq = seq_A

                # Extract chosen progress
                if chosen_seq is None:
                    progress_pred_chosen.append([])
                else:
                    progress_pred_chosen.append(chosen_seq.detach().cpu().flatten().tolist())

                # Extract rejected progress
                if rejected_seq is None:
                    progress_pred_rejected.append([])
                else:
                    progress_pred_rejected.append(rejected_seq.detach().cpu().flatten().tolist())

        return {
            "predictions": predictions,
            "prediction_probs": probs.detach().cpu().tolist(),
            "progress_pred_chosen": progress_pred_chosen,
            "progress_pred_rejected": progress_pred_rejected,
            "preference_labels": preference_labels,
        }


def compute_batch_outputs_progress_only(model, tokenizer, batch_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Compute batch outputs for progress prediction only (trajectory A)."""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        outputs, progress_logits, _ = model(
            input_ids=batch_inputs["input_ids"].to(device),
            attention_mask=batch_inputs["attention_mask"].to(device),
            pixel_values=batch_inputs.get("pixel_values", None).to(device)
            if batch_inputs.get("pixel_values") is not None
            else None,
            pixel_values_videos=batch_inputs.get("pixel_values_videos", None).to(device)
            if batch_inputs.get("pixel_values_videos") is not None
            else None,
            image_grid_thw=batch_inputs.get("image_grid_thw", None).to(device)
            if batch_inputs.get("image_grid_thw") is not None
            else None,
            video_grid_thw=batch_inputs.get("video_grid_thw", None).to(device)
            if batch_inputs.get("video_grid_thw") is not None
            else None,
            second_per_grid_ts=batch_inputs.get("second_per_grid_ts", None).to(device)
            if batch_inputs.get("second_per_grid_ts") is not None
            else None,
            sample_type="progress",
        )

    # For progress-only samples, we only care about trajectory A progress
    progress_pred_A = []

    if isinstance(progress_logits, dict) and "A" in progress_logits:
        for seq_A in progress_logits["A"]:
            if seq_A is None:
                progress_pred_A.append([])
            else:
                progress_pred_A.append(seq_A.detach().cpu().flatten().tolist())
    else:
        # If no progress logits, create empty lists
        progress_pred_A = [[] for _ in range(len(batch_inputs["input_ids"]))]

    return {
        "progress_pred_A": progress_pred_A,
    }


def compute_batch_outputs_vqa(
    model, tokenizer, batch_inputs: Dict[str, torch.Tensor], mode: str = "preference"
) -> Dict[str, Any]:
    """Compute batch outputs for VQA."""
    model.eval()
    device = next(model.parameters()).device

    input_ids = batch_inputs["input_ids"].to(device)
    attention_mask = batch_inputs["attention_mask"].to(device)
    pixel_values = (
        batch_inputs.get("pixel_values", None).to(device) if batch_inputs.get("pixel_values") is not None else None
    )
    pixel_values_videos = (
        batch_inputs.get("pixel_values_videos", None).to(device)
        if batch_inputs.get("pixel_values_videos") is not None
        else None
    )
    image_grid_thw = (
        batch_inputs.get("image_grid_thw", None).to(device) if batch_inputs.get("image_grid_thw") is not None else None
    )
    video_grid_thw = (
        batch_inputs.get("video_grid_thw", None).to(device) if batch_inputs.get("video_grid_thw") is not None else None
    )
    input_to_model = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "pixel_values_videos": pixel_values_videos,
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
    }

    with torch.no_grad():
        output_ids = model.generate(**input_to_model, max_new_tokens=1024)
        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(input_ids, output_ids)]
        generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    if mode == "preference":
        predictions = [_extract_answer_from_text(text) for text in generated_texts]
        predictions_num_labels = []
        for prediction in predictions:
            if prediction == "A":
                predictions_num_labels.append(1)
            elif prediction == "B":
                predictions_num_labels.append(0)
            else:
                predictions_num_labels.append(-1)
        return {
            "predictions": predictions_num_labels,
            "preference_labels": batch_inputs.get("preference_labels").detach().cpu().tolist(),
        }
    elif mode == "progress":
        progress_predictions = [_extract_answer_from_text(text) for text in generated_texts]
        progress_predictions = [ast.literal_eval(prediction) for prediction in progress_predictions]
        return {
            "progress_pred_A": progress_predictions,
        }
    else:
        raise ValueError(f"Mode {mode} not supported")


def _extract_answer_from_text(text):
    m = re.search(r"<ans>(.*?)</ans>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def create_app(cfg: EvaluationConfig, model_config: ModelConfig):
    app = FastAPI(title="RFM Multi-GPU Evaluation Server")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize async GPU pool
    num_gpus = getattr(cfg, "num_gpus", None)
    max_workers = getattr(cfg, "max_workers", None)

    gpu_pool = AsyncGPUPool(model_config, cfg.model_path, num_gpus, max_workers)
    print(f"GPU pool initialized with {gpu_pool.num_gpus} GPUs")

    @app.post("/evaluate_batch")
    async def evaluate_batch(batch: Dict[str, Any]):
        """Evaluate a batch of preference samples using async GPU pool."""
        return await gpu_pool.process_batch(batch)

    @app.post("/evaluate_batch_npy")
    async def evaluate_batch_npy(request: Request):
        """Evaluate a batch with .npy file support for numpy arrays.

        This endpoint handles multipart form data where:
        - numpy arrays are sent as .npy files
        - other data is sent as form fields
        """
        # Parse form data
        form_data = await request.form()

        # Extract numpy arrays from files
        numpy_arrays = {}
        other_data = {}

        for key, value in form_data.items():
            # Check if this is a file upload (UploadFile object)
            if hasattr(value, "filename") and value.filename:
                # This is a file upload
                if value.filename.endswith(".npy"):
                    # Load .npy file
                    content = await value.read()
                    buf = io.BytesIO(content)
                    array = np.load(buf)
                    numpy_arrays[key] = array
                else:
                    # Non-.npy file, skip for now
                    continue
            else:
                # This is a string value (form field)
                try:
                    # Try to parse as JSON
                    other_data[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # Keep as string if not JSON
                    other_data[key] = value

        # Reconstruct the original payload structure
        batch_data = reconstruct_payload_from_npy(numpy_arrays, other_data)

        # Process the batch
        return await gpu_pool.process_batch(batch_data)

    def reconstruct_payload_from_npy(
        numpy_arrays: Dict[str, np.ndarray], other_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Reconstruct the original payload structure from .npy files and form data.

        The client sends data in this format:
        - Files: sample_0_chosen_trajectory_frames.npy, sample_0_chosen_trajectory_lang_vector.npy, etc.
        - Data: sample_0, sample_1, etc. (each containing the full sample JSON with numpy file references)

        We need to reconstruct the original list of sample dictionaries.
        """
        samples = []

        # Process each sample
        for i in range(len(other_data)):
            sample_key = f"sample_{i}"
            if sample_key in other_data:
                # Get the sample data - might already be parsed or might be a string
                sample_data = other_data[sample_key]
                if isinstance(sample_data, str):
                    # Parse the sample JSON if it's a string
                    sample_data = json.loads(sample_data)

                # Replace numpy file references with actual arrays
                for key, value in sample_data.items():
                    if key in [
                        "chosen_trajectory",
                        "rejected_trajectory",
                        "reference_trajectory",
                        "traj_sim_trajectory",
                        "traj_diff_trajectory",
                        "trajectory",
                    ]:
                        if isinstance(value, dict):
                            for traj_key, traj_value in value.items():
                                if isinstance(traj_value, dict) and traj_value.get("__numpy_file__"):
                                    # Replace with actual numpy array
                                    file_key = traj_value["__numpy_file__"]
                                    if file_key in numpy_arrays:
                                        value[traj_key] = numpy_arrays[file_key]

                samples.append(sample_data)

        return samples

    @app.get("/gpu_status")
    def get_gpu_status():
        """Get status of all GPUs and pool."""
        return gpu_pool.get_pool_status()

    @app.get("/health")
    def health_check():
        """Health check endpoint."""
        status = gpu_pool.get_pool_status()
        return {"status": "healthy", "available_gpus": status["available_gpus"], "total_gpus": status["total_gpus"]}

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        gpu_pool.shutdown()

    return app


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="rfm/configs/eval_config.yaml")
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="",
        help="Path to the model config file (Only used if model_path is not set in eval config)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (None for all available)")
    parser.add_argument("--max_workers", type=int, default=None, help="Max worker threads (None for auto)")
    args = parser.parse_args()

    # Load evaluation config manually
    print(f"Loading evaluation config from: {args.config_path}")
    with open(args.config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    cfg = EvaluationConfig(**config_dict)

    # Override config with command line args
    if args.num_gpus is not None:
        cfg.num_gpus = args.num_gpus
    if args.max_workers is not None:
        cfg.max_workers = args.max_workers

    print(f"Evaluation config: {cfg}")

    if cfg.model_path != "":
        # Download model config from Hugging Face
        model_config_path = hf_hub_download(repo_id=cfg.model_path, filename="config.yaml")
        with open(model_config_path, "r") as f:
            model_config_dict = yaml.safe_load(f)

        model_config = ModelConfig(**model_config_dict["model"])
    else:
        print(f"Saved checkpoint is not found, loading model config from local path")
        print(f"Loading model config from local path: {args.model_config_path}")
        # load model config from local path
        assert args.model_config_path != "", "Model config path is required if model path is not set in eval config"
        with open(args.model_config_path, "r") as f:
            model_config_dict = yaml.safe_load(f)
        model_config = ModelConfig(**model_config_dict["model"])
    app = create_app(cfg, model_config)
    print(f"Running async multi-GPU server on {args.host}:{args.port}")
    print(f"Using {cfg.num_gpus or torch.cuda.device_count()} GPUs")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
