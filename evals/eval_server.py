#!/usr/bin/env python3
"""
FastAPI server to evaluate RFM model batches with async multi-GPU support.

Usage example:
    uv run python evals/eval_server.py \
        --config_path rfm/configs/eval_config_server.yaml 
        --num_gpus 1 \
        --model_path reward_fm/ant_rfm_rewind_bs1024_pref_prog

Endpoints:
  POST /evaluate_batch        - JSON payload
  POST /evaluate_batch_npy    - multipart payload with .npy blobs

Response payload per request contains predictions grouped by head:
  {
    "outputs_preference": {...},   # Preference logits + optional progress traces
    "outputs_progress": {...},     # Progress-only trajectories
    "outputs_similarity": {...},   # Similarity logits (if requested)
  }
"""

from __future__ import annotations

import ast
import asyncio
import io
import json
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict
import argparse

import uvicorn

import numpy as np
import torch
import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from rich.console import Console

from evals.eval_utils import extract_answer_from_text, load_model_from_hf
from rfm.configs.eval_configs import EvalServerConfig
from rfm.configs.experiment_configs import ExperimentConfig
from rfm.data.dataset_types import PreferenceSample, ProgressSample, SimilaritySample
from rfm.utils.parser import deep_merge
from rfm.utils.setup_utils import setup_model_and_processor, setup_batch_collator
from rfm.models.utils import ModelOutput


def forward_model(model, batch_inputs: Dict[str, Any], sample_type: str = "progress") -> tuple[ModelOutput, Dict[str, Any]]:
    """Forward pass that mirrors trainer logic (handles ReWiND vs RFM)."""
    with torch.no_grad():
        if getattr(model.config, "base_model_id", "") and "rewind" in model.config.base_model_id:
            model_output, extra = model(
                video_embeddings=batch_inputs.get("video_embeddings"),
                text_embeddings=batch_inputs.get("text_embeddings"),
                sample_type=sample_type,
                timing_raw=None,
            )
        else:
            model_output, extra = model(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                pixel_values=batch_inputs.get("pixel_values", None),
                pixel_values_videos=batch_inputs.get("pixel_values_videos", None),
                image_grid_thw=batch_inputs.get("image_grid_thw", None),
                video_grid_thw=batch_inputs.get("video_grid_thw", None),
                second_per_grid_ts=batch_inputs.get("second_per_grid_ts", None),
                sample_type=sample_type,
                timing_raw=None,
            )
    return model_output, extra


class AsyncGPUPool:
    """Async GPU pool for handling multiple requests efficiently across multiple GPUs."""

    def __init__(
        self,
        exp_config: ExperimentConfig,
        model_path: str,
        num_gpus: int | None = None,
        max_workers: int | None = None,
    ):
        self.exp_config = exp_config
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

        print("GPU pool initialized successfully")

    def _initialize_gpus(self):
        """Initialize models on all GPUs."""
        for gpu_id in range(self.num_gpus):
            device = f"cuda:{gpu_id}"
            print(f"Loading model on GPU {gpu_id} ({device})")

            # Load model on specific GPU
            tokenizer, processor, model = setup_model_and_processor(self.exp_config.model, self.model_path)
            batch_collator = setup_batch_collator(processor, tokenizer, self.exp_config, is_eval=True)

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
            self.gpu_pool.put({
                "model": model,
                "processor": processor,
                "tokenizer": tokenizer,
                "batch_collator": batch_collator,
                "device": device,
                "gpu_id": gpu_id,
                "created_at": time.time(),
            })

            print(f"Successfully loaded model on GPU {gpu_id}")

    async def process_batch(self, batch_data: dict[str, Any] | list[dict[str, Any]]):
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

    def _process_batch_sync(self, gpu_info: dict, batch_data: dict[str, Any] | list[dict[str, Any]]):
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

        batch_collator = gpu_info["batch_collator"]
        input_samples = []
        for sample in samples:
            if isinstance(sample, (PreferenceSample, ProgressSample, SimilaritySample)):
                input_samples.append(sample)
            elif isinstance(sample, dict):
                sample_type = sample.get("sample_type")
                if sample_type == "preference":
                    input_samples.append(PreferenceSample(**sample))
                elif sample_type == "progress":
                    input_samples.append(ProgressSample(**sample))
                elif sample_type == "similarity":
                    input_samples.append(SimilaritySample(**sample))
                else:
                    raise ValueError(f"Unsupported sample_type: {sample_type}")
            else:
                raise ValueError(f"Unsupported sample object type: {type(sample)}")

        batch_inputs = batch_collator(input_samples)

        # Move inputs to the correct GPU
        device = gpu_info["device"]
        for key, value in batch_inputs["preference_inputs"].items():
            if isinstance(value, torch.Tensor):
                batch_inputs["preference_inputs"][key] = value.to(device)
        for key, value in batch_inputs["progress_inputs"].items():
            if isinstance(value, torch.Tensor):
                batch_inputs["progress_inputs"][key] = value.to(device)
        for key, value in batch_inputs["similarity_inputs"].items():
            if isinstance(value, torch.Tensor):
                batch_inputs["similarity_inputs"][key] = value.to(device)

        outputs_preference = None
        outputs_progress = None
        outputs_similarity = None

        num_preferences = batch_inputs.get("num_preferences", 0)
        num_progress = batch_inputs.get("num_progress", 0)
        num_similarities = batch_inputs.get("num_similarities", 0)

        if num_preferences > 0:
            if self.exp_config.model.model_type == "vqa":
                outputs_preference = compute_batch_outputs_vqa(
                    gpu_info["model"],
                    gpu_info["tokenizer"],
                    batch_inputs["preference_inputs"],
                    mode="preference",
                )
            else:
                outputs_preference = compute_batch_outputs(
                    gpu_info["model"],
                    gpu_info["tokenizer"],
                    batch_inputs["preference_inputs"],
                    sample_type="preference",
                )
        else:
            outputs_preference = None

        if num_progress > 0:
            if self.exp_config.model.model_type == "vqa":
                outputs_progress = compute_batch_outputs_vqa(
                    gpu_info["model"], gpu_info["tokenizer"], batch_inputs["progress_inputs"], mode="progress"
                )
            else:
                outputs_progress = compute_batch_outputs(
                    gpu_info["model"],
                    gpu_info["tokenizer"],
                    batch_inputs["progress_inputs"],
                    sample_type="progress",
                )

        if num_similarities > 0:
            if self.exp_config.model.model_type == "vqa":
                raise ValueError("Similarity evaluation is not supported for VQA model type.")
            outputs_similarity = compute_batch_outputs(
                gpu_info["model"],
                gpu_info["tokenizer"],
                batch_inputs["similarity_inputs"],
                sample_type="similarity",
            )

        return {
            "outputs_preference": outputs_preference,
            "outputs_progress": outputs_progress,
            "outputs_similarity": outputs_similarity,
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


def compute_batch_outputs(model, tokenizer, batch_inputs: dict[str, torch.Tensor], sample_type: str) -> dict[str, Any]:
    """
    Run a forward pass for non-VQA models and return the raw head outputs we
    need for eval logging.

    Args:
        model: RFM/ReWiND model on the target device.
        tokenizer: Included for parity with the VQA helper (unused here).
        batch_inputs: Collated inputs for the requested head.
        sample_type: One of {"preference","progress","similarity"}.

    Returns:
        Dict containing logits/derived predictions keyed by head.
    """
    model.eval()
    model_output, _ = forward_model(model, batch_inputs, sample_type=sample_type)

    results: dict[str, Any] = {}

    # Preference logits and metadata
    if sample_type == "preference" and model_output.pref_logits is not None:
        logits = model_output.pref_logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        results.update({
            "predictions": preds.detach().cpu().tolist(),
            "prediction_probs": probs.detach().cpu().tolist(),
            "preference_labels": batch_inputs["preference_labels"].cpu().tolist(),
        })

    # Progress predictions (used by both preference + progress sample types)
    progress_logits = model_output.progress_logits
    if progress_logits is not None and isinstance(progress_logits, dict):
        if sample_type == "preference":
            progress_pred_chosen: list[list[float]] = []
            progress_pred_rejected: list[list[float]] = []
            preference_labels = results.get("preference_labels", batch_inputs["preference_labels"].cpu().tolist())
            seq_A_list = progress_logits.get("A", [])
            seq_B_list = progress_logits.get("B", [])
            for label, seq_A, seq_B in zip(preference_labels, seq_A_list, seq_B_list, strict=False):
                if label == 1.0:
                    chosen_seq, rejected_seq = seq_A, seq_B
                else:
                    chosen_seq, rejected_seq = seq_B, seq_A
                progress_pred_chosen.append([] if chosen_seq is None else chosen_seq.detach().cpu().flatten().tolist())
                progress_pred_rejected.append(
                    [] if rejected_seq is None else rejected_seq.detach().cpu().flatten().tolist()
                )
            results.update({
                "progress_pred_chosen": progress_pred_chosen,
                "progress_pred_rejected": progress_pred_rejected,
            })
        elif sample_type == "progress":
            progress_pred = []
            seq_A_list = progress_logits.get("A", [])
            for seq_A in seq_A_list:
                progress_pred.append([] if seq_A is None else seq_A.detach().cpu().flatten().tolist())
            if not progress_pred:
                batch_size = len(batch_inputs.get("task", []))
                progress_pred = [[] for _ in range(batch_size)]
            results["progress_pred"] = progress_pred

    # Similarity logits
    if sample_type == "similarity" and model_output.sim_logits is not None:
        sim_logits = model_output.sim_logits
        if isinstance(sim_logits, torch.Tensor):
            sim_tensor = sim_logits.squeeze(-1)
        elif isinstance(sim_logits, list):
            sim_tensor = torch.stack(sim_logits).squeeze(-1)
        else:
            sim_tensor = None

        sim_scores_list: list[float] = []
        if sim_tensor is not None:
            sim_scores_list = sim_tensor.detach().cpu().flatten().tolist()

        num_samples = len(batch_inputs.get("task", []))
        if num_samples == 0 and sim_scores_list:
            num_samples = len(sim_scores_list) // 2

        sim_score_ref_sim: list[float | None] = []
        sim_score_ref_diff: list[float | None] = []
        for i in range(num_samples):
            idx_sim = i * 2
            idx_diff = idx_sim + 1
            sim_score_ref_sim.append(sim_scores_list[idx_sim] if idx_sim < len(sim_scores_list) else None)
            sim_score_ref_diff.append(sim_scores_list[idx_diff] if idx_diff < len(sim_scores_list) else None)

        results.update({
            "sim_score_ref_sim": sim_score_ref_sim,
            "sim_score_ref_diff": sim_score_ref_diff,
            "task": batch_inputs.get("task", []),
            "data_source": batch_inputs.get("data_source", []),
            "data_gen_strategy": batch_inputs.get("data_gen_strategy", []),
            "metadata": batch_inputs.get("metadata", []),
        })

    return results


def compute_batch_outputs_vqa(
    model, tokenizer, batch_inputs: dict[str, torch.Tensor], mode: str = "preference"
) -> dict[str, Any]:
    """
    Generate text answers for VQA-style models and post-process into numeric
    predictions so downstream aggregation matches the non-VQA path.

    Args:
        model: VQA model (e.g., Qwen VQA variant).
        tokenizer: Associated tokenizer for decoding.
        batch_inputs: Collated inputs including prompt tokens/images.
        mode: "preference" or "progress" depending on the head requested.

    Returns:
        Dict containing parsed predictions for the requested mode.
    """
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
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(input_ids, output_ids, strict=False)
        ]
        generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    if mode == "preference":
        predictions = [extract_answer_from_text(text) for text in generated_texts]
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
        progress_predictions = [extract_answer_from_text(text) for text in generated_texts]
        progress_predictions = [ast.literal_eval(prediction) for prediction in progress_predictions]
        return {
            "progress_pred_A": progress_predictions,
        }
    else:
        raise ValueError(f"Mode {mode} not supported")


def create_app(cfg: EvaluationConfig, exp_config: ExperimentConfig):
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

    gpu_pool = AsyncGPUPool(exp_config, cfg.model_path, num_gpus, max_workers)
    print(f"GPU pool initialized with {gpu_pool.num_gpus} GPUs")

    @app.post("/evaluate_batch")
    async def evaluate_batch(batch: dict[str, Any]):
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
        numpy_arrays: dict[str, np.ndarray], other_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
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

                                if traj_key in ["video_embeddings", "text_embedding"]:
                                    # convert to tensor
                                    value[traj_key] = torch.tensor(value[traj_key])

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="rfm/configs/eval_config.yaml")
    parser.add_argument(
        "--model_config_paths",
        nargs="*",
        default=[],
        help="Paths to the model config files (Only used if model_path is not set in eval config)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (None for all available)")
    parser.add_argument("--max_workers", type=int, default=None, help="Max worker threads (None for auto)")
    args = parser.parse_args()

    # Load evaluation config manually
    print(f"Loading evaluation config from: {args.config_path}")
    with open(args.config_path) as f:
        config_dict = yaml.safe_load(f)

    cfg = EvalServerConfig(**config_dict)

    # Override config with command line args
    if args.num_gpus is not None:
        cfg.num_gpus = args.num_gpus
    if args.max_workers is not None:
        cfg.max_workers = args.max_workers

    console = Console()
    console.print(cfg)

    # Loading experiment config from prtrained model
    exp_config = None
    if cfg.model_path:
        exp_config, _, _, _ = load_model_from_hf(
            model_path=cfg.model_path,
            device=torch.device("cpu"),
            load_model=False,
        )
    else:
        print("Saved checkpoint is not found, initializing base model")
        print(f"Loading model configs from local paths: {args.model_config_paths}")
        # load model config from local path
        assert args.model_config_paths != "", "Model config path is required if model path is not set in eval config"
        # load & deep-merge YAMLs in order (later files override earlier ones)
        merged: dict[str, Any] = {}
        for path in args.model_config_paths:
            with open(path) as f:
                doc = yaml.safe_load(f) or {}
            deep_merge(merged, doc)
        exp_config = ExperimentConfig(**merged)

    if exp_config is None:
        raise ValueError("Failed to load experiment config. Provide a valid model_path or model_config_paths.")

    console.print(exp_config)

    app = create_app(cfg, exp_config)
    print(f"Running async multi-GPU server on {args.host}:{args.port}")
    print(f"Using {cfg.num_gpus or torch.cuda.device_count()} GPUs")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
