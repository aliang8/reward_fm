#!/usr/bin/env python3
"""
FastAPI server to evaluate RFM model batches with a multi-GPU service layer.

Usage example:
    uv run python rfm/evals/eval_server.py \
        model_path=rewardfm/pref_prog_2frames_all \
        batch_size=32 \
        num_gpus=1 \
        server_port=8000

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
import copy
import io
import json
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import uvicorn

import numpy as np
import torch
from omegaconf import DictConfig
from hydra import main as hydra_main
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from rfm.evals.eval_utils import extract_answer_from_text
from rfm.utils.save import load_model_from_hf
from rfm.configs.eval_configs import EvalServerConfig
from rfm.configs.experiment_configs import ExperimentConfig
from rfm.data.dataset_types import PreferenceSample, ProgressSample, SimilaritySample
from rfm.utils.setup_utils import setup_model_and_processor, setup_batch_collator
from rfm.models.utils import ModelOutput
from rfm.utils.config_utils import display_config, convert_hydra_to_dataclass
from rfm.utils.logger import get_logger, setup_loguru_logging

LOG_LEVEL = "DEBUG"
setup_loguru_logging(log_level=LOG_LEVEL)
logger = get_logger()
logger.info(f"rfm.eval_server logger initialized at level {LOG_LEVEL}")


def log_logits(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        logger.debug(f"{name} shape={tuple(value.shape)} values={value.detach().cpu().tolist()}")
    elif isinstance(value, dict):
        logger.debug(f"{name} keys={list(value.keys())}")
        for key, sub_value in value.items():
            log_logits(f"{name}.{key}", sub_value)
    elif isinstance(value, list):
        logger.debug(f"{name}: {value}")


def forward_model(
    model: Any, batch_inputs: Dict[str, Any], sample_type: str = "progress"
) -> Tuple[ModelOutput, Dict[str, Any]]:
    """Forward pass that mirrors trainer logic (handles ReWiND vs RFM)."""
    with torch.no_grad():
        if "rewind" in model.__class__.__name__.lower():
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


def process_batch_helper(
    model_type: str,
    model: Any,
    tokenizer: Any,
    batch_collator: Any,
    device: torch.device,
    batch_data: List[Dict[str, Any]],
    job_id: int = 0,
) -> Dict[str, Any]:
    """Synchronous batch processing on specific GPU."""
    if not batch_data:
        raise ValueError("No samples found in batch data")

    logger.debug(f"[job {job_id}] Processing {len(batch_data)} samples on device {device}")

    input_samples: List[Any] = []
    for sample in batch_data:
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
    logger.debug(
        f"[job {job_id}] Batch counts — preference: {num_preferences} "
        f"progress: {num_progress} similarity: {num_similarities}"
    )

    if num_preferences > 0:
        if model_type == "vqa":
            outputs_preference = compute_batch_outputs_vqa(
                model,
                tokenizer,
                batch_inputs["preference_inputs"],
                mode="preference",
            )
        else:
            outputs_preference = compute_batch_outputs(
                model,
                tokenizer,
                batch_inputs["preference_inputs"],
                sample_type="preference",
            )
    else:
        outputs_preference = None

    if num_progress > 0:
        if model_type == "vqa":
            outputs_progress = compute_batch_outputs_vqa(
                model, tokenizer, batch_inputs["progress_inputs"], mode="progress"
            )
        else:
            outputs_progress = compute_batch_outputs(
                model,
                tokenizer,
                batch_inputs["progress_inputs"],
                sample_type="progress",
            )

    if num_similarities > 0:
        if model_type == "vqa":
            raise ValueError("Similarity evaluation is not supported for VQA model type.")
        outputs_similarity = compute_batch_outputs(
            model,
            tokenizer,
            batch_inputs["similarity_inputs"],
            sample_type="similarity",
        )

    return {
        "outputs_preference": outputs_preference,
        "outputs_progress": outputs_progress,
        "outputs_similarity": outputs_similarity,
    }


class MultiGPUEvalServer:
    """Multi-GPU inference server that schedules requests across devices."""

    def __init__(
        self,
        model_path: str,
        num_gpus: int | None = None,
        max_workers: int | None = None,
    ):
        self.model_path = model_path
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.max_workers = max_workers or self.num_gpus
        self._active_jobs = 0
        self._job_counter = 0
        self._completed_jobs = 0
        self._active_jobs_lock = Lock()

        logger.info(f"Loading experiment config and base model from {self.model_path}")
        exp_config, tokenizer, processor, reward_model = load_model_from_hf(
            model_path=self.model_path,
            device=torch.device("cpu"),
        )
        self.exp_config: ExperimentConfig = exp_config
        self.base_tokenizer = tokenizer
        self.base_processor = processor
        self.base_model = reward_model
        self.base_batch_collator = setup_batch_collator(processor, tokenizer, self.exp_config, is_eval=True)

        if self.num_gpus == 0:
            raise RuntimeError("No CUDA devices available")

        logger.info(
            f"Initializing multi-GPU eval server: model_path={self.model_path} "
            f"num_gpus={self.num_gpus} max_workers={self.max_workers}"
        )

        # Initialize GPU pool
        self.gpu_pool = queue.Queue(maxsize=self.num_gpus)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.gpu_stats = {}

        # Initialize GPUs
        self._initialize_gpus()

        logger.info("Multi-GPU eval server initialized successfully")

    def _initialize_gpus(self):
        """Initialize models on all GPUs."""
        for gpu_id in range(self.num_gpus):
            device = f"cuda:{gpu_id}"
            logger.info(f"Loading model replica on GPU {gpu_id} ({device})")

            # Load model on specific GPU
            tokenizer = copy.deepcopy(self.base_tokenizer)
            processor = copy.deepcopy(self.base_processor)
            model = copy.deepcopy(self.base_model)
            batch_collator = copy.deepcopy(self.base_batch_collator)

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

            logger.info(f"Successfully loaded model on GPU {gpu_id}")

    async def process_batch(self, batch_data: List[Dict[str, Any]]):
        """Process a batch using whichever GPU is available."""
        loop = asyncio.get_event_loop()

        # Get GPU from pool (this will block until one is available).
        # Use the default executor so worker threads remain available for compute.
        gpu_info = await loop.run_in_executor(None, self.gpu_pool.get)
        queue_size_after_acquire = self.gpu_pool.qsize()
        with self._active_jobs_lock:
            self._job_counter += 1
            job_id = self._job_counter
            self._active_jobs += 1
            active_jobs = self._active_jobs
        logger.debug(
            f"[job {job_id}] Acquired GPU {gpu_info['gpu_id']} "
            f"queue_size={queue_size_after_acquire} active_jobs={active_jobs}"
        )

        start_time = time.time()

        # Update GPU stats
        self.gpu_stats[gpu_info["gpu_id"]]["status"] = "processing"
        self.gpu_stats[gpu_info["gpu_id"]]["last_used"] = start_time

        try:
            # Process batch in thread pool
            result = await loop.run_in_executor(
                self.executor,
                process_batch_helper,
                self.exp_config.model.model_type,
                gpu_info["model"],
                gpu_info["tokenizer"],
                gpu_info["batch_collator"],
                gpu_info["device"],
                batch_data,
                job_id,
            )

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
            queue_size_after_release = self.gpu_pool.qsize()
            with self._active_jobs_lock:
                self._active_jobs -= 1
                self._completed_jobs += 1
                active_jobs = self._active_jobs
                completed_jobs = self._completed_jobs
            logger.debug(
                f"[job {job_id}] Completed on GPU {gpu_info['gpu_id']} "
                f"active_jobs={active_jobs} completed_jobs={completed_jobs} "
                f"queue_size={queue_size_after_release} "
                f"processing_time={processing_time:.3f}s"
            )

    def get_pool_status(self) -> Dict[str, Any]:
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
        logger.info("Shutting down GPU pool...")
        self.executor.shutdown(wait=True)
        logger.info("GPU pool shutdown complete")


def compute_batch_outputs(
    model: Any, tokenizer: Any, batch_inputs: Dict[str, torch.Tensor], sample_type: str
) -> Dict[str, Any]:
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
    logger.debug(f"compute_batch_outputs sample_type={sample_type}")
    model_output, _ = forward_model(model, batch_inputs, sample_type=sample_type)

    results: Dict[str, Any] = {}

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

        logger.debug(f"predictions: {results['predictions']}")
        logger.debug(f"prediction_probs: {results['prediction_probs']}")
        logger.debug(f"preference_labels: {results['preference_labels']}")

    # Progress predictions (used by both preference + progress sample types)
    progress_logits = model_output.progress_logits
    if progress_logits is not None and isinstance(progress_logits, dict):
        if sample_type == "preference":
            progress_pred_chosen: List[List[float]] = []
            progress_pred_rejected: List[List[float]] = []
            preference_labels = results.get("preference_labels", batch_inputs["preference_labels"].cpu().tolist())
            seq_A = progress_logits.get("A")
            seq_B = progress_logits.get("B")

            # Convert tensors to lists
            seq_A_list = [seq_A[i] for i in range(seq_A.shape[0])] if seq_A is not None else []
            seq_B_list = [seq_B[i] for i in range(seq_B.shape[0])] if seq_B is not None else []

            for label, seq_A_item, seq_B_item in zip(preference_labels, seq_A_list, seq_B_list, strict=False):
                if label == 1.0:
                    chosen_seq, rejected_seq = seq_A_item, seq_B_item
                else:
                    chosen_seq, rejected_seq = seq_B_item, seq_A_item
                progress_pred_chosen.append([] if chosen_seq is None else chosen_seq.detach().cpu().flatten().tolist())
                progress_pred_rejected.append(
                    [] if rejected_seq is None else rejected_seq.detach().cpu().flatten().tolist()
                )
            results.update({
                "progress_pred_chosen": progress_pred_chosen,
                "progress_pred_rejected": progress_pred_rejected,
            })
            logger.debug(f"progress_pred_chosen: {progress_pred_chosen}")
            logger.debug(f"progress_pred_rejected: {progress_pred_rejected}")
        elif sample_type == "progress":
            progress_pred = []
            seq_A = progress_logits.get("A")

            # Convert tensor to list
            seq_A_list = [seq_A[i] for i in range(seq_A.shape[0])] if seq_A is not None else []

            for seq_A_item in seq_A_list:
                progress_pred.append([] if seq_A_item is None else seq_A_item.detach().cpu().flatten().tolist())
            if not progress_pred:
                batch_size = len(batch_inputs.get("task", []))
                progress_pred = [[] for _ in range(batch_size)]
            results["progress_pred"] = progress_pred
            logger.debug(f"progress_pred: {progress_pred}")

    # Similarity logits
    if sample_type == "similarity" and model_output.sim_logits is not None:
        sim_logits = model_output.sim_logits
        if isinstance(sim_logits, torch.Tensor):
            sim_tensor = sim_logits.squeeze(-1)
        elif isinstance(sim_logits, list):
            sim_tensor = torch.stack(sim_logits).squeeze(-1)
        else:
            sim_tensor = None

        sim_scores_list: List[float] = []
        if sim_tensor is not None:
            sim_scores_list = sim_tensor.detach().cpu().flatten().tolist()

        num_samples = len(batch_inputs.get("task", []))
        if num_samples == 0 and sim_scores_list:
            num_samples = len(sim_scores_list) // 2

        sim_score_ref_sim: List[Optional[float]] = []
        sim_score_ref_diff: List[Optional[float]] = []
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

        logger.debug(f"sim_score_ref_sim: {sim_score_ref_sim}")
        logger.debug(f"sim_score_ref_diff: {sim_score_ref_diff}")
        logger.debug(f"task: {batch_inputs.get('task', [])}")
        logger.debug(f"data_source: {batch_inputs.get('data_source', [])}")
        logger.debug(f"data_gen_strategy: {batch_inputs.get('data_gen_strategy', [])}")

    return results


def compute_batch_outputs_vqa(
    model: Any, tokenizer: Any, batch_inputs: Dict[str, torch.Tensor], mode: str = "preference"
) -> Dict[str, Any]:
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
    logger.debug(f"VQA generated {len(generated_texts)} sequences (mode={mode})")

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


def create_app(cfg: EvalServerConfig, multi_gpu_server: MultiGPUEvalServer | None = None):
    app = FastAPI(title="RFM Multi-GPU Evaluation Server")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize multi-GPU server
    num_gpus = getattr(cfg, "num_gpus", None)
    max_workers = getattr(cfg, "max_workers", None)

    multi_gpu_server = multi_gpu_server or MultiGPUEvalServer(cfg.model_path, num_gpus, max_workers)
    logger.info(f"Multi-GPU eval server initialized with {multi_gpu_server.num_gpus} GPUs")

    @app.post("/evaluate_batch")
    async def evaluate_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a batch of preference samples using the Multi-GPU server."""
        logger.debug(f"Received /evaluate_batch request with keys: {list(batch.keys())}")
        return await multi_gpu_server.process_batch(batch)

    @app.post("/evaluate_batch_npy")
    async def evaluate_batch_npy(request: Request) -> Dict[str, Any]:
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
        logger.debug(
            f"Received /evaluate_batch_npy request with {len(numpy_arrays)} numpy arrays "
            f"and {len(other_data)} other fields"
        )
        return await multi_gpu_server.process_batch(batch_data)

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

                                if traj_key in ["video_embeddings", "text_embedding"]:
                                    if traj_key in value and value[traj_key] is not None:
                                        if isinstance(value[traj_key], np.ndarray):
                                            value[traj_key] = torch.tensor(value[traj_key])
                                        elif isinstance(value[traj_key], list):
                                            value[traj_key] = torch.tensor(value[traj_key])

                samples.append(sample_data)

        return samples

    @app.get("/gpu_status")
    def get_gpu_status() -> Dict[str, Any]:
        """Get status of all GPUs and pool."""
        return multi_gpu_server.get_pool_status()

    @app.get("/health")
    def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        status = multi_gpu_server.get_pool_status()
        return {"status": "healthy", "available_gpus": status["available_gpus"], "total_gpus": status["total_gpus"]}

    @app.get("/model_info")
    def get_model_info() -> Dict[str, Any]:
        """Get model information and experiment configuration."""

        def serialize_config(config_obj: Any) -> Any:
            """Recursively serialize dataclass to dict, handling nested dataclasses."""
            if hasattr(config_obj, "__dataclass_fields__"):
                result = {}
                for field_name, field_value in asdict(config_obj).items():
                    result[field_name] = serialize_config(field_value)
                return result
            elif isinstance(config_obj, dict):
                return {k: serialize_config(v) for k, v in config_obj.items()}
            elif isinstance(config_obj, list):
                return [serialize_config(item) for item in config_obj]
            elif isinstance(config_obj, (str, int, float, bool, type(None))):
                return config_obj
            else:
                # Fallback: convert to string for non-serializable types
                return str(config_obj)

        def get_model_architecture_info(model: Any) -> Dict[str, Any]:
            """Extract model architecture information."""
            model_info = {
                "model_class": model.__class__.__name__,
                "model_module": model.__class__.__module__,
            }

            # Count parameters
            try:
                all_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                frozen_params = all_params - trainable_params

                model_info.update({
                    "total_parameters": all_params,
                    "trainable_parameters": trainable_params,
                    "frozen_parameters": frozen_params,
                    "trainable_percentage": (100 * trainable_params / all_params) if all_params > 0 else 0.0,
                })
            except Exception as e:
                logger.warning(f"Could not count model parameters: {e}")
                model_info["parameter_count_error"] = str(e)

            # Get model architecture summary (first few layers)
            try:
                architecture_summary = []
                for name, module in list(model.named_children())[:10]:  # First 10 top-level modules
                    module_type = module.__class__.__name__
                    num_params = sum(p.numel() for p in module.parameters())
                    architecture_summary.append({
                        "name": name,
                        "type": module_type,
                        "parameters": num_params,
                    })
                model_info["architecture_summary"] = architecture_summary
            except Exception as e:
                logger.warning(f"Could not get architecture summary: {e}")
                model_info["architecture_summary_error"] = str(e)

            return model_info

        exp_config_dict = serialize_config(multi_gpu_server.exp_config)

        # Get model architecture info from the base model
        model_arch_info = None
        try:
            model_arch_info = get_model_architecture_info(multi_gpu_server.base_model)
        except Exception as e:
            logger.warning(f"Could not get model architecture info: {e}")
            model_arch_info = {"error": str(e)}

        return {
            "model_path": multi_gpu_server.model_path,
            "num_gpus": multi_gpu_server.num_gpus,
            "experiment_config": exp_config_dict,
            "model_architecture": model_arch_info,
        }

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        multi_gpu_server.shutdown()

    return app


@hydra_main(version_base=None, config_path="../configs", config_name="eval_config_server")
def main(cfg: DictConfig):
    """Main entry point for evaluation server using Hydra configuration."""
    # Convert Hydra config to dataclass
    eval_cfg = convert_hydra_to_dataclass(cfg, EvalServerConfig)

    # Display the configuration in a nice Rich format
    display_config(eval_cfg)

    # Ensure pretrained checkpoint is specified
    if not eval_cfg.model_path:
        raise ValueError("Eval config must set model_path to a pretrained checkpoint.")

    multi_gpu_server = MultiGPUEvalServer(
        model_path=eval_cfg.model_path,
        num_gpus=eval_cfg.num_gpus,
        max_workers=eval_cfg.max_workers,
    )
    display_config(multi_gpu_server.exp_config)

    app = create_app(eval_cfg, multi_gpu_server)
    print(f"Running multi-GPU eval server on {eval_cfg.server_url}:{eval_cfg.server_port}")
    print(f"Using {eval_cfg.num_gpus or torch.cuda.device_count()} GPUs")
    uvicorn.run(app, host=eval_cfg.server_url, port=eval_cfg.server_port)


if __name__ == "__main__":
    main()
