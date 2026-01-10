#!/usr/bin/env python3
"""
FastAPI server to evaluate baseline models (RL-VLM-F, GVL, VLAC, RoboReward) with batch processing.

Usage examples:
    # RL-VLM-F baseline server
    uv run python rfm/evals/baseline_eval_server.py \
        reward_model=rlvlmf \
        vlm_provider=gemini \
        server_port=8001
    
    # VLAC baseline server
    uv run python rfm/evals/baseline_eval_server.py \
        reward_model=vlac \
        vlac_model_path=InternRobotics/VLAC \
        server_port=8003
    
    # RoboReward baseline server
    uv run python rfm/evals/baseline_eval_server.py \
        reward_model=roboreward \
        roboreward_model_path=teetone/RoboReward-8B \
        server_port=8003

Endpoints:
  POST /evaluate_batch        - JSON payload with samples
  POST /evaluate_batch_npy    - multipart payload with .npy blobs for numpy arrays
  GET /health                 - Health check
  GET /model_info             - Model information

Response payload per request contains predictions grouped by sample type:
  {
    "outputs_preference": [...],   # Preference predictions
    "outputs_progress": [...],     # Progress predictions
  }
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from threading import Lock
from typing import Any, Dict, List, Optional

import uvicorn
import numpy as np
from omegaconf import DictConfig
from hydra import main as hydra_main
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from rfm.configs.eval_configs import BaselineEvalConfig
from rfm.data.dataset_types import PreferenceSample, ProgressSample
from rfm.evals.baselines.rlvlmf import RLVLMF
from rfm.evals.baselines.gvl import GVL
from rfm.evals.baselines.vlac import VLAC
from rfm.evals.baselines.roboreward import RoboReward
from rfm.evals.run_baseline_eval import process_preference_sample, process_progress_sample
from rfm.evals.eval_utils import parse_npy_form_data, reconstruct_payload_from_npy
from rfm.utils.config_utils import display_config, convert_hydra_to_dataclass
from rfm.utils.logger import get_logger, setup_loguru_logging

LOG_LEVEL = "TRACE"
setup_loguru_logging(log_level=LOG_LEVEL)
logger = get_logger()
logger.info(f"rfm.baseline_eval_server logger initialized at level {LOG_LEVEL}")


def process_batch_helper(
    model: Any,
    reward_model: str,
    batch_data: List[Dict[str, Any]],
    job_id: int = 0,
) -> Dict[str, Any]:
    """Synchronous batch processing."""
    if not batch_data:
        raise ValueError("No samples found in batch data")

    logger.debug(f"[job {job_id}] Processing {len(batch_data)} samples")

    input_samples: List[Any] = []
    for sample in batch_data:
        if isinstance(sample, (PreferenceSample, ProgressSample)):
            input_samples.append(sample)
        elif isinstance(sample, dict):
            sample_type = sample.get("sample_type")
            if sample_type == "preference":
                input_samples.append(PreferenceSample(**sample))
            elif sample_type == "progress":
                input_samples.append(ProgressSample(**sample))
            else:
                raise ValueError(f"Unsupported sample_type: {sample_type}")
        else:
            raise ValueError(f"Unsupported sample object type: {type(sample)}")

    outputs_preference = []
    outputs_progress = []

    # Process samples
    for sample in input_samples:
        if isinstance(sample, PreferenceSample):
            if reward_model != "rlvlmf":
                logger.warning(f"Preference samples only supported for rlvlmf, got {reward_model}")
                continue
            result = process_preference_sample(sample, model)
            if result:
                outputs_preference.append(result)
        elif isinstance(sample, ProgressSample):
            if reward_model not in ["gvl", "vlac", "roboreward"]:
                logger.warning(f"Progress samples only supported for gvl, vlac, roboreward, got {reward_model}")
                continue
            result = process_progress_sample(sample, model)
            if result:
                outputs_progress.append(result)

    return {
        "outputs_preference": outputs_preference if outputs_preference else None,
        "outputs_progress": outputs_progress if outputs_progress else None,
    }


class BaselineEvalServer:
    """Baseline evaluation server that processes batches of samples."""

    def __init__(self, cfg: BaselineEvalConfig):
        self.cfg = cfg
        self.reward_model = cfg.reward_model
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._job_counter = 0
        self._job_counter_lock = Lock()

        logger.info(f"Initializing baseline eval server: reward_model={self.reward_model}")

        # Initialize model
        self._initialize_model()

        logger.info("Baseline eval server initialized successfully")

    def _initialize_model(self):
        """Initialize the baseline model based on config."""
        if self.reward_model == "rlvlmf":
            self.model = RLVLMF(vlm_provider=self.cfg.vlm_provider, temperature=self.cfg.temperature)
        elif self.reward_model == "gvl":
            self.model = GVL(max_frames=self.cfg.gvl_max_frames, offset=self.cfg.gvl_offset)
        elif self.reward_model == "vlac":
            if not self.cfg.vlac_model_path:
                raise ValueError("vlac_model_path is required for VLAC baseline")
            self.model = VLAC(
                model_path=self.cfg.vlac_model_path,
                device=self.cfg.vlac_device,
                model_type=self.cfg.vlac_model_type,
                temperature=self.cfg.vlac_temperature,
                batch_num=self.cfg.vlac_batch_num,
                skip=self.cfg.vlac_skip,
                frame_skip=self.cfg.vlac_frame_skip,
                use_images=self.cfg.vlac_use_images,
            )
        elif self.reward_model == "roboreward":
            self.model = RoboReward(
                model_path=self.cfg.roboreward_model_path,
                max_new_tokens=self.cfg.roboreward_max_new_tokens,
            )
        else:
            raise ValueError(
                f"Unknown reward_model: {self.reward_model}. Must be 'rlvlmf', 'gvl', 'vlac', or 'roboreward'"
            )

        logger.info(f"Loaded {self.reward_model} baseline model")

    async def process_batch(self, batch_data: List[Dict[str, Any]]):
        """Process a batch using the executor."""
        loop = asyncio.get_event_loop()

        with self._job_counter_lock:
            self._job_counter += 1
            job_id = self._job_counter

        logger.debug(f"[job {job_id}] Processing batch with {len(batch_data)} samples")

        start_time = time.time()

        try:
            # Process batch in thread pool
            result = await loop.run_in_executor(
                self.executor,
                process_batch_helper,
                self.model,
                self.reward_model,
                batch_data,
                job_id,
            )

            processing_time = time.time() - start_time
            logger.debug(f"[job {job_id}] Completed in {processing_time:.3f}s")

            return result

        except Exception as e:
            logger.error(f"[job {job_id}] Error processing batch: {e}", exc_info=True)
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "reward_model": self.reward_model,
            "config": asdict(self.cfg),
        }

    def shutdown(self):
        """Shutdown the executor."""
        logger.info("Shutting down baseline eval server...")
        self.executor.shutdown(wait=True)
        logger.info("Baseline eval server shutdown complete")


def create_app(cfg: BaselineEvalConfig, baseline_server: BaselineEvalServer | None = None):
    app = FastAPI(title="Baseline Evaluation Server")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize baseline server
    baseline_server = baseline_server or BaselineEvalServer(cfg)
    logger.info(f"Baseline eval server initialized with model: {baseline_server.reward_model}")

    @app.post("/evaluate_batch")
    async def evaluate_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a batch of samples using the baseline server."""
        logger.debug(f"Received /evaluate_batch request with keys: {list(batch.keys())}")

        # Handle both list and dict formats
        if isinstance(batch, list):
            batch_data = batch
        elif isinstance(batch, dict) and "samples" in batch:
            batch_data = batch["samples"]
        else:
            # Assume it's a single sample wrapped in a dict
            batch_data = [batch]

        return await baseline_server.process_batch(batch_data)

    @app.post("/evaluate_batch_npy")
    async def evaluate_batch_npy(request: Request) -> Dict[str, Any]:
        """Evaluate a batch with .npy file support for numpy arrays.

        This endpoint handles multipart form data where:
        - numpy arrays are sent as .npy files
        - other data is sent as form fields
        """
        # Parse form data
        form_data = await request.form()

        # Extract numpy arrays and other data using shared utility (await async function)
        numpy_arrays, other_data = await parse_npy_form_data(form_data)

        # Reconstruct the original payload structure (baselines don't need torch tensor conversion)
        batch_data = reconstruct_payload_from_npy(
            numpy_arrays,
            other_data,
            trajectory_keys=["chosen_trajectory", "rejected_trajectory", "trajectory"],
            convert_embeddings_to_torch=False,
        )

        # Process the batch
        logger.debug(
            f"Received /evaluate_batch_npy request with {len(numpy_arrays)} numpy arrays "
            f"and {len(other_data)} other fields"
        )
        return await baseline_server.process_batch(batch_data)

    @app.get("/health")
    def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        return {"status": "healthy", "reward_model": baseline_server.reward_model}

    @app.get("/model_info")
    def get_model_info() -> Dict[str, Any]:
        """Get model information."""
        return baseline_server.get_model_info()

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        baseline_server.shutdown()

    return app


@hydra_main(version_base=None, config_path="../configs", config_name="baseline_eval_config")
def main(cfg: DictConfig):
    """Main entry point for baseline evaluation server using Hydra configuration."""
    # Convert Hydra config to dataclass
    baseline_cfg = convert_hydra_to_dataclass(cfg, BaselineEvalConfig)

    # Display the configuration
    display_config(baseline_cfg)

    # Validate reward model
    if baseline_cfg.reward_model not in ["rlvlmf", "gvl", "vlac", "roboreward"]:
        raise ValueError(
            f"reward_model must be 'rlvlmf', 'gvl', 'vlac', or 'roboreward', got {baseline_cfg.reward_model}"
        )

    baseline_server = BaselineEvalServer(baseline_cfg)
    app = create_app(baseline_cfg, baseline_server)

    print(f"Running baseline eval server on {baseline_cfg.server_url}:{baseline_cfg.server_port}")
    print(f"Using {baseline_cfg.reward_model} baseline model")
    uvicorn.run(app, host=baseline_cfg.server_url, port=baseline_cfg.server_port)


if __name__ == "__main__":
    main()
