#!/usr/bin/env python3
"""
FastAPI server to evaluate RFM model batches.

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
from calendar import c
import yaml
import base64
import io
from typing import Any, Dict, List, Optional, Union


import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_serializer
from huggingface_hub import hf_hub_download

from rfm.utils.setup_utils import setup_model_and_processor
from rfm.configs.eval_configs import EvaluationConfig
from rfm.configs.experiment_configs import DataConfig, ModelConfig
from rfm.data.batch_collator import BatchCollator, PreferenceSample
from evals.eval_utils import BatchPayload


def compute_batch_outputs(model, tokenizer, batch_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
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


def create_app(cfg: EvaluationConfig, model_config: ModelConfig):
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load model/processor once
    processor, model = setup_model_and_processor(model_config, cfg.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    batch_collator = BatchCollator(processor, resized_height=cfg.data.resized_height, resized_width=cfg.data.resized_width)

    @app.post("/evaluate_batch")
    def evaluate_batch(batch: Dict[str, Any]):
        # For now we only support preference-style evaluation
        pref_samples = [s for s in batch["samples"] if s["sample_type"] == "preference"]
        # convert to PreferenceSample
        pref_samples = [PreferenceSample(**s) for s in pref_samples]
        batch_inputs = batch_collator(pref_samples)["preference_inputs"]
        outputs = compute_batch_outputs(model, processor.tokenizer, batch_inputs)
        return outputs

    return app


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="rfm/configs/eval_config.yaml")
    args = parser.parse_args()

    # Load evaluation config manually
    print(f"Loading evaluation config from: {args.config_path}")
    with open(args.config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    cfg = EvaluationConfig(**config_dict)
    cfg.data = DataConfig(**config_dict["data"])
    print(f"Evaluation config: {cfg}")

    # Download model config from Hugging Face
    model_config_path = hf_hub_download(repo_id=cfg.model_path, filename="config.yaml")
    with open(model_config_path, "r") as f:
        model_config_dict = yaml.safe_load(f)

    model_config = ModelConfig(**model_config_dict["model"])

    app = create_app(cfg, model_config)
    print(f"Running server on {cfg.server_url}:{cfg.server_port}")
    uvicorn.run(app, host=cfg.server_url, port=cfg.server_port)


if __name__ == "__main__":
    main()
