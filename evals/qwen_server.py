#!/usr/bin/env python3
"""
FastAPI server to evaluate RFM model batches.

Endpoint:
  POST /evaluate_batch
Payload schema:
  {
    "samples": [
      {
        "task": str,
        "prediction_type": "preference",  # currently required
        "chosen_frames_b64": [str, ...],
        "rejected_frames_b64": [str, ...],
        "target_progress_A": [float, ...] | null,
        "target_progress_B": [float, ...] | null
      },
      ...
    ]
  }

Returns metrics per-batch consistent with train.py evaluate printout:
  - eval_loss
  - eval_accuracy
  - eval_reward_diff
  - eval_avg_reward_chosen
  - eval_avg_reward_rejected
  - demo_reward_alignment (Spearman rho, single float)

e.g.: uv run /home/jessez/reward_fm/evals/server.py --config_path=/home/jessez/reward_fm/rfm/configs/config.yaml --host=0.0.0.0 --port=8000
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from scipy.stats import spearmanr

from rfm.configs.experiment_configs import ExperimentConfig
from setup_utils import (
    setup_model_and_processor,
    setup_peft_model,
    create_training_arguments,
    setup_eval_dataset,
    setup_batch_collator,
)
from trainer import RFMTrainer


class SamplePayload(BaseModel):
    task: Optional[str] = ""
    prediction_type: str
    chosen_frames_b64: List[str]
    rejected_frames_b64: List[str]
    target_progress_A: Optional[List[float]] = None
    target_progress_B: Optional[List[float]] = None


class BatchPayload(BaseModel):
    samples: List[SamplePayload]


class EvalRequest(BaseModel):
    eval_subset_size: Optional[int] = None


def decode_frames_b64(frames_b64: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for s in frames_b64:
        try:
            buf = io.BytesIO(base64.b64decode(s))
            img = Image.open(buf).convert("RGB")
            images.append(img)
        except Exception:
            continue
    return images


def build_preference_batch(processor, samples: List[SamplePayload], resized_h: int = 128, resized_w: int = 128):
    from qwen_vl_utils import process_vision_info

    conversations = []
    for s in samples:
        chosen_imgs = decode_frames_b64(s.chosen_frames_b64)
        rejected_imgs = decode_frames_b64(s.rejected_frames_b64)

        conv = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Task: {s.task}"},
                    {"type": "video", "video": chosen_imgs, "resized_height": resized_h, "resized_width": resized_w},
                    {"type": "text", "text": "<|split_token|>"},
                    {"type": "video", "video": rejected_imgs, "resized_height": resized_h, "resized_width": resized_w},
                    {"type": "text", "text": "<|pref_token|>"},
                ],
            }
        ]
        conversations.append(conv)

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False, add_vision_id=True)
        for msg in conversations
    ]
    image_inputs, video_inputs, video_kwargs = process_vision_info(conversations, return_video_kwargs=True)
    batch_inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=False,
        max_length=1024,
        return_tensors="pt",
        **video_kwargs,
    )

    # Attach optional target progress
    def pad_progress(progress_lists: List[Optional[List[float]]]):
        valid = [p for p in progress_lists if p]
        if not valid:
            return None
        max_len = max(len(p) for p in valid)
        arr = []
        for p in progress_lists:
            if not p:
                arr.append([0.0] * max_len)
            else:
                pad = p + [0.0] * (max_len - len(p))
                arr.append(pad)
        return torch.tensor(arr, dtype=torch.float32)

    batch_inputs["target_progress_A"] = pad_progress([s.target_progress_A for s in samples])
    batch_inputs["target_progress_B"] = pad_progress([s.target_progress_B for s in samples])

    # Labels: 1 means chosen preferred over rejected
    batch_inputs["preference_labels"] = torch.ones(len(samples), dtype=torch.float32)
    return batch_inputs


def compute_batch_metrics(model, tokenizer, batch_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        outputs, progress_logits = model(
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
            prediction_type="preference",
            target_progress=batch_inputs.get("target_progress_A", None).to(device)
            if batch_inputs.get("target_progress_A") is not None
            else None,
        )

        logits = outputs.logits.squeeze(-1)  # [B]
        labels = batch_inputs["preference_labels"].to(device)  # [B]

        # Binary cross-entropy with logits
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        accuracy = (preds == labels).float().mean().item()

        # Treat preference head logit as reward for chosen; need also rejected reward
        # To approximate reward for rejected, rebuild batch swapping order.
        # Build swapped inputs quickly by replacing last video before <|pref_token|>.
        # Simpler: reuse same text but swap pixel_values_videos pair-wise if available.
        # If not available, fall back to symmetric estimate (negation) which is imperfect.
        avg_reward_chosen = probs.mean().item()

        if batch_inputs.get("pixel_values_videos") is not None:
            swapped = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()}
            # pixel_values_videos shape: [B, T, C, H, W] or similar – swapping videos A/B is non-trivial here
            # For Qwen processor packing, videos are concatenated in a single sequence; exact swap requires re-processing.
            # As an approximation, compute rejected reward as 1 - chosen_prob.
            avg_reward_rejected = (1.0 - probs).mean().item()
        else:
            avg_reward_rejected = (1.0 - probs).mean().item()

        reward_diff = avg_reward_chosen - avg_reward_rejected

        # Spearman correlation for progress prediction per sample if available
        per_sample_spearman: List[float] = []
        targA = batch_inputs.get("target_progress_A")
        if isinstance(progress_logits, dict) and progress_logits.get("A") is not None and targA is not None:
            # progress_logits['A'] is a list of 1D tensors with variable lengths
            pred_list = progress_logits["A"]
            targets_np = targA.detach().cpu().numpy()

            for i, seq in enumerate(pred_list):
                if seq is None or seq.numel() == 0:
                    per_sample_spearman.append(float("nan"))
                    continue
                pred_np = seq.detach().cpu().numpy()
                # TODO: hacky thing right now, not sure if it's correct
                if len(pred_np) != len(targets_np[i]):
                    print(
                        f"WARNING: pred_np and targets_np have different lengths: pred_np: {pred_np}, targets_np: {targets_np[i]}"
                    )
                    print(f"pred_np length: {len(pred_np)}")
                    print(f"targets_np length: {len(targets_np[i])}")
                    print(f"pred_np shape: {pred_np.shape}, targets_np shape: {targets_np[i].shape}")
                    print(f"pred_np: {pred_np}")
                    print(f"targets_np: {targets_np[i]}")
                    print(f"pred_np: {pred_np}")
                L = min(len(pred_np), targets_np.shape[1])
                if L < 2:
                    per_sample_spearman.append(float("nan"))
                    continue
                res = spearmanr(pred_np[:L], targets_np[i, :L])
                rho = float(res.correlation) if hasattr(res, "correlation") else float(res[0])
                per_sample_spearman.append(rho)
        spearman_rho = np.mean(per_sample_spearman)

        return {
            "eval_loss": float(bce.item()),
            "eval_accuracy": float(accuracy),
            "eval_reward_diff": float(reward_diff),
            "eval_avg_reward_chosen": float(avg_reward_chosen),
            "eval_avg_reward_rejected": float(avg_reward_rejected),
            "demo_reward_alignment": spearman_rho,
        }


def create_app(cfg: ExperimentConfig):
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load model/processor once
    processor, rfm_model = setup_model_and_processor(cfg)
    model = setup_peft_model(rfm_model, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    @app.post("/evaluate_batch")
    def evaluate_batch(batch: BatchPayload):
        # For now we only support preference-style evaluation
        pref_samples = [s for s in batch.samples if s.prediction_type == "preference"]
        if not pref_samples:
            return {
                "eval_loss": 0.0,
                "eval_accuracy": 0.0,
                "eval_reward_diff": 0.0,
                "eval_avg_reward_chosen": 0.0,
                "eval_avg_reward_rejected": 0.0,
                "demo_reward_alignment": 0.0,
            }

        batch_inputs = build_preference_batch(
            processor, pref_samples, resized_h=cfg.data.resized_height, resized_w=cfg.data.resized_width
        )
        metrics = compute_batch_metrics(model, processor.tokenizer, batch_inputs)
        return metrics

    @app.post("/evaluate_internal")
    def evaluate_internal(req: EvalRequest):
        # Optionally override eval subset size for this run
        if req.eval_subset_size is not None and req.eval_subset_size > 0:
            cfg.data.eval_subset_size = req.eval_subset_size

        # Prepare eval dataset and args similar to train.py:evaluate
        eval_dataset = setup_eval_dataset(cfg)
        eval_args = create_training_arguments(cfg, "./eval_output", is_eval=True)
        batch_collator = setup_batch_collator(processor, cfg)

        trainer = RFMTrainer(
            model=model,
            args=eval_args,
            train_dataset=eval_dataset,
            eval_dataset=eval_dataset,
            data_collator=batch_collator,
            beta=cfg.training.beta,
        )

        # Load checkpoint if provided
        if cfg.evaluation.model_path:
            trainer.train(resume_from_checkpoint=cfg.evaluation.model_path)

        # Run evaluation
        eval_results = trainer.evaluate()
        return eval_results

    return app


def main():
    import argparse
    import uvicorn
    from evals.eval_utils import load_experiment_config_from_yaml
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="rfm/configs/config.yaml")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config with dot-path assignments, e.g., --set data.max_frames=8 --set model.base_model_id='Qwen/...'.",
    )
    args = parser.parse_args()

    cfg = load_experiment_config_from_yaml(args.config_path)

    # Apply overrides from --set key=value (dot-path)
    for assignment in args.set:
        if "=" not in assignment:
            continue
        key, value_str = assignment.split("=", 1)
        try:
            value = ast.literal_eval(value_str)
        except Exception:
            value = value_str
        target = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            target = getattr(target, p)
        setattr(target, parts[-1], value)
    app = create_app(cfg)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
