import ast
from re import M, S
import wandb
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import List, Dict, Optional, Union, Any, Tuple
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from transformers.trainer_utils import EvalPrediction
from transformers.trainer import PredictionOutput

from rfm.utils.logging import is_rank_0, rank_0_print
from rfm.utils.metrics import compute_auc, compute_spearman_correlation
from rfm.utils.logging import _timer
from rfm.trainers.rfm_heads_trainer import RFMHeadsTrainer


class VQATrainer(RFMHeadsTrainer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss for VQA tasks."""

        # Extract the separate batches
        preference_inputs = inputs.get("preference_inputs", {})
        similarity_inputs = inputs.get("similarity_inputs", {})
        progress_inputs = inputs.get("progress_inputs", {})

        num_preferences = inputs.get("num_preferences", 0)
        num_similarities = inputs.get("num_similarities", 0)
        num_progress = inputs.get("num_progress", 0)

        # Initialize loss components and metadata
        total_loss = 0.0
        log_metadata = {}

        # Compute VQA loss for each type of input
        if num_preferences > 0 and preference_inputs:
            with _timer("time/compute_vqa_loss", timing_raw=self.timing_raw):
                preference_loss, loss_dict = self._compute_vqa_loss(
                    model, preference_inputs, return_outputs=True, mode="preference"
                )
            total_loss += preference_loss
            log_metadata.update(loss_dict)

        if num_similarities > 0 and similarity_inputs:
            with _timer("time/compute_vqa_loss", timing_raw=self.timing_raw):
                similarity_loss, loss_dict = self._compute_vqa_loss(
                    model, similarity_inputs, return_outputs=True, mode="similarity"
                )
            total_loss += similarity_loss
            log_metadata.update(loss_dict)

        if num_progress > 0 and progress_inputs:
            with _timer("time/compute_vqa_loss", timing_raw=self.timing_raw):
                progress_loss, loss_dict = self._compute_vqa_loss(
                    model, progress_inputs, return_outputs=True, mode="progress"
                )
            total_loss += progress_loss
            log_metadata.update(loss_dict)

        # Always store custom losses for logging (even when return_outputs=False)
        self.log_metadata = log_metadata

        if return_outputs:
            # Combine outputs from all loss functions
            extra_info = {
                **log_metadata,
                "total_loss": total_loss.item(),
                "batch_size": num_preferences + num_similarities + num_progress,
            }
            return total_loss, extra_info

        return total_loss

    def _compute_vqa_loss(self, model, inputs, return_outputs=False, mode=None):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
            labels=inputs["labels"],
        )
        loss = outputs.loss
        loss_dict = {f"{mode}_loss": loss.item()}
        return (loss, loss_dict) if return_outputs else loss
