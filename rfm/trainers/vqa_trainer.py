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


class VQATrainer(Trainer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        # Initialize custom loss tracking
        self.log_metadata = {}
        self.loss_keys = [
            "total_loss",
            "preference_loss",
            "similarity_loss",
            "progress_loss",
            "preference_accuracy",
            "spearman_corr_avg",
        ]

        self.log_keys = [
            "num_rewind_frames_min",
            "num_rewind_frames_max",
            "num_rewind_frames_mean",
            "num_rewound_trajs",
        ]
        self.global_metadata = {
            "total_samples": 0,
            "total_samples_with_rewound_trajs": 0,
        }
        self.timing_raw = {}

        # # ---- generation/decoding helpers ----
        # self.postprocess_fn = getattr(self.config, "postprocess_fn", None)
        # default_stops = ("</s>",)
        # self.stop_strings: Tuple[str, ...] = tuple(
        #     getattr(self.config, "stop_strings", default_stops) or default_stops
        # )

        # # pad token for batch decoding/generation
        # if self.tokenizer is not None and self.tokenizer.pad_token_id is None:
        #     if getattr(self.tokenizer, "eos_token_id", None) is not None:
        #         self.tokenizer.pad_token = self.tokenizer.eos_token

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step and log custom losses.
        """
        self.timing_raw = {}
        with _timer("time/training_step", timing_raw=self.timing_raw):
            # Call the parent training_step to handle all the standard training logic
            loss = super().training_step(model, inputs, num_items_in_batch)

        # Log custom losses at specified intervals
        if self.state.global_step % self.args.logging_steps == 0:
            self._log_metadata()

        return loss

    def _log_metadata(self):
        """Log custom VQA losses to wandb and console."""
        if not self.log_metadata:
            return

        # Aggregate custom losses across all processes if using distributed training
        aggregated_metadata = self._aggregate_log_metadata()
        aggregated_losses = {
            f"train/{key}": aggregated_metadata[key] for key in self.loss_keys if key in aggregated_metadata
        }
        aggregated_log_keys = {
            f"misc/{key}": aggregated_metadata[key] for key in self.log_keys if key in aggregated_metadata
        }

        # Prepare logging data using aggregated losses
        log_data = {
            "step": self.state.global_step,
            **self.timing_raw,
        }
        log_data.update(aggregated_losses)
        log_data.update(aggregated_log_keys)

        # also log the global metadata
        log_global = {f"counts/{key}": self.global_metadata[key] for key in self.global_metadata}
        log_data.update(log_global)

        # Log to wandb if available and configured (only on rank 0)
        if self.args.report_to and "wandb" in self.args.report_to and is_rank_0():
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(log_data)
            except ImportError:
                rank_0_print("Warning: wandb not available for logging custom losses")

        # Log to console on rank 0
        if is_rank_0():
            rank_0_print(f"Step {self.state.global_step} Custom Losses (Aggregated):")
            for key in self.loss_keys:
                if f"train/{key}" in aggregated_losses:
                    rank_0_print(f"  {key}: {aggregated_losses[f'train/{key}']:.6f}")

            rank_0_print("-" * 50)
            for key in self.log_keys:
                if f"misc/{key}" in aggregated_log_keys:
                    rank_0_print(f"  {key}: {aggregated_log_keys[f'misc/{key}']:.6f}")

            rank_0_print("-" * 50)
            for key in log_global:
                rank_0_print(f"  {key}: {log_global[key]}")

            rounded_times = {k: round(v, 2) for k, v in self.timing_raw.items()}
            rank_0_print(f"Timing raw: {rounded_times}")

    def _aggregate_log_metadata(self):
        """Aggregate custom losses across all processes using all_reduce."""
        if not self.log_metadata:
            return {}

        # If not using distributed training, return losses as-is
        if not dist.is_initialized():
            return self.log_metadata.copy()

        aggregated = {}

        # Aggregate loss values (averages) across all processes

        for key in self.loss_keys + self.log_keys:
            if key in self.log_metadata:
                # Convert to tensor for all_reduce
                loss_tensor = torch.tensor(self.log_metadata[key], device=self.accelerator.device)

                # Sum across all processes
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

                # Average by world size
                aggregated[key] = (loss_tensor / dist.get_world_size()).item()

        # Aggregate count values (mean) across all processes
        for key in self.log_keys:
            if key in self.log_metadata:
                # Convert to tensor for all_reduce
                count_tensor = torch.tensor(self.log_metadata[key], device=self.accelerator.device)

                # Sum across all processes
                dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

                # Average by world size
                aggregated[key] = (count_tensor / dist.get_world_size()).item()

        return aggregated

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval") -> Dict[str, float]:
        """
        Override evaluate method to implement custom RFM evaluation metrics.
        """
        # Get the evaluation dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # Set model to eval mode
        self.model.eval()

        # Run evaluation
        outputs = []
        with _timer("time/evaluate", timing_raw=self.timing_raw):
            with torch.no_grad():
                for step, inputs in tqdm(
                    enumerate(eval_dataloader),
                    total=len(eval_dataloader),
                    desc="Evaluating",
                ):
                    # Move inputs to device
                    inputs = self._prepare_inputs(inputs)

                    _, loss_dicts = self.compute_loss(self.model, inputs, return_outputs=True, training=False)
                    outputs.append(loss_dicts)

        # Aggregate outputs
        aggregated_outputs = {}

        # assume that we already called .item() on the outputs
        for key in self.loss_keys:
            if key in outputs[0]:
                aggregated_outputs[key] = [output[key] for output in outputs if key in output]
                aggregated_outputs[key] = np.array(aggregated_outputs[key]).mean()

        # Compute metrics
        metrics = {f"eval/{key}": aggregated_outputs[key] for key in aggregated_outputs}

        # Log metrics
        if is_rank_0():
            rank_0_print(f"\n=== Custom RFM Evaluation Results ===")
            for key, value in metrics.items():
                rank_0_print(f"{key}: {value:.6f}")
            rank_0_print("=" * 50)

        # Also log to wandb if available and configured (only on rank 0)
        if self.args.report_to and "wandb" in self.args.report_to and is_rank_0():
            if wandb.run is not None:
                wandb.log(metrics)

        return metrics

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

        # Log rewind length stats if available in preference inputs
        rewind_stats = {}
        if num_preferences > 0 and preference_inputs:
            rewind_lengths = preference_inputs.get("rewind_lengths", None)

            if rewind_lengths is not None:
                rewind_lengths = rewind_lengths.tolist()
                num_rewind_frames_min = min(rewind_lengths)
                num_rewind_frames_max = max(rewind_lengths)
                num_rewind_frames_mean = np.mean(rewind_lengths)
                num_rewound_trajs = np.array(rewind_lengths).nonzero()[0].size
                rewind_stats = {
                    "num_rewind_frames_min": num_rewind_frames_min,
                    "num_rewind_frames_max": num_rewind_frames_max,
                    "num_rewind_frames_mean": num_rewind_frames_mean,
                    "num_rewound_trajs": num_rewound_trajs,
                }
                log_metadata.update(rewind_stats)

        # Always store custom losses for logging (even when return_outputs=False)
        self.log_metadata = log_metadata

        # Update global metadata for training
        # Keep sum counts over all processes
        if kwargs.get("training", True) and dist.is_initialized():
            # add to total batch size and sum across all processes
            batch_size = torch.tensor(num_preferences + num_similarities + num_progress, device=self.accelerator.device)
            dist.all_reduce(batch_size, op=dist.ReduceOp.SUM)
            self.global_metadata["total_samples"] += batch_size.item()

            # total rewounded trajectories
            if "num_rewound_trajs" in rewind_stats:
                total_samples_with_rewound_trajs = torch.tensor(
                    rewind_stats["num_rewound_trajs"], device=self.accelerator.device
                )
                dist.all_reduce(total_samples_with_rewound_trajs, op=dist.ReduceOp.SUM)
                self.global_metadata["total_samples_with_rewound_trajs"] += total_samples_with_rewound_trajs.item()

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

    # # -------------------- generation-based eval/predict --------------------

    # def _extract_answer_from_text(self, text):
    #     import re
    #     m = re.search(r"<ans>(.*?)</ans>", text, re.DOTALL)
    #     return m.group(1).strip() if m else ""

    # def _cfg(self, name: str, default: Any) -> Any:
    #     if hasattr(self.config, name):
    #         return getattr(self.config, name)
    #     if isinstance(self.config, dict) and name in self.config:
    #         return self.config[name]
    #     return default

    # def _build_gen_kwargs(self) -> Dict[str, Any]:
    #     g = dict(
    #         max_new_tokens=int(self._cfg("generation_max_new_tokens", 64)),
    #         num_beams=int(self._cfg("generation_num_beams", 1)),
    #         do_sample=bool(self._cfg("generation_do_sample", False)),
    #     )
    #     tp = self._cfg("generation_top_p", None)
    #     temp = self._cfg("generation_temperature", None)
    #     if tp is not None:
    #         g["top_p"] = float(tp)
    #     if temp is not None:
    #         g["temperature"] = float(temp)
    #     # ZeRO-3 multi-GPU nicety
    #     if getattr(self, "deepspeed", None) is not None:
    #         g["synced_gpus"] = True
    #     return g

    # def _truncate_at_stops(self, text: str) -> str:
    #     for s in self.stop_strings:
    #         if s and s in text:
    #             text = text.split(s, 1)[0]
    #     return text

    # def _decode(self, sequences: np.ndarray) -> List[str]:
    #     if self.tokenizer is None:
    #         return [" ".join(map(str, row.tolist())) for row in sequences]
    #     texts = self.tokenizer.batch_decode(
    #         sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #     )
    #     out = []
    #     for t in texts:
    #         t = self._truncate_at_stops(t)
    #         if self.postprocess_fn is not None:
    #             t = self.postprocess_fn(t)
    #         out.append(t.strip())
    #     return out

    # # ---------- utilities for nested eval batches & device harmonization ----------

    # def _collect_generate_inputs(self, src: Dict[str, Any]) -> Dict[str, Any]:
    #     keep = {
    #         "input_ids", "attention_mask", "position_ids",
    #         "pixel_values", "pixel_values_videos",
    #         "image_grid_thw", "video_grid_thw",
    #     }
    #     return {k: v for k, v in src.items() if k in keep and v is not None}

    # def _move_to_device(self, obj, device):
    #     if torch.is_tensor(obj):
    #         return obj.to(device, non_blocking=True)
    #     if isinstance(obj, (list, tuple)):
    #         t = [self._move_to_device(o, device) for o in obj]
    #         return type(obj)(t)
    #     if isinstance(obj, dict):
    #         return {k: self._move_to_device(v, device) for k, v in obj.items()}
    #     return obj

    # # -------------------- core override: generation-only eval --------------------

    # def prediction_step(
    #     self,
    #     model: torch.nn.Module,
    #     inputs: Dict[str, Any],
    #     prediction_loss_only: bool,
    #     ignore_keys: Optional[List[str]] = None,
    # ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    #     # standard HF prep first
    #     inputs = inputs.get("preference_inputs", {})
    #     inputs = self._prepare_inputs(inputs)

    #     # only generate during evaluation/prediction
    #     if not self.model.training:
    #         inputs_for_generate = self._collect_generate_inputs(inputs)

    #         # *** KEY: move every tensor passed to generate() to the *same* device as input_ids ***
    #         target_device = 'cpu'
    #         inputs_for_generate = self._move_to_device(inputs_for_generate, target_device)

    #         gen_kwargs = self._build_gen_kwargs()

    #         with torch.inference_mode():
    #             generated_tokens = model.generate(**inputs_for_generate, **gen_kwargs)

    #         if isinstance(generated_tokens, torch.Tensor):
    #             generated_tokens = generated_tokens.detach()

    #         # pick labels from the same view (optional, for metrics)
    #         labels = inputs.get("labels", None)
    #         return (None, generated_tokens, labels)

    #     # fallback during training
    #     return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    # def evaluate(
    #     self,
    #     eval_dataset: Optional[Any] = None,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    #     return_predictions: bool = True,
    # ) -> Dict[str, Any]:
    #     eval_dataloader = self.get_eval_dataloader(eval_dataset)
    #     output = self.evaluation_loop(
    #         eval_dataloader,
    #         description="Evaluation",
    #         prediction_loss_only=False,
    #         ignore_keys=ignore_keys,
    #         metric_key_prefix=metric_key_prefix,
    #     )

    #     metrics = dict(output.metrics) if output.metrics is not None else {}
    #     for k in list(metrics.keys()):
    #         if k.endswith("_loss"):
    #             del metrics[k]

    #     decoded_preds: Optional[List[str]] = None
    #     decoded_refs: Optional[List[str]] = None

    #     if output.predictions is not None:
    #         preds = output.predictions
    #         if isinstance(preds, torch.Tensor):
    #             preds = preds.cpu().numpy()
    #         decoded_preds = self._decode(preds)

    #     if output.label_ids is not None:
    #         labels = output.label_ids
    #         if isinstance(labels, torch.Tensor):
    #             labels = labels.cpu().numpy()
    #         if self.tokenizer is not None and self.tokenizer.pad_token_id is not None:
    #             labels = np.where(labels == -100, self.tokenizer.pad_token_id, labels)
    #         decoded_refs = self._decode(labels)

    #     if self.compute_metrics is not None and decoded_preds is not None:
    #         m = self.compute_metrics(
    #             EvalPrediction(
    #                 predictions=np.array(decoded_preds, dtype=object),
    #                 label_ids=np.array(decoded_refs or [], dtype=object),
    #             )
    #         )
    #         if m:
    #             metrics.update(m)

    #     self.log(metrics)
    #     if return_predictions and decoded_preds is not None:
    #         metrics[f"{metric_key_prefix}_predictions"] = decoded_preds
    #         if decoded_refs is not None:
    #             metrics[f"{metric_key_prefix}_references"] = decoded_refs
    #     return metrics

    # def predict(
    #     self,
    #     test_dataset: Any,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "test",
    #     return_predictions: bool = True,
    # ) -> PredictionOutput:
    #     test_dataloader = self.get_test_dataloader(test_dataset)
    #     output = self.evaluation_loop(
    #         test_dataloader,
    #         description="Prediction",
    #         prediction_loss_only=False,
    #         ignore_keys=ignore_keys,
    #         metric_key_prefix=metric_key_prefix,
    #     )

    #     metrics = dict(output.metrics) if output.metrics is not None else {}
    #     for k in list(metrics.keys()):
    #         if k.endswith("_loss"):
    #             del metrics[k]

    #     decoded_preds = None
    #     decoded_refs = None
    #     if output.predictions is not None:
    #         preds = output.predictions
    #         if isinstance(preds, torch.Tensor):
    #             preds = preds.cpu().numpy()
    #         decoded_preds = self._decode(preds)

    #     if output.label_ids is not None:
    #         labels = output.label_ids
    #         if isinstance(labels, torch.Tensor):
    #             labels = labels.cpu().numpy()
    #         if self.tokenizer is not None and self.tokenizer.pad_token_id is not None:
    #             labels = np.where(labels == -100, self.tokenizer.pad_token_id, labels)
    #         decoded_refs = self._decode(labels)

    #     if self.compute_metrics is not None and decoded_preds is not None:
    #         m = self.compute_metrics(
    #             EvalPrediction(
    #                 predictions=np.array(decoded_preds, dtype=object),
    #                 label_ids=np.array(decoded_refs or [], dtype=object),
    #             )
    #         )
    #         if m:
    #             metrics.update(m)

    #     if return_predictions and decoded_preds is not None:
    #         metrics[f"{metric_key_prefix}_predictions"] = decoded_preds
    #         if decoded_refs is not None:
    #             metrics[f"{metric_key_prefix}_references"] = decoded_refs

    #     return PredictionOutput(
    #         predictions=output.predictions,
    #         label_ids=output.label_ids,
    #         metrics=metrics,
    #     )