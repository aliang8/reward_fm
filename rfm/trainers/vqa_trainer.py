import torch
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
import ast
from rfm.utils.metrics import compute_spearman_correlation
from rfm.utils.logging import is_rank_0, rank_0_print, _timer
from transformers import Trainer
from typing import Dict

from rfm.trainers.trainer import RFMTrainer


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
        Override evaluate to compute custom metrics and loss:
        - preference accuracy via generate from prompt-only prefix
        - progress Spearman correlation
        """
        self.model = self.accelerator.prepare_model(self.model, evaluation_mode=True)
        self.model.eval()
        self.model = self.model.to(self.accelerator.device)
        # Get the evaluation dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # tokenizer for decoding
        tokenizer = None
        # Get the processor/tokenizer from the model
        if hasattr(self.model, "processor"):
            tokenizer = self.model.processor.tokenizer
            print(f"Tokenizer found in model.")
        elif hasattr(self.model, "tokenizer"):
            tokenizer = self.model.tokenizer
            print(f"Tokenizer found in model.")
        else:
            # Try to get from the trainer's config or data collator
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None and hasattr(self, "data_collator") and hasattr(self.data_collator, "processor"):
                tokenizer = self.data_collator.processor.tokenizer
                print(f"Tokenizer found in data collator.")
        if tokenizer is None:
            raise ValueError("Tokenizer not found for evaluation decoding.")

        outputs = []
        # local tallies (reduced after loop if DDP)
        pref_correct_local = 0
        pref_total_local = 0
        prog_sum_rho_local = 0.0
        prog_count_local = 0
        with _timer("time/evaluate", timing_raw=self.timing_raw):
            with torch.no_grad():
                for step, batch in tqdm(
                    enumerate(eval_dataloader),
                    total=len(eval_dataloader),
                    desc="Evaluating",
                ):
                    # move to device
                    batch = self._prepare_inputs(batch)

                    # Process different types of samples
                    preference_inputs = batch.get("preference_inputs", {})
                    similarity_inputs = batch.get("similarity_inputs", {})
                    progress_inputs = batch.get("progress_inputs", {})
                    num_preferences = batch.get("num_preferences", 0)
                    num_similarities = batch.get("num_similarities", 0)
                    num_progress = batch.get("num_progress", 0)

                    # ---- Handle preference samples ----
                    if num_preferences > 0 and preference_inputs:
                        # Extract ground truth preference labels
                        preference_labels = preference_inputs.get("preference_labels")  # tensor of 0s and 1s

                        # Create generation inputs (keep only model input fields)
                        valid_model_keys = [
                            "input_ids",
                            "attention_mask",
                            "pixel_values",
                            "pixel_values_videos",
                            "image_grid_thw",
                            "video_grid_thw",
                            "second_per_grid_ts",
                        ]
                        gen_inputs = {k: v for k, v in preference_inputs.items() if k in valid_model_keys}

                        # Use accelerator's autocast for consistent dtype handling
                        with self.accelerator.autocast():
                            generation_outputs = self.model.generate(
                                **gen_inputs,
                                max_new_tokens=10,  # Short answers like "A" or "B"
                                do_sample=False,
                                temperature=1.0,
                                pad_token_id=tokenizer.pad_token_id
                                if tokenizer.pad_token_id is not None
                                else tokenizer.eos_token_id,
                            )

                        # Decode generated texts
                        input_length = gen_inputs["input_ids"].shape[1]
                        generated_tokens = generation_outputs[:, input_length:]
                        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                        # Extract answers and compare with ground truth
                        for i, (generated_text, true_label) in enumerate(zip(generated_texts, preference_labels)):
                            predicted_answer = self._extract_answer_from_text(generated_text)

                            # Convert prediction to label (A=1, B=0)
                            predicted_label = None
                            if predicted_answer.strip().upper() == "A":
                                predicted_label = 1.0
                            elif predicted_answer.strip().upper() == "B":
                                predicted_label = 0.0

                            if predicted_label is not None:
                                pref_total_local += 1
                                if abs(predicted_label - true_label.item()) < 0.5:  # Allow for floating point errors
                                    pref_correct_local += 1

                    # ---- Handle progress samples ----
                    if num_progress > 0 and progress_inputs:
                        # Extract ground truth progress
                        target_progress = progress_inputs.get("target_progress")  # tensor of progress lists
                        quality_labels = progress_inputs.get("quality_labels")  # tensor of quality flags

                        # Create generation inputs (keep only model input fields)
                        valid_model_keys = [
                            "input_ids",
                            "attention_mask",
                            "pixel_values",
                            "pixel_values_videos",
                            "image_grid_thw",
                            "video_grid_thw",
                            "second_per_grid_ts",
                        ]
                        gen_inputs = {k: v for k, v in progress_inputs.items() if k in valid_model_keys}

                        # Use accelerator's autocast for consistent dtype handling
                        with self.accelerator.autocast():
                            generation_outputs = self.model.generate(
                                **gen_inputs,
                                max_new_tokens=100,  # Longer answers for progress lists
                                do_sample=False,
                                temperature=1.0,
                                pad_token_id=tokenizer.pad_token_id
                                if tokenizer.pad_token_id is not None
                                else tokenizer.eos_token_id,
                            )

                        # Decode generated texts
                        input_length = gen_inputs["input_ids"].shape[1]
                        generated_tokens = generation_outputs[:, input_length:]
                        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                        # Extract progress arrays and compute Spearman correlation
                        for i, (generated_text, true_progress, quality) in enumerate(
                            zip(generated_texts, target_progress, quality_labels)
                        ):
                            if quality.item() > 0.5:  # Only evaluate successful trajectories
                                predicted_answer = self._extract_answer_from_text(generated_text)
                                predicted_progress = self._safe_parse_array(predicted_answer)

                                if predicted_progress is not None and true_progress is not None:
                                    # Ensure we only compare non-zero elements of true progress
                                    true_progress_clean = true_progress[true_progress > 0]
                                    if len(true_progress_clean) > 1 and len(predicted_progress) > 1:
                                        # Truncate to minimum length for fair comparison
                                        min_len = min(len(true_progress_clean), len(predicted_progress))
                                        true_slice = true_progress_clean[:min_len]
                                        pred_slice = predicted_progress[:min_len]

                                        if min_len > 1:  # Need at least 2 points for correlation
                                            spearman_corr = compute_spearman_correlation(pred_slice, true_slice)
                                            if not torch.isnan(spearman_corr):
                                                prog_sum_rho_local += spearman_corr.item()
                                                prog_count_local += 1

        # ---- Aggregate losses ----
        aggregated_outputs = {}
        if len(outputs) > 0:
            # assume that we already called .item() on the outputs
            for key in self.loss_keys:
                if key in outputs[0]:
                    aggregated_outputs[key] = [o[key] for o in outputs if key in o]
                    aggregated_outputs[key] = np.array(aggregated_outputs[key]).mean()

        # ---- Reduce metrics across ranks (if DDP) ----
        device = self.accelerator.device
        if dist.is_initialized():
            t_pref = torch.tensor([pref_correct_local, pref_total_local], device=device, dtype=torch.long)
            dist.all_reduce(t_pref, op=dist.ReduceOp.SUM)
            pref_correct, pref_total = int(t_pref[0].item()), int(t_pref[1].item())

            t_prog = torch.tensor([prog_sum_rho_local, prog_count_local], device=device, dtype=torch.float32)
            dist.all_reduce(t_prog, op=dist.ReduceOp.SUM)
            prog_sum_rho, prog_count = float(t_prog[0].item()), float(t_prog[1].item())
        else:
            pref_correct, pref_total = pref_correct_local, pref_total_local
            prog_sum_rho, prog_count = float(prog_sum_rho_local), float(prog_count_local)

        # Compute metrics
        metrics = {f"{metric_key_prefix}/{k}": v for k, v in aggregated_outputs.items()}

        if pref_total > 0:
            metrics[f"{metric_key_prefix}/preference_accuracy"] = pref_correct / max(1, pref_total)

        if prog_count > 0:
            metrics[f"{metric_key_prefix}/spearman_corr_avg"] = prog_sum_rho / max(1.0, prog_count)

        # Log metrics
        if is_rank_0():
            rank_0_print(f"\n=== Custom VQA Evaluation Results ===")
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
            batch_size = torch.tensor(num_preferences + num_similarities, device=self.accelerator.device)
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
                "total_loss": total_loss,
                "batch_size": num_preferences + num_similarities,
            }
            return total_loss, extra_info

        return total_loss

    def _extract_answer_from_text(self, text):
        """
        Extract content between <ans></ans> tags from text.

        Args:
            text (str): Text containing <ans></ans> tags

        Returns:
            str: Content between <ans></ans> tags, or empty string if not found
        """
        import re

        match = re.search(r"<ans>(.*?)</ans>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _compute_vqa_loss(self, model, inputs, return_outputs=False, mode=None):
        """
        Compute VQA loss for given inputs.
        """
        # Extract inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        # Handle vision inputs - the processor might return different keys
        pixel_values = inputs.get("pixel_values")
        pixel_values_videos = inputs.get("pixel_values_videos")
        image_grid_thw = inputs.get("image_grid_thw")
        video_grid_thw = inputs.get("video_grid_thw")
        second_per_grid_ts = inputs.get("second_per_grid_ts")

        # Move to device if they exist
        if pixel_values is not None:
            pixel_values = pixel_values
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw
        if second_per_grid_ts is not None:
            second_per_grid_ts = second_per_grid_ts

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            labels=labels,
        )

        # The model's forward method should return a loss
        loss = outputs.loss

        # Prepare loss dictionary for logging
        loss_dict = {f"{mode}_loss": loss.item()}

        return (loss, loss_dict) if return_outputs else loss

    def _safe_parse_array(self, s: str):
        """Safely parse a Python list/tuple string into a 1D float tensor; return None on failure."""
        try:
            arr = ast.literal_eval(s)
            if isinstance(arr, (list, tuple)):
                t = torch.tensor(arr, dtype=torch.float32)
                return t.view(-1)
        except Exception:
            pass
        return None
