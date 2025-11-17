import ast
import gc
import math
from rfm.utils.distributed import rank_0_print
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics
from evals.eval_utils import extract_answer_from_text
from .rfm_heads_trainer import RFMHeadsTrainer
from rfm.utils.timer import _timer
from rfm.models.utils import ModelOutput
from rfm.models.rfm_vqa import RFMVQA
from rfm.data.collators.vqa import IGNORE_INDEX
import numpy as np


# copied because the original function forces the metric reduction
def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    shift_labels: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss


class RFMVQATrainer(RFMHeadsTrainer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._ddp_static_graph_set = False
        self.model_type_checked = False

    def evaluate(self, eval_dataset=None, ignore_keys=None) -> dict[str, float]:
        """Override evaluate to add aggressive memory cleanup after evaluation."""
        # Run parent evaluation
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys)
        
        # Aggressive memory cleanup after evaluation to prevent OOM in next training step
        if torch.cuda.is_available():
            # Empty cache multiple times to ensure fragmented memory is freed
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Force garbage collection
        gc.collect()
        
        return metrics

    def _aggregate_progress_logits(self, progress_logits, target_progress) -> list[list[float]]:
        # ensures all progress logits are the same length as each other

        # get the mode of the target progress lengths
        target_progress_lengths = []
        for progress in target_progress:
            if hasattr(progress, "shape"):
                if progress.shape[-1] > 0:
                    target_progress_lengths.append(progress.shape[-1])
            else:
                target_progress_lengths.append(len(progress))

        if not target_progress_lengths:
            return []

        target_progress_length_mode = statistics.mode(target_progress_lengths)

        # aggregate by padding and truncating to the mode length
        aggregated_progress_logits = []
        for prediction in progress_logits:
            parsed = self._parse_progress_prediction(prediction)
            normalized = self._pad_or_truncate_progress(parsed, target_progress_length_mode)
            aggregated_progress_logits.append(normalized)
        return aggregated_progress_logits
        

    def _check_model_type(self, model):
        """
        Check if the model is an instance of RFMVQA.
        Works with DDP/FSDP by unwrapping the model first.
        """
        if self.model_type_checked:
            return
        # Unwrap DDP/FSDP wrapper to get the actual model
        real_model = model.module if hasattr(model, "module") else model
        assert isinstance(real_model, RFMVQA), f"Model must be an instance of RFMVQA, got {type(real_model)}"
        self.model_type_checked = True

    def forward_model(self, model, inputs, sample_type="progress"):
        """
        Forward model for VQA - uses generate() for proper autoregressive prediction.
        This is used during evaluation to get actual model predictions.
        """
        progress_logits = None
        pref_logits = None
        self._check_model_type(model)
        with _timer("time/forward_vqa", timing_raw=self.timing_raw):
            # Use generate() for proper autoregressive text generation
            # Note: dtype casting is handled in trainer's _prepare_inputs
            gen_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "pixel_values": inputs.get("pixel_values"),
                "pixel_values_videos": inputs.get("pixel_values_videos"),
                "image_grid_thw": inputs.get("image_grid_thw"),
                "video_grid_thw": inputs.get("video_grid_thw"),
            }

            if gen_inputs["pixel_values"] is not None:
                gen_inputs["pixel_values"] = gen_inputs["pixel_values"].to(dtype=self._get_dtype(model))
            if gen_inputs["pixel_values_videos"] is not None:
                gen_inputs["pixel_values_videos"] = gen_inputs["pixel_values_videos"].to(dtype=self._get_dtype(model))

            # Generate with reasonable parameters for short structured answers
            with torch.no_grad():
                generated_ids = model.generate(
                    **gen_inputs,
                    max_new_tokens=100,  # enough for a list of 16 floats
                    do_sample=False,  # Greedy decoding for reproducibility
                    pad_token_id=model.tokenizer.pad_token_id,
                    eos_token_id=model.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV caching for faster generation
                )

            # Decode only the generated part (not the input prompt)
            rfm_model = self.model.module if hasattr(self.model, "module") else self.model
            tokenizer = rfm_model.tokenizer

            # Get input length to slice only generated tokens
            input_len = inputs["input_ids"].shape[1]
            generated_ids_sliced = generated_ids[:, input_len:]  # Only new tokens

            pred_texts = tokenizer.batch_decode(generated_ids_sliced, skip_special_tokens=True)
            predictions = [extract_answer_from_text(text) for text in pred_texts]

            if sample_type == "progress":
                progress_logits = self._aggregate_progress_logits(predictions, inputs["target_progress"])
                progress_logits = {"A": progress_logits, "B": None}
            elif sample_type == "preference":
                pref_logits = []
                for i, prediction in enumerate(predictions):
                    if prediction == "A":
                        pref_logits.append(1)
                    elif prediction == "B":
                        pref_logits.append(0)
                    else:
                        pref_logits.append(-1)
                pref_logits = {"A": pref_logits, "B": None}
            
            # Explicitly free generation tensors to prevent memory accumulation (for all sample types)
            del generated_ids, generated_ids_sliced, gen_inputs, pred_texts, predictions
            
            # Clear CUDA cache after generation to free KV cache memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Create ModelOutput with all expected fields to match parent class expectations
        model_output = ModelOutput(
            progress_logits=progress_logits,
            success_logits=None,  # VQA doesn't use success head
            pref_logits=None,  # VQA doesn't use preference head
            sim_logits=None,  # VQA doesn't use similarity head
        )
        return model_output, self.timing_raw

    def compute_loss(self, model, inputs, return_outputs=False, training=True, **kwargs):
        """Compute loss for VQA tasks."""
        # check model is right type
        self._check_model_type(model)

        # Set static graph for DDP on first training step to handle multiple forward passes. This is needed
        # when combining gradient checkpointing with multiple forward passes.
        if self.config.training.gradient_checkpointing and (
            training and not self._ddp_static_graph_set and hasattr(model, "module")
        ):
            # Check if model is wrapped in DDP
            if hasattr(model.module, "_set_static_graph"):
                model.module._set_static_graph()
                self._ddp_static_graph_set = True
            elif hasattr(model, "_set_static_graph"):
                model._set_static_graph()
                self._ddp_static_graph_set = True

        # Extract the separate batches
        preference_inputs = inputs.get("preference_inputs", {})
        similarity_inputs = inputs.get("similarity_inputs", {})
        progress_inputs = inputs.get("progress_inputs", {})

        num_preferences = inputs.get("num_preferences", 0)
        num_similarities = inputs.get("num_similarities", 0)
        num_progress = inputs.get("num_progress", 0)

        # Simple combining: concatenate all non-empty batches into one
        batches_to_combine = []
        modes_per_sample = []  # Track mode for each sample
        
        if num_preferences > 0:
            batches_to_combine.append(preference_inputs)
            modes_per_sample.extend(['preference'] * num_preferences)
        if num_similarities > 0:
            batches_to_combine.append(similarity_inputs)
            modes_per_sample.extend(['similarity'] * num_similarities)
        if num_progress > 0:
            batches_to_combine.append(progress_inputs)
            modes_per_sample.extend(['progress'] * num_progress)

        if len(batches_to_combine) > 1:
            # Combine all batches - pad and concatenate tensors, single forward pass
            combined = {}
            for key in batches_to_combine[0].keys():
                if isinstance(batches_to_combine[0][key], torch.Tensor):
                    tensors = [b[key] for b in batches_to_combine if key in b]
                    
                    # Check if tensors need padding (different sizes in dim 1)
                    if len(tensors[0].shape) > 1 and any(t.shape[1] != tensors[0].shape[1] for t in tensors):
                        # Pad to max length
                        max_len = max(t.shape[1] for t in tensors)
                        padded_tensors = []
                        
                        for t in tensors:
                            if t.shape[1] < max_len:
                                # Pad with zeros (or IGNORE_INDEX for labels)
                                pad_value = IGNORE_INDEX if key == "labels" else 0
                                padding = torch.full(
                                    (t.shape[0], max_len - t.shape[1]) + t.shape[2:],
                                    pad_value,
                                    dtype=t.dtype,
                                    device=t.device
                                )
                                padded_tensors.append(torch.cat([t, padding], dim=1))
                            else:
                                padded_tensors.append(t)
                        
                        combined[key] = torch.cat(padded_tensors, dim=0)
                    else:
                        # No padding needed
                        combined[key] = torch.cat(tensors, dim=0)
            
            # Pass per-sample modes for reusing existing mode logic
            combined['modes_per_sample'] = modes_per_sample
            
            with _timer("time/compute_vqa_loss_combined", timing_raw=self.timing_raw):
                loss, loss_dict = self._compute_vqa_loss(model, combined, return_outputs=True, mode=modes_per_sample, training=training)
            
            self.log_metadata = loss_dict
            if return_outputs:
                return loss, {**loss_dict, "total_loss": loss.item()}
            return loss
        
        else:
            # Single batch type - use original code
            total_loss = torch.tensor(0.0, device=self.accelerator.device)
            log_metadata = {}

            if num_preferences > 0 and preference_inputs:
                with _timer("time/compute_vqa_loss", timing_raw=self.timing_raw):
                    preference_loss, loss_dict = self._compute_vqa_loss(
                        model, preference_inputs, return_outputs=True, mode="preference", training=training
                    )
                total_loss += preference_loss
                log_metadata.update(loss_dict)

            if num_similarities > 0 and similarity_inputs:
                with _timer("time/compute_vqa_loss", timing_raw=self.timing_raw):
                    similarity_loss, loss_dict = self._compute_vqa_loss(
                        model, similarity_inputs, return_outputs=True, mode="similarity", training=training
                    )
                total_loss += similarity_loss
                log_metadata.update(loss_dict)

            if num_progress > 0 and progress_inputs:
                with _timer("time/compute_vqa_loss", timing_raw=self.timing_raw):
                    progress_loss, loss_dict = self._compute_vqa_loss(
                        model, progress_inputs, return_outputs=True, mode="progress", training=training
                    )
                total_loss += progress_loss
                log_metadata.update(loss_dict)

            self.log_metadata = log_metadata

            if return_outputs:
                extra_info = {**log_metadata, "total_loss": total_loss.item()}
                return total_loss, extra_info

        return total_loss

    def _get_dtype(self, model):
        """Get the dtype of the model."""
        if hasattr(model, "module"):
            return model.module.model.dtype
        else:
            return model.model.dtype
    
    def _pad_or_truncate_progress(self, values: list[float], expected_len: int | None) -> list[float]:
        if not values:
            values = [0.0]

        if expected_len is None or expected_len <= 0:
            return values

        if len(values) < expected_len:
            pad_val = values[-1]
            values = values + [pad_val] * (expected_len - len(values))
        elif len(values) > expected_len:
            values = values[:expected_len]
        return values

    def _update_preference_metrics(
        self,
        sample_indices: list[int],
        index_tensor: torch.Tensor,
        extracted_answers: list[str],
        inputs: dict,
        mode_loss_examples: torch.Tensor,
        loss_dict: dict,
        prefix: str,
        mode_name: str,
    ) -> None:
        device = self.accelerator.device
        mode_predictions = [extracted_answers[idx] for idx in sample_indices]

        label_map = {"A": 1, "B": 0}
        predictions_num_labels = torch.tensor(
            [label_map.get(pred, -1) for pred in mode_predictions],
            device=device,
            dtype=torch.long,
        )

        gt_labels = inputs["preference_labels"].index_select(
            0, index_tensor.to(inputs["preference_labels"].device)
        )

        preference_correct = (predictions_num_labels == gt_labels).float()
        loss_dict[f"{prefix}/preference_acc"] = preference_correct.mean().item()

        rejected_strategy = inputs.get("rejected_data_gen_strategy")
        if rejected_strategy:
            subset_strats = [rejected_strategy[idx] for idx in sample_indices]
            for strat in set(subset_strats):
                mask = torch.tensor(
                    [s == strat for s in subset_strats],
                    device=mode_loss_examples.device,
                    dtype=torch.bool,
                )
                if mask.any():
                    loss_dict[f"{prefix}_strat/{mode_name}_loss_{strat}"] = mode_loss_examples[mask].mean().item()
                    pref_mask = mask.to(preference_correct.device)
                    loss_dict[f"{prefix}_strat/{mode_name}_acc_{strat}"] = preference_correct[pref_mask].mean().item()

        data_sources = inputs.get("data_source")
        if data_sources:
            subset_sources = [data_sources[idx] for idx in sample_indices]
            for source in set(subset_sources):
                mask = torch.tensor(
                    [s == source for s in subset_sources],
                    device=mode_loss_examples.device,
                    dtype=torch.bool,
                )
                if mask.any():
                    loss_dict[f"{prefix}_ds/{mode_name}_loss_{source}"] = mode_loss_examples[mask].mean().item()
                    pref_mask = mask.to(preference_correct.device)
                    loss_dict[f"{prefix}_ds/{mode_name}_acc_{source}"] = preference_correct[pref_mask].mean().item()

    def _update_progress_metrics(
        self,
        sample_indices: list[int],
        index_tensor: torch.Tensor,
        extracted_answers: list[str],
        inputs: dict,
        mode_loss_examples: torch.Tensor,
        loss_dict: dict,
        prefix: str,
        mode_name: str,
    ) -> None:
        mode_predictions = [extracted_answers[idx] for idx in sample_indices]
        target_progress = inputs["target_progress"].index_select(
            0, index_tensor.to(inputs["target_progress"].device)
        )

        progress_losses = []
        for rel_idx, prediction in enumerate(mode_predictions):
            expected_len = target_progress[rel_idx].shape[-1] if target_progress[rel_idx].ndim > 0 else None
            # parse 
            try:
                parsed = ast.literal_eval(prediction)
                normalized = self._pad_or_truncate_progress(parsed, expected_len)

                progress_pred_tensor = torch.tensor(
                    normalized,
                    dtype=torch.float32,
                )
                gt_tensor = target_progress[rel_idx]

                progress_losses.append(F.mse_loss(progress_pred_tensor, gt_tensor).item())

            except Exception:
                rank_0_print(f"Warning: Failed to parse progress prediction for sample {rel_idx}: {prediction}")
                continue

        if progress_losses:
            loss_dict[f"{prefix}/progress_mse"] = np.mean(progress_losses)

        data_gen_strategy = inputs.get("data_gen_strategy")
        if data_gen_strategy:
            subset_strats = [data_gen_strategy[idx] for idx in sample_indices]
            for strat in set(subset_strats):
                mask = torch.tensor(
                    [s == strat for s in subset_strats],
                )
                if mask.any():
                    loss_dict[f"{prefix}_strat/{mode_name}_loss_{strat}"] = mode_loss_examples[mask].mean().item()

        data_sources = inputs.get("data_source")
        if data_sources:
            subset_sources = [data_sources[idx] for idx in sample_indices]
            for source in set(subset_sources):
                mask = torch.tensor(
                    [s == source for s in subset_sources],
                )
                if mask.any():
                    loss_dict[f"{prefix}_ds/{mode_name}_loss_{source}"] = mode_loss_examples[mask].mean().item()

    def _compute_vqa_loss(self, model, inputs, return_outputs=False, mode=None, training=True):
        B = inputs["input_ids"].shape[0]

        # cast to correct dtype
        if "pixel_values" in inputs and inputs["pixel_values"] is not None and inputs["pixel_values"].dtype != self._get_dtype(model):
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self._get_dtype(model))
        if "pixel_values_videos" in inputs and inputs["pixel_values_videos"] is not None and inputs["pixel_values_videos"].dtype != self._get_dtype(model):
            inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(dtype=self._get_dtype(model))
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
            use_cache=False,  # Disable KV caching for training
            return_dict=True,
        )

        # RFMVQA has model directly, handle DDP wrapping
        rfm_model = self.model.module if hasattr(self.model, "module") else self.model
        # Handle different config structures for different models
        # Qwen has text_config.vocab_size, SmolVLM has vocab_size directly
        if hasattr(rfm_model.model.config, "text_config"):
            vocab_size = rfm_model.model.config.text_config.vocab_size
        else:
            vocab_size = rfm_model.model.config.vocab_size

        loss = ForCausalLMLoss(
            logits=outputs["logits"],
            labels=inputs["labels"],
            vocab_size=vocab_size,
            reduction="none",
        )
        # reshape
        loss = loss.reshape(B, -1)
        loss_per_example = loss.mean(dim=1)
        loss = loss.mean()

        prefix = "train" if training else "eval"
        loss_dict = {}
        combined_batch = isinstance(mode, list)
        modes_per_sample = mode if combined_batch else [mode] * B

        if combined_batch:
            loss_dict[f"{prefix}/combined_loss"] = loss.item()

        mode_name_map = {"preference": "pref", "progress": "prog", "similarity": "sim"}

        pred_ids = outputs["logits"].argmax(dim=-1)
        rfm_model = self.model.module if hasattr(self.model, "module") else self.model
        tokenizer = rfm_model.tokenizer
        pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        extracted_answers = [extract_answer_from_text(text) for text in pred_texts]

        unique_modes = sorted(set(modes_per_sample))

        for sample_mode in unique_modes:
            sample_indices = [idx for idx, m in enumerate(modes_per_sample) if m == sample_mode]
            if not sample_indices:
                continue

            index_tensor = torch.tensor(sample_indices, device=loss_per_example.device, dtype=torch.long)
            mode_loss_examples = loss_per_example.index_select(0, index_tensor)
            loss_dict[f"{prefix}/{sample_mode}_loss"] = mode_loss_examples.mean().item()

            mode_name = mode_name_map.get(sample_mode, sample_mode[:4])

            if sample_mode == "preference":
                self._update_preference_metrics(
                    sample_indices,
                    index_tensor,
                    extracted_answers,
                    inputs,
                    mode_loss_examples,
                    loss_dict,
                    prefix,
                    mode_name,
                )
            elif sample_mode == "progress":
                self._update_progress_metrics(
                    sample_indices,
                    index_tensor,
                    extracted_answers,
                    inputs,
                    mode_loss_examples,
                    loss_dict,
                    prefix,
                    mode_name,
                )

        return (loss, loss_dict) if return_outputs else loss
