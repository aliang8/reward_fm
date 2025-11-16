import ast
import gc
from rfm.utils.distributed import rank_0_print
import torch
import torch.nn as nn
import torch.nn.functional as F

from evals.eval_utils import extract_answer_from_text
from .rfm_heads_trainer import RFMHeadsTrainer
from rfm.utils.timer import _timer
from rfm.models.utils import ModelOutput
from rfm.models.rfm_vqa import RFMVQA
from rfm.data.collators.vqa import IGNORE_INDEX


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

    def forward_model(self, model, inputs, sample_type="progress"):
        """
        Forward model for VQA - uses generate() for proper autoregressive prediction.
        This is used during evaluation to get actual model predictions.
        """
        assert isinstance(model, RFMVQA), "Model must be an instance of RFMVQA"
        progress_logits = None
        pref_logits = None
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
                gen_inputs["pixel_values"] = gen_inputs["pixel_values"].to(dtype=model.model.dtype)
            if gen_inputs["pixel_values_videos"] is not None:
                gen_inputs["pixel_values_videos"] = gen_inputs["pixel_values_videos"].to(dtype=model.model.dtype)

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
                progress_logits = []
                for i, prediction in enumerate(predictions):
                    try:
                        progress_logits.append(ast.literal_eval(prediction))
                    except Exception as e:
                        # Log parsing failures for debugging
                        if self.state.global_step % 100 == 0:
                            rank_0_print(f"Failed to parse prediction: {prediction[:50]}, inserting all 0s")

                        # insert fake progress logits
                        progress_logits.append([0] * len(inputs["target_progress"][i]))
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

        assert isinstance(model, RFMVQA), "Model must be an instance of RFMVQA"

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

    def _compute_vqa_loss(self, model, inputs, return_outputs=False, mode=None, training=True):
        B = inputs["input_ids"].shape[0]

        if "pixel_values" in inputs and inputs["pixel_values"] is not None and inputs["pixel_values"].dtype != model.model.dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.model.dtype)
        if "pixel_values_videos" in inputs and inputs["pixel_values_videos"] is not None and inputs["pixel_values_videos"].dtype != model.model.dtype:
            inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(dtype=model.model.dtype)
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
        vocab_size = rfm_model.model.config.text_config.vocab_size

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
        
        # Check if mode is a list (per-sample modes from combined batch)
        is_combined = isinstance(mode, list)
        
        if is_combined:
            # Compute per-type losses using masks
            modes_per_sample = mode
            unique_modes = set(modes_per_sample)
            
            loss_dict[f"{prefix}/combined_loss"] = loss.item()
            
            for sample_mode in unique_modes:
                mask = torch.tensor([m == sample_mode for m in modes_per_sample], device=loss_per_example.device)
                type_loss = loss_per_example[mask].mean()
                loss_dict[f"{prefix}/{sample_mode}_loss"] = type_loss.item()
            
            # For combined batches, skip detailed per-sample metrics (would be complex to parse)
            return (loss, loss_dict) if return_outputs else loss
        else:
            # Single mode - use original logic
            loss_dict[f"{prefix}/{mode}_loss"] = loss.item()
            mode_name = "pref" if mode == "preference" else "prog" if mode == "progress" else "sim"
        
        # compute accuracy
        pred_ids = outputs["logits"].argmax(dim=-1)
        # RFMVQA has tokenizer directly, handle DDP wrapping
        rfm_model = self.model.module if hasattr(self.model, "module") else self.model
        tokenizer = rfm_model.tokenizer
        pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        if mode == "preference":
            predictions = [extract_answer_from_text(text) for text in pred_texts]
            predictions_num_labels = []
            for prediction in predictions:
                if prediction == "A":
                    predictions_num_labels.append(1)
                elif prediction == "B":
                    predictions_num_labels.append(0)
                else:
                    predictions_num_labels.append(-1)

            predictions_num_labels = torch.tensor(predictions_num_labels, device=self.accelerator.device)
            gt_labels = inputs["preference_labels"]

            preference_correct = (predictions_num_labels == gt_labels).float()
            loss_dict.update({f"{prefix}/{mode}_acc": preference_correct.mean().item()})
        elif mode == "progress":
            predictions = [extract_answer_from_text(text) for text in pred_texts]
            gt_labels = inputs["target_progress"]

            # Compute progress prediction accuracy (for logging)
            progress_losses = []
            for i, prediction in enumerate(predictions):
                try:
                    progress_pred = ast.literal_eval(prediction)
                    # Ensure tensors are on correct device
                    progress_pred_tensor = torch.tensor(
                        progress_pred, device=self.accelerator.device, dtype=torch.float32
                    )
                    gt_tensor = torch.tensor(gt_labels[i], device=self.accelerator.device, dtype=torch.float32)
                    progress_losses.append(F.mse_loss(progress_pred_tensor, gt_tensor).item())
                except Exception:
                    progress_losses.append(float("inf"))  # Mark failed predictions

            # Log average progress MSE if we have valid predictions
            valid_losses = [l for l in progress_losses if l != float("inf")]
            if valid_losses:
                loss_dict.update({f"{prefix}/{mode}_mse": sum(valid_losses) / len(valid_losses)})
        else:
            pass

        # split the acc and loss by data gen strategy and data source
        if mode == "preference":
            rejected_data_gen_strategy = inputs.get("rejected_data_gen_strategy", None)

            for strat in set(rejected_data_gen_strategy):
                mask = [1 if s == strat else 0 for s in rejected_data_gen_strategy]
                mask = torch.tensor(mask, device=self.accelerator.device)
                loss_dict.update({
                    f"{prefix}_strat/{mode_name}_loss_{strat}": (loss_per_example[mask == 1]).mean().item()
                })
                loss_dict.update({
                    f"{prefix}_strat/{mode_name}_acc_{strat}": (preference_correct[mask == 1]).mean().item()
                })

        elif mode == "progress":
            data_gen_strategy = inputs["data_gen_strategy"]

            for strat in set(data_gen_strategy):
                mask = [1 if s == strat else 0 for s in data_gen_strategy]
                mask = torch.tensor(mask, device=self.accelerator.device)
                loss_dict.update({
                    f"{prefix}_strat/{mode_name}_loss_{strat}": (loss_per_example[mask == 1]).mean().item()
                })

        data_sources = inputs.get("data_source", [])

        for data_source in set(data_sources):
            mask = [1 if s == data_source else 0 for s in inputs["data_source"]]
            mask = torch.tensor(mask, device=self.accelerator.device)
            loss_dict.update({
                f"{prefix}_ds/{mode_name}_loss_{data_source}": (loss_per_example[mask == 1]).mean().item()
            })

            if mode == "preference":
                loss_dict.update({
                    f"{prefix}_ds/{mode_name}_acc_{data_source}": (preference_correct[mask == 1]).mean().item()
                })

        return (loss, loss_dict) if return_outputs else loss
