import ast
from rfm.utils.distributed import rank_0_print
import torch
import torch.nn as nn
import torch.nn.functional as F

from evals.eval_utils import extract_answer_from_text
from .rfm_heads_trainer import RFMHeadsTrainer
from rfm.utils.timer import _timer
from rfm.models.utils import ModelOutput


# copied because the original function forces the metric reduction
def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
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
    ignore_index: int = -100,
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

    def forward_model(self, model, inputs, sample_type="progress"):
        """
        Forward model for VQA - uses generate() for proper autoregressive prediction.
        This is used during evaluation to get actual model predictions.
        """
        with _timer("time/forward_vqa", timing_raw=self.timing_raw):
            if sample_type == "progress":
                # Use generate() for proper autoregressive text generation
                # Remove labels from inputs if present (we're doing inference)
                gen_inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "pixel_values": inputs.get("pixel_values"),
                    "pixel_values_videos": inputs.get("pixel_values_videos"),
                    "image_grid_thw": inputs.get("image_grid_thw"),
                    "video_grid_thw": inputs.get("video_grid_thw"),
                }
                
                # Generate with reasonable parameters for short structured answers
                with torch.no_grad():
                    generated_ids = model.generate(
                        **gen_inputs,
                        max_new_tokens=80,  # Enough for a list like [0.0, 0.1, ..., 1.0]
                        do_sample=False,  # Greedy decoding for reproducibility
                        pad_token_id=model.tokenizer.pad_token_id,
                        eos_token_id=model.tokenizer.eos_token_id,
                        use_cache=True,  # Enable KV caching for faster generation
                    )
                
                # Decode only the generated part (not the input prompt)
                rfm_model = self.model.module if hasattr(self.model, 'module') else self.model
                tokenizer = rfm_model.tokenizer
                
                # Get input length to slice only generated tokens
                input_len = inputs["input_ids"].shape[1]
                generated_ids = generated_ids[:, input_len:]  # Only new tokens
                
                pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                predictions = [extract_answer_from_text(text) for text in pred_texts]
                
                progress_logits = []
                for prediction in predictions:
                    try:
                        progress_logits.append(ast.literal_eval(prediction))
                    except Exception as e:
                        # Log parsing failures for debugging
                        if self.state.global_step % 100 == 0:
                            rank_0_print(f"Failed to parse prediction: {prediction[:100]}")
                        progress_logits.append(None)
                progress_logits = {"A": progress_logits, "B": None}
            else:
                progress_logits = None
        
        # Create ModelOutput with all expected fields to match parent class expectations
        model_output = ModelOutput(
            progress_logits=progress_logits,
            success_logits=None,  # VQA doesn't use success head
            pref_logits=None,     # VQA doesn't use preference head
            sim_logits=None,      # VQA doesn't use similarity head
        )
        return model_output, self.timing_raw

    def compute_loss(self, model, inputs, return_outputs=False, training=True, **kwargs):
        """Compute loss for VQA tasks."""

        # Set static graph for DDP on first training step to handle multiple forward passes. This is needed
        # when combining gradient checkpointing with multiple forward passes.
        if self.config.training.gradient_checkpointing and (training and not self._ddp_static_graph_set and hasattr(model, 'module')):
            # Check if model is wrapped in DDP
            if hasattr(model.module, '_set_static_graph'):
                model.module._set_static_graph()
                self._ddp_static_graph_set = True
            elif hasattr(model, '_set_static_graph'):
                model._set_static_graph()
                self._ddp_static_graph_set = True

        # Extract the separate batches
        preference_inputs = inputs.get("preference_inputs", {})
        similarity_inputs = inputs.get("similarity_inputs", {})
        progress_inputs = inputs.get("progress_inputs", {})

        num_preferences = inputs.get("num_preferences", 0)
        num_similarities = inputs.get("num_similarities", 0)
        num_progress = inputs.get("num_progress", 0)

        # Initialize loss components and metadata
        # Initialize as tensor to avoid issues when no samples exist
        # Don't set requires_grad=True on leaf tensor - it will get gradients from operations
        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        log_metadata = {}

        # Compute VQA loss for each type of input
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

        # Always store custom losses for logging (even when return_outputs=False)
        self.log_metadata = log_metadata

        if return_outputs:
            # Combine outputs from all loss functions
            extra_info = {
                **log_metadata,
                "total_loss": total_loss.item()
            }
            return total_loss, extra_info

        return total_loss

    def _compute_vqa_loss(self, model, inputs, return_outputs=False, mode=None, training=True):
        B = inputs["input_ids"].shape[0]
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
            use_cache=False,  # Disable KV caching for training
        )

        # RFMVQA has model directly, handle DDP wrapping
        rfm_model = self.model.module if hasattr(self.model, 'module') else self.model
        vocab_size = rfm_model.model.config.text_config.vocab_size
        
        loss = ForCausalLMLoss(
            logits=outputs.logits,
            labels=inputs["labels"],
            vocab_size=vocab_size,
            reduction="none",
        )
        # reshape
        loss = loss.reshape(B, -1)
        loss_per_example = loss.mean(dim=1)
        loss = loss.mean()

        prefix = "train" if training else "eval"
        loss_dict = {f"{prefix}/{mode}_loss": loss.item()}

        mode_name = "pref" if mode == "preference" else "prog" if mode == "progress" else "sim"

        # compute accuracy
        pred_ids = outputs.logits.argmax(dim=-1)
        # RFMVQA has tokenizer directly, handle DDP wrapping
        rfm_model = self.model.module if hasattr(self.model, 'module') else self.model
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
                    progress_pred_tensor = torch.tensor(progress_pred, device=self.accelerator.device, dtype=torch.float32)
                    gt_tensor = torch.tensor(gt_labels[i], device=self.accelerator.device, dtype=torch.float32)
                    progress_losses.append(F.mse_loss(progress_pred_tensor, gt_tensor).item())
                except Exception:
                    progress_losses.append(float('inf'))  # Mark failed predictions
            
            # Log average progress MSE if we have valid predictions
            valid_losses = [l for l in progress_losses if l != float('inf')]
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
