import ast

import torch
import torch.nn as nn
import torch.nn.functional as F

from evals.eval_utils import extract_answer_from_text
from .rfm_heads_trainer import RFMHeadsTrainer
from rfm.utils.logging import _timer


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

    def compute_loss(self, model, inputs, return_outputs=False, training=True, **kwargs):
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
                "total_loss": total_loss.item(),
                "batch_size": num_preferences + num_similarities + num_progress,
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
        )

        loss = ForCausalLMLoss(
            logits=outputs.logits,
            labels=inputs["labels"],
            vocab_size=self.model.base_model.model.config.text_config.vocab_size,
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
        tokenizer = self.model.base_model.processor.tokenizer
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

            progress_losses = []
            for prediction in predictions:
                try:
                    progress_pred = ast.literal_eval(prediction)
                    progress_losses.append(F.mse_loss(torch.tensor(progress_pred), torch.tensor(gt_labels)))
                except:
                    progress_losses.append(torch.tensor(float(0)))
        else:
            pass

        # split the acc and loss by data gen strategy and data source
        if mode == "preference":
            rejected_data_gen_strategy = inputs.get("rejected_data_gen_strategy", None)

            for strat in set(rejected_data_gen_strategy):
                mask = [1 if s == strat else 0 for s in rejected_data_gen_strategy]
                mask = torch.tensor(mask, device=self.accelerator.device)
                loss_dict.update({f"{prefix}_strat/{mode_name}_loss_{strat}": (loss_per_example[mask == 1]).mean().item()})
                loss_dict.update({
                    f"{prefix}_strat/{mode_name}_acc_{strat}": (preference_correct[mask == 1]).mean().item()
                })

        elif mode == "progress":
            data_gen_strategy = inputs["data_gen_strategy"]

            for strat in set(data_gen_strategy):
                mask = [1 if s == strat else 0 for s in data_gen_strategy]
                mask = torch.tensor(mask, device=self.accelerator.device)
                loss_dict.update({f"{prefix}_strat/{mode_name}_loss_{strat}": (loss_per_example[mask == 1]).mean().item()})

        data_source = inputs.get("data_source", [])

        for data_source in set(data_source):
            mask = [1 if s == data_source else 0 for s in inputs["data_source"]]
            mask = torch.tensor(mask, device=self.accelerator.device)
            loss_dict.update({f"{prefix}_ds/{mode_name}_loss_{data_source}": (loss_per_example[mask == 1]).mean().item()})

            if mode == "preference":
                loss_dict.update({
                    f"{prefix}_ds/{mode_name}_acc_{data_source}": (preference_correct[mask == 1]).mean().item()
                })

        return (loss, loss_dict) if return_outputs else loss
