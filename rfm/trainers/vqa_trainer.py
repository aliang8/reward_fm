import ast
import gc
import math
from rfm.utils.distributed import rank_0_print
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics
from rfm.evals.eval_utils import extract_answer_from_text
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

def process_progress_pred(progress):
    return progress / 100


class RFMVQATrainer(RFMHeadsTrainer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._ddp_static_graph_set = False
        self.model_type_checked = False

    def _get_model(self):
        # Clear any existing past_key_values in model if present
        return self.model.module if hasattr(self.model, "module") else self.model

    def evaluate(self, eval_dataset=None, ignore_keys=None) -> dict[str, float]:
        """Override evaluate to add aggressive memory cleanup after evaluation."""
        # Run parent evaluation
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys)

        # Aggressive memory cleanup after evaluation
        # Don't move model to CPU with DDP/FSDP as it can cause issues
        rank_0_print("🧹 Aggressive CUDA memory cleanup after evaluation...")

        rfm_model = self._get_model()
        if hasattr(rfm_model.model, "past_key_values"):
            rfm_model.model.past_key_values = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()

        # Put model back in training mode (this can help clear eval-specific state)
        self.model.train()

        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        rank_0_print("✅ Memory cleanup complete")

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

        # target_progress_length_mode = statistics.mode(target_progress_lengths)
        max_frames = self.config.data.max_frames

        # aggregate by padding and truncating to the mode length
        aggregated_progress_logits = []
        for prediction in progress_logits:
            # Parse progress prediction
            try:
                parsed = ast.literal_eval(prediction)
                if isinstance(parsed, (int, float)):
                    parsed = [float(parsed)]
                elif isinstance(parsed, str):
                    parsed = [float(parsed)] if parsed.strip() else []
                elif hasattr(parsed, "__iter__"):
                    parsed = list(parsed)
                else:
                    parsed = []
                # Convert to float
                parsed = [float(val) for val in parsed if isinstance(val, (int, float))]
            except Exception:
                parsed = []
            # Pad/truncate to target length
            if not parsed:
                parsed = [0.0]
            if len(parsed) < max_frames:
                pad_val = parsed[-1]
                parsed = parsed + [pad_val] * (max_frames - len(parsed))
            elif len(parsed) > max_frames:
                parsed = parsed[:max_frames]
            aggregated_progress_logits.append(parsed)
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
            # Generate with reasonable parameters for short structured answers
            with torch.no_grad():
                generated_ids = model.generate(
                    **gen_inputs,
                    max_new_tokens=10,  # Reduced from 100 to save memory - enough for structured answers
                    do_sample=False,  # Greedy decoding for reproducibility
                    pad_token_id=model.tokenizer.pad_token_id,
                    eos_token_id=model.tokenizer.eos_token_id,
                    use_cache=True,  # Disable KV caching to prevent OOM - slower but memory safe
                )

            # Decode only the generated part (not the input prompt)
            tokenizer = self._get_model().tokenizer

            # Get input length to slice only generated tokens
            input_len = inputs["input_ids"].shape[1]
            generated_ids_sliced = generated_ids[:, input_len:].clone()  # Clone to break any references

            # Free original generated_ids immediately
            del generated_ids

            pred_texts = tokenizer.batch_decode(generated_ids_sliced, skip_special_tokens=True)
            predictions = [extract_answer_from_text(text) for text in pred_texts]

            # Free intermediate tensors immediately
            del generated_ids_sliced, pred_texts

            if sample_type == "progress":
                progress_logits = self._aggregate_progress_logits(predictions, inputs["target_progress"])
                progress_logits = process_progress_pred(torch.tensor(progress_logits, dtype=torch.float32))
                progress_logits = {"A": progress_logits, "B": None}
            elif sample_type == "preference":
                pref_logits = []
                for i, prediction in enumerate(predictions):
                    if prediction == 1:
                        pref_logits.append(1)
                    elif prediction == 2:
                        pref_logits.append(0)
                    else:
                        pref_logits.append(-1)
                pref_logits = {"A": pref_logits, "B": None}

            # Explicitly free all remaining references
            del gen_inputs, predictions

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

        # Set static graph for DDP on first training step
        if (
            training
            and not self._ddp_static_graph_set
            and getattr(self.accelerator.gradient_state, "sync_gradients", True)
            and hasattr(model, "module")
        ):
            if hasattr(model.module, "_set_static_graph"):
                model.module._set_static_graph()
                self._ddp_static_graph_set = True
            elif hasattr(model, "_set_static_graph"):
                model._set_static_graph()
                self._ddp_static_graph_set = True

        # Combine all batch types for single forward pass
        preference_inputs = inputs.get("preference_inputs") or {}
        similarity_inputs = inputs.get("similarity_inputs") or {}
        progress_inputs = inputs.get("progress_inputs") or {}

        batches_to_combine = []
        modes_per_sample = []

        if self._get_batch_size(preference_inputs) > 0:
            batches_to_combine.append(preference_inputs)
            modes_per_sample.extend(["preference"] * self._get_batch_size(preference_inputs))
        if self._get_batch_size(similarity_inputs) > 0:
            batches_to_combine.append(similarity_inputs)
            modes_per_sample.extend(["similarity"] * self._get_batch_size(similarity_inputs))
        if self._get_batch_size(progress_inputs) > 0:
            batches_to_combine.append(progress_inputs)
            modes_per_sample.extend(["progress"] * self._get_batch_size(progress_inputs))

        if not batches_to_combine:
            return torch.tensor(0.0, device=self.accelerator.device)

        # Combine batches - pad and concatenate tensors
        combined = {}
        for key in batches_to_combine[0].keys():
            if isinstance(batches_to_combine[0][key], torch.Tensor):
                tensors = [b[key] for b in batches_to_combine if key in b]

                # Check if tensors need padding
                if len(tensors[0].shape) > 1 and any(t.shape[1] != tensors[0].shape[1] for t in tensors):
                    max_len = max(t.shape[1] for t in tensors)
                    padded_tensors = []

                    for t in tensors:
                        if t.shape[1] < max_len:
                            pad_value = IGNORE_INDEX if key == "labels" else 0
                            padding = torch.full(
                                (t.shape[0], max_len - t.shape[1]) + t.shape[2:],
                                pad_value,
                                dtype=t.dtype,
                                device=t.device,
                            )
                            padded_tensors.append(torch.cat([t, padding], dim=1))
                        else:
                            padded_tensors.append(t)

                    combined[key] = torch.cat(padded_tensors, dim=0)
                else:
                    combined[key] = torch.cat(tensors, dim=0)

        with _timer("time/compute_vqa_loss", timing_raw=self.timing_raw):
            loss, loss_dict = self._compute_vqa_loss(
                model,
                combined,
                modes_per_sample,
                preference_inputs,
                progress_inputs,
                similarity_inputs,
                return_outputs=True,
                training=training,
            )

        self.log_metadata = loss_dict

        if return_outputs:
            return loss, {**loss_dict, "total_loss": loss.item()}

        return loss

    def _get_dtype(self, model):
        """Get the dtype of the model."""
        if hasattr(model, "module"):
            return model.module.model.dtype
        else:
            return model.model.dtype

    def _get_batch_size(self, batch_inputs: dict | None) -> int:
        if not batch_inputs:
            return 0
        candidates = [
            batch_inputs.get("input_ids"),
            batch_inputs.get("labels"),
        ]
        for tensor in candidates:
            if isinstance(tensor, torch.Tensor):
                return tensor.shape[0]
            if isinstance(tensor, list):
                return len(tensor)
        return 0

    def _compute_vqa_loss(
        self, model, inputs, modes_per_sample, pref_inputs, prog_inputs, sim_inputs, return_outputs=False, training=True
    ):
        B = inputs["input_ids"].shape[0]

        # cast to correct dtype
        if (
            "pixel_values" in inputs
            and inputs["pixel_values"] is not None
            and inputs["pixel_values"].dtype != self._get_dtype(model)
        ):
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self._get_dtype(model))
        if (
            "pixel_values_videos" in inputs
            and inputs["pixel_values_videos"] is not None
            and inputs["pixel_values_videos"].dtype != self._get_dtype(model)
        ):
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
        loss = ForCausalLMLoss(
            logits=outputs["logits"],
            labels=inputs["labels"],
            vocab_size=outputs["logits"].shape[-1],
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
        # reshape
        #loss = loss.reshape(B, -1)

        #loss_per_example = loss[loss != IGNORE_INDEX]
        #loss_per_example = loss.mean(dim=1)
        #loss = loss.mean()


        prefix = "train" if training else "eval"
        loss_dict = {f"{prefix}/combined_loss": loss.item()}

        # Compute predictions for all samples
        pred_ids = outputs["logits"].argmax(dim=-1)
        rfm_model = self.model.module if hasattr(self.model, "module") else self.model
        tokenizer = rfm_model.tokenizer
        pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        extracted_answers = [extract_answer_from_text(text) for text in pred_texts]

        # Aggregate metrics per mode via simple for loop
        mode_name_map = {"preference": "pref", "progress": "prog", "similarity": "sim"}

        # Collect per-mode data with breakdown by source/strategy
        pref_data = []  # (loss, correct, source, strategy)
        prog_data = []  # (loss, mse, source, strategy)

        for i, mode in enumerate(modes_per_sample):
            #mode_loss = loss_per_example[i].item()

            if mode == "preference":
                pred = extracted_answers[i]
                print(f"PREFERENCE: {pred}")
                label_map = {1: 1, 2: 0} # video 1 is A, video 2 is B
                pred_label = label_map.get(pred, -1)
                # Get from original batch (index within preference batch)
                pref_idx = sum(1 for j, m in enumerate(modes_per_sample[:i]) if m == "preference")
                gt_label = pref_inputs["preference_labels"][pref_idx].item()
                correct = 1.0 if pred_label == gt_label else 0.0

                # Get metadata
                source = pref_inputs.get("data_source", [None] * len(pref_inputs["preference_labels"]))[pref_idx]
                strategy = pref_inputs.get(
                    "rejected_data_gen_strategy", [None] * len(pref_inputs["preference_labels"])
                )[pref_idx]
                #pref_data.append(dict(loss=mode_loss, correct=correct, source=source, strategy=strategy))
                pref_data.append(dict(correct=correct, source=source, strategy=strategy))

            elif mode == "progress":
                pred = extracted_answers[i]
                # Get from original batch
                prog_idx = sum(1 for j, m in enumerate(modes_per_sample[:i]) if m == "progress")
                gt = prog_inputs["target_progress"][prog_idx][-1]

                mse = None
                try:
                    parsed = ast.literal_eval(pred)
                    pred_tensor = process_progress_pred(torch.tensor([parsed], dtype=torch.float32))
                    gt_tensor = torch.tensor([gt], dtype=torch.float32)
                    print(f"PROGRESS: {pred_tensor}, {gt_tensor}")
                    mse = F.mse_loss(pred_tensor, gt_tensor).item()
                except Exception:
                    mse = None
                # Get metadata
                source = prog_inputs.get("data_source", [None] * len(prog_inputs["target_progress"]))[prog_idx]
                strategy = prog_inputs.get("data_gen_strategy", [None] * len(prog_inputs["target_progress"]))[prog_idx]
                #prog_data.append(dict(loss=mode_loss, mse=mse, source=source, strategy=strategy))
                prog_data.append(dict(mse=mse, source=source, strategy=strategy))

        # Aggregate overall metrics
        if pref_data:
            #loss_dict[f"{prefix}/preference_loss"] = np.mean([x["loss"] for x in pref_data])
            loss_dict[f"{prefix}/preference_acc"] = np.mean([x["correct"] for x in pref_data])

            # By data source
            sources = set(x["source"] for x in pref_data if x["source"] is not None)
            for source in sources:
                source_data = [x for x in pref_data if x["source"] == source]
                #loss_dict[f"{prefix}_ds/pref_loss_{source}"] = np.mean([x["loss"] for x in source_data])
                loss_dict[f"{prefix}_ds/pref_acc_{source}"] = np.mean([x["correct"] for x in source_data])

            # By strategy
            strategies = set(x["strategy"] for x in pref_data if x["strategy"] is not None)
            for strategy in strategies:
                strat_data = [x for x in pref_data if x["strategy"] == strategy]
                #loss_dict[f"{prefix}_strat/pref_loss_{strategy}"] = np.mean([x["loss"] for x in strat_data])
                loss_dict[f"{prefix}_strat/pref_acc_{strategy}"] = np.mean([x["correct"] for x in strat_data])

        if prog_data:
            #loss_dict[f"{prefix}/progress_loss"] = np.mean([x["loss"] for x in prog_data])
            mses = [x["mse"] for x in prog_data if x["mse"] is not None]
            if mses:
                loss_dict[f"{prefix}/progress_mse"] = np.mean(mses)

            # By data source
            sources = set(x["source"] for x in prog_data if x["source"] is not None)
            for source in sources:
                source_data = [x for x in prog_data if x["source"] == source]
                #loss_dict[f"{prefix}_ds/prog_loss_{source}"] = np.mean([x["loss"] for x in source_data])
                prog_mse = [x["mse"] for x in source_data if x["mse"] is not None]
                if prog_mse:
                    loss_dict[f"{prefix}_ds/prog_mse_{source}"] = np.mean(prog_mse)

            # By strategy
            strategies = set(x["strategy"] for x in prog_data if x["strategy"] is not None)
            for strategy in strategies:
                strat_data = [x for x in prog_data if x["strategy"] == strategy]
                #loss_dict[f"{prefix}_strat/prog_loss_{strategy}"] = np.mean([x["loss"] for x in strat_data])
                prog_mse = [x["mse"] for x in strat_data if x["mse"] is not None]
                if prog_mse:
                    loss_dict[f"{prefix}_strat/prog_mse_{strategy}"] = np.mean(prog_mse)

        return (loss, loss_dict) if return_outputs else loss
