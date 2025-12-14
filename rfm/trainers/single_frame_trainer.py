import torch
from rfm.utils.timer import _timer
from rfm.utils.logger import get_logger, log_memory_usage
from rfm.trainers.rfm_heads_trainer import RFMHeadsTrainer

logger = get_logger()


class SingleFrameTrainer(RFMHeadsTrainer):
    """Trainer for single-frame progress prediction (no data strategies)."""

    def __init__(self, config, *args, logger=None, **kwargs):
        # Ensure max_frames is set to 1 for single frame training
        if config.data.max_frames != 1:
            logger.warning(
                f"SingleFrameTrainer requires max_frames=1, but config has max_frames={config.data.max_frames}. "
                "Setting max_frames=1 for training."
            )
            config.data.max_frames = 1

        super().__init__(config, *args, logger=logger, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step for single-frame progress prediction.
        Simplified version that only handles progress samples.
        """
        logger.trace("training_step: Starting (single frame)")

        if not self._fsdp_diagnostics_logged:
            from rfm.utils.distributed import log_fsdp_diagnostics
            log_fsdp_diagnostics(model, accelerator=self.accelerator, logger=logger)
            self._fsdp_diagnostics_logged = True

        # Check if we just resumed from checkpoint
        if hasattr(self, "_just_resumed_from_checkpoint") and self._just_resumed_from_checkpoint:
            self._post_checkpoint_load_reset()
            self._just_resumed_from_checkpoint = False

        self.timing_raw = {}
        self.log_metadata = {}

        # Safety check: ensure model is in training mode
        if not model.training:
            logger.warning("Model not in training mode, setting to train mode")
            model.train()

        # Clear any stale gradients before starting
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        with _timer("time/training_step", timing_raw=self.timing_raw):
            loss = super().training_step(model, inputs, num_items_in_batch)

        # Extract progress batch
        progress_inputs = inputs.get("progress_inputs", {})
        num_progress = inputs.get("num_progress", 0)

        logger.trace(f"num_progress: {num_progress}")

        if num_progress > 0 and progress_inputs:
            data_sources = progress_inputs.get("data_source", None)
            if data_sources is not None:
                for ds in data_sources:
                    self.global_metadata[f"total_{ds}"] += 1.0

        # Update global metadata
        self.global_metadata["total_samples"] += num_progress
        self.global_metadata["total_progress"] += num_progress

        logger.trace("finished updating global metadata")

        # Log custom losses at specified intervals
        if self.state.global_step % self.args.logging_steps == 0:
            self._log_metadata()

        # Log GPU memory usage
        log_memory_usage(f"Step {self.state.global_step}")

        return loss

    def compute_loss(self, model, inputs, return_outputs=False, training=True, **kwargs):
        """
        Compute loss for single-frame progress prediction only.
        Simplified version that only handles progress samples.
        """
        logger.trace("compute_loss: Starting (single frame)")

        progress_inputs = inputs.get("progress_inputs", {})
        num_progress = inputs.get("num_progress", 0)

        total_loss = 0
        log_metadata = {}

        logger.trace(f"Num progress: {num_progress}")

        # Only compute progress loss
        if num_progress > 0 and progress_inputs and self.config.model.train_progress_head:
            with _timer("time/compute_progress_loss", timing_raw=self.timing_raw):
                progress_loss, loss_dict = self._compute_progress_loss(
                    model, progress_inputs, return_outputs=True, training=training
                )
                total_loss += progress_loss
                log_metadata.update(loss_dict)

        # Check for NaN in total loss
        if torch.isnan(total_loss).any():
            logger.warning(f"NaN detected in total_loss, replacing with 0.0")
            total_loss = torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype)

        # Store custom losses for logging
        self.log_metadata = log_metadata

        if return_outputs:
            extra_info = {**log_metadata, "total_loss": total_loss.item()}
            return total_loss, extra_info

        return total_loss

    def _compute_progress_loss(self, model, inputs, return_outputs=False, training=True):
        """
        Compute progress prediction loss for single frames.
        Reuses parent's helper function but simplified for single frames.
        """
        model_output, _ = self.forward_model(model, inputs, sample_type="progress")
        progress_logits = model_output.progress_logits
        progress_pred = progress_logits["A"]
        progress_target = inputs["target_progress"]
        progress_target_mask = inputs["target_progress_mask"].unsqueeze(-1)

        # Use parent's helper function
        progress_loss, spearman_corr, progress_metrics = self._compute_progress_loss_helper(
            progress_pred, progress_target, progress_target_mask
        )

        final_loss = progress_loss
        if self.config.model.train_success_head:
            success_logits = model_output.success_logits
            success_pred = success_logits["A"]
            success_labels = inputs["success_labels"]

            success_loss, success_accuracy, success_auprc, success_metrics = self._compute_success_loss_helper(
                success_pred,
                progress_target,
                success_labels,
                progress_loss_mask=progress_target_mask,
            )
            final_loss = progress_loss + success_loss

        # Check for NaN
        if torch.isnan(final_loss).any():
            if training:
                import ipdb
                ipdb.set_trace()
            logger.warning(f"NaN detected in progress loss, replacing with 0.0")
            final_loss = torch.tensor(0.0, device=final_loss.device, dtype=final_loss.dtype)

        if return_outputs:
            outputs_dict = {}
            prefix = "train" if training else "eval"

            # Simplified metrics (no data_gen_strategy stratification since we don't use strategies)
            data_sources = inputs.get("data_source", [])
            if data_sources:
                stratified_metrics = {
                    "spearman_corr": progress_metrics["masked_spearman_corr"],
                    "prog_loss": progress_metrics["masked_loss"],
                }
                # Only stratify by data source (no strategy)
                self._add_stratified_metrics(
                    outputs_dict,
                    prefix,
                    strategy_values=None,  # No strategies
                    data_source_values=data_sources,
                    metrics=stratified_metrics,
                )

            outputs_dict.update({
                f"{prefix}/prog_loss": progress_loss.item(),
                f"{prefix}/spearman_corr": spearman_corr.item(),
            })

            if self.config.model.train_success_head:
                outputs_dict.update({
                    f"{prefix}/success_loss": success_loss.item(),
                    f"{prefix}/success_accuracy": success_accuracy.item(),
                    f"{prefix}/success_auprc": success_auprc.item(),
                })

            return final_loss, outputs_dict

        return final_loss
