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
        return super()._compute_progress_loss(
            model, inputs, return_outputs=return_outputs, training=training, stratify_by_strategy=False
        )
