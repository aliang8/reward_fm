from .rfm_heads_trainer import RFMHeadsTrainer
from rfm.utils.timer import _timer


class ReWiNDTrainer(RFMHeadsTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_model(self, model, inputs, sample_type="progress"):
        """Forward pass for the model."""
        with _timer("time/forward", timing_raw=self.timing_raw):
            model_outputs, progress_logits, model_timing_raw = model(
                video_embeddings=inputs["video_embeddings"],
                text_embeddings=inputs["text_embeddings"],
                sample_type=sample_type,
                timing_raw=self.timing_raw,
            )
            self.timing_raw.update(model_timing_raw)
            return model_outputs, progress_logits, model_timing_raw
