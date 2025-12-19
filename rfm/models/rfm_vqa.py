#!/usr/bin/env python3
"""
RFM (Reward Foundation Model) VQA version implementation.
Contains the RFM class by using the standard Qwen2.5-VL model, training it with VQA data.
"""

from transformers import PreTrainedModel, Qwen2_5_VLForConditionalGeneration, SmolVLMForConditionalGeneration
from transformers import AutoModelForImageTextToText as Molmo2VLForConditionalGeneration
# Try to import Qwen3 if available
try:
    from transformers import Qwen3VLForConditionalGeneration
    HAS_QWEN3 = True
except ImportError:
    HAS_QWEN3 = False
    Qwen3VLForConditionalGeneration = None

import torch


class RFMVQA(PreTrainedModel):
    """RFM Model for VQA using base VLM outputs as naive baseline."""

    config_class = Qwen2_5_VLForConditionalGeneration.config_class

    # Declare support for SDPA and Flash Attention (will delegate to underlying model), needed for Qwen3
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def __init__(self, config, processor, tokenizer, base_model=None, base_model_id=None, model_config=None):
        super().__init__(config)
        # Use Qwen2_5_VLForConditionalGeneration for VQA (language generation)
        if base_model is not None:
            self.model = base_model
        elif "SmolVLM" in base_model_id:
            self.model = SmolVLMForConditionalGeneration(config)
        elif "Qwen2.5" in base_model_id:
            self.model = Qwen2_5_VLForConditionalGeneration(config)
        elif "Qwen3" in base_model_id:
            if HAS_QWEN3 and Qwen3VLForConditionalGeneration is not None:
                self.model = Qwen3VLForConditionalGeneration(config)
            else:
                raise ImportError("Qwen3VLForConditionalGeneration not available. Please update transformers.")
        elif "Molmo" in base_model_id:
            self.model = Molmo2VLForConditionalGeneration(config)
        else:
            raise ValueError(f"Base model id not supported in RFMVQA yet: {base_model_id}")

        self.processor = processor
        self.tokenizer = tokenizer
        self.base_model_id = base_model_id

        # Inherit attention implementation support from underlying model
        if hasattr(self.model, "_supports_sdpa"):
            self._supports_sdpa = self.model._supports_sdpa
        if hasattr(self.model, "_supports_flash_attn_2"):
            self._supports_flash_attn_2 = self.model._supports_flash_attn_2

    def _sdpa_can_dispatch(self, is_init_check=False):
        """Delegate SDPA dispatch check to underlying model."""
        # If underlying model doesn't have this method, default to True since we declared support
        return True
        # if hasattr(self.model, '_sdpa_can_dispatch'):
        #    return self.model._sdpa_can_dispatch(is_init_check)

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegates gradient checkpointing enabling to the base model."""
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        """Delegates gradient checkpointing disabling to the base model."""
        self.model.gradient_checkpointing_disable(**kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        labels=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        """
        Forward pass for VQA using base VLM outputs as naive baseline.

        This implementation directly uses the base VLM's language modeling head
        without adding separate prediction heads, serving as a naive baseline.

        Args:
            input_ids (torch.LongTensor, optional):
                Indices of input sequence tokens in the vocabulary. Shape: [batch_size, sequence_length]

            attention_mask (torch.Tensor, optional):
                Mask to avoid performing attention on padding token indices. Shape: [batch_size, sequence_length]

            pixel_values (torch.FloatTensor, optional):
                Pixel values for images. Shape: [batch_size, num_channels, height, width]

            pixel_values_videos (torch.FloatTensor, optional):
                Pixel values for video frames. Shape: [batch_size, num_frames, num_channels, height, width]

            image_grid_thw (torch.LongTensor, optional):
                Image grid dimensions (N, 3) for image processing

            video_grid_thw (torch.LongTensor, optional):
                Video grid dimensions (N, 3) for video processing

            labels (torch.LongTensor, optional):
                Labels for computing the language modeling loss. Shape: [batch_size, sequence_length]
                If provided, the model will compute the loss for VQA training.

            second_per_grid_ts (torch.FloatTensor, optional):
                Time stamps for video grid processing

            **kwargs: Additional keyword arguments passed to the base model.

        Returns:
            CausalLMOutputWithPast:
                - loss (torch.FloatTensor, optional): Language modeling loss if labels are provided
                - logits (torch.FloatTensor): Language modeling logits [batch_size, sequence_length, vocab_size]
                - past_key_values (tuple, optional): Past key values for generation
                - hidden_states (tuple, optional): Hidden states from all layers
                - attentions (tuple, optional): Attention weights from all layers
        """
        # Forward pass through the base VLM
        # Note: dtype casting is handled in the trainer's _prepare_inputs method
        # Only pass second_per_grid_ts if it's not None (some models may not support it)
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "labels": labels,
        }

        if second_per_grid_ts is not None:
            forward_kwargs["second_per_grid_ts"] = second_per_grid_ts

        outputs = self.model(**forward_kwargs, **kwargs)

        # Return the outputs directly - this is the naive baseline approach
        # The base VLM's language modeling head will handle VQA generation
        return outputs

    def generate(self, *args, **kwargs):
        """
        Generate VQA answers using the base VLM's language modeling head.
        Note: dtype casting is handled in the trainer's _prepare_inputs method.
        """
        return self.model.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)
