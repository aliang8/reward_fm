#!/usr/bin/env python3
"""
RFM (Reward Foundation Model) VQA version implementation.
Contains the RFMModel class by using the standard Qwen2.5-VL model, training it with VQA data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin


class RFMModelVQA(PreTrainedModel):
    """RFM Model for VQA using base VLM outputs as naive baseline."""

    config_class = Qwen2_5_VLForConditionalGeneration.config_class

    def __init__(self, config, processor, base_model=None):
        super().__init__(config)
        # Use Qwen2_5_VLForConditionalGeneration for VQA (language generation)
        if base_model is not None:
            self.model = base_model
        else:
            self.model = Qwen2_5_VLForConditionalGeneration(config)

        self.processor = processor

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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            labels=labels,
            **kwargs,
        )

        # Return the outputs directly - this is the naive baseline approach
        # The base VLM's language modeling head will handle VQA generation
        return outputs

    def generate(self, *args, **kwargs):
        """Generate VQA answers using the base VLM's language modeling head."""
        return self.model.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)
