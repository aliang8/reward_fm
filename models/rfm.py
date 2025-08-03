#!/usr/bin/env python3
"""
RFM (Reward Foundation Model) implementation.
Contains the RFMModel class with three prediction heads for different objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import Optional, Dict, Any


class RFMModel(PreTrainedModel):
    """Reward Foundation Model with three prediction heads for different objectives."""
    
    config_class = Qwen2_5_VLForConditionalGeneration.config_class

    def __init__(self, config, tokenizer):
        super().__init__(config)
        # The RFMModel now owns and creates its submodules.
        # This is the standard pattern for PreTrainedModel.
        self.model = Qwen2_5_VLForConditionalGeneration(config)
        
        # Three prediction heads for different objectives
        self.progress_head = nn.Linear(config.hidden_size, 1, bias=False)  # Progress prediction (0-1)
        self.preference_head = nn.Linear(config.hidden_size, 1, bias=False)  # Preference prediction (binary)
        self.similarity_head = nn.Linear(config.hidden_size, 1, bias=False)  # Similarity scoring (reward)

        # Ensure all heads have the same dtype as the base model
        self.model_dtype = self.model.dtype
        self.progress_head = self.progress_head.to(dtype=self.model_dtype)
        self.preference_head = self.preference_head.to(dtype=self.model_dtype)
        self.similarity_head = self.similarity_head.to(dtype=self.model_dtype)
        
        self.tokenizer = tokenizer

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
        prediction_type=None,  # "preference" or "similarity"
        target_progress=None,  # For progress prediction on trajectory A
        **kwargs,
    ):
        # Handle case where input_ids and attention_mask might be in kwargs
        if input_ids is None and "input_ids" in kwargs:
            input_ids = kwargs.pop("input_ids")
        if attention_mask is None and "attention_mask" in kwargs:
            attention_mask = kwargs.pop("attention_mask")
        
        # Handle nested argument structures (common in evaluation)
        if input_ids is None and "input_ids_chosen" in kwargs:
            input_ids = kwargs.pop("input_ids_chosen")
        if attention_mask is None and "attention_mask_chosen" in kwargs:
            attention_mask = kwargs.pop("attention_mask_chosen")
        
        # Handle case where arguments might be in a nested dict
        if input_ids is None and "inputs" in kwargs:
            inputs = kwargs.pop("inputs")
            if isinstance(inputs, dict):
                input_ids = inputs.get("input_ids")
                attention_mask = inputs.get("attention_mask")
        
        # Ensure required arguments are provided
        if input_ids is None:
            raise ValueError("input_ids is required")
        if attention_mask is None:
            raise ValueError("attention_mask is required")
        
        # Prepare model kwargs with all available inputs
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            **kwargs,
        }
        
        # Add vision inputs based on what's available
        if pixel_values is not None:
            model_kwargs["pixel_values"] = pixel_values
        if pixel_values_videos is not None:
            model_kwargs["pixel_values_videos"] = pixel_values_videos
        if image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = image_grid_thw
        if video_grid_thw is not None:
            model_kwargs["video_grid_thw"] = video_grid_thw
        
        # Forward pass through the model
        outputs = self.model(**model_kwargs)
        last_hidden_state = outputs.hidden_states[-1]

        # Always compute progress for trajectory A if target_progress is provided
        progress_logits = None
        if target_progress is not None:
            # Use the last token position for each sequence
            last_positions = attention_mask.sum(dim=1) - 1  # [batch_size]
            token_hidden_states = torch.gather(
                last_hidden_state,
                1,
                last_positions.view(-1, 1, 1).expand(
                    -1, -1, last_hidden_state.size(-1)
                ).long(),
            ).squeeze(1)  # [batch_size, hidden_size]
            
            progress_logits = self.progress_head(token_hidden_states)  # [batch_size, 1]

        # For preference and similarity, use specific tokens
        logits = None
        if prediction_type is not None:
            if prediction_type == "preference":
                token_id = self.tokenizer.convert_tokens_to_ids("<|pref_token|>")
            else:  # similarity (default)
                token_id = self.tokenizer.convert_tokens_to_ids("<|reward_token|>")
            
            # Find all positions where the target token appears
            token_positions = []
            for i, seq_ids in enumerate(input_ids):
                # Find the last occurrence of token_id in this sequence
                positions = (seq_ids == token_id).nonzero(as_tuple=True)[0]
                if len(positions) > 0:
                    token_positions.append(positions[-1].item())
                else:
                    # Fallback to last token if target token not found
                    token_positions.append(attention_mask[i].sum().item() - 1)
            token_positions = torch.tensor(token_positions, device=input_ids.device, dtype=torch.long)
            
            # Extract hidden states at the target token positions
            token_hidden_states = torch.gather(
                last_hidden_state,
                1,
                token_positions.view(-1, 1, 1).expand(
                    -1, -1, last_hidden_state.size(-1)
                ),
            ).squeeze(1)
            
            # Apply the appropriate head
            if prediction_type == "preference":
                logits = self.preference_head(token_hidden_states)
            else:  # similarity (default)
                logits = self.similarity_head(token_hidden_states)
        
        if logits is not None:
            return SequenceClassifierOutputWithPast(logits=logits), progress_logits
        else:
            # No prediction requested
            return SequenceClassifierOutputWithPast(logits=torch.zeros(input_ids.shape[0], 1, device=input_ids.device)), progress_logits    