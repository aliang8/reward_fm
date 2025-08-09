#!/usr/bin/env python3
"""
RFM (Reward Foundation Model) implementation.
Contains the RFMModel class with three prediction heads for different objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import Optional, Dict, Any
from dataclasses import dataclass


class RFMModel(PreTrainedModel):
    """Reward Foundation Model with three prediction heads for different objectives."""
    
    config_class = Qwen2_5_VLModel.config_class

    def __init__(self, config, tokenizer, base_model=None):
        super().__init__(config)
        # Use the provided base model or create a new one
        if base_model is not None:
            # Use the provided base model to preserve embedding layer configuration
            from rfm.utils.logging import rank_0_print
            rank_0_print(f"Using provided base model with embedding shape: {base_model.get_input_embeddings().weight.shape}")
            self.model = base_model
        else:
            # Create a new base model if none provided
            from rfm.utils.logging import rank_0_print
            rank_0_print(f"Creating new base model")
            self.model = Qwen2_5_VLModel(config)

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
        second_per_grid_ts=None,
        **kwargs,
    ):
        """
        Forward pass that computes base model hidden states and applies prediction heads.
        This is the main entry point that maintains backward compatibility.
        """
        # Get base model hidden states
        hidden_states = self.compute_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )
        
        # Apply prediction heads based on type
        logits = None
        
        if prediction_type == "preference":
            logits = self.compute_preference_logits(hidden_states, input_ids, attention_mask)
        elif prediction_type == "similarity":
            logits = self.compute_similarity_logits(hidden_states, input_ids, attention_mask)
        
        return logits

    def compute_hidden_states(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
    ):
        """
        Compute base model hidden states without applying prediction heads.
        
        Returns:
            BaseModelOutput with hidden_states and last_hidden_state
        """      
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "second_per_grid_ts": second_per_grid_ts,
        }

        # Forward pass through the base model
        outputs = self.model(**model_kwargs)
        last_hidden_state = outputs.hidden_states[-1]
        
        return type('BaseModelOutput', (), {
            'hidden_states': outputs.hidden_states,
            'last_hidden_state': last_hidden_state
        })()

    def compute_progress_logits(self, hidden_states, attention_mask):
        """
        Compute progress prediction logits from hidden states.
        
        Args:
            hidden_states: BaseModelOutput with hidden_states and last_hidden_state
            attention_mask: Attention mask tensor
            
        Returns:
            progress_logits: Progress prediction logits [batch_size, 1]
        """
        last_hidden_state = hidden_states.last_hidden_state
        
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
        return progress_logits

    def compute_preference_logits(self, hidden_states, input_ids, attention_mask):
        """
        Compute preference prediction logits from hidden states.
        
        Args:
            hidden_states: BaseModelOutput with hidden_states and last_hidden_state
            input_ids: Input token IDs
            attention_mask: Attention mask tensor
            
        Returns:
            preference_logits: Preference prediction logits [batch_size, 1]
        """
        last_hidden_state = hidden_states.last_hidden_state
        
        # Find preference token positions
        token_id = self.tokenizer.convert_tokens_to_ids("<|pref_token|>")
        token_positions = self._find_token_positions(input_ids, attention_mask, token_id)
        
        # Extract hidden states at the target token positions
        token_hidden_states = torch.gather(
            last_hidden_state,
            1,
            token_positions.view(-1, 1, 1).expand(
                -1, -1, last_hidden_state.size(-1)
            ),
        ).squeeze(1)
        
        preference_logits = self.preference_head(token_hidden_states)
        return preference_logits

    def compute_similarity_logits(self, hidden_states, input_ids, attention_mask):
        """
        Compute similarity prediction logits from hidden states.
        
        Args:
            hidden_states: BaseModelOutput with hidden_states and last_hidden_state
            input_ids: Input token IDs
            attention_mask: Attention mask tensor
            
        Returns:
            similarity_logits: Similarity prediction logits [batch_size, 1]
        """
        last_hidden_state = hidden_states.last_hidden_state
        
        # Find similarity token positions
        token_id = self.tokenizer.convert_tokens_to_ids("<|reward_token|>")
        token_positions = self._find_token_positions(input_ids, attention_mask, token_id)
        
        # Extract hidden states at the target token positions
        token_hidden_states = torch.gather(
            last_hidden_state,
            1,
            token_positions.view(-1, 1, 1).expand(
                -1, -1, last_hidden_state.size(-1)
            ),
        ).squeeze(1)
        
        similarity_logits = self.similarity_head(token_hidden_states)
        return similarity_logits

    def _find_token_positions(self, input_ids, attention_mask, token_id):
        """
        Find the positions of a specific token in each sequence.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask tensor
            token_id: Token ID to find
            
        Returns:
            token_positions: Tensor of token positions [batch_size]
        """
        token_positions = []
        for i, seq_ids in enumerate(input_ids):
            # Find the last occurrence of token_id in this sequence
            positions = (seq_ids == token_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                token_positions.append(positions[-1].item())
            else:
                # Fallback to last token if target token not found
                token_positions.append(attention_mask[i].sum().item() - 1)
        
        return torch.tensor(token_positions, device=input_ids.device, dtype=torch.long)    