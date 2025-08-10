#!/usr/bin/env python3
"""
RFM (Reward Foundation Model) implementation.
Contains the RFMModel class with three prediction heads for different objectives.

Note: make sure that the forward pass uses all of the
heads or there will be some problems with FSDP sharding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import Optional, Dict, Any


class RFMModel(PreTrainedModel):
    """Reward Foundation Model with three prediction heads for different objectives."""
    
    config_class = Qwen2_5_VLModel.config_class

    def __init__(self, config, tokenizer, base_model=None):
        super().__init__(config)
        # The RFMModel now owns and creates its submodules.
        # This is the standard pattern for PreTrainedModel.
        if base_model is not None:
            self.model = base_model
        else:
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
        target_progress=None,  # For progress prediction on trajectory A
        second_per_grid_ts=None,
        **kwargs,
    ):   
        """
        Forward pass for the RFM (Reward Foundation Model).
        
        This method handles three types of predictions:
        1. **Preference prediction**: Binary classification comparing two trajectories
        2. **Similarity prediction**: Scoring how similar a trajectory is to a reference
        3. **Progress prediction**: Regression predicting task completion progress (0-1)
        
        Args:
            input_ids (torch.LongTensor, optional): 
                Indices of input sequence tokens in the vocabulary. Shape: [batch_size, sequence_length]
                
            attention_mask (torch.Tensor, optional): 
                Mask to avoid performing attention on padding token indices. Shape: [batch_size, sequence_length]
                Values: 1 for tokens that are NOT masked, 0 for tokens that are masked.
                
            pixel_values_videos (torch.FloatTensor, optional): 
                Pixel values for video frames. Shape: [sequence_length, embedding_dim]
                
            image_grid_thw (torch.LongTensor, optional): 
                Image grid dimensions (N, 3) for image processing
                
            video_grid_thw (torch.LongTensor, optional): 
                Video grid dimensions (N, 3) for video processing
                
            prediction_type (str, optional): 
                Type of prediction to perform:
                - "preference": Uses preference head with <|pref_token|> for binary trajectory comparison
                - "similarity": Uses similarity head with <|reward_token|> for trajectory-reference scoring
                - None: No specific prediction, returns zero logits
                
            target_progress (torch.FloatTensor, optional): 
                Target progress values for progress prediction. Shape: [batch_size, sequence_length]
                If provided, progress prediction will be computed using the last token position.
                
            second_per_grid_ts (torch.FloatTensor, optional): 
                Time stamps for video grid processing.
                
            **kwargs: Additional keyword arguments passed to the base model.
        
        Returns:
            tuple: (model_outputs, progress_logits)
                - model_outputs (SequenceClassifierOutputWithPast): 
                    Contains logits for the specified prediction type:
                    - For preference: Binary logits [batch_size, 1] 
                    - For similarity: Continuous similarity scores [batch_size, 1]
                    - For none: Zero tensor [batch_size, 1]
                    
                - progress_logits (Dict[str, List[torch.Tensor]] or None):
                    Progress prediction logits split by trajectory:
                    - 'A': List of tensors for trajectory A (before split token), each [seq_len_A]
                    - 'B': List of tensors for trajectory B (after split token), each [seq_len_B]
                    Values should be in range [0, 1] representing task completion percentage at each timestep.
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
            **kwargs,
        }

        # Forward pass through the model
        outputs = self.model(**model_kwargs)
        last_hidden_state = outputs.hidden_states[-1]

        # Always compute progress for all timesteps if target_progress is provided
        progress_logits = None
        if target_progress is not None:
            # Apply progress head to all timesteps
            progress_logits_full = self.progress_head(last_hidden_state)  # [batch_size, seq_len, 1]
            progress_logits_full = progress_logits_full.squeeze(-1)  # [batch_size, seq_len]
            
            # Split progress predictions based on split token
            progress_logits_A = []
            progress_logits_B = []
            
            split_token_id = self.tokenizer.convert_tokens_to_ids("<|split_token|>")
            
            for i, seq_ids in enumerate(input_ids):
                # Find the position of the split token
                split_positions = (seq_ids == split_token_id).nonzero(as_tuple=True)[0]
                
                if len(split_positions) > 0:
                    split_pos = split_positions[0].item()
                    
                    # Split the sequence at the split token
                    seq_A = progress_logits_full[i, :split_pos]  # Before split token
                    seq_B = progress_logits_full[i, split_pos+1:]  # After split token
                    
                    progress_logits_A.append(seq_A)
                    progress_logits_B.append(seq_B)
                else:
                    # No split token found, treat entire sequence as trajectory A
                    progress_logits_A.append(progress_logits_full[i])
                    progress_logits_B.append(torch.empty(0, device=progress_logits_full.device))
            
            progress_logits = {
                'A': progress_logits_A,
                'B': progress_logits_B
            }

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
        
        return SequenceClassifierOutputWithPast(logits=logits), progress_logits
