#!/usr/bin/env python3
"""
RFM Text Model implementation.
Contains the RFMText class for generating text descriptions of trajectory progress.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, Qwen2_5_VLForConditionalGeneration
from typing import Optional, Dict, Any


class RFMText(PreTrainedModel):
    """RFM Text Model that generates text descriptions of trajectory progress."""
    
    config_class = Qwen2_5_VLForConditionalGeneration.config_class

    def __init__(self, config, processor, base_model=None):
        super().__init__(config)
        # Use Qwen2_5VLForConditionalGeneration for text generation
        if base_model is not None:
            self.model = base_model
        else:
            self.model = Qwen2_5_VLForConditionalGeneration(config)
        
        # Only progress head for text generation
        self.progress_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # Progress text generation
        
        # Ensure progress head has the same dtype as the base model
        self.model_dtype = self.model.dtype
        self.progress_head = self.progress_head.to(dtype=self.model_dtype)
        
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
        second_per_grid_ts=None,
        **kwargs,
    ):   
        """
        Forward pass for the RFM Text Model.
        
        This method generates text descriptions of trajectory progress for both trajectories A and B.
        It uses the progress head to generate text at frame boundaries.
        
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
                
            labels (torch.LongTensor, optional): 
                Labels for computing the language modeling loss. Shape: [batch_size, sequence_length]
                
            second_per_grid_ts (torch.FloatTensor, optional): 
                Time stamps for video grid processing.
                
            **kwargs: Additional keyword arguments passed to the base model.
        
        Returns:
            Qwen2VLCausalLMOutputWithPast or tuple(torch.FloatTensor):
                - If labels is provided: Returns the language modeling loss
                - If no labels: Returns the generated text logits
                - Also returns progress_logits for both trajectories
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

        # [batch_size, seq_len, hidden_size]
        last_hidden_state = outputs.hidden_states[-1]

        # Find vision_start_token and split_token for trajectory separation
        vision_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        split_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|split_token|>")
        
        progress_logits_A = []
        progress_logits_B = []

        tps = self.model.config.vision_config.temporal_patch_size
        
        for i, seq_ids in enumerate(input_ids):
            # Find the position of the vision_start token
            vision_start_positions = (seq_ids == vision_start_token_id).nonzero(as_tuple=True)[0]
            if len(vision_start_positions) <= 0:
                raise ValueError(f"vision_start_token not found in sequence {i}")
            
            # Find the position of the split token after vision_start
            vision_start_pos = vision_start_positions[0].item()
            split_positions = (seq_ids == split_token_id).nonzero(as_tuple=True)[0]
            # Filter split positions to only those after vision_start
            split_positions = split_positions[split_positions > vision_start_pos]
            
            if len(split_positions) <= 0:
                raise ValueError(f"split_token not found after vision_start in sequence {i}")
            
            # Get video grid dimensions for this sample
            if video_grid_thw is None or i >= len(video_grid_thw):
                raise ValueError(f"video_grid_thw is required for progress prediction. Got: {video_grid_thw}")
            
            # For trajectory A: use video_grid_thw[i]
            current_video_grid_A = video_grid_thw[i * tps]  # [T, H, W]
            T_A, H_A, W_A = current_video_grid_A
            
            # For trajectory B: use video_grid_thw[i+1] 
            if i + 1 >= len(video_grid_thw):
                raise ValueError(f"video_grid_thw index {i+1} out of bounds for trajectory B")
            current_video_grid_B = video_grid_thw[i * tps + 1]  # [T, H, W]
            T_B, H_B, W_B = current_video_grid_B
            
            # Calculate tokens per frame for trajectory A: (H_A * W_A) // merge_size^2
            merge_size = self.processor.video_processor.merge_size
            tokens_per_frame_A = (H_A * W_A) // (merge_size ** 2)
            
            # Calculate tokens per frame for trajectory B: (H_B * W_B) // merge_size^2
            tokens_per_frame_B = (H_B * W_B) // (merge_size ** 2)
            
            # Calculate frame boundary positions for trajectory A
            frame_boundary_positions_A = []
            current_pos = vision_start_pos + 1  # Start after vision_start token
            
            # Find where each frame ends in trajectory A
            for frame_idx in range(T_A):
                # Each frame takes tokens_per_frame_A tokens
                frame_end = current_pos + tokens_per_frame_A
                frame_boundary_positions_A.append(frame_end)
                current_pos = frame_end
            
            # Get split_pos before using it in trajectory B calculations
            split_pos = split_positions[0].item()
            
            # Calculate frame boundary positions for trajectory B (after split_token)
            frame_boundary_positions_B = []
            current_pos = split_pos + 1  # Start after split_token
            
            # Find where each frame ends in trajectory B
            for frame_idx in range(T_B):
                # Each frame takes tokens_per_frame_B tokens
                frame_end = current_pos + tokens_per_frame_B
                frame_boundary_positions_B.append(frame_end)
                current_pos = frame_end
            
            # For trajectory A: extract hidden states at frame boundaries before split_token
            trajectory_A_boundaries = torch.tensor([pos for pos in frame_boundary_positions_A if pos < split_pos])
            trajectory_B_boundaries = torch.tensor([pos for pos in frame_boundary_positions_B if pos > split_pos])
            
            # Apply progress head to hidden states at frame boundary positions for trajectory A
            if trajectory_A_boundaries.numel() > 0:
                boundary_hidden_states_A = last_hidden_state[i][trajectory_A_boundaries]  # [num_frames_A, hidden_dim]
                progress_A = self.progress_head(boundary_hidden_states_A)  # [num_frames_A, vocab_size]
                progress_logits_A.append(progress_A)
            else:
                progress_logits_A.append(torch.empty(0, device=last_hidden_state.device))
            
            # Apply progress head to hidden states at frame boundary positions for trajectory B
            if trajectory_B_boundaries.numel() > 0:
                boundary_hidden_states_B = last_hidden_state[i][trajectory_B_boundaries]  # [num_frames_B, hidden_dim]
                progress_B = self.progress_head(boundary_hidden_states_B)  # [num_frames_B, vocab_size]
                progress_logits_B.append(progress_B)
            else:
                progress_logits_B.append(torch.empty(0, device=last_hidden_state.device))

        
        progress_logits = {
            'A': progress_logits_A,
            'B': progress_logits_B
        }

        # If labels are provided, compute language modeling loss
        if labels is not None:
            # Use the base model's language modeling head for the main loss
            lm_loss = outputs.loss
            
            # Add progress text generation loss if we have progress targets
            progress_loss = 0.0
            if hasattr(self, 'progress_labels') and self.progress_labels is not None:
                # This would be implemented if we want to supervise the progress text generation
                pass
            
            total_loss = lm_loss + progress_loss
            
            # Return the loss
            return total_loss, progress_logits
        
        # If no labels, return the generated text logits
        return outputs.logits, progress_logits

    def generate(self, *args, **kwargs):
        """Delegates text generation to the base model."""
        return self.model.generate(*args, **kwargs)