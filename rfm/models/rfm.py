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
from transformers import PreTrainedModel, Qwen2_5_VLModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import Optional, Dict, Any
from rfm.utils.logging import _timer


class RFMModel(PreTrainedModel):
    """Reward Foundation Model with three prediction heads for different objectives."""

    config_class = Qwen2_5_VLModel.config_class

    def __init__(self, config, processor, base_model=None):
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
        sample_type=None,  # "preference", "similarity"
        second_per_grid_ts=None,
        timing_raw=None,
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

            sample_type (str, optional):
                Type of sample to process:
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
                    Contains logits for the specified sample type:
                    - For preference: Binary logits [batch_size, 1]
                    - For similarity: Continuous similarity scores [batch_size, 1]
                    - For paired_video: Continuous similarity scores [batch_size, 1]
                    - For none: Zero tensor [batch_size, 1]

                - progress_logits (Dict[str, List[torch.Tensor]] or None):
                    Progress prediction logits split by trajectory:
                    - 'A': List of tensors for trajectory A (before vision_end token), each [seq_len_A]
                    - 'B': List of tensors for trajectory B (after vision_end token), each [seq_len_B]
                    Values should be in range [0, 1] representing task completion percentage at each timestep.

                - timing_raw (Dict[str, float]):
                    Timing information for the forward pass.
        """
        if timing_raw is None:
            timing_raw = {}

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
        with _timer("time/rfm_forward", timing_raw=timing_raw):
            outputs = self.model(**model_kwargs)

        # [batch_size, seq_len, hidden_size]
        last_hidden_state = outputs.hidden_states[-1]

        # Always compute progress for all timesteps if target_progress is provided
        progress_logits = None
        # Find vision_start_token and split_token for trajectory separation
        vision_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        split_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|split_token|>")

        progress_logits_A = []
        progress_logits_B = []

        tps = self.processor.video_processor.merge_size

        with _timer("time/progress_logits", timing_raw=timing_raw):
            for i, seq_ids in enumerate(input_ids):
                # the input_ids is structured as follows
                # the split demarcates the end of trajectory A and the start of trajectory B
                # NOTE: base Qwen2.5_VL model has a temporal_patch_size of 2, so we can only
                # predict the progress for every 2 frames.
                # [vision_start, frame_1/2_tokens, ..., frame_N/2_tokens, split_token, frame_1/2_tokens, frame_2/2_tokens, ..., frame_N/2_tokens]
                # we want to extract the hidden_states at the frame boundaries for both trajectories
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
                    raise ValueError(f"video_grid_thw index {i + 1} out of bounds for trajectory B")
                current_video_grid_B = video_grid_thw[i * tps + 1]  # [T, H, W]
                T_B, H_B, W_B = current_video_grid_B

                # Calculate tokens per frame for trajectory A: (H_A * W_A) // merge_size^2
                tokens_per_frame_A = (H_A * W_A) // (tps**2)

                # Calculate tokens per frame for trajectory B: (H_B * W_B) // merge_size^2
                tokens_per_frame_B = (H_B * W_B) // (tps**2)

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
                    boundary_hidden_states_A = last_hidden_state[i][
                        trajectory_A_boundaries
                    ]  # [num_frames_A, hidden_dim]
                    progress_A = self.progress_head(boundary_hidden_states_A).squeeze(-1)  # [num_frames_A]
                    progress_logits_A.append(progress_A)
                else:
                    progress_logits_A.append(torch.empty(0, device=last_hidden_state.device))

                # Apply progress head to hidden states at frame boundary positions for trajectory B
                if trajectory_B_boundaries.numel() > 0:
                    boundary_hidden_states_B = last_hidden_state[i][
                        trajectory_B_boundaries
                    ]  # [num_frames_B, hidden_dim]
                    progress_B = self.progress_head(boundary_hidden_states_B).squeeze(-1)  # [num_frames_B]
                    progress_logits_B.append(progress_B)
                else:
                    progress_logits_B.append(torch.empty(0, device=last_hidden_state.device))

            progress_logits = {"A": progress_logits_A, "B": progress_logits_B}

        # For preference and similarity, use specific tokens
        with _timer("time/logits", timing_raw=timing_raw):
            logits = None
            if sample_type is not None:
                if sample_type == "preference":
                    token_id = self.processor.tokenizer.convert_tokens_to_ids("<|pref_token|>")
                else:  # similarity (default)
                    token_id = self.processor.tokenizer.convert_tokens_to_ids("<|reward_token|>")

                # Find all positions where the target token appears
                token_positions = []
                for i, seq_ids in enumerate(input_ids):
                    # Find all occurrences of token_id in this sequence
                    positions = (seq_ids == token_id).nonzero(as_tuple=True)[0]
                    if len(positions) == 0:
                        raise ValueError(f"token_id {token_id} not found in sequence {i}")
                    elif len(positions) > 1:
                        raise ValueError(
                            f"token_id {token_id} appears {len(positions)} times in sequence {i}, expected exactly 1"
                        )
                    else:
                        # Exactly one occurrence
                        token_positions.append(positions[0].item())
                token_positions = torch.tensor(token_positions, device=input_ids.device, dtype=torch.long)

                # Extract hidden states at the target token positions
                token_hidden_states = torch.gather(
                    last_hidden_state,
                    1,
                    token_positions.view(-1, 1, 1).expand(-1, -1, last_hidden_state.size(-1)),
                ).squeeze(1)

                # Apply the appropriate head
                if sample_type == "preference":
                    logits = self.preference_head(token_hidden_states)
                else:  # similarity (default)
                    logits = self.similarity_head(token_hidden_states)

        return SequenceClassifierOutputWithPast(logits=logits), progress_logits, timing_raw
