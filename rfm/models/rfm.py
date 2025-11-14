#!/usr/bin/env python3
"""
RFM (Reward Foundation Model) implementation.
Contains the RFM class with three prediction heads for different objectives.

Note: make sure that the forward pass uses all of the
heads or there will be some problems with FSDP sharding.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, Qwen2_5_VLModel
from transformers import SmolVLMModel

from rfm.models.utils import ModelOutput
from rfm.utils.timer import _timer


class RFM(PreTrainedModel):
    """Reward Foundation Model with three prediction heads for different objectives.

    Supports multiple base model architectures:
    - Qwen2.5-VL (Qwen2_5_VLModel)
    - SmolVLM (AutoModelForImageTextToText)
    """
    config_class = Qwen2_5_VLModel.config_class

    config_class = Qwen2_5_VLModel.config_class

    def __init__(self, config, processor, tokenizer, base_model=None, base_model_id=None, model_config=None):
        super().__init__(config)

        if "SmolVLM" in base_model_id:
            hidden_size = config.text_config.hidden_size
            self.model_cls = SmolVLMModel
        else:
            hidden_size = config.hidden_size
            self.model_cls = Qwen2_5_VLModel

        if base_model is not None:
            self.model = base_model
        else:
            self.model = self.model_class(config)

        self.config_class = self.model_cls.config_class
        self.base_model_id = base_model_id

        self.progress_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
        self.preference_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )
        self.similarity_head = nn.Linear(hidden_size, 1)

        self.success_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )

        self.model_dtype = self.model.dtype
        self.progress_head = self.progress_head.to(dtype=self.model_dtype)
        self.preference_head = self.preference_head.to(dtype=self.model_dtype)
        self.similarity_head = self.similarity_head.to(dtype=self.model_dtype)
        self.success_head = self.success_head.to(dtype=self.model_dtype)

        self.processor = processor
        self.tokenizer = tokenizer
        
        # Store config for averaging temporal patches
        # model_config is the ModelConfig, config is the base model config
        if model_config is not None:
            self.average_temporal_patches = getattr(model_config, "average_temporal_patches", False)
        else:
            self.average_temporal_patches = getattr(config, "average_temporal_patches", False)

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegates gradient checkpointing enabling to the base model."""
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        """Delegates gradient checkpointing disabling to the base model."""
        self.model.gradient_checkpointing_disable(**kwargs)

    def _extract_hidden_states_for_smolvlm(
        self,
        hidden_state: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract image/video frame embeddings from hidden states for SmolVLM.
        
        SmolVLM uses <fake_token_around_image> tokens to wrap images/videos.
        This function finds all pairs of these tokens and mean pools the hidden states
        between them to get per-frame/image embeddings.
        
        Args:
            hidden_state: Hidden states tensor [seq_len, hidden_dim]
            input_ids: Input token IDs [seq_len]
            
        Returns:
            frame_embeddings: Tensor [num_frames, hidden_dim] containing mean-pooled
                            embeddings for each frame/image between token pairs
        """
        # Get the token ID for <fake_token_around_image>
        fake_token_id = self.tokenizer.convert_tokens_to_ids("<fake_token_around_image>")
                   
        # Find all positions where the fake token appears
        token_positions = (input_ids == fake_token_id).nonzero(as_tuple=True)[0]
        
        if len(token_positions) == 0:
            raise ValueError(
                f"No <fake_token_around_image> tokens found in input_ids. "
                f"Token ID {fake_token_id} not found in sequence."
            )
        
        if len(token_positions) % 2 != 0:
            raise ValueError(
                f"Expected even number of <fake_token_around_image> tokens (pairs), "
                f"but found {len(token_positions)} tokens."
            )
        
        # Group tokens into pairs and extract hidden states between them
        frame_embeddings = []
        for i in range(0, len(token_positions), 2):
            start_pos = token_positions[i].item()
            end_pos = token_positions[i + 1].item()
            
            # Extract hidden states between the token pair (exclusive of the tokens themselves)
            # Add 1 to start_pos to exclude the start token, end_pos is exclusive
            frame_tokens = hidden_state[start_pos + 1 : end_pos]
            
            if frame_tokens.shape[0] == 0:
                # If no tokens between the pair, use the token positions themselves
                # This shouldn't happen normally, but handle it gracefully
                frame_embedding = (hidden_state[start_pos] + hidden_state[end_pos]) / 2.0
            else:
                # Mean pool all tokens between the pair
                frame_embedding = frame_tokens.mean(dim=0)  # [hidden_dim]
            
            frame_embeddings.append(frame_embedding)
        
        if len(frame_embeddings) == 0:
            return torch.empty(0, hidden_state.shape[-1], device=hidden_state.device)
        
        return torch.stack(frame_embeddings)  # [num_frames, hidden_dim]

    def _extract_progress_from_trajectory(
        self,
        hidden_state: torch.Tensor,
        start_position: int,
        video_grid_thw: list[int],  # [T, H, W]
        merge_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract progress and success predictions from a trajectory's hidden states.
        
        Args:
            hidden_state: Hidden states tensor [seq_len, hidden_dim]
            start_position: Starting position in the sequence for this trajectory
            video_grid_thw: Video grid dimensions [T, H, W] where T is number of temporal patch groups,
                           H and W are spatial grid dimensions
            merge_size: Merge size for patch grouping
            
        Returns:
            tuple: (progress_logits [T], success_logits [T])
        """
        T, H, W = video_grid_thw
        
        if T == 0:
            return torch.empty(0, device=hidden_state.device), torch.empty(0, device=hidden_state.device)
        
        # Calculate tokens per frame: (H * W) // merge_size^2
        tokens_per_frame = (H * W) // (merge_size**2)
        
        if self.average_temporal_patches:
            # Average all tokens within each temporal patch group
            temporal_patch_tokens = []
            current_pos = start_position
            for t_idx in range(T):
                start_idx = current_pos
                end_idx = current_pos + tokens_per_frame
                patch_tokens = hidden_state[start_idx:end_idx]  # [tokens_per_frame, hidden_dim]
                patch_embedding = patch_tokens.mean(dim=0)  # [hidden_dim] - averaged
                temporal_patch_tokens.append(patch_embedding)
                current_pos = end_idx
            boundary_hidden_states = torch.stack(temporal_patch_tokens)  # [T, hidden_dim]
        else:
            # Use last token (boundary) of each temporal patch group
            frame_boundary_positions = []
            current_pos = start_position
            for _frame_idx in range(T):
                frame_end = current_pos + tokens_per_frame
                frame_boundary_positions.append(frame_end)
                current_pos = frame_end
            
            trajectory_boundaries = torch.tensor(frame_boundary_positions, device=hidden_state.device)
            boundary_hidden_states = hidden_state[trajectory_boundaries]  # [T, hidden_dim]
        
        assert boundary_hidden_states.shape[0] == T, (
            f"Expected {T} frames, got {boundary_hidden_states.shape[0]}"
        )
        progress = self.progress_head(boundary_hidden_states).squeeze(-1)  # [T]
        success = self.success_head(boundary_hidden_states).squeeze(-1)  # [T]
        
        return progress, success

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        sample_type=None,  # "preference", "progress", "similarity"
        second_per_grid_ts=None,
        timing_raw=None,
        **kwargs,
    ):
        """
        Forward pass for the RFM (Reward Foundation Model).

        This method handles three types of predictions:
        1. **Preference prediction**: Binary classification comparing two trajectories
        2. **Progress prediction**: Regression predicting task completion progress (0-1)
        3. **Similarity prediction**: Scoring how similar a trajectory is to a reference

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
            tuple: (model_output, timing_raw)
                - model_output (ModelOutput):
                    Contains predictions for the specified sample type:
                    - pref_logits: Binary logits [batch_size, 1] for preference
                    - sim_logits: Continuous similarity scores [batch_size, 1] for similarity
                    - progress_logits: Dict with 'A' and 'B' trajectories
                        - 'A': List of tensors for trajectory A (before vision_end token), each [seq_len_A]
                        - 'B': List of tensors for trajectory B (after vision_end token), each [seq_len_B]
                    Values should be in range [0, 1] representing task completion percentage at each timestep.

                - timing_raw (Dict[str, float]):
                    Timing information for the forward pass.
        """
        if timing_raw is None:
            timing_raw = {}

        if "SmolVLM" in self.base_model_id:
            # SmolVLM uses different parameter names and doesn't need some Qwen-specific params
            model_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,  # SmolVLM uses pixel_values for both images and videos
                **kwargs,
            }
        else:
            # Qwen2.5-VL parameters
            model_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "second_per_grid_ts": second_per_grid_ts,
                **kwargs,
            }

        # Always compute progress for all timesteps if target_progress is provided
        progress_logits = None
        success_logits = None

        if "SmolVLM" in self.base_model_id:
            with _timer("time/rfm_forward", timing_raw=timing_raw):
                outputs = self.model(**model_kwargs, output_hidden_states=True, return_dict=True)

            B = input_ids.shape[0]
            # from the last layer of the text transformer
            # these hidden states are post multimodal fusion
            hidden_state = outputs.hidden_states[-1]  # [B, seq_len, hidden_dim]

            progress_logits_A = []
            progress_logits_B = []
            success_logits_A = []
            success_logits_B = []

            with _timer("time/progress_logits", timing_raw=timing_raw):
                for i in range(B):
                    # Extract frame embeddings for this sample using <fake_token_around_image> tokens
                    # This returns embeddings for all frames from all videos in the sequence
                    frame_embeddings = self._extract_hidden_states_for_smolvlm(
                        hidden_state[i],  # [seq_len, hidden_dim]
                        input_ids[i],     # [seq_len]
                    )  # [num_frames, hidden_dim]

                    if frame_embeddings.shape[0] == 0:
                        raise ValueError(f"No frame embeddings extracted for sample {i}")

                    # For progress samples, there's only one video/trajectory (V=1)
                    # For preference/similarity samples, there are two videos/trajectories (V=2)
                    if sample_type == "progress":
                        # All frames belong to trajectory A
                        trajectory_A_frames = frame_embeddings
                        trajectory_B_frames = None
                    else:
                        # this is find because we assume both videos are
                        # padded to the same number of frames
                        mid_point = frame_embeddings.shape[0] // 2
                        trajectory_A_frames = frame_embeddings[:mid_point]
                        trajectory_B_frames = frame_embeddings[mid_point:]

                    # Apply heads to trajectory A frames
                    progress_A = self.progress_head(trajectory_A_frames).squeeze(-1)  # [T_A]
                    success_A = self.success_head(trajectory_A_frames).squeeze(-1)  # [T_A]
                    progress_logits_A.append(progress_A)
                    success_logits_A.append(success_A)

                    # Apply heads to trajectory B frames (if available)
                    if trajectory_B_frames is not None:
                        progress_B = self.progress_head(trajectory_B_frames).squeeze(-1)  # [T_B]
                        success_B = self.success_head(trajectory_B_frames).squeeze(-1)  # [T_B]
                        progress_logits_B.append(progress_B)
                        success_logits_B.append(success_B)
                    else:
                        progress_logits_B.append(None)
                        success_logits_B.append(None)

            progress_logits = {"A": progress_logits_A, "B": progress_logits_B}
            success_logits = {"A": success_logits_A, "B": success_logits_B}
        else:
            with _timer("time/rfm_forward", timing_raw=timing_raw):
                outputs = self.model(**model_kwargs)

            # [batch_size, seq_len, hidden_size]
            hidden_state = outputs.last_hidden_state

            # Original Qwen2.5-VL progress computation logic
            # Find vision_start_token and split_token for trajectory separation
            vision_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")

            progress_logits_A = []
            progress_logits_B = []
            success_logits_A = []
            success_logits_B = []

            # temporal patch size
            tps = self.processor.video_processor.temporal_patch_size
            merge_size = self.processor.video_processor.merge_size

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

                    # Get video grid dimensions for this sample
                    if video_grid_thw is None or i >= len(video_grid_thw):
                        raise ValueError(f"video_grid_thw is required for progress prediction. Got: {video_grid_thw}")

                    # For trajectory A: use video_grid_thw[i]
                    if sample_type == "progress":
                        current_video_grid_A = video_grid_thw[i]  # [T, H, W]
                    else:
                        current_video_grid_A = video_grid_thw[i * tps]  # [T, H, W]

                    # Extract progress and success from trajectory A
                    start_position_A = vision_start_positions[0].item()
                    progress_A, success_A = self._extract_progress_from_trajectory(
                        hidden_state[i],
                        start_position_A,
                        current_video_grid_A,
                        merge_size,
                    )
                    progress_logits_A.append(progress_A)
                    success_logits_A.append(success_A)

                    # For progress-only samples, we don't need trajectory B
                    if sample_type != "progress":
                        # For trajectory B: use video_grid_thw[i+1]
                        if (i * tps) + 1 >= len(video_grid_thw):
                            raise ValueError(f"video_grid_thw index {(i * tps) + 1} out of bounds for trajectory B")
                        current_video_grid_B = video_grid_thw[i * tps + 1]  # [T, H, W]

                        # Extract progress and success from trajectory B
                        start_position_B = vision_start_positions[1].item()
                        progress_B, success_B = self._extract_progress_from_trajectory(
                            hidden_state[i],
                            start_position_B,
                            current_video_grid_B,
                            merge_size,
                        )
                        progress_logits_B.append(progress_B)
                        success_logits_B.append(success_B)
                    else:
                        # For progress-only samples, trajectory B is None
                        progress_logits_B.append(None)
                        success_logits_B.append(None)

                progress_logits = {"A": progress_logits_A, "B": progress_logits_B}
                success_logits = {"A": success_logits_A, "B": success_logits_B}

        # Create ModelOutput
        output = ModelOutput()
        output.progress_logits = progress_logits
        output.success_logits = success_logits

        # For preference and similarity, use specific tokens
        if sample_type in ["preference", "similarity"]:
            if sample_type == "preference":
                token_id = self.processor.tokenizer.convert_tokens_to_ids("<|pref_token|>")
            elif sample_type == "similarity":
                token_id = self.processor.tokenizer.convert_tokens_to_ids("<|sim_token|>")
            else:
                import ipdb

                ipdb.set_trace()
                raise ValueError(f"Invalid sample type: {sample_type}")

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
                hidden_state,
                1,
                token_positions.view(-1, 1, 1).expand(-1, -1, hidden_state.size(-1)),
            ).squeeze(1)

            # Apply the appropriate head
            if sample_type == "preference":
                output.pref_logits = self.preference_head(token_hidden_states)
            else:  # similarity
                output.sim_logits = self.similarity_head(token_hidden_states)

        return output, timing_raw
