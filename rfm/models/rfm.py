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

    def __init__(self, config, processor, tokenizer, base_model=None, base_model_id=None):
        super().__init__(config)

        if "SmolVLM" in base_model_id:
            # hidden_size = config.vision_config.hidden_size
            hidden_size = 960
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

        # Three prediction heads for different objectives
        # self.progress_head = nn.Linear(hidden_size, 1)  # Progress prediction (0-1)
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
        # self.preference_head = nn.Linear(hidden_size, 1)  # Preference prediction (binary)
        self.preference_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )
        self.similarity_head = nn.Linear(hidden_size, 1)  # Similarity scoring (reward)

        # Ensure all heads have the same dtype as the base model
        self.model_dtype = self.model.dtype
        self.progress_head = self.progress_head.to(dtype=self.model_dtype)
        self.preference_head = self.preference_head.to(dtype=self.model_dtype)
        self.similarity_head = self.similarity_head.to(dtype=self.model_dtype)

        self.processor = processor
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

        if "SmolVLM" in self.base_model_id:
            with _timer("time/rfm_forward", timing_raw=timing_raw):
                outputs = self.model(**model_kwargs, output_hidden_states=True, return_dict=True)

            B = input_ids.shape[0]
            if sample_type == "progress":
                V = 1
            else:
                V = 2
            T = 16  # TODO: fix this hardcoding
            D = outputs.image_hidden_states.shape[-1]

            # from the last layer of the text transformer
            hidden_state = outputs.hidden_states[-1]
            vision_hidden_states = outputs.image_hidden_states
            vision_hidden_states = vision_hidden_states.reshape(B, V, T, -1, D)

            # per frame hidden state
            # [B, V, T, D]
            per_frame_hidden_states = vision_hidden_states.mean(dim=3)

            # apply progress head to per frame hidden states
            progress_logits = self.progress_head(per_frame_hidden_states)

            progress_logits_A = progress_logits[:, 0, :, :]
            progress_logits_B = None
            if sample_type != "progress":
                progress_logits_B = progress_logits[:, 1, :, :]
                progress_logits_B = progress_logits_B.squeeze(-1)

            progress_logits = {"A": progress_logits_A.squeeze(-1), "B": progress_logits_B}
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

            # temporal patch size
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

                    # Get video grid dimensions for this sample
                    if video_grid_thw is None or i >= len(video_grid_thw):
                        raise ValueError(f"video_grid_thw is required for progress prediction. Got: {video_grid_thw}")

                    # For trajectory A: use video_grid_thw[i]
                    if sample_type == "progress":
                        current_video_grid_A = video_grid_thw[i]  # [T, H, W]
                    else:
                        current_video_grid_A = video_grid_thw[i * tps]  # [T, H, W]

                    T_A, H_A, W_A = current_video_grid_A

                    # Calculate tokens per frame for trajectory A: (H_A * W_A) // merge_size^2
                    tokens_per_frame_A = (H_A * W_A) // (tps**2)

                    # Calculate frame boundary positions for trajectory A
                    frame_boundary_positions_A = []
                    current_pos = vision_start_positions[0].item()

                    # Find where each frame ends in trajectory A
                    for _frame_idx in range(T_A):
                        # Each frame takes tokens_per_frame_A tokens
                        frame_end = current_pos + tokens_per_frame_A
                        frame_boundary_positions_A.append(frame_end)
                        current_pos = frame_end

                    # For trajectory A: extract hidden states at frame boundaries
                    trajectory_A_boundaries = torch.tensor(frame_boundary_positions_A)

                    # Apply progress head to hidden states at frame boundary positions for trajectory A
                    if trajectory_A_boundaries.numel() > 0:
                        boundary_hidden_states_A = hidden_state[i][
                            trajectory_A_boundaries
                        ]  # [num_frames_A, hidden_dim]

                        assert boundary_hidden_states_A.shape[0] == T_A, (
                            f"Expected {T_A} frames, got {boundary_hidden_states_A.shape[0]}"
                        )
                        progress_A = self.progress_head(boundary_hidden_states_A).squeeze(-1)  # [num_frames_A]
                        progress_logits_A.append(progress_A)
                    else:
                        progress_logits_A.append(torch.empty(0, device=hidden_state.device))

                    # For progress-only samples, we don't need trajectory B
                    if sample_type != "progress":
                        # For trajectory B: use video_grid_thw[i+1]
                        if (i * tps) + 1 >= len(video_grid_thw):
                            raise ValueError(f"video_grid_thw index {(i * tps) + 1} out of bounds for trajectory B")
                        current_video_grid_B = video_grid_thw[i * tps + 1]  # [T, H, W]
                        T_B, H_B, W_B = current_video_grid_B

                        # Calculate tokens per frame for trajectory B: (H_B * W_B) // merge_size^2
                        tokens_per_frame_B = (H_B * W_B) // (tps**2)

                        # Calculate frame boundary positions for trajectory B (after split_token)
                        frame_boundary_positions_B = []
                        current_pos = vision_start_positions[1].item()

                        # Find where each frame ends in trajectory B
                        for _frame_idx in range(T_B):
                            # Each frame takes tokens_per_frame_B tokens
                            frame_end = current_pos + tokens_per_frame_B
                            frame_boundary_positions_B.append(frame_end)
                            current_pos = frame_end

                        trajectory_B_boundaries = torch.tensor(frame_boundary_positions_B)

                        # Apply progress head to hidden states at frame boundary positions for trajectory B
                        if trajectory_B_boundaries.numel() > 0:
                            boundary_hidden_states_B = hidden_state[i][
                                trajectory_B_boundaries
                            ]  # [num_frames_B, hidden_dim]

                            assert boundary_hidden_states_B.shape[0] == T_B, (
                                f"Expected {T_B} frames, got {boundary_hidden_states_B.shape[0]}"
                            )
                            progress_B = self.progress_head(boundary_hidden_states_B).squeeze(-1)  # [num_frames_B]
                            progress_logits_B.append(progress_B)
                        else:
                            progress_logits_B.append(torch.empty(0, device=hidden_state.device))
                    else:
                        # For progress-only samples, trajectory B is None
                        progress_logits_B.append(None)

                progress_logits = {"A": progress_logits_A, "B": progress_logits_B}

        # Create ModelOutput
        output = ModelOutput()
        output.progress_logits = progress_logits

        # For preference and similarity, use specific tokens
        with _timer("time/logits", timing_raw=timing_raw):
            if sample_type in ["preference", "similarity"]:
                if sample_type == "preference":
                    token_id = self.processor.tokenizer.convert_tokens_to_ids("<|pref_token|>")
                else:  # similarity
                    token_id = self.processor.tokenizer.convert_tokens_to_ids("<|sim_token|>")

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
