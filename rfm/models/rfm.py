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

try:
    from transformers import Qwen3VLModel
except ImportError:
    Qwen3VLModel = None

# from transformers import AutoModelForImageTextToText as Molmo2VLModel  # Molmo2 uses AutoModelForImageTextToText
from transformers import SmolVLMModel
import torch.distributed as dist

from rfm.models.utils import ModelOutput
from rfm.models.heads import PredictionHeadsMixin
from rfm.utils.timer import _timer
from rfm.utils.logger import get_logger

logger = get_logger()


class RFM(PredictionHeadsMixin, PreTrainedModel):
    """Reward Foundation Model with three prediction heads for different objectives.

    Supports multiple base model architectures:
    - Qwen2.5-VL (Qwen2_5_VLModel)
    - SmolVLM (AutoModelForImageTextToText)
    """

    config_class = Qwen2_5_VLModel.config_class

    # Declare support for SDPA and Flash Attention (will delegate to underlying model), needed for Qwen3
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def __init__(self, config, processor, tokenizer, base_model=None, base_model_id=None, model_config=None):
        if "SmolVLM" in base_model_id:
            hidden_size = config.text_config.hidden_size
            self.model_cls = SmolVLMModel
        elif "Qwen2.5" in base_model_id:
            hidden_size = config.hidden_size
            self.model_cls = Qwen2_5_VLModel
        elif "Qwen3" in base_model_id:
            hidden_size = config.text_config.hidden_size
            self.model_cls = Qwen3VLModel
        elif "Molmo" in base_model_id:
            # Molmo2 is based on Qwen3 architecture
            hidden_size = config.text_config.hidden_size
            self.model_cls = Qwen3VLModel
            # self.model_cls = Molmo2VLModel
        else:
            raise ValueError(f"Unsupported base model: {base_model_id}")

        super().__init__(
            config,
            hidden_dim=hidden_size,
            model_config=model_config,
            dropout=0.1,
        )

        if base_model is not None:
            self.model = base_model
        else:
            self.model = self.model_cls(config)

        self.config_class = self.model_cls.config_class
        self.base_model_id = base_model_id

        self.model_dtype = self.model.dtype
        self.progress_head = self.progress_head.to(dtype=self.model_dtype)
        self.preference_head = self.preference_head.to(dtype=self.model_dtype)
        self.similarity_head = self.similarity_head.to(dtype=self.model_dtype)
        self.success_head = self.success_head.to(dtype=self.model_dtype)

        self.processor = processor
        self.tokenizer = tokenizer
        self.model_config = model_config

        self.average_temporal_patches = self.model_config.average_temporal_patches
        self.use_progress_token = self.model_config.use_progress_token
        self.use_multi_image = self.model_config.use_multi_image

        # Molmo2 only supports multi-image mode, not video
        if "Molmo" in self.base_model_id and not self.use_multi_image:
            raise ValueError(
                "Molmo2 does not support video mode (use_multi_image=False). "
                "Please set data.use_multi_image=True to use Molmo2 with multi-image input."
            )

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegates gradient checkpointing enabling to the base model."""
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        """Delegates gradient checkpointing disabling to the base model."""
        self.model.gradient_checkpointing_disable(**kwargs)

    def _extract_hidden_states_from_token_pairs(
        self,
        hidden_state: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract image/video frame embeddings from hidden states by finding token pairs and mean pooling.

        This is a general function that works for both SmolVLM and Qwen multi-image mode.
        It automatically detects which model is being used based on the base_model_id:
        - SmolVLM: Uses the same token for start and end: <fake_token_around_image>
        - Qwen: Uses different tokens: <|vision_start|> and <|vision_end|>

        Args:
            hidden_state: Hidden states tensor [seq_len, hidden_dim]
            input_ids: Input token IDs [seq_len]

        Returns:
            frame_embeddings: Tensor [num_frames, hidden_dim] containing mean-pooled
                            embeddings for each frame/image between token pairs
        """
        # Detect model type and get appropriate tokenizer and tokens
        is_molmo = "Molmo" in self.base_model_id
        if "SmolVLM" in self.base_model_id:
            # SmolVLM mode: same token appears in pairs
            tokenizer = self.tokenizer
            start_token = "<fake_token_around_image>"
            end_token = None  # Same token for both start and end
            use_same_token = True
            use_molmo_mode = False
        elif is_molmo:
            # Molmo2 mode: <low_res_im_start> followed by <im_patch> tokens
            tokenizer = self.processor.tokenizer
            start_token = "<low_res_im_start>"
            end_token = None  # No explicit end token
            patch_token = "<im_patch>"
            use_same_token = False
            use_molmo_mode = True
        else:
            # Qwen mode: different start and end tokens
            tokenizer = self.processor.tokenizer
            start_token = "<|vision_start|>"
            end_token = "<|vision_end|>"
            use_same_token = False
            use_molmo_mode = False

        # Get token IDs
        start_token_id = tokenizer.convert_tokens_to_ids(start_token)

        # Find all positions where start tokens appear
        start_positions = (input_ids == start_token_id).nonzero(as_tuple=True)[0]

        if len(start_positions) == 0:
            raise ValueError(
                f"No {start_token} tokens found in input_ids. Token ID {start_token_id} not found in sequence."
            )

        # Handle different pairing modes
        if use_same_token:
            # SmolVLM mode: same token appears in pairs
            if len(start_positions) % 2 != 0:
                raise ValueError(
                    f"Expected even number of {start_token} tokens (pairs), but found {len(start_positions)} tokens."
                )

            # Group tokens into pairs (every two consecutive tokens form a pair)
            token_pairs = []
            for i in range(0, len(start_positions), 2):
                token_pairs.append((start_positions[i].item(), start_positions[i + 1].item()))
        elif use_molmo_mode:
            # Molmo2 mode: <low_res_im_start> followed by <im_patch> tokens
            patch_token_id = tokenizer.convert_tokens_to_ids(patch_token)
            im_patch_positions = (input_ids == patch_token_id).nonzero(as_tuple=True)[0]

            token_pairs = []
            for start_idx, start_pos in enumerate(start_positions):
                start_pos_val = start_pos.item()
                # Find the last consecutive im_patch token after this start
                patches_after_start = im_patch_positions[im_patch_positions > start_pos]
                if len(patches_after_start) > 0:
                    # Find where patches stop (at next image start or end of sequence)
                    if start_idx + 1 < len(start_positions):
                        next_start = start_positions[start_idx + 1].item()
                        patches_for_this_image = patches_after_start[patches_after_start < next_start]
                    else:
                        patches_for_this_image = patches_after_start
                    if len(patches_for_this_image) > 0:
                        end_pos = patches_for_this_image[-1].item()
                        token_pairs.append((start_pos_val, end_pos))
        else:
            # Qwen mode: different start and end tokens
            end_token_id = tokenizer.convert_tokens_to_ids(end_token)

            # Find all positions where end tokens appear
            end_positions = (input_ids == end_token_id).nonzero(as_tuple=True)[0]

            if len(end_positions) == 0:
                raise ValueError(
                    f"No {end_token} tokens found in input_ids. Token ID {end_token_id} not found in sequence."
                )

            if len(start_positions) != len(end_positions):
                raise ValueError(
                    f"Mismatched number of tokens: "
                    f"found {len(start_positions)} {start_token} tokens "
                    f"and {len(end_positions)} {end_token} tokens."
                )

            # Pair up start and end tokens (they should appear in order: start, end, start, end, ...)
            token_pairs = []
            for i in range(len(start_positions)):
                start_pos = start_positions[i].item()
                end_pos = end_positions[i].item()

                if start_pos >= end_pos:
                    raise ValueError(
                        f"Invalid token pair at index {i}: "
                        f"{start_token} at {start_pos}, {end_token} at {end_pos}. "
                        f"Start must come before end."
                    )
                token_pairs.append((start_pos, end_pos))

        # Extract hidden states between token pairs
        frame_embeddings = []
        for start_pos, end_pos in token_pairs:
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
            return torch.empty(0, hidden_state.shape[-1], device=hidden_state.device, dtype=hidden_state.dtype)

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

        assert boundary_hidden_states.shape[0] == T, f"Expected {T} frames, got {boundary_hidden_states.shape[0]}"
        progress_output = self.progress_head(boundary_hidden_states)  # [T, 1] or [T, num_bins] for discrete
        if self.use_discrete_progress:
            progress = progress_output  # [T, num_bins] - keep logits
        else:
            progress = progress_output.squeeze(-1)  # [T]
        success = self.success_head(boundary_hidden_states).squeeze(-1)  # [T]

        return progress, success

    def _extract_hidden_state_from_token(
        self,
        hidden_state: torch.Tensor,
        input_ids: torch.Tensor,
        token_name: str,
    ) -> torch.Tensor:
        """
        Extract hidden states at specific token positions.

        Args:
            hidden_state: Hidden states tensor [B, seq_len, hidden_dim] or [seq_len, hidden_dim]
            input_ids: Input token IDs [B, seq_len] or [seq_len]
            token_name: Name of the token to find (e.g., "<|prog_token|>", "<|pref_token|>")

        Returns:
            token_hidden_states: Hidden states at token positions [B, hidden_dim]
        """
        # Handle both batched and unbatched inputs
        is_batched = hidden_state.dim() == 3
        if not is_batched:
            hidden_state = hidden_state.unsqueeze(0)  # [1, seq_len, hidden_dim]
            input_ids = input_ids.unsqueeze(0)  # [1, seq_len]

        B = input_ids.shape[0]

        # Get tokenizer (works for both SmolVLM and Qwen)
        if "SmolVLM" in self.base_model_id:
            tokenizer = self.tokenizer
        else:
            tokenizer = self.processor.tokenizer

        # Get token ID
        token_id = tokenizer.convert_tokens_to_ids(token_name)

        # Find all positions where the token appears
        token_positions = []
        for i, seq_ids in enumerate(input_ids):
            positions = (seq_ids == token_id).nonzero(as_tuple=True)[0]
            if len(positions) == 0:
                raise ValueError(f"{token_name} not found in sequence {i}")
            elif len(positions) > 1:
                raise ValueError(f"{token_name} appears {len(positions)} times in sequence {i}, expected exactly 1")
            else:
                token_positions.append(positions[0].item())

        token_positions = torch.tensor(token_positions, device=input_ids.device, dtype=torch.long)

        # Extract hidden states at the token positions
        token_hidden_states = torch.gather(
            hidden_state,
            1,
            token_positions.view(-1, 1, 1).expand(-1, -1, hidden_state.size(-1)),
        ).squeeze(1)  # [B, hidden_dim]

        return token_hidden_states

    def _forward_smolvlm(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        sample_type,
        timing_raw,
        **kwargs,
    ):
        """Forward pass for SmolVLM model."""
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            **kwargs,
        }
        with _timer("time/rfm_forward", timing_raw=timing_raw):
            outputs = self.model(**model_kwargs, output_hidden_states=True, return_dict=True)

        B = input_ids.shape[0]
        hidden_state = outputs.hidden_states[-1]  # [B, seq_len, hidden_dim]

        progress_logits_A = []
        progress_logits_B = []
        success_logits_A = []
        success_logits_B = []

        # Skip all frame extraction when using progress token
        skip_frame_extraction = self.use_progress_token

        with _timer("time/progress_logits", timing_raw=timing_raw):
            if not skip_frame_extraction:
                # Compute per-frame embeddings and predictions
                for i in range(B):
                    # Extract frame embeddings for this sample
                    frame_embeddings = self._extract_hidden_states_from_token_pairs(
                        hidden_state[i],  # [seq_len, hidden_dim]
                        input_ids[i],  # [seq_len]
                    )  # [num_frames, hidden_dim]

                    if frame_embeddings.shape[0] == 0:
                        raise ValueError(f"No frame embeddings extracted for sample {i}")

                    # For progress samples, there's only one video/trajectory (V=1)
                    # For preference/similarity samples, there are two videos/trajectories (V=2)
                    if sample_type == "progress":
                        trajectory_A_frames = frame_embeddings
                        trajectory_B_frames = None
                    else:
                        mid_point = frame_embeddings.shape[0] // 2
                        trajectory_A_frames = frame_embeddings[:mid_point]
                        trajectory_B_frames = frame_embeddings[mid_point:]

                    # Apply heads to trajectory A frames
                    progress_A_output = self.progress_head(
                        trajectory_A_frames
                    )  # [T_A, 1] or [T_A, num_bins] for discrete
                    if self.use_discrete_progress:
                        progress_A = progress_A_output  # [T_A, num_bins] - keep logits
                    else:
                        progress_A = progress_A_output.squeeze(-1)  # [T_A]
                    success_A = self.success_head(trajectory_A_frames).squeeze(-1)  # [T_A]
                    progress_logits_A.append(progress_A)
                    success_logits_A.append(success_A)

                    # Apply heads to trajectory B frames (if available)
                    if trajectory_B_frames is not None:
                        progress_B_output = self.progress_head(
                            trajectory_B_frames
                        )  # [T_B, 1] or [T_B, num_bins] for discrete
                        if self.use_discrete_progress:
                            progress_B = progress_B_output  # [T_B, num_bins] - keep logits
                        else:
                            progress_B = progress_B_output.squeeze(-1)  # [T_B]
                        success_B = self.success_head(trajectory_B_frames).squeeze(-1)  # [T_B]
                        progress_logits_B.append(progress_B)
                        success_logits_B.append(success_B)
                    else:
                        progress_logits_B.append(None)
                        success_logits_B.append(None)

        progress_logits = {
            "A": torch.stack(progress_logits_A) if progress_logits_A else None,
            "B": torch.stack(progress_logits_B) if progress_logits_B else None,
        }
        success_logits = {
            "A": torch.stack(success_logits_A) if success_logits_A else None,
            "B": torch.stack(success_logits_B) if success_logits_B else None,
        }

        return outputs, progress_logits, success_logits

    def _forward_qwen(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_values_videos,
        image_grid_thw,
        video_grid_thw,
        sample_type,
        timing_raw,
        **kwargs,
    ):
        """Forward pass for Qwen model."""
        batch_size = input_ids.shape[0] if input_ids is not None else 0
        logger.trace(f"RFM._forward_qwen: Starting, sample_type={sample_type}, batch_size={batch_size}")

        # Extract second_per_grid_ts from kwargs if present
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        # Log all input shapes to check for inconsistencies across ranks
        logger.trace(
            f"RFM._forward_qwen: Input shapes - input_ids: {input_ids.shape if input_ids is not None else 'None'}, "
            f"attention_mask: {attention_mask.shape if attention_mask is not None else 'None'}, "
            f"pixel_values: {pixel_values.shape if pixel_values is not None else 'None'}, "
            f"pixel_values_videos: {pixel_values_videos.shape if pixel_values_videos is not None else 'None'}"
        )
        if image_grid_thw is not None:
            logger.trace(f"RFM._forward_qwen: image_grid_thw shape: {image_grid_thw.shape}, len: {len(image_grid_thw)}")
        if video_grid_thw is not None:
            logger.trace(f"RFM._forward_qwen: video_grid_thw shape: {video_grid_thw.shape}, len: {len(video_grid_thw)}")

        torch.cuda.synchronize()
        logger.trace(f"attention mask sum: {attention_mask.sum()}")
        torch.cuda.synchronize()

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
        logger.trace("RFM._forward_qwen: About to call base model forward")
        with _timer("time/rfm_forward", timing_raw=timing_raw):
            outputs = self.model(**model_kwargs)
        logger.trace("RFM._forward_qwen: Base model forward completed")

        hidden_state = outputs.last_hidden_state  # [B, seq_len, hidden_dim]
        logger.trace(f"RFM._forward_qwen: hidden_state shape: {hidden_state.shape}")

        # Get token IDs for vision tokens
        # Qwen uses <|vision_start|> and <|vision_end|>
        # Molmo2 uses <low_res_im_start> and <im_patch> tokens instead
        is_molmo = "Molmo" in self.base_model_id

        if is_molmo:
            # Molmo2 uses different tokens for images
            vision_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<low_res_im_start>")
            vision_end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<low_res_im_end>")
            im_patch_token_id = self.processor.tokenizer.convert_tokens_to_ids("<im_patch>")
        else:
            vision_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            vision_end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            im_patch_token_id = None
        split_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|split_token|>")

        progress_logits_A = []
        progress_logits_B = []
        success_logits_A = []
        success_logits_B = []

        # temporal patch size (only needed for video mode)
        # Check both that video_processor exists AND has the required attributes (Molmo2 doesn't have these)
        has_tps = hasattr(self.processor, "video_processor") and hasattr(
            self.processor.video_processor, "temporal_patch_size"
        )
        has_merge = hasattr(self.processor, "video_processor") and hasattr(self.processor.video_processor, "merge_size")
        tps = self.processor.video_processor.temporal_patch_size if has_tps else 2
        merge_size = self.processor.video_processor.merge_size if has_merge else 14

        # Skip all frame extraction when using progress token
        skip_frame_extraction = self.use_progress_token

        with _timer("time/progress_logits", timing_raw=timing_raw):
            if not skip_frame_extraction:
                is_multi_image = self.use_multi_image
                
                if is_multi_image:
                    # Multi-image mode: collect all frames first, then batch process
                    all_trajectory_A_frames = []
                    all_trajectory_B_frames = []
                    trajectory_A_lengths = []
                    trajectory_B_lengths = []
                    has_trajectory_B = sample_type != "progress"
                    
                    # First pass: extract all frame embeddings
                    for i, seq_ids in enumerate(input_ids):
                        # Find all vision token positions
                        vision_start_positions = (seq_ids == vision_start_token_id).nonzero(as_tuple=True)[0]

                        # For Molmo2, vision_end_token_id is None, so we need to find image regions differently
                        if is_molmo and im_patch_token_id is not None:
                            # For Molmo2: find where <im_patch> tokens are
                            im_patch_positions = (seq_ids == im_patch_token_id).nonzero(as_tuple=True)[0]
                            # Find boundaries: where patches end for each image (where non-patch token appears)
                            # Each <low_res_im_start> marks a new image
                            vision_end_positions = []
                            for start_idx, start_pos in enumerate(vision_start_positions):
                                start_pos_val = start_pos.item()
                                # Find the last consecutive im_patch token after this start
                                patches_after_start = im_patch_positions[im_patch_positions > start_pos]
                                if len(patches_after_start) > 0:
                                    # Find where patches stop being consecutive or hit next image start
                                    if start_idx + 1 < len(vision_start_positions):
                                        next_start = vision_start_positions[start_idx + 1].item()
                                        patches_for_this_image = patches_after_start[patches_after_start < next_start]
                                    else:
                                        patches_for_this_image = patches_after_start
                                    if len(patches_for_this_image) > 0:
                                        vision_end_positions.append(patches_for_this_image[-1])
                            vision_end_positions = torch.tensor(vision_end_positions, device=seq_ids.device)
                        elif vision_end_token_id is not None:
                            vision_end_positions = (seq_ids == vision_end_token_id).nonzero(as_tuple=True)[0]
                        else:
                            vision_end_positions = torch.tensor([], device=seq_ids.device)

                        if len(vision_start_positions) == 0:
                            raise ValueError(f"vision_start_token (id={vision_start_token_id}) not found in sequence {i}")

                        # Extract embeddings from each vision_start/end pair
                        frame_embeddings = self._extract_hidden_states_from_token_pairs(
                            hidden_state[i],  # [seq_len, hidden_dim]
                            seq_ids,  # [seq_len]
                        )  # [num_frames, hidden_dim]

                        if frame_embeddings.shape[0] == 0:
                            raise ValueError(f"No frame embeddings extracted for sample {i} in multi-image mode")

                        # For progress samples, all frames belong to trajectory A
                        if sample_type == "progress":
                            trajectory_A_frames = frame_embeddings
                            trajectory_B_frames = None
                        else:
                            # For preference/similarity, find the split token to separate trajectories
                            split_positions = (seq_ids == split_token_id).nonzero(as_tuple=True)[0]
                            if len(split_positions) == 0:
                                raise ValueError(
                                    f"split_token not found in sequence {i} for preference/similarity sample"
                                )

                            split_pos = split_positions[0].item()
                            traj_a_pairs = sum(1 for pos in vision_start_positions if pos.item() < split_pos)
                            trajectory_A_frames = frame_embeddings[:traj_a_pairs]
                            trajectory_B_frames = frame_embeddings[traj_a_pairs:]

                        # Collect frames for batch processing
                        all_trajectory_A_frames.append(trajectory_A_frames)
                        trajectory_A_lengths.append(trajectory_A_frames.shape[0])
                        
                        if trajectory_B_frames is not None:
                            all_trajectory_B_frames.append(trajectory_B_frames)
                            trajectory_B_lengths.append(trajectory_B_frames.shape[0])
                        else:
                            all_trajectory_B_frames.append(None)
                            trajectory_B_lengths.append(0)

                    # Batch process trajectory A frames
                    if len(all_trajectory_A_frames) > 0:
                        # Concatenate all trajectory A frames
                        batched_trajectory_A = torch.cat(all_trajectory_A_frames, dim=0)  # [sum(T_A), hidden_dim]
                        
                        # Apply heads in batch
                        progress_A_output_batched = self.progress_head(batched_trajectory_A)
                        success_A_output_batched = self.success_head(batched_trajectory_A)
                        
                        # Split results back to individual samples
                        if self.use_discrete_progress:
                            # progress_A_output_batched: [sum(T_A), num_bins]
                            progress_A_split = torch.split(progress_A_output_batched, trajectory_A_lengths, dim=0)
                        else:
                            # progress_A_output_batched: [sum(T_A), 1]
                            progress_A_output_batched = progress_A_output_batched.squeeze(-1)  # [sum(T_A)]
                            progress_A_split = torch.split(progress_A_output_batched, trajectory_A_lengths, dim=0)
                        
                        success_A_output_batched = success_A_output_batched.squeeze(-1)  # [sum(T_A)]
                        success_A_split = torch.split(success_A_output_batched, trajectory_A_lengths, dim=0)
                        
                        # Append to output lists
                        for progress_A, success_A in zip(progress_A_split, success_A_split):
                            progress_logits_A.append(progress_A)
                            success_logits_A.append(success_A)

                    # Batch process trajectory B frames (where available)
                    if has_trajectory_B and any(f is not None for f in all_trajectory_B_frames):
                        # Filter out None entries and track which samples have trajectory B
                        valid_B_frames = [f for f in all_trajectory_B_frames if f is not None]
                        valid_B_lengths = [traj_len for traj_len, frame in zip(trajectory_B_lengths, all_trajectory_B_frames) if frame is not None]
                        
                        if len(valid_B_frames) > 0:
                            # Concatenate all trajectory B frames
                            batched_trajectory_B = torch.cat(valid_B_frames, dim=0)  # [sum(T_B), hidden_dim]
                            
                            # Apply heads in batch
                            progress_B_output_batched = self.progress_head(batched_trajectory_B)
                            success_B_output_batched = self.success_head(batched_trajectory_B)
                            
                            # Split results back to individual samples
                            if self.use_discrete_progress:
                                # progress_B_output_batched: [sum(T_B), num_bins]
                                progress_B_split = torch.split(progress_B_output_batched, valid_B_lengths, dim=0)
                            else:
                                # progress_B_output_batched: [sum(T_B), 1]
                                progress_B_output_batched = progress_B_output_batched.squeeze(-1)  # [sum(T_B)]
                                progress_B_split = torch.split(progress_B_output_batched, valid_B_lengths, dim=0)
                            
                            success_B_output_batched = success_B_output_batched.squeeze(-1)  # [sum(T_B)]
                            success_B_split = torch.split(success_B_output_batched, valid_B_lengths, dim=0)
                            
                            # Map back to original sample order (some may be None)
                            valid_idx = 0
                            for frame in all_trajectory_B_frames:
                                if frame is not None:
                                    progress_logits_B.append(progress_B_split[valid_idx])
                                    success_logits_B.append(success_B_split[valid_idx])
                                    valid_idx += 1
                                else:
                                    progress_logits_B.append(None)
                                    success_logits_B.append(None)
                        else:
                            # No valid trajectory B frames
                            for _ in all_trajectory_B_frames:
                                progress_logits_B.append(None)
                                success_logits_B.append(None)
                    else:
                        # No trajectory B for progress samples
                        for _ in all_trajectory_A_frames:
                            progress_logits_B.append(None)
                            success_logits_B.append(None)
                else:
                    # Video mode: use existing per-sample processing (not batched in this iteration)
                    for i, seq_ids in enumerate(input_ids):
                        # Find all vision token positions
                        vision_start_positions = (seq_ids == vision_start_token_id).nonzero(as_tuple=True)[0]

                        if len(vision_start_positions) == 0:
                            raise ValueError(f"vision_start_token (id={vision_start_token_id}) not found in sequence {i}")

                        # Video mode: use existing temporal patch logic
                        if video_grid_thw is None or i >= len(video_grid_thw):
                            raise ValueError(
                                f"video_grid_thw is required for progress prediction in video mode. Got: {video_grid_thw}"
                            )

                        # For trajectory A
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
                            # For trajectory B
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
                            progress_logits_B.append(None)
                            success_logits_B.append(None)

        # logger.trace(
        #     f"RFM._forward_qwen: Stacking progress/success logits, len_A={len(progress_logits_A)}, len_B={len(progress_logits_B)}"
        # )
        progress_logits = {
            "A": torch.stack(progress_logits_A) if progress_logits_A else None,
            "B": torch.stack(progress_logits_B) if progress_logits_B[0] is not None else None,
        }
        success_logits = {
            "A": torch.stack(success_logits_A) if success_logits_A else None,
            "B": torch.stack(success_logits_B) if success_logits_B[0] is not None else None,
        }
        # logger.trace("RFM._forward_qwen: Completed, returning outputs")

        return outputs, progress_logits, success_logits

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
        # Molmo2-specific parameters
        image_grids=None,
        image_token_pooling=None,
        image_num_crops=None,
        video_grids=None,
        video_token_pooling=None,
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
                        - 'A': Tensor for trajectory A [batch_size, max_seq_len_A] (padded to max length)
                        - 'B': Tensor for trajectory B [batch_size, max_seq_len_B] or None (padded to max length)
                    Values should be in range [0, 1] representing task completion percentage at each timestep.

                - timing_raw (Dict[str, float]):
                    Timing information for the forward pass.
        """
        logger.trace(
            f"RFM.forward: Starting, sample_type={sample_type}, batch_size={input_ids.shape[0] if input_ids is not None else 'N/A'}"
        )

        if timing_raw is None:
            timing_raw = {}

        # Call appropriate forward method based on model type
        if "SmolVLM" in self.base_model_id:
            logger.trace("RFM.forward: Calling _forward_smolvlm")
            outputs, progress_logits, success_logits = self._forward_smolvlm(
                input_ids, attention_mask, pixel_values, sample_type, timing_raw, **kwargs
            )
            logger.trace("RFM.forward: _forward_smolvlm completed")
        else:
            logger.trace("RFM.forward: Calling _forward_qwen")
            outputs, progress_logits, success_logits = self._forward_qwen(
                input_ids,
                attention_mask,
                pixel_values,
                pixel_values_videos,
                image_grid_thw,
                video_grid_thw,
                sample_type,
                timing_raw,
                second_per_grid_ts=second_per_grid_ts,
                # Molmo2-specific parameters
                image_grids=image_grids,
                image_token_pooling=image_token_pooling,
                image_num_crops=image_num_crops,
                video_grids=video_grids,
                video_token_pooling=video_token_pooling,
                **kwargs,
            )
            logger.trace("RFM.forward: _forward_qwen completed")

        # Create ModelOutput
        logger.trace("RFM.forward: Creating ModelOutput")
        output = ModelOutput()
        output.progress_logits = progress_logits
        output.success_logits = success_logits

        # For token-based predictions (progress with use_progress_token, preference, similarity)
        # Get hidden states once if needed
        need_token_extraction = (
            (sample_type == "progress" and self.use_progress_token and outputs is not None)
            or (sample_type in ["preference", "similarity"] and self.use_progress_token and outputs is not None)
            or sample_type in ["preference", "similarity"]
        )
        logger.trace(
            f"RFM.forward: need_token_extraction={need_token_extraction}, sample_type={sample_type}, use_progress_token={self.use_progress_token}"
        )

        if need_token_extraction:
            # Get hidden states (works for both SmolVLM and Qwen)
            logger.trace("RFM.forward: Extracting hidden states for token-based predictions")
            if "SmolVLM" in self.base_model_id:
                hidden_state_for_token = outputs.hidden_states[-1]  # [B, seq_len, hidden_dim]
            else:
                hidden_state_for_token = outputs.last_hidden_state  # [B, seq_len, hidden_dim]
            logger.trace(f"RFM.forward: hidden_state_for_token shape: {hidden_state_for_token.shape}")

            # For progress samples with use_progress_token, extract hidden state from <|prog_token_A|> and <|succ_token_A|>
            # This overrides the per-frame prediction above
            if sample_type == "progress" and self.use_progress_token:
                logger.trace("RFM.forward: Processing progress with use_progress_token")
                # Extract hidden states at <|prog_token_A|> positions
                logger.trace("RFM.forward: Extracting <|prog_token_A|> hidden states")
                prog_token_A_hidden_states = self._extract_hidden_state_from_token(
                    hidden_state_for_token,
                    input_ids,
                    "<|prog_token_A|>",
                )  # [B, hidden_dim]
                logger.trace(f"RFM.forward: prog_token_A_hidden_states shape: {prog_token_A_hidden_states.shape}")

                # Extract hidden states at <|succ_token_A|> positions
                logger.trace("RFM.forward: Extracting <|succ_token_A|> hidden states")
                succ_token_A_hidden_states = self._extract_hidden_state_from_token(
                    hidden_state_for_token,
                    input_ids,
                    "<|succ_token_A|>",
                )  # [B, hidden_dim]
                logger.trace(f"RFM.forward: succ_token_A_hidden_states shape: {succ_token_A_hidden_states.shape}")

                # Apply heads to get predictions
                logger.trace("RFM.forward: Applying progress and success heads")
                progress_pred_output = self.progress_head(
                    prog_token_A_hidden_states
                )  # [B, 1] or [B, num_bins] for discrete
                if self.use_discrete_progress:
                    progress_pred = progress_pred_output  # [B, num_bins] - keep logits
                else:
                    progress_pred = progress_pred_output.squeeze(-1)  # [B]
                success_pred = self.success_head(succ_token_A_hidden_states).squeeze(-1)  # [B]
                logger.trace(
                    f"RFM.forward: progress_pred shape: {progress_pred.shape}, success_pred shape: {success_pred.shape}"
                )

                progress_logits["A"] = progress_pred.unsqueeze(-1)
                success_logits["A"] = success_pred.unsqueeze(-1)
            # For preference and similarity with use_progress_token, extract from both prog_token_A/B and succ_token_A/B
            elif sample_type in ["preference", "similarity"] and self.use_progress_token:
                logger.trace(f"RFM.forward: Processing {sample_type} with use_progress_token")
                # Extract hidden states at <|prog_token_A|> and <|prog_token_B|> positions
                logger.trace("RFM.forward: Extracting <|prog_token_A|> hidden states")
                prog_token_A_hidden_states = self._extract_hidden_state_from_token(
                    hidden_state_for_token,
                    input_ids,
                    "<|prog_token_A|>",
                )  # [B, hidden_dim]
                logger.trace(f"RFM.forward: prog_token_A_hidden_states shape: {prog_token_A_hidden_states.shape}")

                logger.trace("RFM.forward: Extracting <|prog_token_B|> hidden states")
                prog_token_B_hidden_states = self._extract_hidden_state_from_token(
                    hidden_state_for_token,
                    input_ids,
                    "<|prog_token_B|>",
                )  # [B, hidden_dim]
                logger.trace(f"RFM.forward: prog_token_B_hidden_states shape: {prog_token_B_hidden_states.shape}")

                # Extract hidden states at <|succ_token_A|> and <|succ_token_B|> positions
                logger.trace("RFM.forward: Extracting <|succ_token_A|> hidden states")
                succ_token_A_hidden_states = self._extract_hidden_state_from_token(
                    hidden_state_for_token,
                    input_ids,
                    "<|succ_token_A|>",
                )  # [B, hidden_dim]
                logger.trace(f"RFM.forward: succ_token_A_hidden_states shape: {succ_token_A_hidden_states.shape}")

                logger.trace("RFM.forward: Extracting <|succ_token_B|> hidden states")
                succ_token_B_hidden_states = self._extract_hidden_state_from_token(
                    hidden_state_for_token,
                    input_ids,
                    "<|succ_token_B|>",
                )  # [B, hidden_dim]
                logger.trace(f"RFM.forward: succ_token_B_hidden_states shape: {succ_token_B_hidden_states.shape}")

                # Apply heads to get progress and success values for both trajectories
                logger.trace("RFM.forward: Applying progress and success heads for A and B")
                progress_pred_A_output = self.progress_head(
                    prog_token_A_hidden_states
                )  # [B, 1] or [B, num_bins] for discrete
                progress_pred_B_output = self.progress_head(
                    prog_token_B_hidden_states
                )  # [B, 1] or [B, num_bins] for discrete
                if self.use_discrete_progress:
                    progress_pred_A = progress_pred_A_output  # [B, num_bins] - keep logits
                    progress_pred_B = progress_pred_B_output  # [B, num_bins] - keep logits
                else:
                    progress_pred_A = progress_pred_A_output.squeeze(-1)  # [B]
                    progress_pred_B = progress_pred_B_output.squeeze(-1)  # [B]
                success_pred_A = self.success_head(succ_token_A_hidden_states).squeeze(-1)  # [B]
                success_pred_B = self.success_head(succ_token_B_hidden_states).squeeze(-1)  # [B]
                logger.trace(f"RFM.forward: Progress/success predictions completed")

                progress_logits["A"] = progress_pred_A.unsqueeze(-1)  # [B, 1]
                progress_logits["B"] = progress_pred_B.unsqueeze(-1)  # [B, 1]
                success_logits["A"] = success_pred_A.unsqueeze(-1)  # [B, 1]
                success_logits["B"] = success_pred_B.unsqueeze(-1)  # [B, 1]

                # Also extract preference/similarity token for the main prediction
                if sample_type == "preference":
                    token_name = "<|pref_token|>"
                elif sample_type == "similarity":
                    token_name = "<|sim_token|>"
                else:
                    raise ValueError(f"Invalid sample type: {sample_type}")

                # Extract hidden states at the target token positions
                logger.trace(f"RFM.forward: Extracting {token_name} hidden states")
                token_hidden_states = self._extract_hidden_state_from_token(
                    hidden_state_for_token,
                    input_ids,
                    token_name,
                )  # [B, hidden_dim]
                logger.trace(f"RFM.forward: token_hidden_states shape: {token_hidden_states.shape}")

                # Apply the appropriate head
                logger.trace(f"RFM.forward: Applying {sample_type} head")
                if sample_type == "preference":
                    output.pref_logits = self.preference_head(token_hidden_states)
                else:  # similarity
                    output.sim_logits = self.similarity_head(token_hidden_states)
                logger.trace(f"RFM.forward: {sample_type} head completed")

            # For preference and similarity without use_progress_token, use specific tokens for main prediction
            elif sample_type in ["preference", "similarity"]:
                logger.trace(f"RFM.forward: Processing {sample_type} without use_progress_token")
                # Determine which token to use
                if sample_type == "preference":
                    token_name = "<|pref_token|>"
                elif sample_type == "similarity":
                    token_name = "<|sim_token|>"
                else:
                    raise ValueError(f"Invalid sample type: {sample_type}")

                # Extract hidden states at the target token positions
                logger.trace(f"RFM.forward: Extracting {token_name} hidden states")
                token_hidden_states = self._extract_hidden_state_from_token(
                    hidden_state_for_token,
                    input_ids,
                    token_name,
                )  # [B, hidden_dim]
                logger.trace(f"RFM.forward: token_hidden_states shape: {token_hidden_states.shape}")

                # Apply the appropriate head
                logger.trace(f"RFM.forward: Applying {sample_type} head")
                if sample_type == "preference":
                    output.pref_logits = self.preference_head(token_hidden_states)
                else:  # similarity
                    output.sim_logits = self.similarity_head(token_hidden_states)
                logger.trace(f"RFM.forward: {sample_type} head completed")

        return output, timing_raw
