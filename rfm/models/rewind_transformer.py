#!/usr/bin/env python3
"""
ReWiND Transformer implementation.
Contains the ReWINDTransformer class with three prediction heads for different objectives.

Note: make sure that the forward pass uses all of the
heads or there will be some problems with FSDP sharding.
"""

import einops
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class ReWiNDTransformer(PreTrainedModel):
    """ReWiND Transformer with three prediction heads for different objectives."""

    def __init__(self, config, image_encoder=None, text_encoder=None):
        super().__init__(config)
        rewind_tfm_config = config.rewind

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.video_proj = nn.Linear(image_encoder.config.hidden_size, rewind_tfm_config.hidden_dim)
        self.text_proj = nn.Linear(text_encoder.config.hidden_size, rewind_tfm_config.hidden_dim)

        self.first_embedding_A = nn.Parameter(torch.randn(1, 1, rewind_tfm_config.hidden_dim))
        self.first_embedding_B = nn.Parameter(torch.randn(1, 1, rewind_tfm_config.hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=rewind_tfm_config.hidden_dim,
            nhead=rewind_tfm_config.num_attention_heads,
            dim_feedforward=rewind_tfm_config.hidden_dim * 4,
            dropout=rewind_tfm_config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=rewind_tfm_config.num_layers)

        # Class token
        self.preference_token = nn.Parameter(torch.randn(1, 1, rewind_tfm_config.hidden_dim))
        self.similarity_token = nn.Parameter(torch.randn(1, 1, rewind_tfm_config.hidden_dim))
        self.progress_token = nn.Parameter(torch.randn(1, 1, rewind_tfm_config.hidden_dim))

        # Positional embeddings (for vision sequence length)
        self.pos_embed = nn.Parameter(torch.randn(1, rewind_tfm_config.max_len, rewind_tfm_config.hidden_dim))

        self.progress_head = nn.Linear(rewind_tfm_config.hidden_dim, 1, bias=False)
        self.preference_head = nn.Linear(rewind_tfm_config.hidden_dim, 1, bias=False)
        self.similarity_head = nn.Linear(rewind_tfm_config.hidden_dim, 1, bias=False)

        # Ensure all heads have the same dtype as the base model
        self.model_dtype = next(self.video_proj.parameters()).dtype
        self.transformer = self.transformer.to(dtype=self.model_dtype)
        self.text_proj = self.text_proj.to(dtype=self.model_dtype)
        self.progress_head = self.progress_head.to(dtype=self.model_dtype)
        self.preference_head = self.preference_head.to(dtype=self.model_dtype)
        self.similarity_head = self.similarity_head.to(dtype=self.model_dtype)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        sample_type=None,  # "preference", "similarity", "progress"
        second_per_grid_ts=None,
        timing_raw=None,
        **kwargs,
    ):
        """Forward pass for ReWiND Transformer."""
        if timing_raw is None:
            timing_raw = {}

        if pixel_values_videos is None:
            raise ValueError("pixel_values_videos is required")

        B, T, C, H, W = pixel_values_videos.shape

        # processing text inputs
        with torch.no_grad():
            text_embeddings = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeddings = mean_pooling(text_embeddings, attention_mask)  # [B, text_hidden_dim]
            text_embeddings = self.text_proj(text_embeddings)  # [B, D]

        # processing video inputs
        # T should contain both chosen and rejected trajectories concatenated together
        pixel_values_videos = pixel_values_videos.view(B * T, C, H, W)
        video_embeddings = self.image_encoder(
            pixel_values=pixel_values_videos
        ).pooler_output  # [B, vision_hidden_dim]
        video_embeddings = self.video_proj(video_embeddings)  # [B * T, D]
        video_embeddings = video_embeddings.view(B, T, -1)  # [B, T, D]

        if sample_type == "preference" or sample_type == "similarity":
            video_embeddings_A = video_embeddings[:, : T // 2]
            video_embeddings_B = video_embeddings[:, T // 2 :]

            # Add the first embedding to the beginning of embedding A
            first_frame_emb_A = einops.repeat(self.first_embedding_A, "1 1 d -> b 1 d", b=B)  # [B, 1, D]
            first_frame_emb_B = einops.repeat(self.first_embedding_B, "1 1 d -> b 1 d", b=B)  # [B, 1, D]

            video_embeddings_A[:, 0:1] += first_frame_emb_A
            video_embeddings_B[:, 0:1] += first_frame_emb_B
            
            pref_token = einops.repeat(self.preference_token, "1 1 d -> b 1 d", b=B)  # [B, 1, D]
            sim_token = einops.repeat(self.similarity_token, "1 1 d -> b 1 d", b=B)  # [B, 1, D]

            token_sequence = torch.cat(
                [text_embeddings.unsqueeze(1), video_embeddings_A, video_embeddings_B, pref_token, sim_token], dim=1
            )  # shape: [B, 2*T + 2, D]

            token_embeddings = self.transformer(token_sequence)
            D = token_embeddings.shape[-1]

            final_embeddings_A = token_embeddings[:, 1 : 1 + T // 2, :]  # avoid the text embedding
            final_embeddings_B = token_embeddings[:, 1 + T // 2 : -2, :]  # avoid the text embedding

            progress_A_logits = self.progress_head(final_embeddings_A.reshape(-1, D))
            progress_A_logits = einops.rearrange(progress_A_logits, "(b t) 1 -> b t", b=B)

            progress_B_logits = self.progress_head(final_embeddings_B.reshape(-1, D))
            progress_B_logits = einops.rearrange(progress_B_logits, "(b t) 1 -> b t", b=B)

            progress_logits = {"A": progress_A_logits, "B": progress_B_logits}

            preference_class_token = token_embeddings[:, -2, :]  # [B, D]

            logits = None
            if sample_type == "preference":
                logits = self.preference_head(preference_class_token)
            else:  # similarity
                pass
        elif sample_type == "progress":
            first_frame_emb = einops.repeat(self.first_embedding_A, "1 1 d -> b 1 d", b=B)  # [B, 1, D]
            
            # [B, T, D]
            video_embeddings[:, 0:1] += first_frame_emb
            
            token_sequence = torch.cat(
                [text_embeddings.unsqueeze(1), video_embeddings], dim=1
            )  # shape: [B, T, D]
            token_embeddings = self.transformer(token_sequence)
            D = token_embeddings.shape[-1]
            final_embeddings = token_embeddings[:, 1 :, :]  # avoid the text embedding
            progress_logits = self.progress_head(final_embeddings)
            progress_logits = progress_logits.squeeze(-1)

            logits = None
            progress_logits = {"A": progress_logits, "B": None}

        return SequenceClassifierOutputWithPast(logits=logits), progress_logits, timing_raw
