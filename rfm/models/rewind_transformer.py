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
from transformers import PreTrainedModel, AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers import PretrainedConfig


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class ReWINDTransformerConfig(PretrainedConfig):
    model_type = "rewind_transformer"

    def __init__(
        self,
        video_feature_dim: int = 768,
        text_feature_dim: int = 384,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.video_feature_dim = video_feature_dim
        self.text_feature_dim = text_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.max_len = max_len


class ReWiNDTransformer(PreTrainedModel):
    """ReWiND Transformer with three prediction heads for different objectives."""

    config_class = ReWINDTransformerConfig

    def __init__(self, config, image_encoder=None, text_encoder=None, tokenizer=None):
        super().__init__(config)

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        video_feature_dim = config.video_feature_dim
        text_feature_dim = config.text_feature_dim
        hidden_dim = config.hidden_dim

        if image_encoder is not None:
            video_feature_dim = image_encoder.config.hidden_size
        if text_encoder is not None:
            text_feature_dim = text_encoder.config.hidden_size

        self.video_proj = nn.Linear(video_feature_dim, hidden_dim)
        self.text_proj = nn.Linear(text_feature_dim, hidden_dim)

        self.first_embedding_A = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.first_embedding_B = nn.Parameter(torch.randn(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Prediction tokens
        self.preference_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        self.similarity_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))

        # self.progress_head = nn.Linear(config.hidden_dim, 1, bias=False)
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.preference_head = nn.Linear(hidden_dim, 1, bias=False)
        self.similarity_head = nn.Linear(hidden_dim, 1, bias=False)

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
        pixel_values_videos=None,
        video_embeddings=None,
        text_embeddings=None,
        sample_type=None,  # "preference", "similarity", "progress"
        timing_raw=None,
        **kwargs,
    ):
        """Forward pass for ReWiND Transformer."""
        if timing_raw is None:
            timing_raw = {}

        use_precomputed = video_embeddings is not None and text_embeddings is not None

        if use_precomputed:
            B, T, D_video = video_embeddings.shape
            D_text = text_embeddings.shape[1]

            # Project embeddings to hidden dimension
            video_embeddings = self.video_proj(video_embeddings.view(-1, D_video)).view(B, T, -1)  # [B, T, hidden_dim]
            text_embeddings = self.text_proj(text_embeddings)  # [B, hidden_dim]
        else:
            # Use raw inputs with encoders
            if pixel_values_videos is None:
                raise ValueError("pixel_values_videos is required when not using precomputed embeddings")

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

            if sample_type == "preference":
                pred_token = einops.repeat(self.preference_token, "1 1 d -> b 1 d", b=B)  # [B, 1, D]
            else:
                pred_token = einops.repeat(self.similarity_token, "1 1 d -> b 1 d", b=B)  # [B, 1, D]

            token_sequence = torch.cat(
                [text_embeddings.unsqueeze(1), video_embeddings_A, video_embeddings_B, pred_token], dim=1
            )  # shape: [B, 2*T + 1, D]

            token_embeddings = self.transformer(token_sequence)
            D = token_embeddings.shape[-1]

            final_embeddings_A = token_embeddings[:, 1 : 1 + T // 2, :]  # avoid the text embedding
            final_embeddings_B = token_embeddings[:, 1 + T // 2 : -1, :]  # avoid the text embedding

            progress_A_logits = self.progress_head(final_embeddings_A.reshape(-1, D))
            progress_A_logits = einops.rearrange(progress_A_logits, "(b t) 1 -> b t", b=B)

            progress_B_logits = self.progress_head(final_embeddings_B.reshape(-1, D))
            progress_B_logits = einops.rearrange(progress_B_logits, "(b t) 1 -> b t", b=B)

            progress_logits = {"A": progress_A_logits, "B": progress_B_logits}

            pred_class_token = token_embeddings[:, -1, :]  # [B, D]

            logits = None
            if sample_type == "preference":
                logits = self.preference_head(pred_class_token)
            else:  # similarity
                pass
        elif sample_type == "progress":
            first_frame_emb = einops.repeat(self.first_embedding_A, "1 1 d -> b 1 d", b=B)  # [B, 1, D]

            # [B, T, D]
            video_embeddings[:, 0:1] += first_frame_emb

            token_sequence = torch.cat([text_embeddings.unsqueeze(1), video_embeddings], dim=1)  # shape: [B, T, D]
            token_embeddings = self.transformer(token_sequence)
            D = token_embeddings.shape[-1]
            final_embeddings = token_embeddings[:, 1:, :]  # avoid the text embedding
            progress_logits = self.progress_head(final_embeddings)
            progress_logits = progress_logits.squeeze(-1)

            logits = None
            progress_logits = {"A": progress_logits, "B": None}

        return SequenceClassifierOutputWithPast(logits=logits), progress_logits, timing_raw


# Register the model and config with transformers
AutoConfig.register("rewind_transformer", ReWINDTransformerConfig)
AutoModel.register(ReWINDTransformerConfig, ReWiNDTransformer)
