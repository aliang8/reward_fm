#!/usr/bin/env python3
"""
ReWiND Transformer implementation.
Contains the ReWINDTransformer class with three prediction heads for different objectives.

Note: make sure that the forward pass uses all of the
heads or there will be some problems with FSDP sharding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import Optional, Dict, Any
from rfm.utils.logging import _timer
from transformers import AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


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
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=rewind_tfm_config.num_layers)

        # Class token
        self.preference_token = nn.Parameter(torch.randn(1, 1, rewind_tfm_config.hidden_dim))
        self.similarity_token = nn.Parameter(torch.randn(1, 1, rewind_tfm_config.hidden_dim))
        
        # Positional embeddings (for vision sequence length)
        self.pos_embed = nn.Parameter(torch.randn(1, rewind_tfm_config.max_len, rewind_tfm_config.hidden_dim))
        
        self.progress_head = nn.Linear(rewind_tfm_config.hidden_dim, 1, bias=False)  # Progress prediction (0-1)
        self.preference_head = nn.Linear(rewind_tfm_config.hidden_dim, 1, bias=False)  # Preference prediction (binary)
        self.similarity_head = nn.Linear(rewind_tfm_config.hidden_dim, 1, bias=False)  # Similarity scoring (reward)
        
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
        """Forward pass for ReWiND Transformer. See RFM model for more details"""
        if timing_raw is None:
            timing_raw = {}
        
        import ipdb; ipdb.set_trace()

        # processing text inputs
        with torch.no_grad():
            text_embeddings = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeddings = mean_pooling(text_embeddings, attention_mask)  # [batch_size, text_hidden_dim]
            text_embeddings = self.text_proj(text_embeddings)  # [batch_size, hidden_dim]

            # processing video inputs
            B = pixel_values_videos.shape[0]
            T = pixel_values_videos.shape[1]
            C = pixel_values_videos.shape[2]
            H = pixel_values_videos.shape[3]
            W = pixel_values_videos.shape[4]
            pixel_values_videos = pixel_values_videos.view(B * T, C, H, W)  # merge batch & time
            pixel_values_videos = pixel_values_videos.view(B * T, C, H, W).to(self.image_encoder.device)
            video_embeddings = self.image_encoder(pixel_values=pixel_values_videos).pooler_output  # [batch_size, vision_hidden_dim]
            video_embeddings = self.video_proj(video_embeddings)  # [batch_size * T, hidden_dim]
            video_embeddings = video_embeddings.view(B, T, -1)  # [batch_size, T, hidden_dim]
        
        # concatenate video and text embeddings
        embedding_A = video_embeddings[::2].clone()  # [batch_size/2, T, hidden_dim]
        embedding_B = video_embeddings[1::2].clone()  # [batch_size/2, T, hidden_dim]
        
        # Add the first embedding to the beginning of embedding A
        first_frame_emb_A = self.first_embedding_A.expand(embedding_A.size(0), -1, -1)  # [batch_size/2, 1, hidden_dim]
        first_frame_emb_B = self.first_embedding_B.expand(embedding_B.size(0), -1, -1)  # [batch_size/2, 1, hidden_dim]

        embedding_A[:,0:1] += first_frame_emb_A
        embedding_B[:,0:1] += first_frame_emb_B
        # cat text embeddings together
        length_A = embedding_A.size(1)
        length_B = embedding_B.size(1)
        total_embedding = torch.cat([
            text_embeddings.unsqueeze(1), 
            embedding_A, 
            embedding_B,
            self.preference_token.expand(embedding_A.size(0), -1, -1),
            self.similarity_token.expand(embedding_A.size(0), -1, -1)
            ], dim=1)  # [batch_size/2, 2*T + 1, hidden_dim]

        embedding_A = total_embedding[:,1:1+length_A,:] # avoid the text embedding
        embedding_B = total_embedding[:,1+length_A:-2,:] # avoid the text embedding

        batch_size = total_embedding.size(0)
        feature_dim = total_embedding.size(-1)
        progress_A_logits = self.progress_head(
            embedding_A.reshape(-1, feature_dim)
            ).squeeze(-1).view(batch_size, -1)  # [batch_size/2, T]
        

        if sample_type != "progress":
            progress_B_logits = self.progress_head(
                embedding_B.reshape(-1, feature_dim)
                ).squeeze(-1).view(batch_size, -1)  # [batch_size/2, T]
        else:
            # for progress only samples, we don't need trajectory B
            progress_B_logits = None
        progress_logits = {"A": progress_A_logits, "B": progress_B_logits}

        preference_class_token = total_embedding[:,-2,:]  # [batch_size/2, hidden_dim]
        similarity_class_token = total_embedding[:,-1,:]  # [batch_size/2, hidden_dim]
        if sample_type == "preference":
            logits = self.preference_head(preference_class_token)
        else:  # similarity
            logits = self.similarity_head(similarity_class_token)

        return SequenceClassifierOutputWithPast(logits=logits), progress_logits, timing_raw