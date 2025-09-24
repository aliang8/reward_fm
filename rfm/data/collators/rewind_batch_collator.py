#!/usr/bin/env python3
"""
Batch collator for processing Sample objects through the processor and returning processed tensors.
This collator handles the conversion from PreferenceSample and SimilaritySample objects to processed tensors.
"""

import numpy as np
import torch
import os

from .rfm_batch_collator import RFMBatchCollator
from .utils import convert_frames_to_pil_images
from rfm.data.dataset_types import PreferenceSample, ProgressSample


class ReWiNDBatchCollator(RFMBatchCollator):
    """Batch collator that processes Sample objects through the processor."""

    def __init__(self, load_embeddings: bool = False, **kwargs):
        """
        Initialize the batch collator.

        Args:
            load_embeddings: Whether to use precomputed embeddings from trajectories instead of processing frames
            processor: HuggingFace processor for text and vision processing
            max_length: Maximum sequence length for text
            resized_height: Height to resize images/videos to (default: 128)
            resized_width: Width to resize images/videos to (default: 128)
        """
        super().__init__(**kwargs)
        self.load_embeddings = load_embeddings
    

    def _process_preference_batch(self, preference_samples: list[PreferenceSample]) -> dict[str, torch.Tensor]:
        """Process a batch of preference samples."""
        # Randomly decide whether chosen trajectory goes first or second
        preference_labels = np.random.randint(0, 2, len(preference_samples))

        if self.load_embeddings:
            # Use embeddings directly from trajectories (already loaded by dataset)
            all_chosen_video_embeddings = []
            all_rejected_video_embeddings = []
            all_chosen_text_embeddings = []
            all_rejected_text_embeddings = []
            
            for sample in preference_samples:
                # Get embeddings directly from trajectories
                chosen_video_emb = sample.chosen_trajectory.video_embeddings
                chosen_text_emb = sample.chosen_trajectory.text_embeddings
                rejected_video_emb = sample.rejected_trajectory.video_embeddings
                rejected_text_emb = sample.rejected_trajectory.text_embeddings
                
                if any(emb is None for emb in [chosen_video_emb, chosen_text_emb, rejected_video_emb, rejected_text_emb]):
                    raise ValueError("Sample trajectories are missing embeddings")
                
                all_chosen_video_embeddings.append(chosen_video_emb)
                all_chosen_text_embeddings.append(chosen_text_emb)
                all_rejected_video_embeddings.append(rejected_video_emb)
                all_rejected_text_embeddings.append(rejected_text_emb)
            
            # Stack embeddings into batches
            chosen_video_embeddings = torch.stack(all_chosen_video_embeddings)  # [B, T, D]
            rejected_video_embeddings = torch.stack(all_rejected_video_embeddings)  # [B, T, D]
            chosen_text_embeddings = torch.stack(all_chosen_text_embeddings)  # [B, D]
            rejected_text_embeddings = torch.stack(all_rejected_text_embeddings)  # [B, D]
            
            # Interleave embeddings based on preference_labels
            frame_len = chosen_video_embeddings.shape[1]
            video_embeddings = torch.empty(len(preference_samples), frame_len * 2, chosen_video_embeddings.shape[2])
            text_embeddings = torch.empty(len(preference_samples), chosen_text_embeddings.shape[1])
            
            for i in range(len(preference_samples)):
                if preference_labels[i] == 1:  # chosen first
                    video_embeddings[i] = torch.cat([chosen_video_embeddings[i], rejected_video_embeddings[i]], dim=0)
                    text_embeddings[i] = chosen_text_embeddings[i]  # Use chosen text embedding
                else:
                    video_embeddings[i] = torch.cat([rejected_video_embeddings[i], chosen_video_embeddings[i]], dim=0)
                    text_embeddings[i] = chosen_text_embeddings[i]  # Use chosen text embedding
            
            batch_inputs = {
                "video_embeddings": video_embeddings,  # [B, T*2, D]
                "text_embeddings": text_embeddings,    # [B, D]
            }
        else:
            all_chosen_frames = []
            all_rejected_frames = []
            all_tasks = []

            for i, sample in enumerate(preference_samples):
                # Convert frames to appropriate format using stored shapes
                # NOTE: these should already be padded to max_frames by the data generator
                chosen_frames = convert_frames_to_pil_images(
                    sample.chosen_trajectory.frames, sample.chosen_trajectory.frames_shape
                )
                rejected_frames = convert_frames_to_pil_images(
                    sample.rejected_trajectory.frames, sample.rejected_trajectory.frames_shape
                )
                all_chosen_frames.append(chosen_frames)
                all_rejected_frames.append(rejected_frames)
                all_tasks.append(sample.chosen_trajectory.task)

            frame_len = len(all_chosen_frames[0])
            # [(B*T), C, H, W]
            chosen_video_inputs = self.processor(images=all_chosen_frames, return_tensors="pt")["pixel_values"]
            _, C, H, W = chosen_video_inputs.shape
            chosen_video_inputs = chosen_video_inputs.view(len(preference_samples), frame_len, C, H, W)
            rejected_video_inputs = self.processor(images=all_rejected_frames, return_tensors="pt")["pixel_values"]
            rejected_video_inputs = rejected_video_inputs.view(len(preference_samples), frame_len, C, H, W)

            # interleave them based on preference_labels
            video_inputs = torch.empty(len(preference_samples), frame_len * 2, C, H, W)
            for i in range(len(preference_samples)):
                if preference_labels[i] == 1:  # means chosen first
                    video_inputs[i] = torch.cat([chosen_video_inputs[i], rejected_video_inputs[i]], dim=0)
                else:
                    video_inputs[i] = torch.cat([rejected_video_inputs[i], chosen_video_inputs[i]], dim=0)

            encodings = self.tokenizer(
                all_tasks,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch_inputs = {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "pixel_values_videos": video_inputs,
            }

        batch_inputs["preference_labels"] = torch.tensor(preference_labels, dtype=torch.float32)
        batch_inputs = self._add_preference_meta(batch_inputs, preference_samples)
        return batch_inputs

    def _process_progress_batch(self, progress_samples: list[ProgressSample]) -> dict[str, torch.Tensor]:
        """Process a batch of progress samples."""
        if self.load_embeddings:
            # Use embeddings directly from trajectory (already loaded by dataset)
            all_video_embeddings = []
            all_text_embeddings = []
            
            for sample in progress_samples:
                # Get embeddings directly from the trajectory
                video_emb = sample.trajectory.video_embeddings
                text_emb = sample.trajectory.text_embedding
                
                if video_emb is None or text_emb is None:
                    raise ValueError(f"Sample trajectory is missing embeddings. video_embeddings: {video_emb is not None}, text_embeddings: {text_emb is not None}")
                
                all_video_embeddings.append(video_emb)
                all_text_embeddings.append(text_emb)
            
            # Stack embeddings into batches
            video_embeddings = torch.stack(all_video_embeddings)  # [B, T, D]
            text_embeddings = torch.stack(all_text_embeddings)    # [B, D]
            
            batch_inputs = {
                "video_embeddings": video_embeddings,  # [B, T, D]
                "text_embeddings": text_embeddings,    # [B, D]
            }
        else:
            all_frames = []
            for sample in progress_samples:
                frames = convert_frames_to_pil_images(sample.trajectory.frames, sample.trajectory.frames_shape)
                all_frames.append(frames)

            # here we directly use dino processor process the images and videos to tensors
            video_inputs = self.processor(images=all_frames, return_tensors="pt")["pixel_values"]
            frame_len = len(all_frames[0])
            _, C, H, W = video_inputs.shape
            video_inputs = video_inputs.view(len(progress_samples), frame_len, C, H, W)

            # here we directly use the tokenizer to process the texts to input_ids and attention_mask
            texts = [sample.trajectory.task for sample in progress_samples]
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            batch_inputs = {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "pixel_values_videos": video_inputs,
            }

        batch_inputs = self._add_progress_meta(batch_inputs, progress_samples)
        return batch_inputs
