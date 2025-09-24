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
            load_embeddings: Whether to load precomputed embeddings instead of processing frames
            processor: HuggingFace processor for text and vision processing
            max_length: Maximum sequence length for text
            resized_height: Height to resize images/videos to (default: 128)
            resized_width: Width to resize images/videos to (default: 128)
        """
        super().__init__(**kwargs)
        self.load_embeddings = load_embeddings
    
    def _load_precomputed_embeddings(self, trajectory) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load precomputed video and text embeddings from .pt file.
        
        Args:
            trajectory: Trajectory object containing embeddings_path
            
        Returns:
            Tuple of (video_embeddings, text_embedding)
        """
        if not hasattr(trajectory, 'embeddings_path') or not trajectory.embeddings_path:
            raise ValueError("Trajectory does not have embeddings_path. Make sure embeddings were precomputed.")
            
        if not os.path.exists(trajectory.embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found: {trajectory.embeddings_path}")
            
        embeddings_data = torch.load(trajectory.embeddings_path, map_location='cpu')
        return embeddings_data['video_embeddings'], embeddings_data['text_embedding']

    def _process_preference_batch(self, preference_samples: list[PreferenceSample]) -> dict[str, torch.Tensor]:
        """Process a batch of preference samples."""
        # Randomly decide whether chosen trajectory goes first or second
        preference_labels = np.random.randint(0, 2, len(preference_samples))

        if self.load_embeddings:
            # Load precomputed embeddings
            all_chosen_video_embeddings = []
            all_rejected_video_embeddings = []
            all_chosen_text_embeddings = []
            all_rejected_text_embeddings = []
            
            for sample in preference_samples:
                # Load embeddings for chosen trajectory
                chosen_video_emb, chosen_text_emb = self._load_precomputed_embeddings(sample.chosen_trajectory)
                all_chosen_video_embeddings.append(chosen_video_emb)
                all_chosen_text_embeddings.append(chosen_text_emb)
                
                # Load embeddings for rejected trajectory  
                rejected_video_emb, rejected_text_emb = self._load_precomputed_embeddings(sample.rejected_trajectory)
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
            # Load precomputed embeddings
            all_video_embeddings = []
            all_text_embeddings = []
            
            for sample in progress_samples:
                video_emb, text_emb = self._load_precomputed_embeddings(sample.trajectory)
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
