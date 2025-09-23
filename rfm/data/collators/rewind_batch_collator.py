#!/usr/bin/env python3
"""
Batch collator for processing Sample objects through the processor and returning processed tensors.
This collator handles the conversion from PreferenceSample and SimilaritySample objects to processed tensors.
"""

import numpy as np
import torch

from .rfm_batch_collator import RFMBatchCollator
from .utils import convert_frames_to_pil_images
from rfm.data.dataset_types import PreferenceSample, ProgressSample


class ReWiNDBatchCollator(RFMBatchCollator):
    """Batch collator that processes Sample objects through the processor."""

    def __init__(self, **kwargs):
        """
        Initialize the batch collator.

        Args:
            processor: HuggingFace processor for text and vision processing
            max_length: Maximum sequence length for text
            resized_height: Height to resize images/videos to (default: 128)
            resized_width: Width to resize images/videos to (default: 128)
        """
        super().__init__(**kwargs)

    def _process_preference_batch(self, preference_samples: list[PreferenceSample]) -> dict[str, torch.Tensor]:
        """Process a batch of preference samples."""
        # Randomly decide whether chosen trajectory goes first or second
        preference_labels = np.random.randint(0, 2, len(preference_samples))

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
        """Process a batch of progress samples with VQA-style question."""
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
