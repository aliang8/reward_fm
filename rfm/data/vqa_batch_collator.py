#!/usr/bin/env python3
"""
VQA Batch collator for processing Sample objects through the processor and returning processed tensors.
This collator handles the conversion from PreferenceSample, SimilaritySample, and ProgressSample objects to processed tensors
for VQA-based reward modeling with different question types.
"""

import torch
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, field
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np
import random
from pydantic import BaseModel, field_serializer

from rfm.data.dataset_types import PreferenceSample, SimilaritySample, ProgressSample
from rfm.data.batch_collator import BatchCollator


class VQABatchCollator(BatchCollator):
    """Batch collator that processes Sample objects through the processor for VQA-based reward modeling."""

    def __init__(self, training: bool = True, inference: bool = False, **kwargs):
        """
        Initialize the VQA batch collator.

        Args:
            inference: Whether to return the labels for the batch
            processor: HuggingFace processor for text and vision processing
            max_length: Maximum sequence length for text
            resized_height: Height to resize images/videos to (default: 128)
            resized_width: Width to resize images/videos to (default: 128)
        """
        self.training = training
        self.inference = inference
        super().__init__(**kwargs)

    def _process_preference_batch(self, preference_samples: List[PreferenceSample]) -> Dict[str, torch.Tensor]:
        """Process a batch of preference samples with VQA-style question."""
        # Collect all messages for batch processing
        all_messages = []

        # Randomly decide whether chosen trajectory goes first or second
        preference_labels = np.random.randint(0, 2, len(preference_samples))

        for i, sample in enumerate(preference_samples):
            # Convert frames to appropriate format using stored shapes
            chosen_frames = self._convert_frames_to_pil_images(
                sample.chosen_trajectory.frames, sample.chosen_trajectory.frames_shape
            )
            rejected_frames = self._convert_frames_to_pil_images(
                sample.rejected_trajectory.frames, sample.rejected_trajectory.frames_shape
            )
            prompt = f"Given these two trajectories for the task '{sample.chosen_trajectory.task}', which one best corresponds to solving the task? Trajectory A or B? Format your answer enclosed by <ans> and </ans> tags. For example, if you prefer trajectory A, your answer should be <ans>A</ans>."

            if preference_labels[i] == 1.0:
                # Chosen trajectory first: Trajectory A (chosen) + Trajectory B (rejected)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "text", "text": "Trajectory A. "},
                            {
                                "type": "video",
                                "video": chosen_frames,
                                "resized_height": self.resized_height,
                                "resized_width": self.resized_width,
                            },
                            {"type": "text", "text": "Trajectory B. "},
                            {
                                "type": "video",
                                "video": rejected_frames,
                                "resized_height": self.resized_height,
                                "resized_width": self.resized_width,
                            },
                        ],
                    },
                ]
                if not self.inference:
                    conversation.append({"role": "assistant", "content": "<ans>A</ans>"})

            else:
                # Chosen trajectory second: Trajectory A (rejected) + Trajectory B (chosen)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "text", "text": "Trajectory A. "},
                            {
                                "type": "video",
                                "video": rejected_frames,
                                "resized_height": self.resized_height,
                                "resized_width": self.resized_width,
                            },
                            {"type": "text", "text": "Trajectory B. "},
                            {
                                "type": "video",
                                "video": chosen_frames,
                                "resized_height": self.resized_height,
                                "resized_width": self.resized_width,
                            },
                        ],
                    }
                ]
                if not self.inference:
                    conversation.append({"role": "assistant", "content": "<ans>B</ans>"})

            all_messages.append(conversation)

        # Create input with generation prompt and answer for proper label setting, if it is evaluation, we don't need to set the labels
        texts = [
            self.processor.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=self.inference,  # include assistant prefix tokens
                add_vision_id=True,
                fps=1,
            )
            for conv in all_messages
        ]

        image_inputs, video_inputs, video_kwargs = process_vision_info(all_messages, return_video_kwargs=True)

        batch_inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,  # pad so we can batch
            truncation=False,  # keep everything; truncate only at the "full" step if you must
            max_length=self.max_length,
            return_tensors="pt",
        )

        if not self.inference:
            labels = batch_inputs["input_ids"].clone()

            # mask out the prompt
            assistant_id = self.processor.tokenizer.encode("assistant", add_special_tokens=False)[0] 
            for i in range(len(labels)):
                token_after_assistant = (labels[i] == assistant_id).nonzero()[0][0] + 1
                labels[i][:token_after_assistant] = -100

            batch_inputs["labels"] = labels

        # Use the dynamically generated preference labels based on trajectory order
        batch_inputs["preference_labels"] = torch.tensor(preference_labels, dtype=torch.float32)

        batch_inputs = self._add_preference_meta(batch_inputs, preference_samples)

        return batch_inputs

    def _process_progress_batch(self, progress_samples: List[ProgressSample]) -> Dict[str, torch.Tensor]:
        """Process a batch of progress samples with VQA-style question."""
        # Collect all messages for batch processing
        all_messages = []

        for i, sample in enumerate(progress_samples):
            target_progress = sample.trajectory.target_progress
            
            # Let's round the target progress to 2 decimal places
            target_progress = np.round(target_progress, 2)

            # Convert frames to appropriate format using stored shapes
            frames = self._convert_frames_to_pil_images(sample.trajectory.frames, sample.trajectory.frames_shape)

            prompt = f"For the task '{sample.trajectory.task}', estimate the progress at each frame in the trajectory. Give a list of numbers between 0 and 1 where 0 means no progress and 1 means successful completion of the task. Format your answer enclosed by <ans> and </ans> tags. For example, if you think the progress at each frame is [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], your answer should be <ans>[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]</ans>."
            # Create conversation for progress evaluation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "video",
                            "video": frames,
                            "resized_height": self.resized_height,
                            "resized_width": self.resized_width,
                        },
                    ],
                }
            ]
            if not self.inference:
                conversation.append({"role": "assistant", "content": f"<ans>{target_progress}</ans>"})

            all_messages.append(conversation)

        # Create input with generation prompt and answer for proper label setting, if it is evaluation, we don't need to set the labels
        texts = [
            self.processor.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=self.inference,  # include assistant prefix tokens
                add_vision_id=True,
                fps=1,
            )
            for conv in all_messages
        ]

        image_inputs, video_inputs, video_kwargs = process_vision_info(all_messages, return_video_kwargs=True)

        batch_inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=False,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if not self.inference:
            labels = batch_inputs["input_ids"].clone()

            # mask out the prompt
            assistant_id = self.processor.tokenizer.encode("assistant", add_special_tokens=False)[0] 
            for i in range(len(labels)):
                token_after_assistant = (labels[i] == assistant_id).nonzero()[0][0] + 1
                labels[i][:token_after_assistant] = -100

            batch_inputs["labels"] = labels

        batch_inputs["data_gen_strategy"] = [sample.trajectory.data_gen_strategy for sample in progress_samples]

        batch_inputs = self._add_progress_meta(batch_inputs, progress_samples)

        return batch_inputs