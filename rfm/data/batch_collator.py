#!/usr/bin/env python3
"""
Batch collator for processing Sample objects through the processor and returning processed tensors.
This collator handles the conversion from PreferenceSample and SimilaritySample objects to processed tensors.
"""

import base64
import torch
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, field
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np
import random

from rfm.data.dataset_types import PreferenceSample, SimilaritySample, ProgressSample
from rfm.data.base_collator import BaseCollator


class BatchCollator(BaseCollator):
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

    def _add_preference_meta(self, batch_inputs: Dict[str, torch.Tensor], preference_samples: List[PreferenceSample]) -> Dict[str, torch.Tensor]:
        """Add metadata to the batch inputs."""
        batch_inputs["data_source"] = [sample.chosen_trajectory.data_source for sample in preference_samples]
        batch_inputs["sample_type"] = ["preference"] * len(preference_samples)
        batch_inputs["chosen_data_gen_strategy"] = [sample.chosen_trajectory.data_gen_strategy for sample in preference_samples]
        batch_inputs["rejected_data_gen_strategy"] = [sample.rejected_trajectory.data_gen_strategy for sample in preference_samples]
        
        # Add target progress for both trajectories based on conversation order
        target_progress_chosen = [sample.chosen_trajectory.target_progress for sample in preference_samples]
        target_progress_rejected = [sample.rejected_trajectory.target_progress for sample in preference_samples]
        target_progress_chosen_mask = [
            1.0
            if sample.chosen_trajectory.quality_label == "successful"
            or sample.chosen_trajectory.data_gen_strategy == "rewind_same_task"
            else 0.0
            for sample in preference_samples
        ]
        target_progress_rejected_mask = [
            1.0
            if sample.rejected_trajectory.quality_label == "successful"
            or sample.rejected_trajectory.data_gen_strategy == "rewind_same_task"
            else 0.0
            for sample in preference_samples
        ]

        # Pad target progress tensors to max length in last dimension
        batch_inputs["target_progress_chosen"] = self._pad_target_progress(target_progress_chosen)
        batch_inputs["target_progress_rejected"] = self._pad_target_progress(target_progress_rejected)
        batch_inputs["target_progress_chosen_mask"] = torch.tensor(target_progress_chosen_mask, dtype=torch.float32)
        batch_inputs["target_progress_rejected_mask"] = torch.tensor(target_progress_rejected_mask, dtype=torch.float32)

        # Also add the frame_shapes
        batch_inputs["chosen_frames_shape"] = torch.tensor(
            [sample.chosen_trajectory.frames_shape for sample in preference_samples], dtype=torch.int32
        )
        batch_inputs["rejected_frames_shape"] = torch.tensor(
            [sample.rejected_trajectory.frames_shape for sample in preference_samples], dtype=torch.int32
        )
        return batch_inputs

    def _add_progress_meta(self, batch_inputs: Dict[str, torch.Tensor], progress_samples: List[ProgressSample]) -> Dict[str, torch.Tensor]:
        """Add metadata to the batch inputs."""
        # Add target progress and quality labels
        target_progress_list = []
        quality_labels = []

        for sample in progress_samples:
            if sample.trajectory.target_progress is not None:
                target_progress_list.append(sample.trajectory.target_progress)
            quality_labels.append(1.0 if sample.trajectory.quality_label == "successful" else 0.0)

        # Add metadata
        batch_inputs["sample_type"] = ["progress"] * len(progress_samples)

        # Pad target progress tensors to max length in last dimension
        batch_inputs["target_progress"] = self._pad_target_progress(target_progress_list)
        batch_inputs["quality_labels"] = torch.tensor(quality_labels, dtype=torch.float32)

        return batch_inputs

    def _process_preference_batch(self, preference_samples: List[PreferenceSample]) -> Dict[str, torch.Tensor]:
        """Process a batch of preference samples."""
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

            if preference_labels[i] == 1.0:
                # Chosen trajectory first: task + video A (chosen) + <|split_token|> + video B (rejected) + <|pref_token|>
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Task: {sample.chosen_trajectory.task}"},
                            {
                                "type": "video",
                                "video": chosen_frames,
                                "resized_height": self.resized_height,
                                "resized_width": self.resized_width,
                            },
                            {"type": "text", "text": "<|split_token|>"},
                            {
                                "type": "video",
                                "video": rejected_frames,
                                "resized_height": self.resized_height,
                                "resized_width": self.resized_width,
                            },
                            {"type": "text", "text": "<|pref_token|>"},
                        ],
                    }
                ]
            else:
                # Chosen trajectory second: task + video A (rejected) + <|split_token|> + video B (chosen) + <|pref_token|>
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Task: {sample.chosen_trajectory.task}"},
                            {
                                "type": "video",
                                "video": rejected_frames,
                                "resized_height": self.resized_height,
                                "resized_width": self.resized_width,
                            },
                            {"type": "text", "text": "<|split_token|>"},
                            {
                                "type": "video",
                                "video": chosen_frames,
                                "resized_height": self.resized_height,
                                "resized_width": self.resized_width,
                            },
                            {"type": "text", "text": "<|pref_token|>"},
                        ],
                    }
                ]

            all_messages.append(conversation)

        # Process all messages in one batch
        texts = [
            self.processor.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=False,
                add_vision_id=True,
                fps=1,
            )
            for msg in all_messages
        ]

        image_inputs, video_inputs, video_kwargs = process_vision_info(all_messages, return_video_kwargs=True)

        # Process through the processor in one batch
        batch_inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=False,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Use the dynamically generated preference labels based on trajectory order
        batch_inputs["preference_labels"] = torch.tensor(preference_labels, dtype=torch.float32)
        batch_inputs = self._add_preference_meta(batch_inputs, preference_samples)
        return batch_inputs

    def _process_similarity_batch(self, similarity_samples: List[SimilaritySample]) -> Dict[str, torch.Tensor]:
        """Process a batch of similarity samples."""
        # Collect all messages for batch processing (ref and traj_sim for each sample)
        all_messages = []

        for sample in similarity_samples:
            # Convert frames to appropriate format using stored shapes
            reference_frames = self._convert_frames_to_pil_images(
                sample.reference_trajectory.frames, sample.reference_trajectory.frames_shape
            )
            traj_sim_frames = self._convert_frames_to_pil_images(
                sample.traj_sim_trajectory.frames, sample.traj_sim_trajectory.frames_shape
            )
            traj_diff_frames = self._convert_frames_to_pil_images(
                sample.traj_diff_trajectory.frames, sample.traj_diff_trajectory.frames_shape
            )

            # Process reference vs trajectory sim
            conversation_ref_sim = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Reference task: {sample.reference_trajectory.task}"},
                        {
                            "type": "video",
                            "video": reference_frames,
                            "resized_height": self.resized_height,
                            "resized_width": self.resized_width,
                        },
                        {"type": "text", "text": "<|split_token|>"},
                        {
                            "type": "video",
                            "video": traj_sim_frames,
                            "resized_height": self.resized_height,
                            "resized_width": self.resized_width,
                        },
                        {"type": "text", "text": "<|reward_token|>"},
                    ],
                }
            ]

            # Process reference vs trajectory diff
            conversation_ref_diff = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Reference task: {sample.reference_trajectory.task}"},
                        {
                            "type": "video",
                            "video": reference_frames,
                            "resized_height": self.resized_height,
                            "resized_width": self.resized_width,
                        },
                        {"type": "text", "text": "<|split_token|>"},
                        {
                            "type": "video",
                            "video": traj_diff_frames,
                            "resized_height": self.resized_height,
                            "resized_width": self.resized_width,
                        },
                        {"type": "text", "text": "<|reward_token|>"},
                    ],
                }
            ]

            all_messages.extend([conversation_ref_sim, conversation_ref_diff])

        # Process all messages in one batch
        texts = [
            self.processor.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=False,
                add_vision_id=True,
                fps=1,
            )
            for msg in all_messages
        ]

        image_inputs, video_inputs, video_kwargs = process_vision_info(all_messages, return_video_kwargs=True)

        # Process through the processor in one batch
        batch_inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=False,
            max_length=self.max_length,
            return_tensors="pt",
            # **video_kwargs,
        )

        # Split the batch inputs back into ref_A and ref_B
        num_samples = len(similarity_samples)
        ref_sim_inputs = {}
        ref_diff_inputs = {}

        for key, value in batch_inputs.items():
            if isinstance(value, torch.Tensor):
                # Split into ref_A and ref_B (alternating)
                ref_sim_inputs[key] = value[::2]  # Even indices (0, 2, 4, ...)
                ref_diff_inputs[key] = value[1::2]  # Odd indices (1, 3, 5, ...)
            else:
                ref_sim_inputs[key] = value
                ref_diff_inputs[key] = value

        # Combine into single batch with ref_A and ref_B suffixes
        combined_inputs = {"sample_type": ["similarity"] * num_samples}

        # Add ref_sim inputs
        for key, value in ref_sim_inputs.items():
            combined_inputs[f"{key}_ref_sim"] = value

        # Add ref_diff inputs
        for key, value in ref_diff_inputs.items():
            combined_inputs[f"{key}_ref_diff"] = value

        # Add target progress for both trajectories
        target_progress_sim_list = []
        target_progress_diff_list = []
        target_progress_ref_list = []

        for sample in similarity_samples:
            if sample.traj_sim_trajectory.target_progress is not None:
                target_progress_sim_list.append(sample.traj_sim_trajectory.target_progress)

            if sample.traj_diff_trajectory.target_progress is not None:
                target_progress_diff_list.append(sample.traj_diff_trajectory.target_progress)

            if sample.reference_trajectory.target_progress is not None:
                target_progress_ref_list.append(sample.reference_trajectory.target_progress)

        # Pad target progress tensors to max length in last dimension
        combined_inputs["target_progress_sim"] = self._pad_target_progress(target_progress_sim_list)
        combined_inputs["target_progress_diff"] = self._pad_target_progress(target_progress_diff_list)
        combined_inputs["target_progress_ref"] = self._pad_target_progress(target_progress_ref_list)

        # Also add the frame_shapes
        combined_inputs["ref_frames_shape"] = torch.tensor(
            [sample.reference_trajectory.frames_shape for sample in similarity_samples], dtype=torch.int32
        )
        combined_inputs["traj_sim_frames_shape"] = torch.tensor(
            [sample.traj_sim_trajectory.frames_shape for sample in similarity_samples], dtype=torch.int32
        )
        combined_inputs["traj_diff_frames_shape"] = torch.tensor(
            [sample.traj_diff_trajectory.frames_shape for sample in similarity_samples], dtype=torch.int32
        )
        return combined_inputs

    def _process_progress_batch(self, progress_samples: List[ProgressSample]) -> Dict[str, torch.Tensor]:
        """Process a batch of progress samples with VQA-style question."""
        # Collect all messages for batch processing
        all_messages = []

        for sample in progress_samples:
            # Convert frames to appropriate format using stored shapes
            frames = self._convert_frames_to_pil_images(sample.trajectory.frames, sample.trajectory.frames_shape)

            # Create conversation for progress evaluation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Task: {sample.trajectory.task}",
                        },
                        {
                            "type": "video",
                            "video": frames,
                            "resized_height": self.resized_height,
                            "resized_width": self.resized_width,
                        },
                    ],
                }
            ]

            all_messages.append(conversation)

        texts = [
            self.processor.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=False,
                add_vision_id=True,
                fps=1,
            )
            for msg in all_messages
        ]

        image_inputs, video_inputs, video_kwargs = process_vision_info(all_messages, return_video_kwargs=True)

        # Process through the processor in one batch
        batch_inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=False,
            max_length=self.max_length,
            return_tensors="pt",
            **video_kwargs,
        )
        batch_inputs = self._add_progress_meta(batch_inputs, progress_samples)
        return batch_inputs
        
