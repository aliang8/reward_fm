#!/usr/bin/env python3
"""
VQA Batch collator for processing Sample objects through the processor and returning processed tensors.
This collator handles the conversion from PreferenceSample, SimilaritySample, and ProgressSample objects to processed tensors
for VQA-based reward modeling with different question types.
"""

import numpy as np
import torch
import tempfile
from pathlib import Path

from .rfm_heads import RFMBatchCollator
from .utils import convert_frames_to_pil_images, write_mp4
from rfm.data.dataset_types import PreferenceSample, ProgressSample


class VQABatchCollator(RFMBatchCollator):
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

    def _process_preference_batch(self, preference_samples: list[PreferenceSample]) -> dict[str, torch.Tensor]:
        """Process a batch of preference samples with VQA-style question."""
        # Collect all messages for batch processing
        all_messages = []

        # Randomly decide whether chosen trajectory goes first or second
        preference_labels = np.random.randint(0, 2, len(preference_samples))

        for i, sample in enumerate(preference_samples):
            # Convert frames to appropriate format using stored shapes
            chosen_frames = convert_frames_to_pil_images(
                sample.chosen_trajectory.frames, sample.chosen_trajectory.frames_shape
            )
            rejected_frames = convert_frames_to_pil_images(
                sample.rejected_trajectory.frames, sample.rejected_trajectory.frames_shape
            )
            prompt = f"Given these two trajectories for the task '{sample.chosen_trajectory.task}', which one best corresponds to solving the task? Trajectory A or B? Format your answer enclosed by <ans> and </ans> tags. For example, if you prefer trajectory A, your answer should be <ans>A</ans>."

            if "Qwen" in self.base_model_id:
                content_extras = {
                    "resized_height": self.resized_height,
                    "resized_width": self.resized_width,
                }
            elif "SmolVLM" in self.base_model_id:
                # we need to write the frames to a temporary file
                tmp = Path(tempfile.gettempdir()) / f"tmp_chosen.mp4"
                write_mp4(chosen_frames, tmp)
                chosen_frames = str(tmp)
                tmp = Path(tempfile.gettempdir()) / f"tmp_rejected.mp4"
                write_mp4(rejected_frames, tmp)
                rejected_frames = str(tmp)
                content_extras = {}
            else:
                content_extras = {}

            if preference_labels[i] == 1.0:
                # Chosen trajectory first: Trajectory A (chosen) + Trajectory B (rejected)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "text", "text": "This is Trajectory A. "},
                            {
                                "type": "video",
                                "video": chosen_frames,
                                **content_extras,
                            },
                            {"type": "text", "text": "This is Trajectory B. "},
                            {
                                "type": "video",
                                "video": rejected_frames,
                                **content_extras,
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
                            {"type": "text", "text": "This is Trajectory A. "},
                            {
                                "type": "video",
                                "video": rejected_frames,
                                **content_extras,
                            },
                            {"type": "text", "text": "This is Trajectory B. "},
                            {
                                "type": "video",
                                "video": chosen_frames,
                                **content_extras,
                            },
                        ],
                    }
                ]
                if not self.inference:
                    conversation.append({"role": "assistant", "content": "<ans>B</ans>"})

            all_messages.append(conversation)

        batch_inputs = self._process_conversation(all_messages)
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

        # Add preference metadata (includes all the misc fields like target_progress, masks, frames_shape, etc.)
        batch_inputs = self._add_preference_meta(batch_inputs, preference_samples)

        return batch_inputs

    def _process_progress_batch(self, progress_samples: list[ProgressSample]) -> dict[str, torch.Tensor]:
        """Process a batch of progress samples with VQA-style question."""
        # Collect all messages for batch processing
        all_messages = []

        for i, sample in enumerate(progress_samples):
            target_progress = sample.trajectory.target_progress

            # Convert frames to appropriate format using stored shapes
            frames = convert_frames_to_pil_images(sample.trajectory.frames, sample.trajectory.frames_shape)

            prompt = f"For the task '{sample.trajectory.task}', estimate the progress at each frame in the trajectory. Give a list of numbers between 0 and 1 where 0 means no progress and 1 means successful completion of the task. Format your answer enclosed by <ans> and </ans> tags. For example, if you think the progress at each frame is [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], your answer should be <ans>[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]</ans>."

            if "Qwen" in self.base_model_id:
                content_extras = {
                    "resized_height": self.resized_height,
                    "resized_width": self.resized_width,
                }
            elif "SmolVLM" in self.base_model_id:
                # we need to write the frames to a temporary file
                tmp = Path(tempfile.gettempdir()) / f"tmp_progress.mp4"
                write_mp4(frames, tmp)
                frames = str(tmp)
                content_extras = {}
            else:
                content_extras = {}

            # Create conversation for progress evaluation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "video",
                            "video": frames,
                            **content_extras,
                        },
                    ],
                }
            ]
            # Add assistant response only if not in inference mode and target_progress exists
            if not self.inference and target_progress is not None:
                # Round target progress to 2 decimal places for the response
                target_progress_rounded = np.round(target_progress, 2)
                conversation.append({"role": "assistant", "content": f"<ans>{target_progress_rounded}</ans>"})

            all_messages.append(conversation)

        batch_inputs = self._process_conversation(all_messages)

        if not self.inference:
            labels = batch_inputs["input_ids"].clone()

            # mask out the prompt
            assistant_id = self.processor.tokenizer.encode("assistant", add_special_tokens=False)[0]
            for i in range(len(labels)):
                token_after_assistant = (labels[i] == assistant_id).nonzero()[0][0] + 1
                labels[i][:token_after_assistant] = -100

            batch_inputs["labels"] = labels

        # Add progress metadata (includes all the misc fields like target_progress, masks, frames_shape, etc.)
        # Only call if target_progress exists (matches RFM batch collator behavior)
        if progress_samples[0].trajectory.target_progress is not None:
            batch_inputs = self._add_progress_meta(batch_inputs, progress_samples)
        
        # Add resample_attempts (always added, like RFM batch collator)
        batch_inputs["resample_attempts"] = [sample.resample_attempts for sample in progress_samples]

        return batch_inputs
