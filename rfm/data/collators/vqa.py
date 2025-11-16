#!/usr/bin/env python3
"""
VQA Batch collator for processing Sample objects through the processor and returning processed tensors.
This collator handles the conversion from PreferenceSample, SimilaritySample, and ProgressSample objects to processed tensors
for VQA-based reward modeling with different question types.
"""

import numpy as np
import torch

from .rfm_heads import RFMBatchCollator
from .utils import convert_frames_to_pil_images
from rfm.data.dataset_types import PreferenceSample, ProgressSample

IGNORE_INDEX = -100


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
            # prompt = f"Given these two trajectories for the task '{sample.chosen_trajectory.task}', evaluate which one better demonstrates successful completion of the task. Compare the trajectories and determine which is preferred."
            prompt = f"Given these two trajectories for the task '{sample.chosen_trajectory.task}', which one best corresponds to solving the task? Trajectory A or B? Format your answer enclosed by <ans> and </ans> tags."

            # Prepare frames for conversation (handles multi-image vs video conversion)
            chosen_video_field, content_extras = self._prepare_frames_for_conversation(
                chosen_frames, prefix="tmp_chosen"
            )
            rejected_video_field, _ = self._prepare_frames_for_conversation(rejected_frames, prefix="tmp_rejected")

            # Determine which trajectory is A and which is B based on preference label
            if preference_labels[i] == 1.0:
                # Chosen trajectory first: Trajectory A (chosen) + Trajectory B (rejected)
                traj_a_field = chosen_video_field
                traj_b_field = rejected_video_field
                answer = "A"
            else:
                # Chosen trajectory second: Trajectory A (rejected) + Trajectory B (chosen)
                traj_a_field = rejected_video_field
                traj_b_field = chosen_video_field
                answer = "B"

            # Build content list
            content_list = [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "This is Trajectory A. "},
            ]
            self._add_vision_content_to_list(content_list, traj_a_field, content_extras)
            content_list.append({"type": "text", "text": "This is Trajectory B. "})
            self._add_vision_content_to_list(content_list, traj_b_field, content_extras)

            conversation = [
                {
                    "role": "user",
                    "content": content_list,
                },
            ]
            # Only add assistant response during training, not inference
            if not self.inference:
                # SmolVLM requires list format for all messages, Qwen accepts both but we use list for consistency
                if "SmolVLM" in self.base_model_id:
                    # SmolVLM requires content as list of dicts
                    conversation.append({
                        "role": "assistant", 
                        "content": [{"type": "text", "text": f"<ans>{answer}</ans>"}]
                    })
                else:
                    # Qwen accepts simple string content for text-only assistant messages
                    conversation.append({
                        "role": "assistant", 
                        "content": f"<ans>{answer}</ans>"
                    })

            all_messages.append(conversation)

        batch_inputs = self._process_conversation(all_messages, add_generation_prompt=self.inference)
        if not self.inference:
            labels = batch_inputs["input_ids"].clone()

            # Mask out the prompt - only train on assistant response
            # Locate <ans> tag positions directly in token space
            for i in range(len(labels)):
                seq_len = labels[i].shape[0]
                max_window = 8  # number of tokens to decode when searching for <ans>
                ans_token_positions: list[int] = []

                for idx in range(seq_len):
                    end_idx = min(seq_len, idx + max_window)
                    window_tokens = labels[i][idx:end_idx]
                    window_text = self.processor.tokenizer.decode(window_tokens, skip_special_tokens=False)
                    if window_text.lstrip().startswith("<ans>"):
                        ans_token_positions.append(idx)

                if ans_token_positions:
                    start_idx = ans_token_positions[-1]
                    labels[i][:start_idx] = IGNORE_INDEX
                else:
                    labels[i][:] = IGNORE_INDEX

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

            prompt = f"For the task '{sample.trajectory.task}', estimate task progress at each frame in the video trajectory. Give a list of numbers between 0 and 1 where 0 means no progress and 1 means successful completion of the task. Format your answer as a python list enclosed by <ans> and </ans> tags."

            # Prepare frames for conversation (handles multi-image vs video conversion)
            video_field, content_extras = self._prepare_frames_for_conversation(frames, prefix="tmp_progress")

            # Build content list
            content_list = [{"type": "text", "text": prompt}]
            self._add_vision_content_to_list(content_list, video_field, content_extras)

            # Create conversation for progress evaluation
            conversation = [
                {
                    "role": "user",
                    "content": content_list,
                }
            ]
            # Add assistant response only if not in inference mode and target_progress exists
            if not self.inference and target_progress is not None:
                # Round target progress to 2 decimal places for the response
                # Convert to Python list to get proper comma-separated format
                target_progress_rounded = np.round(target_progress, 2).tolist()
                # SmolVLM requires list format for all messages, Qwen accepts both but we use string for simplicity
                if "SmolVLM" in self.base_model_id:
                    # SmolVLM requires content as list of dicts
                    conversation.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"<ans>{target_progress_rounded}</ans>"}]
                    })
                else:
                    # Qwen accepts simple string content for text-only assistant messages
                    conversation.append({
                        "role": "assistant",
                        "content": f"<ans>{target_progress_rounded}</ans>"
                    })

            all_messages.append(conversation)

        batch_inputs = self._process_conversation(all_messages, add_generation_prompt=self.inference)

        if not self.inference:
            labels = batch_inputs["input_ids"].clone()

            # Mask out the prompt - only train on assistant response
            # Locate <ans> token positions directly
            for i in range(len(labels)):
                seq_len = labels[i].shape[0]
                max_window = 8
                ans_token_positions: list[int] = []

                for idx in range(seq_len):
                    end_idx = min(seq_len, idx + max_window)
                    window_tokens = labels[i][idx:end_idx]
                    window_text = self.processor.tokenizer.decode(window_tokens, skip_special_tokens=False)
                    if window_text.lstrip().startswith("<ans>"):
                        ans_token_positions.append(idx)

                if ans_token_positions:
                    start_idx = ans_token_positions[-1]
                    labels[i][:start_idx] = IGNORE_INDEX
                else:
                    labels[i][:] = IGNORE_INDEX

            batch_inputs["labels"] = labels

        # Add progress metadata (includes all the misc fields like target_progress, masks, frames_shape, etc.)
        # Only call if target_progress exists (matches RFM batch collator behavior)
        if progress_samples[0].trajectory.target_progress is not None:
            batch_inputs = self._add_progress_meta(batch_inputs, progress_samples)

        # Add resample_attempts (always added, like RFM batch collator)
        batch_inputs["resample_attempts"] = [sample.resample_attempts for sample in progress_samples]

        return batch_inputs
