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

    def __init__(self, training: bool = True, inference: bool = False, shuffle_progress_frames: bool = False, **kwargs):
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
        super().__init__(inference=inference, shuffle_progress_frames=shuffle_progress_frames, **kwargs)

    def _mask_labels(self, labels):
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
                if window_text.lstrip().startswith("ANSWER:"):
                    ans_token_positions.append(idx)

            if ans_token_positions:
                start_idx = ans_token_positions[-1]
                labels[i][:start_idx] = IGNORE_INDEX
            else:
                labels[i][:] = IGNORE_INDEX
        return labels

    def _process_preference_batch(self, preference_samples: list[PreferenceSample]) -> dict[str, torch.Tensor]:
        """Process a batch of preference samples with VQA-style question."""
        # Collect all messages for batch processing
        all_messages = []

        # During inference, keep original order (chosen=A, rejected=B)
        # During training, randomly decide whether chosen trajectory goes first or second
        if self.inference:
            # Keep original order: chosen is always A (preference_label=1.0)
            preference_labels = np.ones(len(preference_samples), dtype=np.int32)
        else:
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
            prompt = f"""Given these two robot and/or human trajectory videos, which one makes the most progress towards solving the task, Video 1 or 2? Format your answer as: ANSWER: 1/2

Task: {sample.chosen_trajectory.task}"""

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
                answer = "1"
            else:
                # Chosen trajectory second: Trajectory A (rejected) + Trajectory B (chosen)
                traj_a_field = rejected_video_field
                traj_b_field = chosen_video_field
                answer = "2"

            # Build content list
            content_list = [
                #{"type": "text", "text": "This is Trajectory A. "},
            ]
            self._add_vision_content_to_list(content_list, traj_a_field, content_extras)
            #content_list.append({"type": "text", "text": "This is Trajectory B. "})
            self._add_vision_content_to_list(content_list, traj_b_field, content_extras)
            content_list.append({"type": "text", "text": prompt})

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
                        "content": [{"type": "text", "text": f"ANSWER: {answer}"}],
                    })
                else:
                    # Qwen accepts simple string content for text-only assistant messages
                    conversation.append({"role": "assistant", "content": f"ANSWER: {answer}"})

            all_messages.append(conversation)

        batch_inputs = self._process_conversation(all_messages, add_generation_prompt=self.inference)
        if not self.inference:
            labels = batch_inputs["input_ids"].clone()
            batch_inputs["labels"] = self._mask_labels(labels)

        # Use the dynamically generated preference labels based on trajectory order
        batch_inputs["preference_labels"] = torch.tensor(preference_labels, dtype=torch.float32)

        # Add preference metadata (includes all the misc fields like target_progress, masks, frames_shape, etc.)
        batch_inputs = self._add_preference_meta(batch_inputs, preference_samples)

        return batch_inputs

    def _process_progress_batch(self, progress_samples: list[ProgressSample]) -> dict[str, torch.Tensor]:
        """Process a batch of progress samples with VQA-style question."""
        # Collect all messages for batch processing
        all_messages = []
        assert not self.shuffle_progress_frames, "Currently shuffling is not in the new prompt so not supported"

        for i, sample in enumerate(progress_samples):
            target_progress = sample.trajectory.target_progress
            

            # Convert frames to appropriate format using stored shapes
            frames = convert_frames_to_pil_images(sample.trajectory.frames, sample.trajectory.frames_shape)

            # Shuffle frames and their corresponding target progress values (only during training)
            if self.shuffle_progress_frames and target_progress is not None and not self.inference:
                if len(target_progress) > 1 and len(target_progress) == len(frames):
                    shuffle_indices = np.random.permutation(range(1, len(frames)))
                    frames = [frames[0]] + [frames[idx] for idx in shuffle_indices]
                    target_progress = [target_progress[0]] + [target_progress[idx] for idx in shuffle_indices]

                else:
                    raise ValueError(
                        f"Target progress must be a list of at least 1 float for shuffling, got {len(target_progress)}"
                    )
            prompt = """Given the task, assign a python list of integer-valued progress scores from 0 to 100 for each frame of the video in the format: ANSWER: [scores]
End of episode progress should be judged only on the final state, without time limits.
Rubric for end-of-episode progress (judge only the final state without time limits):
0 - No Progress: Final state shows no goal-relevant change for the command.
100 - Perfect Completion: Final state satisfies all requirements to solve the task.
Anything in between represents partial progress towards the goal.

Task: {task}""".format(task=sample.trajectory.task)
            #prompt = f"For the task '{sample.trajectory.task}', estimate task progress at each frame in the video trajectory."
            #if self.shuffle_progress_frames:
            #    prompt += " These frames are possibly shuffled, so pay attention to individual frames when reasoning about progress."
            #prompt += " The first frame is the starting frame, with 0 progress."
            #prompt += (
            #    " Format your answer as a python list with floats between 0 and 1 enclosed by <ans> and </ans> tags."
            #)

            # Prepare frames for conversation (handles multi-image vs video conversion)
            video_field, content_extras = self._prepare_frames_for_conversation(frames, prefix="tmp_progress")

            # Build content list
            content_list = []
            self._add_vision_content_to_list(content_list, video_field, content_extras)
            content_list.append({"type": "text", "text": prompt})

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
                target_progress_rounded = np.round(np.array(target_progress) * 100).astype(np.uint8).tolist()


                ## TODO: unhardcode this: for now, just use last frame target progress
                #target_progress_rounded = target_progress_rounded[-1]
                # SmolVLM requires list format for all messages, Qwen accepts both but we use string for simplicity
                if "SmolVLM" in self.base_model_id:
                    # SmolVLM requires content as list of dicts
                    conversation.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"ANSWER: {target_progress_rounded}"}],
                    })
                else:
                    # Qwen accepts simple string content for text-only assistant messages
                    conversation.append({"role": "assistant", "content": f"ANSWER: {target_progress_rounded}"})

            all_messages.append(conversation)

        batch_inputs = self._process_conversation(all_messages, add_generation_prompt=self.inference)
        if not self.inference:
            labels = batch_inputs["input_ids"].clone()
            batch_inputs["labels"] = self._mask_labels(labels)

        # Add progress metadata (includes all the misc fields like target_progress, masks, frames_shape, etc.)
        # Only call if target_progress exists (matches RFM batch collator behavior)
        if progress_samples[0].trajectory.target_progress is not None:
            batch_inputs = self._add_progress_meta(batch_inputs, progress_samples)

        # Add resample_attempts (always added, like RFM batch collator)
        batch_inputs["resample_attempts"] = [sample.resample_attempts for sample in progress_samples]

        return batch_inputs
