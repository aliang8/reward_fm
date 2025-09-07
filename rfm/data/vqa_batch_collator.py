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
from rfm.data.base_collator import BaseCollator


class VQABatchCollator(BaseCollator):
    """Batch collator that processes Sample objects through the processor for VQA-based reward modeling."""

    def __init__(self, training: bool = True, **kwargs):
        """
        Initialize the VQA batch collator.

        Args:
            processor: HuggingFace processor for text and vision processing
            max_length: Maximum sequence length for text
            resized_height: Height to resize images/videos to (default: 128)
            resized_width: Width to resize images/videos to (default: 128)
        """
        self.training = training
        super().__init__(**kwargs)

    def _create_vqa_inputs_with_labels(self, conversations, answer_texts):
        """
        Create VQA inputs with proper labels that only calculate loss on answer tokens. If it is evaluation, we don't need to set the labels

        Args:
            conversations: list of message-lists (each item is the "conversation" you built for a sample)
            answer_texts:  list[str], e.g., "<ans>A</ans>" or "<ans>[...progress...]</ans>"

        Returns:
            Dict with tokenized inputs and labels masked before the assistant answer. If it is evaluation, we don't need to set the labels
        """
        if self.training:
            assert len(conversations) == len(answer_texts), "conversations and answer_texts must align"

            # 1) Build assistant-augmented conversations for the *full* example
            conversations_full = []
            for conv, ans in zip(conversations, answer_texts):
                # Append an assistant turn holding the gold answer as text
                assistant_turn = [{"role": "assistant", "content": [{"type": "text", "text": ans}]}]
                # NOTE: `conv` is already a list of {role, content} dicts; keep structure consistent
                conversations_full.append(conv + assistant_turn)

            # 2) Render text with chat template
            # Prompt text includes the “assistant header” via add_generation_prompt=True
            prompt_texts = [
                self.processor.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=True,  # include assistant prefix tokens
                    add_vision_id=True,
                    fps=1,
                )
                for conv in conversations
            ]

            full_texts = [
                self.processor.apply_chat_template(
                    conv_full,
                    tokenize=False,
                    add_generation_prompt=False,  # full already includes assistant turn
                    add_vision_id=True,
                    fps=1,
                )
                for conv_full in conversations_full
            ]

            # 3) Pack vision once, reuse for both tokenizations to keep token alignment identical
            image_inputs, video_inputs, video_kwargs = process_vision_info(conversations, return_video_kwargs=True)

            # 4) Tokenize prompt-only (so we know exactly how many tokens precede the answer)
            prompt_inputs = self.processor(
                text=prompt_texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,  # pad so we can batch
                truncation=False,  # keep everything; truncate only at the "full" step if you must
                max_length=self.max_length,
                return_tensors="pt",
            )

            # 5) Tokenize full (prompt + assistant answer)
            full_inputs = self.processor(
                text=full_texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                truncation=False,  # prefer no truncation so we don't chop off the answer
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = full_inputs["input_ids"]
            attn_mask = full_inputs["attention_mask"]
            batch_size, seq_len = input_ids.shape

            # Resolve pad token id (some Qwen tokenizers set it to eos if None)
            pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(self.processor.tokenizer, "eos_token_id", None)
            if pad_id is None:
                raise ValueError("Tokenizer must have pad_token_id or eos_token_id defined for masking.")

            # 6) Build labels: -100 for everything up to the prompt length; copy tokens for the answer span
            labels = torch.full_like(input_ids, fill_value=-100)

            # We'll compute the *non-padded* length of the prompt sequence for each item, then
            # label tokens in the full sequence strictly after that index.
            for i in range(batch_size):
                # length of tokens in prompt example (ignore pads)
                # Use attention_mask from the prompt encoding for robustness
                prompt_len = int(prompt_inputs["attention_mask"][i].sum().item())

                # length of tokens in full example (ignore pads)
                full_len = int(attn_mask[i].sum().item())

                # guard rails
                prompt_len = min(prompt_len, seq_len)
                full_len = min(full_len, seq_len)

                if full_len > prompt_len:
                    labels[i, prompt_len:full_len] = input_ids[i, prompt_len:full_len]
                # else: entirely masked (e.g., if truncation made the answer vanish)

            # 7) Return final dict with any extra vision fields preserved
            result = {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "labels": labels,
            }
            # Carry over vision tensors if present
            for k in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"]:
                if k in full_inputs:
                    result[k] = full_inputs[k]

        else:
            prompt_texts = [
                self.processor.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=True,  # include assistant prefix tokens
                    add_vision_id=True,
                    fps=1,
                )
                for conv in conversations
            ]
            # 3) Pack vision once, reuse for both tokenizations to keep token alignment identical
            image_inputs, video_inputs, video_kwargs = process_vision_info(conversations, return_video_kwargs=True)

            # 4) Tokenize prompt-only (so we know exactly how many tokens precede the answer)
            prompt_inputs = self.processor(
                text=prompt_texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,  # pad so we can batch
                truncation=False,  # keep everything; truncate only at the "full" step if you must
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = prompt_inputs["input_ids"]
            attn_mask = prompt_inputs["attention_mask"]
            batch_size, seq_len = input_ids.shape
            result = {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "labels": torch.full_like(input_ids, fill_value=-100),
            }
            for k in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"]:
                if k in prompt_inputs:
                    result[k] = prompt_inputs[k]

        return result

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

            if preference_labels[i] == 1.0:
                # Chosen trajectory first: Trajectory A (chosen) + Trajectory B (rejected)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Given these two trajectories for the task '{sample.chosen_trajectory.task}', which one best corresponds to solving the task? Trajectory A or B? Format your answer enclosed by <ans> and </ans> tags. For example, if you prefer trajectory A, your answer should be <ans>A</ans>.",
                            },
                            {
                                "type": "video",
                                "video": chosen_frames,
                                "resized_height": self.resized_height,
                                "resized_width": self.resized_width,
                            },
                            {"type": "text", "text": "Trajectory A. "},
                            {
                                "type": "video",
                                "video": rejected_frames,
                                "resized_height": self.resized_height,
                                "resized_width": self.resized_width,
                            },
                            {"type": "text", "text": "Trajectory B. "},
                        ],
                    }
                ]
            else:
                # Chosen trajectory second: Trajectory A (rejected) + Trajectory B (chosen)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Given these two trajectories for the task '{sample.chosen_trajectory.task}', which one best corresponds to solving the task? Trajectory A or B? Format your answer enclosed by <ans> and </ans> tags. For example, if you prefer trajectory A, your answer should be <ans>A</ans>.",
                            },
                            {
                                "type": "video",
                                "video": rejected_frames,
                                "resized_height": self.resized_height,
                                "resized_width": self.resized_width,
                            },
                            {"type": "text", "text": "Trajectory A. "},
                            {
                                "type": "video",
                                "video": chosen_frames,
                                "resized_height": self.resized_height,
                                "resized_width": self.resized_width,
                            },
                            {"type": "text", "text": "Trajectory B. "},
                        ],
                    }
                ]

            all_messages.append(conversation)

        # Convert preference labels to text answers
        preference_labels_text = ["<ans>A</ans>" if label == 1 else "<ans>B</ans>" for label in preference_labels]

        # Create input with generation prompt and answer for proper label setting, if it is evaluation, we don't need to set the labels
        batch_inputs = self._create_vqa_inputs_with_labels(all_messages, preference_labels_text)

        # Add metadata
        batch_inputs["sample_type"] = ["preference"] * len(preference_samples)
        # Use the dynamically generated preference labels based on trajectory order
        batch_inputs["preference_labels"] = torch.tensor(preference_labels, dtype=torch.float32)

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

    def _process_similarity_batch(
        self, similarity_samples: List[SimilaritySample]
    ) -> Dict[str, torch.Tensor]:  # Redundant for now
        """Process a batch of similarity samples with VQA-style question."""
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

            # Create conversation for similarity comparison
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Given the following reference trajectory for the task '{sample.reference_trajectory.task}', which one of the two trajectories are more similar to it? Trajectory A or B? Format your answer enclosed by <ans> and </ans> tags. For example, if you think trajectory A is more similar to the reference trajectory, your answer should be <ans>A</ans>.",
                        },
                        {
                            "type": "video",
                            "video": reference_frames,
                            "resized_height": self.resized_height,
                            "resized_width": self.resized_width,
                        },
                        {"type": "text", "text": "Reference Trajectory. "},
                        {
                            "type": "video",
                            "video": traj_sim_frames,
                            "resized_height": self.resized_height,
                            "resized_width": self.resized_width,
                        },
                        {"type": "text", "text": "Trajectory A. "},
                        {
                            "type": "video",
                            "video": traj_diff_frames,
                            "resized_height": self.resized_height,
                            "resized_width": self.resized_width,
                        },
                        {"type": "text", "text": "Trajectory B. "},
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

        # Add metadata
        batch_inputs["sample_type"] = ["similarity"] * len(similarity_samples)

        # Add target progress for all trajectories
        target_progress_ref_list = []
        target_progress_sim_list = []
        target_progress_diff_list = []

        for sample in similarity_samples:
            if sample.traj_sim_trajectory.target_progress is not None:
                target_progress_sim_list.append(sample.traj_sim_trajectory.target_progress)

            if sample.traj_diff_trajectory.target_progress is not None:
                target_progress_diff_list.append(sample.traj_diff_trajectory.target_progress)

            if sample.reference_trajectory.target_progress is not None:
                target_progress_ref_list.append(sample.reference_trajectory.target_progress)

        # Pad target progress tensors to max length in last dimension
        batch_inputs["target_progress_ref"] = self._pad_target_progress(target_progress_ref_list)
        batch_inputs["target_progress_sim"] = self._pad_target_progress(target_progress_sim_list)
        batch_inputs["target_progress_diff"] = self._pad_target_progress(target_progress_diff_list)

        # Also add the frame_shapes
        batch_inputs["ref_frames_shape"] = torch.tensor(
            [sample.reference_trajectory.frames_shape for sample in similarity_samples], dtype=torch.int32
        )
        batch_inputs["traj_sim_frames_shape"] = torch.tensor(
            [sample.traj_sim_trajectory.frames_shape for sample in similarity_samples], dtype=torch.int32
        )
        batch_inputs["traj_diff_frames_shape"] = torch.tensor(
            [sample.traj_diff_trajectory.frames_shape for sample in similarity_samples], dtype=torch.int32
        )
        return batch_inputs

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
                            "text": f"For the task '{sample.trajectory.task}', estimate the progress at each frame in the trajectory. Give a list of numbers between 0 and 1 where 0 means no progress and 1 means successful completion of the task. Format your answer enclosed by <ans> and </ans> tags. For example, if you think the progress at each frame is [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], your answer should be <ans>[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]</ans>.",
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

        # Add target progress and quality labels
        target_progress_list = []
        quality_labels = []

        for sample in progress_samples:
            if sample.trajectory.target_progress is not None:
                target_progress_list.append(sample.trajectory.target_progress)
            quality_labels.append(1.0 if sample.trajectory.quality_label == "successful" else 0.0)

        # Convert progress labels to text answers
        progress_labels_text = []
        for progress in target_progress_list:
            if progress:
                progress_labels_text.append(f"<ans>{progress}</ans>")
            else:
                progress_labels_text.append(f"<ans>{[0] * len(progress)}</ans>")

        # Create input with generation prompt and answer for proper label setting, if it is evaluation, we don't need to set the labels
        batch_inputs = self._create_vqa_inputs_with_labels(all_messages, progress_labels_text)

        # Add metadata
        batch_inputs["sample_type"] = ["progress"] * len(progress_samples)

        # Pad target progress tensors to max length in last dimension
        batch_inputs["target_progress"] = self._pad_target_progress(target_progress_list)
        batch_inputs["quality_labels"] = torch.tensor(quality_labels, dtype=torch.float32)

        return batch_inputs

    def collate_fn(
        self, batch: List[Union[PreferenceSample, SimilaritySample, ProgressSample]]
    ) -> Dict[str, torch.Tensor]:
        """
        Alternative method name for compatibility with PyTorch DataLoader.

        Args:
            batch: List of Sample objects

        Returns:
            Dictionary containing the processed tensors
        """
        return self(batch)
