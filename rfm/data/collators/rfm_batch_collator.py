#!/usr/bin/env python3
"""
Batch collator for processing list of samples.
"""

import tempfile
from pathlib import Path
import numpy as np
import torch
from qwen_vl_utils import process_vision_info

from .base_collator import BaseCollator
from .utils import convert_frames_to_pil_images, pad_target_progress, write_mp4
from rfm.data.dataset_types import PreferenceSample, ProgressSample, SimilaritySample
from typing import List, Dict


class RFMBatchCollator(BaseCollator):
    def __process_conversation(self, conversations: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        """
        Process a list of conversations into a batch of inputs.

        Args:
            conversations: List of conversations

        Returns:
            Batch of inputs
        """

        if "Qwen" in self.base_model_id:
            # Process all messages in one batch
            texts = [
                self.processor.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=False,
                    add_vision_id=True,
                    fps=1,
                )
                for msg in conversations
            ]

            image_inputs, video_inputs, _video_kwargs = process_vision_info(conversations, return_video_kwargs=True)

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
        elif "SmolVLM" in self.base_model_id:
            batch_inputs = self.processor.apply_chat_template(
                conversations,
                add_generation_prompt=False,
                tokenize=True,
                padding=True,
                truncation=False,
                max_length=self.max_length,
                return_dict=True,
                return_tensors="pt",
                fps=4,  # this should be same as fps for write_mp4
            )
        else:
            raise ValueError(f"Invalid base model id: {self.base_model_id}")

        return batch_inputs

    def _add_preference_meta(
        self, batch_inputs: dict[str, torch.Tensor], preference_samples: list[PreferenceSample]
    ) -> dict[str, torch.Tensor]:
        """Add metadata to the batch inputs."""
        batch_inputs["data_source"] = [sample.chosen_trajectory.data_source for sample in preference_samples]
        batch_inputs["sample_type"] = ["preference"] * len(preference_samples)
        batch_inputs["task"] = [sample.chosen_trajectory.task for sample in preference_samples]

        batch_inputs["chosen_data_gen_strategy"] = [
            sample.chosen_trajectory.data_gen_strategy for sample in preference_samples
        ]
        batch_inputs["rejected_data_gen_strategy"] = [
            sample.rejected_trajectory.data_gen_strategy for sample in preference_samples
        ]

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
        batch_inputs["target_progress_chosen"] = pad_target_progress(target_progress_chosen)
        batch_inputs["target_progress_rejected"] = pad_target_progress(target_progress_rejected)
        batch_inputs["target_progress_chosen_mask"] = torch.tensor(target_progress_chosen_mask, dtype=torch.float32)
        batch_inputs["target_progress_rejected_mask"] = torch.tensor(target_progress_rejected_mask, dtype=torch.float32)

        # Also add the frame_shapes
        if not self.load_embeddings:
            batch_inputs["chosen_frames_shape"] = torch.tensor(
                [sample.chosen_trajectory.frames_shape for sample in preference_samples], dtype=torch.int32
            )
            batch_inputs["rejected_frames_shape"] = torch.tensor(
                [sample.rejected_trajectory.frames_shape for sample in preference_samples], dtype=torch.int32
            )
        else:
            batch_inputs["chosen_frames_shape"] = torch.tensor(
                [sample.chosen_trajectory.video_embeddings.shape for sample in preference_samples], dtype=torch.int32
            )
            batch_inputs["rejected_frames_shape"] = torch.tensor(
                [sample.rejected_trajectory.video_embeddings.shape for sample in preference_samples], dtype=torch.int32
            )
        return batch_inputs

    def _add_progress_meta(
        self, batch_inputs: dict[str, torch.Tensor], progress_samples: list[ProgressSample]
    ) -> dict[str, torch.Tensor]:
        """Add metadata to the batch inputs."""
    
        # Add metadata
        batch_inputs["sample_type"] = ["progress"] * len(progress_samples)
        batch_inputs["task"] = [sample.trajectory.task for sample in progress_samples]
        batch_inputs["metadata"] = [sample.trajectory.metadata for sample in progress_samples]

        # Pad target progress tensors to max length in last dimension
        target_progress_list = [sample.trajectory.target_progress for sample in progress_samples]
        batch_inputs["target_progress"] = pad_target_progress(target_progress_list)
        batch_inputs["quality_labels"] = [sample.trajectory.quality_label for sample in progress_samples]

        if not self.load_embeddings:
            batch_inputs["frame_shapes"] = torch.tensor(
                [sample.trajectory.frames_shape for sample in progress_samples], dtype=torch.int32
            )
        else:
            batch_inputs["frame_shapes"] = torch.tensor(
                [sample.trajectory.video_embeddings.shape for sample in progress_samples], dtype=torch.int32
            )

        batch_inputs["data_source"] = [sample.trajectory.data_source for sample in progress_samples]
        batch_inputs["data_gen_strategy"] = [sample.trajectory.data_gen_strategy for sample in progress_samples]
        target_progress_mask = [
            1.0
            if sample.trajectory.quality_label == "successful"
            or sample.trajectory.data_gen_strategy == "rewind_same_task"
            else 0.0
            for sample in progress_samples
        ]
        batch_inputs["target_progress_mask"] = torch.tensor(target_progress_mask, dtype=torch.float32)
        return batch_inputs

    def _process_preference_batch(self, preference_samples: list[PreferenceSample]) -> dict[str, torch.Tensor]:
        """Process a batch of preference samples."""
        # Collect all messages for batch processing
        all_messages = []

        # Randomly decide whether chosen trajectory goes first or second
        preference_labels = np.random.randint(0, 2, len(preference_samples))

        # Build batch of conversations
        for i, sample in enumerate(preference_samples):
            # Convert frames to appropriate format using stored shapes
            chosen_frames = convert_frames_to_pil_images(
                sample.chosen_trajectory.frames, sample.chosen_trajectory.frames_shape
            )
            rejected_frames = convert_frames_to_pil_images(
                sample.rejected_trajectory.frames, sample.rejected_trajectory.frames_shape
            )

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
                # Chosen trajectory first: task + video A (chosen) + <|split_token|> + video B (rejected) + <|pref_token|>
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Task: {sample.chosen_trajectory.task}"},
                            {
                                "type": "video",
                                "video": chosen_frames,
                                **content_extras,
                            },
                            {"type": "text", "text": "<|split_token|>"},
                            {
                                "type": "video",
                                "video": rejected_frames,
                                **content_extras,
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
                                **content_extras,
                            },
                            {"type": "text", "text": "<|split_token|>"},
                            {
                                "type": "video",
                                "video": chosen_frames,
                                **content_extras,
                            },
                            {"type": "text", "text": "<|pref_token|>"},
                        ],
                    }
                ]

            all_messages.append(conversation)

        batch_inputs = self.__process_conversation(all_messages)
        # Use the dynamically generated preference labels based on trajectory order
        batch_inputs["preference_labels"] = torch.tensor(preference_labels, dtype=torch.float32)
        batch_inputs = self._add_preference_meta(batch_inputs, preference_samples)
        return batch_inputs

    def _process_similarity_batch(self, similarity_samples: list[SimilaritySample]) -> dict[str, torch.Tensor]:
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

        image_inputs, video_inputs, _video_kwargs = process_vision_info(all_messages, return_video_kwargs=True)

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

    def _process_progress_batch(self, progress_samples: list[ProgressSample]) -> dict[str, torch.Tensor]:
        """Process a batch of progress samples with VQA-style question."""
        # Collect all messages for batch processing
        all_messages = []

        for sample in progress_samples:
            # Convert frames to appropriate format using stored shapes
            frames = convert_frames_to_pil_images(sample.trajectory.frames, sample.trajectory.frames_shape)

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
                        {
                            "type": "text",
                            "text": f"Task: {sample.trajectory.task}",
                        },
                        {
                            "type": "video",
                            "video": frames,
                            **content_extras,
                        },
                    ],
                }
            ]

            all_messages.append(conversation)

        batch_inputs = self.__process_conversation(all_messages)
        batch_inputs = self._add_progress_meta(batch_inputs, progress_samples)
        return batch_inputs
