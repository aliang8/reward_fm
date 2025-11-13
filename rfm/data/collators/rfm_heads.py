#!/usr/bin/env python3
"""
Batch collator for processing list of samples.
"""

import tempfile
from pathlib import Path
import numpy as np
import torch
from qwen_vl_utils import process_vision_info

from .base import BaseCollator
from .utils import convert_frames_to_pil_images, pad_target_progress, write_mp4
from rfm.data.dataset_types import PreferenceSample, ProgressSample, SimilaritySample
from rfm.data.dataset_category import is_preference_only
from typing import List, Dict


def should_compute_progress(quality_label: str, data_gen_strategy: str, data_source: str = None) -> float:
    """
    Check if progress should be computed for a trajectory.

    Includes if it is successful, rewound, rewind_same_task, or different_task,
    but NOT suboptimal or failure. Also masks out progress if data_source is in preference_only category.

    Args:
        quality_label: The quality label of the trajectory
        data_gen_strategy: The data generation strategy
        data_source: The data source name (optional)

    Returns:
        1.0 if progress should be computed, 0.0 otherwise
    """
    # Mask out progress if data_source is in preference_only category
    if data_source is not None and is_preference_only(data_source):
        return 0.0

    if quality_label in ["suboptimal", "failure"]:
        return 0.0

    if (
        quality_label == "successful"
        or quality_label == "rewound"
        or data_gen_strategy == "rewind_same_task"
        or data_gen_strategy == "different_task"
    ):
        return 1.0

    return 0.0


def create_padding_mask(frames_shapes: torch.Tensor, max_length: int = None) -> torch.Tensor:
    """
    Create padding mask based on frames_shape.

    Args:
        frames_shapes: Tensor of shape (batch_size, ...) where first dim of each row is num_frames
        max_length: Maximum length for padding. If None, uses max of first dim in frames_shapes

    Returns:
        Tensor of shape (batch_size, max_length) with 1.0 for valid frames, 0.0 for padding
    """
    # Extract num_frames from first dimension of each shape
    if frames_shapes.dim() > 1:
        num_frames = frames_shapes[:, 0].float()
    else:
        num_frames = frames_shapes.float()

    if max_length is None:
        max_length = int(num_frames.max().item())

    # Create range tensor: [0, 1, 2, ..., max_length-1]
    range_tensor = torch.arange(max_length, dtype=torch.float32, device=frames_shapes.device)

    # Broadcast comparison: (batch_size, 1) vs (1, max_length) -> (batch_size, max_length)
    # For each sample, positions < num_frames are valid (1.0), others are padding (0.0)
    masks = (range_tensor.unsqueeze(0) < num_frames.unsqueeze(1)).float()

    return masks


class RFMBatchCollator(BaseCollator):
    def __init__(
        self,
        processor,
        tokenizer=None,
        max_length: int = 1024,
        resized_height: int = 128,
        resized_width: int = 128,
        base_model_id: str = None,
        load_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(
            processor=processor,
            tokenizer=tokenizer,
            max_length=max_length,
            resized_height=resized_height,
            resized_width=resized_width,
            base_model_id=base_model_id,
            load_embeddings=load_embeddings,
            **kwargs,
        )

    def _process_conversation(self, conversations: List[List[Dict]]) -> Dict[str, torch.Tensor]:
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
                do_resize=False
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
            prompt = f"For the task '{sample.trajectory.task}', evaluate the progress shown in this trajectory video. Assess how well the trajectory demonstrates completion of the task at each frame."
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
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

        batch_inputs = self._process_conversation(all_messages)
        if progress_samples[0].trajectory.target_progress is not None:
            batch_inputs = self._add_progress_meta(batch_inputs, progress_samples)
        batch_inputs["resample_attempts"] = [sample.resample_attempts for sample in progress_samples]
        return batch_inputs

    def _add_progress_meta(
        self, batch_inputs: dict[str, torch.Tensor], progress_samples: list[ProgressSample]
    ) -> dict[str, torch.Tensor]:
        batch_inputs["sample_type"] = ["progress"] * len(progress_samples)
        batch_inputs["task"] = [sample.trajectory.task for sample in progress_samples]
        batch_inputs["metadata"] = [sample.trajectory.metadata for sample in progress_samples]

        # Pad target progress tensors to max length in last dimension
        target_progress_list = [sample.trajectory.target_progress for sample in progress_samples]
        batch_inputs["target_progress"] = pad_target_progress(target_progress_list)
        batch_inputs["quality_labels"] = [sample.trajectory.quality_label for sample in progress_samples]

        frames_shape_list = [sample.trajectory.frames_shape for sample in progress_samples]
        batch_inputs["frames_shape"] = torch.tensor(frames_shape_list, dtype=torch.int32)

        max_length = batch_inputs["target_progress"].shape[-1]
        batch_inputs["padding_mask"] = create_padding_mask(batch_inputs["frames_shape"], max_length)

        batch_inputs["data_source"] = [sample.trajectory.data_source for sample in progress_samples]
        batch_inputs["data_gen_strategy"] = [sample.data_gen_strategy for sample in progress_samples]
        target_progress_mask = [
            should_compute_progress(
                sample.trajectory.quality_label,
                sample.data_gen_strategy,
                data_source=sample.trajectory.data_source,
            )
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

            prompt = f"Given these two trajectories for the task '{sample.chosen_trajectory.task}', evaluate which one better demonstrates successful completion of the task. Compare the trajectories and determine which is preferred."
            
            if preference_labels[i] == 1.0:
                # Chosen trajectory first: task + video A (chosen) + <|split_token|> + video B (rejected) + <|pref_token|>
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
                            {"type": "text", "text": "<|split_token|>"},
                            {"type": "text", "text": "This is Trajectory B. "},
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
                            {"type": "text", "text": prompt},
                            {"type": "text", "text": "This is Trajectory A. "},
                            {
                                "type": "video",
                                "video": rejected_frames,
                                **content_extras,
                            },
                            {"type": "text", "text": "<|split_token|>"},
                            {"type": "text", "text": "This is Trajectory B. "},
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

        batch_inputs = self._process_conversation(all_messages)
        # Use the dynamically generated preference labels based on trajectory order
        batch_inputs["preference_labels"] = torch.tensor(preference_labels, dtype=torch.float32)
        batch_inputs = self._add_preference_meta(batch_inputs, preference_samples)
        return batch_inputs

    def _add_preference_meta(
        self, batch_inputs: dict[str, torch.Tensor], preference_samples: list[PreferenceSample]
    ) -> dict[str, torch.Tensor]:
        batch_inputs["data_source"] = [sample.chosen_trajectory.data_source for sample in preference_samples]
        batch_inputs["sample_type"] = ["preference"] * len(preference_samples)
        batch_inputs["task"] = [sample.chosen_trajectory.task for sample in preference_samples]
        batch_inputs["data_gen_strategy"] = [sample.data_gen_strategy for sample in preference_samples]

        # Determine which trajectory is A and which is B based on preference_label
        # Trajectory A is chosen if preference_label==1.0, otherwise rejected is A
        trajectory_A_list = [
            sample.chosen_trajectory
            if batch_inputs["preference_labels"][i].item() == 1.0
            else sample.rejected_trajectory
            for i, sample in enumerate(preference_samples)
        ]
        trajectory_B_list = [
            sample.rejected_trajectory
            if batch_inputs["preference_labels"][i].item() == 1.0
            else sample.chosen_trajectory
            for i, sample in enumerate(preference_samples)
        ]

        batch_inputs["trajectory_A_quality_label"] = [traj.quality_label for traj in trajectory_A_list]

        trajectory_A_data_gen_strategy = []
        trajectory_B_data_gen_strategy = []
        for i, sample in enumerate(preference_samples):
            if batch_inputs["preference_labels"][i].item() == 1.0:
                trajectory_A_data_gen_strategy.append("subsample_task")
                trajectory_B_data_gen_strategy.append(sample.data_gen_strategy)
            else:
                trajectory_A_data_gen_strategy.append(sample.data_gen_strategy)
                trajectory_B_data_gen_strategy.append("subsample_task")

        batch_inputs["trajectory_A_data_gen_strategy"] = trajectory_A_data_gen_strategy

        # Add target progress for both trajectories using list comprehensions
        target_progress_A = [traj.target_progress for traj in trajectory_A_list]
        target_progress_B = [traj.target_progress for traj in trajectory_B_list]
        target_progress_A_mask = [
            should_compute_progress(
                traj.quality_label,
                strategy,
                data_source=traj.data_source,
            )
            for traj, strategy in zip(trajectory_A_list, trajectory_A_data_gen_strategy)
        ]
        target_progress_B_mask = [
            should_compute_progress(
                traj.quality_label,
                strategy,
                data_source=traj.data_source,
            )
            for traj, strategy in zip(trajectory_B_list, trajectory_B_data_gen_strategy)
        ]

        batch_inputs["target_progress_A"] = pad_target_progress(target_progress_A)
        batch_inputs["target_progress_B"] = pad_target_progress(target_progress_B)
        batch_inputs["target_progress_A_mask"] = torch.tensor(target_progress_A_mask, dtype=torch.float32)
        batch_inputs["target_progress_B_mask"] = torch.tensor(target_progress_B_mask, dtype=torch.float32)

        frames_shape_A = [traj.frames_shape for traj in trajectory_A_list]
        frames_shape_B = [traj.frames_shape for traj in trajectory_B_list]
        batch_inputs["frames_shape_A"] = torch.tensor(frames_shape_A, dtype=torch.int32)
        batch_inputs["frames_shape_B"] = torch.tensor(frames_shape_B, dtype=torch.int32)

        max_length_A = batch_inputs["target_progress_A"].shape[-1]
        max_length_B = batch_inputs["target_progress_B"].shape[-1]
        batch_inputs["padding_mask_A"] = create_padding_mask(batch_inputs["frames_shape_A"], max_length_A)
        batch_inputs["padding_mask_B"] = create_padding_mask(batch_inputs["frames_shape_B"], max_length_B)

        batch_inputs["chosen_data_gen_strategy"] = ["subsample_task"] * len(preference_samples)
        batch_inputs["rejected_data_gen_strategy"] = [sample.data_gen_strategy for sample in preference_samples]
        batch_inputs["chosen_quality_label"] = [sample.chosen_trajectory.quality_label for sample in preference_samples]

        target_progress_chosen = [sample.chosen_trajectory.target_progress for sample in preference_samples]
        target_progress_rejected = [sample.rejected_trajectory.target_progress for sample in preference_samples]
        target_progress_chosen_mask = [
            should_compute_progress(
                sample.chosen_trajectory.quality_label,
                "subsample_task",
                data_source=sample.chosen_trajectory.data_source,
            )
            for sample in preference_samples
        ]
        target_progress_rejected_mask = [
            should_compute_progress(
                sample.rejected_trajectory.quality_label,
                sample.data_gen_strategy,
                data_source=sample.rejected_trajectory.data_source,
            )
            for sample in preference_samples
        ]

        # Pad target progress tensors to max length in last dimension
        batch_inputs["target_progress_chosen"] = pad_target_progress(target_progress_chosen)
        batch_inputs["target_progress_rejected"] = pad_target_progress(target_progress_rejected)
        batch_inputs["target_progress_chosen_mask"] = torch.tensor(target_progress_chosen_mask, dtype=torch.float32)
        batch_inputs["target_progress_rejected_mask"] = torch.tensor(target_progress_rejected_mask, dtype=torch.float32)

        batch_inputs["chosen_frames_shape"] = torch.tensor(
            [sample.chosen_trajectory.frames_shape for sample in preference_samples], dtype=torch.int32
        )
        batch_inputs["rejected_frames_shape"] = torch.tensor(
            [sample.rejected_trajectory.frames_shape for sample in preference_samples], dtype=torch.int32
        )
        batch_inputs["resample_attempts"] = [sample.resample_attempts for sample in preference_samples]
        return batch_inputs

    def _process_similarity_batch(self, similarity_samples: list[SimilaritySample]) -> dict[str, torch.Tensor]:
        """Process a batch of similarity samples."""
        # Collect all messages for batch processing (ref_sim and ref_diff for each sample)
        all_messages = []

        for sample in similarity_samples:
            # Convert frames to appropriate format using stored shapes
            reference_frames = convert_frames_to_pil_images(
                sample.ref_trajectory.frames, sample.ref_trajectory.frames_shape
            )
            sim_frames = convert_frames_to_pil_images(sample.sim_trajectory.frames, sample.sim_trajectory.frames_shape)
            diff_frames = convert_frames_to_pil_images(
                sample.diff_trajectory.frames, sample.diff_trajectory.frames_shape
            )

            if "Qwen" in self.base_model_id:
                content_extras = {
                    "resized_height": self.resized_height,
                    "resized_width": self.resized_width,
                }
            elif "SmolVLM" in self.base_model_id:
                # Write frames to temporary files for SmolVLM
                tmp_ref = Path(tempfile.gettempdir()) / f"tmp_ref.mp4"
                write_mp4(reference_frames, tmp_ref)
                reference_frames_video = str(tmp_ref)

                tmp_sim = Path(tempfile.gettempdir()) / f"tmp_sim.mp4"
                write_mp4(sim_frames, tmp_sim)
                sim_frames_video = str(tmp_sim)

                tmp_diff = Path(tempfile.gettempdir()) / f"tmp_diff.mp4"
                write_mp4(diff_frames, tmp_diff)
                diff_frames_video = str(tmp_diff)

                content_extras = {}
            else:
                content_extras = {}

            # For SmolVLM, use the video paths; for Qwen, use the PIL images
            if "SmolVLM" in self.base_model_id:
                ref_video = reference_frames_video
                sim_video = sim_frames_video
                diff_video = diff_frames_video
            else:
                ref_video = reference_frames
                sim_video = sim_frames
                diff_video = diff_frames

            # Process reference vs trajectory sim
            prompt_sim = f"For the task '{sample.ref_trajectory.task}', compare these two trajectories and evaluate how similar they are in terms of task completion and behavior."
            conversation_ref_sim = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_sim},
                        {"type": "text", "text": "This is the reference trajectory. "},
                        {
                            "type": "video",
                            "video": ref_video,
                            **content_extras,
                        },
                        {"type": "text", "text": "<|split_token|>"},
                        {"type": "text", "text": "This is the comparison trajectory. "},
                        {
                            "type": "video",
                            "video": sim_video,
                            **content_extras,
                        },
                        {"type": "text", "text": "<|sim_token|>"},
                    ],
                }
            ]

            # Process reference vs trajectory diff
            prompt_diff = f"For the task '{sample.ref_trajectory.task}', compare these two trajectories and evaluate how similar they are in terms of task completion and behavior."
            conversation_ref_diff = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_diff},
                        {"type": "text", "text": "This is the reference trajectory. "},
                        {
                            "type": "video",
                            "video": ref_video,
                            **content_extras,
                        },
                        {"type": "text", "text": "<|split_token|>"},
                        {"type": "text", "text": "This is the comparison trajectory. "},
                        {
                            "type": "video",
                            "video": diff_video,
                            **content_extras,
                        },
                        {"type": "text", "text": "<|sim_token|>"},
                    ],
                }
            ]

            all_messages.extend([conversation_ref_sim, conversation_ref_diff])

        # Process all conversations
        batch_inputs = self._process_conversation(all_messages)

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

        # Add similarity-specific metadata
        combined_inputs = self._add_similarity_meta(combined_inputs, similarity_samples)
        return combined_inputs

    def _add_similarity_meta(
        self, batch_inputs: dict[str, torch.Tensor], similarity_samples: list[SimilaritySample]
    ) -> dict[str, torch.Tensor]:
        batch_inputs["data_source"] = [sample.ref_trajectory.data_source for sample in similarity_samples]
        batch_inputs["sample_type"] = ["similarity"] * len(similarity_samples)
        batch_inputs["task"] = [sample.ref_trajectory.task for sample in similarity_samples]
        batch_inputs["data_gen_strategy"] = [sample.data_gen_strategy for sample in similarity_samples]

        # Add target progress for all three trajectories
        target_progress_ref = [sample.ref_trajectory.target_progress for sample in similarity_samples]
        target_progress_sim = [sample.sim_trajectory.target_progress for sample in similarity_samples]
        target_progress_diff = [sample.diff_trajectory.target_progress for sample in similarity_samples]

        # Create masks for progress loss (only compute for successful trajectories or rewinds)
        # For similarity samples, ref is always successful, sim and diff depend on sample's data_gen_strategy
        target_progress_ref_mask = [
            should_compute_progress(
                sample.ref_trajectory.quality_label,
                "successful",
                data_source=sample.ref_trajectory.data_source,
            )
            for sample in similarity_samples
        ]
        # sim_trajectory strategy depends on sample's data_gen_strategy (e.g., "rewind_same_task" -> sim is rewound)
        target_progress_sim_mask = [
            should_compute_progress(
                sample.sim_trajectory.quality_label,
                sample.data_gen_strategy,
                data_source=sample.sim_trajectory.data_source,
            )
            for sample in similarity_samples
        ]
        # diff_trajectory is usually from different task or suboptimal
        target_progress_diff_mask = [
            should_compute_progress(
                sample.diff_trajectory.quality_label,
                "different_task",
                data_source=sample.diff_trajectory.data_source,
            )
            for sample in similarity_samples
        ]

        batch_inputs["target_progress_ref"] = pad_target_progress(target_progress_ref)
        batch_inputs["target_progress_sim"] = pad_target_progress(target_progress_sim)
        batch_inputs["target_progress_diff"] = pad_target_progress(target_progress_diff)

        batch_inputs["target_progress_ref_mask"] = torch.tensor(target_progress_ref_mask, dtype=torch.float32)
        batch_inputs["target_progress_sim_mask"] = torch.tensor(target_progress_sim_mask, dtype=torch.float32)
        batch_inputs["target_progress_diff_mask"] = torch.tensor(target_progress_diff_mask, dtype=torch.float32)

        ref_frames_shape_list = [sample.ref_trajectory.frames_shape for sample in similarity_samples]
        traj_sim_frames_shape_list = [sample.sim_trajectory.frames_shape for sample in similarity_samples]
        traj_diff_frames_shape_list = [sample.diff_trajectory.frames_shape for sample in similarity_samples]

        batch_inputs["ref_frames_shape"] = torch.tensor(ref_frames_shape_list, dtype=torch.int32)
        batch_inputs["traj_sim_frames_shape"] = torch.tensor(traj_sim_frames_shape_list, dtype=torch.int32)
        batch_inputs["traj_diff_frames_shape"] = torch.tensor(traj_diff_frames_shape_list, dtype=torch.int32)

        max_length_ref = batch_inputs["target_progress_ref"].shape[-1]
        max_length_sim = batch_inputs["target_progress_sim"].shape[-1]
        max_length_diff = batch_inputs["target_progress_diff"].shape[-1]
        batch_inputs["padding_mask_ref"] = create_padding_mask(batch_inputs["ref_frames_shape"], max_length_ref)
        batch_inputs["padding_mask_sim"] = create_padding_mask(batch_inputs["traj_sim_frames_shape"], max_length_sim)
        batch_inputs["padding_mask_diff"] = create_padding_mask(batch_inputs["traj_diff_frames_shape"], max_length_diff)
        batch_inputs["resample_attempts"] = [sample.resample_attempts for sample in similarity_samples]

        return batch_inputs
