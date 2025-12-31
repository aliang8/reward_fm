#!/usr/bin/env python3
"""
Batch collator for processing list of samples.
"""

import tempfile
import uuid
from pathlib import Path
import numpy as np
import torch
from qwen_vl_utils import process_vision_info

from .base import BaseCollator
from .utils import convert_frames_to_pil_images, pad_list_to_max, write_mp4
from rfm.data.dataset_types import PreferenceSample, ProgressSample, SimilaritySample
from rfm.data.dataset_category import is_preference_only_ds
from rfm.data.datasets.helpers import DataGenStrat
from typing import List, Dict, Union


def should_compute_progress(quality_label: str, data_gen_strategy: str, data_source: str = None) -> float:
    """
    Check if progress should be computed for a trajectory.

    Includes if it is successful or rewound
    but NOT suboptimal or failure. Also masks out progress if data_source is in preference_only category.

    Args:
        quality_label: The quality label of the trajectory
        data_gen_strategy: The data generation strategy
        data_source: The data source name (optional)

    Returns:
        1.0 if progress should be computed, 0.0 otherwise
    """
    # Mask out progress if data_source is in preference_only category
    if data_source is not None and is_preference_only_ds(data_source):
        return 0.0

    if quality_label in ["suboptimal", "failure", "failed"]:
        return 0.0

    if quality_label == "successful" or data_gen_strategy == DataGenStrat.REWIND.value:
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
        use_multi_image: bool = False,
        prog_pref: bool = False,
        prog_sim: bool = False,
        use_progress_token: bool = False,
        shuffle_progress_frames: bool = False,
        inference: bool = False,
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
        self.use_multi_image = use_multi_image
        self.prog_pref = prog_pref
        self.prog_sim = prog_sim

        # Molmo2 only supports multi-image mode, not video
        if "Molmo" in self.base_model_id and not self.use_multi_image:
            raise ValueError(
                "Molmo2 does not support video mode (use_multi_image=False). "
                "Please set data.use_multi_image=True to use Molmo2 with multi-image input."
            )
        self.use_progress_token = use_progress_token
        self.shuffle_progress_frames = shuffle_progress_frames
        self.inference = inference

    def _prepare_frames_for_conversation(self, frames: List, prefix: str = "tmp") -> tuple[Union[List, str], dict]:
        """
        Prepare frames for conversation based on use_multi_image flag.

        Args:
            frames: List of PIL Images
            prefix: Prefix for temporary video file (if needed)

        Returns:
            tuple: (video_field, content_extras)
                - video_field: Either list of PIL Images (if use_multi_image) or video file path (str)
                - content_extras: Dictionary with resized_height/width or empty dict
        """
        if self.use_multi_image:
            # Use images directly - return list of PIL Images
            return frames, {
                "resized_height": self.resized_height,
                "resized_width": self.resized_width,
            }
        elif "Qwen" in self.base_model_id or "Molmo" in self.base_model_id:
            # Qwen and Molmo accept list of PIL Images directly
            return frames, {
                "resized_height": self.resized_height,
                "resized_width": self.resized_width,
            }
        elif "SmolVLM" in self.base_model_id:
            # Convert to video file for SmolVLM
            unique_id = uuid.uuid4().hex
            tmp = Path(tempfile.gettempdir()) / f"{prefix}_{unique_id}.mp4"
            write_mp4(frames, tmp, fps=1)
            return str(tmp), {}
        else:
            # Default: return frames as-is
            return frames, {}

    def _add_vision_content_to_list(
        self, content_list: List[Dict], frames_or_video: Union[List, str], content_extras: dict
    ) -> None:
        """
        Add vision content (images or video) to a content list.

        Args:
            content_list: List to append vision content to
            frames_or_video: Either list of PIL Images (if use_multi_image) or video file path (str)
            content_extras: Dictionary with additional content parameters
        """
        if self.use_multi_image:
            # Add each image as a separate entry
            for img in frames_or_video:
                content_list.append({
                    "type": "image",
                    "image": img,
                    **content_extras,
                })
        else:
            # Add video entry
            content_list.append({
                "type": "video",
                "video": frames_or_video,
                **content_extras,
            })

    def _process_conversation(
        self, conversations: List[List[Dict]], add_generation_prompt: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Process a list of conversations into a batch of inputs.

        Args:
            conversations: List of conversations
            add_generation_prompt: Whether to add generation prompt (used for VQA inference otherwise conversation will be closed with </im_end> token)
        Returns:
            Batch of inputs
        """
        if "Qwen" in self.base_model_id or "Molmo" in self.base_model_id:
            # Process all messages in one batch
            texts = [
                self.processor.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    add_vision_id=True,
                    enable_thinking=False,
                    fps=1,
                )
                for msg in conversations
            ]

            is_qwen3 = "Qwen3" in self.base_model_id or "Molmo2" in self.base_model_id

            # For Qwen3, pass image_patch_size to process_vision_info
            process_kwargs = {
                "return_video_kwargs": True,
                "return_video_metadata": is_qwen3,
            }
            if (
                is_qwen3
                and hasattr(self.processor, "image_processor")
                and hasattr(self.processor.image_processor, "patch_size")
            ):
                process_kwargs["image_patch_size"] = self.processor.image_processor.patch_size

            image_inputs, video_inputs, video_kwargs = process_vision_info(conversations, **process_kwargs)

            # For Qwen3, video_inputs is a list of (video, video_metadata) tuples
            # that need to be split before passing to processor
            if is_qwen3 and video_inputs is not None and len(video_inputs) > 0:
                # Check if video_inputs contains tuples (Qwen3 format) or is already split
                if isinstance(video_inputs[0], tuple) and len(video_inputs[0]) == 2:
                    videos, video_metadatas = zip(*video_inputs)
                    videos, video_metadatas = list(videos), list(video_metadatas)
                else:
                    # Already in the correct format
                    videos = video_inputs
                    video_metadatas = None
            else:
                videos = video_inputs if video_inputs else None
                video_metadatas = None

            # Process through the processor in one batch
            processor_kwargs = {
                "text": texts,
                "images": image_inputs,
                "padding": True,
                "truncation": False,
                "max_length": self.max_length,
                "return_tensors": "pt",
                "do_resize": False,
            }

            # Only add videos if they exist
            if videos is not None:
                processor_kwargs["videos"] = videos

            # Add video_metadata and video_kwargs for Qwen3
            # Note: video_kwargs may contain keys like 'videos_kwargs' that need special handling
            if is_qwen3:
                if video_metadatas is not None:
                    processor_kwargs["video_metadata"] = video_metadatas
                # Pass video_kwargs - these contain important metadata for video processing
                # The processor will handle 'videos_kwargs' internally if present
                if video_kwargs:
                    processor_kwargs.update(video_kwargs)

            batch_inputs = self.processor(**processor_kwargs)
        elif "SmolVLM" in self.base_model_id:
            batch_inputs = self.processor.apply_chat_template(
                conversations,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                padding=True,
                truncation=False,
                max_length=self.max_length,
                return_dict=True,
                return_tensors="pt",
                fps=1,  # this should be same as fps for write_mp4
            )
        else:
            raise ValueError(f"Invalid base model id: {self.base_model_id}")

        return batch_inputs

    def _process_progress_batch(self, progress_samples: list[ProgressSample]) -> dict[str, torch.Tensor]:
        """Process a batch of progress samples with VQA-style question."""
        # Collect all messages for batch processing
        all_messages = []

        target_progress_overrides: list[list[float] | None] = []

        for sample in progress_samples:
            # Convert frames to appropriate format using stored shapes
            frames = convert_frames_to_pil_images(sample.trajectory.frames, sample.trajectory.frames_shape)
            target_progress = sample.trajectory.target_progress

            # Optionally shuffle frames (except the first frame) and keep target progress aligned
            if self.shuffle_progress_frames and target_progress is not None and not self.inference:
                if len(target_progress) > 1 and len(target_progress) == len(frames):
                    shuffle_indices = np.random.permutation(range(1, len(frames)))
                    frames = [frames[0]] + [frames[idx] for idx in shuffle_indices]
                    target_progress = [target_progress[0]] + [target_progress[idx] for idx in shuffle_indices]
                else:
                    raise ValueError(
                        "Target progress must be a list with at least 1 element and match the number of frames "
                        f"for shuffling, got {len(target_progress)} entries for {len(frames)} frames"
                    )

            target_progress_overrides.append(target_progress)

            video_field, content_extras = self._prepare_frames_for_conversation(frames, prefix="tmp_progress")

            # Create conversation for progress evaluation
            prompt = f"The task for the robot is '{sample.trajectory.task}'. Given the trajectory video, predict the task progress at each frame, how far along the robot is towards completing the task, a float between 0 and 1, where 0 is the starting state and 1 is when the task is completed. If the robot is not performing the same task, predict 0 progress."

            # Build content list
            content_list = [{"type": "text", "text": prompt}]
            self._add_vision_content_to_list(content_list, video_field, content_extras)

            # Add progress and success tokens if use_progress_token is enabled
            # For progress samples, we only use prog_token_A and succ_token_A (single trajectory)
            if self.use_progress_token:
                content_list.append({"type": "text", "text": "<|prog_token_A|>"})
                content_list.append({"type": "text", "text": "<|succ_token_A|>"})

            conversation = [
                {
                    "role": "user",
                    "content": content_list,
                }
            ]

            all_messages.append(conversation)

        batch_inputs = self._process_conversation(all_messages)
        all_have_target_progress = all(tp is not None for tp in target_progress_overrides)
        if all_have_target_progress:
            batch_inputs = self._add_progress_meta(
                batch_inputs,
                progress_samples,
                target_progress_override=target_progress_overrides,
            )
        batch_inputs["resample_attempts"] = [sample.resample_attempts for sample in progress_samples]
        return batch_inputs

    def _add_progress_meta(
        self,
        batch_inputs: dict[str, torch.Tensor],
        progress_samples: list[ProgressSample],
        target_progress_override: list[list[float] | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_inputs["sample_type"] = ["progress"] * len(progress_samples)
        batch_inputs["task"] = [sample.trajectory.task for sample in progress_samples]
        batch_inputs["metadata"] = [sample.trajectory.metadata for sample in progress_samples]

        # Pad target progress tensors to max length in last dimension
        if target_progress_override is not None:
            target_progress_list = target_progress_override
        else:
            target_progress_list = [sample.trajectory.target_progress for sample in progress_samples]
        batch_inputs["target_progress"] = pad_list_to_max(target_progress_list)
        batch_inputs["quality_labels"] = [sample.trajectory.quality_label for sample in progress_samples]

        frames_shape_list = [sample.trajectory.frames_shape for sample in progress_samples]
        batch_inputs["frames_shape"] = torch.tensor(frames_shape_list, dtype=torch.int32)

        max_length = batch_inputs["target_progress"].shape[-1]
        batch_inputs["padding_mask"] = create_padding_mask(batch_inputs["frames_shape"], max_length)

        batch_inputs["data_source"] = [sample.trajectory.data_source for sample in progress_samples]
        batch_inputs["partial_success"] = [sample.trajectory.partial_success for sample in progress_samples]
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

        success_label_list = [sample.trajectory.success_label for sample in progress_samples]
        batch_inputs["success_labels"] = pad_list_to_max(success_label_list)

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

            chosen_video_field, content_extras = self._prepare_frames_for_conversation(
                chosen_frames, prefix="tmp_chosen"
            )
            rejected_video_field, _ = self._prepare_frames_for_conversation(rejected_frames, prefix="tmp_rejected")

            prompt = f"Given these two trajectories for the task '{sample.chosen_trajectory.task}', evaluate which one makes more progress towards the task. Return A for the first trajectory and B for the second trajectory."

            if self.prog_pref:
                # We ask the model to predict both of the task progress and preference
                task_prompt = f" Also predict the task progress at each frame of the first trajectory, how far along the robot is towards completing the task, a float between 0 and 1, where 0 is the starting state and 1 is when the task is completed. If the robot is not performing the same task, predict 0 progress."
                prompt += task_prompt

            # Determine which trajectory is A and which is B based on preference label
            if preference_labels[i] == 1.0:
                # Chosen trajectory first: task + video A (chosen) + <|split_token|> + video B (rejected) + <|pref_token|>
                traj_a_field = chosen_video_field
                traj_b_field = rejected_video_field
            else:
                # Chosen trajectory second: task + video A (rejected) + <|split_token|> + video B (chosen) + <|pref_token|>
                traj_a_field = rejected_video_field
                traj_b_field = chosen_video_field

            # Build content list
            content_list = [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "This is Trajectory A. "},
            ]
            self._add_vision_content_to_list(content_list, traj_a_field, content_extras)
            # Add progress and success tokens for trajectory A if use_progress_token is enabled
            if self.use_progress_token:
                content_list.append({"type": "text", "text": "<|prog_token_A|>"})
                content_list.append({"type": "text", "text": "<|succ_token_A|>"})

            content_list.extend([
                {"type": "text", "text": "<|split_token|>"},
                {"type": "text", "text": "This is Trajectory B. "},
            ])
            self._add_vision_content_to_list(content_list, traj_b_field, content_extras)

            # Add progress and success tokens for trajectory B if use_progress_token is enabled
            if self.use_progress_token:
                content_list.append({"type": "text", "text": "<|prog_token_B|>"})
                content_list.append({"type": "text", "text": "<|succ_token_B|>"})

            content_list.append({"type": "text", "text": "<|pref_token|>"})

            conversation = [
                {
                    "role": "user",
                    "content": content_list,
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

        batch_inputs["target_progress_A"] = pad_list_to_max(target_progress_A)
        batch_inputs["target_progress_B"] = pad_list_to_max(target_progress_B)
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

        batch_inputs["chosen_data_gen_strategy"] = [DataGenStrat.FORWARD_PROGRESS.value] * len(preference_samples)
        batch_inputs["rejected_data_gen_strategy"] = [sample.data_gen_strategy for sample in preference_samples]
        batch_inputs["chosen_quality_label"] = [sample.chosen_trajectory.quality_label for sample in preference_samples]

        target_progress_chosen = [sample.chosen_trajectory.target_progress for sample in preference_samples]
        target_progress_rejected = [sample.rejected_trajectory.target_progress for sample in preference_samples]
        target_progress_chosen_mask = [
            should_compute_progress(
                sample.chosen_trajectory.quality_label,
                DataGenStrat.FORWARD_PROGRESS.value,
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
        batch_inputs["target_progress_chosen"] = pad_list_to_max(target_progress_chosen)
        batch_inputs["target_progress_rejected"] = pad_list_to_max(target_progress_rejected)
        batch_inputs["target_progress_chosen_mask"] = torch.tensor(target_progress_chosen_mask, dtype=torch.float32)
        batch_inputs["target_progress_rejected_mask"] = torch.tensor(target_progress_rejected_mask, dtype=torch.float32)

        batch_inputs["chosen_frames_shape"] = torch.tensor(
            [sample.chosen_trajectory.frames_shape for sample in preference_samples], dtype=torch.int32
        )
        batch_inputs["rejected_frames_shape"] = torch.tensor(
            [sample.rejected_trajectory.frames_shape for sample in preference_samples], dtype=torch.int32
        )
        batch_inputs["resample_attempts"] = [sample.resample_attempts for sample in preference_samples]

        # Aggregate success labels for trajectory A from trajectories
        success_label_A_list = [traj.success_label for traj in trajectory_A_list]
        batch_inputs["success_labels_A"] = pad_list_to_max(success_label_A_list)

        # Add metadata structure for evaluation
        metadata_list = []
        for sample in preference_samples:
            metadata = {
                "chosen_metadata": {
                    "quality_label": sample.chosen_trajectory.quality_label,
                    "data_gen_strategy": "subsample_task",
                    "id": sample.chosen_trajectory.id,
                    "video_path": sample.chosen_trajectory.metadata.get("video_path")
                    if sample.chosen_trajectory.metadata
                    else None,
                },
                "rejected_metadata": {
                    "quality_label": sample.rejected_trajectory.quality_label,
                    "data_gen_strategy": sample.data_gen_strategy,
                    "id": sample.rejected_trajectory.id,
                    "video_path": sample.rejected_trajectory.metadata.get("video_path")
                    if sample.rejected_trajectory.metadata
                    else None,
                },
            }
            metadata_list.append(metadata)
        batch_inputs["metadata"] = metadata_list

        return batch_inputs

    def _process_similarity_batch(self, similarity_samples: list[SimilaritySample]) -> dict[str, torch.Tensor]:
        """Process a batch of similarity samples."""
        # Collect all messages for batch processing (ref_sim and ref_diff for each sample)
        all_messages = []

        # Randomly decide order for each comparison (ref first or sim/diff first)
        # Store which trajectory is A (first) for each comparison
        ref_sim_order = []  # True if ref is first (A), False if sim is first (A)
        ref_diff_order = []  # True if ref is first (A), False if diff is first (A)

        for sample in similarity_samples:
            # Convert frames to appropriate format using stored shapes
            reference_frames = convert_frames_to_pil_images(
                sample.ref_trajectory.frames, sample.ref_trajectory.frames_shape
            )
            sim_frames = convert_frames_to_pil_images(sample.sim_trajectory.frames, sample.sim_trajectory.frames_shape)
            diff_frames = convert_frames_to_pil_images(
                sample.diff_trajectory.frames, sample.diff_trajectory.frames_shape
            )

            # Prepare frames for conversation (handles multi-image vs video conversion)
            ref_video, content_extras = self._prepare_frames_for_conversation(reference_frames, prefix="tmp_ref")
            sim_video, _ = self._prepare_frames_for_conversation(sim_frames, prefix="tmp_sim")
            diff_video, _ = self._prepare_frames_for_conversation(diff_frames, prefix="tmp_diff")

            # Randomly decide order for ref_sim comparison
            ref_sim_ref_first = np.random.randint(0, 2) == 0
            ref_sim_order.append(ref_sim_ref_first)

            # Process reference vs trajectory sim
            prompt_sim = f"For the task '{sample.ref_trajectory.task}', compare these two trajectories and evaluate how similar they are in terms of task completion and behavior."
            content_list_sim = [
                {"type": "text", "text": prompt_sim},
            ]

            if ref_sim_ref_first:
                # Ref is first (A), sim is second (B)
                content_list_sim.append({"type": "text", "text": "This is the first trajectory. "})
                self._add_vision_content_to_list(content_list_sim, ref_video, content_extras)
                if self.use_progress_token:
                    content_list_sim.append({"type": "text", "text": "<|prog_token_A|>"})
                    content_list_sim.append({"type": "text", "text": "<|succ_token_A|>"})
                content_list_sim.extend([
                    {"type": "text", "text": "<|split_token|>"},
                    {"type": "text", "text": "This is the second trajectory. "},
                ])
                self._add_vision_content_to_list(content_list_sim, sim_video, content_extras)
                if self.use_progress_token:
                    content_list_sim.append({"type": "text", "text": "<|prog_token_B|>"})
                    content_list_sim.append({"type": "text", "text": "<|succ_token_B|>"})
            else:
                # Sim is first (A), ref is second (B)
                content_list_sim.append({"type": "text", "text": "This is the first trajectory. "})
                self._add_vision_content_to_list(content_list_sim, sim_video, content_extras)
                if self.use_progress_token:
                    content_list_sim.append({"type": "text", "text": "<|prog_token_A|>"})
                    content_list_sim.append({"type": "text", "text": "<|succ_token_A|>"})
                content_list_sim.extend([
                    {"type": "text", "text": "<|split_token|>"},
                    {"type": "text", "text": "This is the second trajectory. "},
                ])
                self._add_vision_content_to_list(content_list_sim, ref_video, content_extras)
                if self.use_progress_token:
                    content_list_sim.append({"type": "text", "text": "<|prog_token_B|>"})
                    content_list_sim.append({"type": "text", "text": "<|succ_token_B|>"})
            content_list_sim.append({"type": "text", "text": "<|sim_token|>"})

            conversation_ref_sim = [
                {
                    "role": "user",
                    "content": content_list_sim,
                }
            ]

            # Randomly decide order for ref_diff comparison
            ref_diff_ref_first = np.random.randint(0, 2) == 0
            ref_diff_order.append(ref_diff_ref_first)

            # Process reference vs trajectory diff
            prompt_diff = f"For the task '{sample.ref_trajectory.task}', compare these two trajectories and evaluate how similar they are in terms of task completion and behavior."
            content_list_diff = [
                {"type": "text", "text": prompt_diff},
            ]

            if ref_diff_ref_first:
                # Ref is first (A), diff is second (B)
                content_list_diff.append({"type": "text", "text": "This is the first trajectory. "})
                self._add_vision_content_to_list(content_list_diff, ref_video, content_extras)
                if self.use_progress_token:
                    content_list_diff.append({"type": "text", "text": "<|prog_token_A|>"})
                    content_list_diff.append({"type": "text", "text": "<|succ_token_A|>"})
                content_list_diff.extend([
                    {"type": "text", "text": "<|split_token|>"},
                    {"type": "text", "text": "This is the second trajectory. "},
                ])
                self._add_vision_content_to_list(content_list_diff, diff_video, content_extras)
                if self.use_progress_token:
                    content_list_diff.append({"type": "text", "text": "<|prog_token_B|>"})
                    content_list_diff.append({"type": "text", "text": "<|succ_token_B|>"})
            else:
                # Diff is first (A), ref is second (B)
                content_list_diff.append({"type": "text", "text": "This is the first trajectory. "})
                self._add_vision_content_to_list(content_list_diff, diff_video, content_extras)
                if self.use_progress_token:
                    content_list_diff.append({"type": "text", "text": "<|prog_token_A|>"})
                    content_list_diff.append({"type": "text", "text": "<|succ_token_A|>"})
                content_list_diff.extend([
                    {"type": "text", "text": "<|split_token|>"},
                    {"type": "text", "text": "This is the second trajectory. "},
                ])
                self._add_vision_content_to_list(content_list_diff, ref_video, content_extras)
                if self.use_progress_token:
                    content_list_diff.append({"type": "text", "text": "<|prog_token_B|>"})
                    content_list_diff.append({"type": "text", "text": "<|succ_token_B|>"})
            content_list_diff.append({"type": "text", "text": "<|sim_token|>"})

            conversation_ref_diff = [
                {
                    "role": "user",
                    "content": content_list_diff,
                }
            ]

            all_messages.extend([conversation_ref_sim, conversation_ref_diff])

        # This creates a single batched input with [ref_sim_0, ref_diff_0, ref_sim_1, ref_diff_1, ...]
        batch_inputs = self._process_conversation(all_messages)

        # Keep the batched inputs as-is (don't split them)
        # The trainer will handle extracting ref_sim and ref_diff from the batched output
        num_samples = len(similarity_samples)
        combined_inputs = {"sample_type": ["similarity"] * num_samples}

        # The batch is structured as [ref_sim_0, ref_diff_0, ref_sim_1, ref_diff_1, ...]
        for key, value in batch_inputs.items():
            combined_inputs[key] = value

        # Add similarity-specific metadata with order information
        combined_inputs = self._add_similarity_meta(combined_inputs, similarity_samples, ref_sim_order, ref_diff_order)
        return combined_inputs

    def _add_similarity_meta(
        self,
        batch_inputs: dict[str, torch.Tensor],
        similarity_samples: list[SimilaritySample],
        ref_sim_order: list[bool],
        ref_diff_order: list[bool],
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

        batch_inputs["target_progress_ref"] = pad_list_to_max(target_progress_ref)
        batch_inputs["target_progress_sim"] = pad_list_to_max(target_progress_sim)
        batch_inputs["target_progress_diff"] = pad_list_to_max(target_progress_diff)

        batch_inputs["target_progress_ref_mask"] = torch.tensor(target_progress_ref_mask, dtype=torch.float32)
        batch_inputs["target_progress_sim_mask"] = torch.tensor(target_progress_sim_mask, dtype=torch.float32)
        batch_inputs["target_progress_diff_mask"] = torch.tensor(target_progress_diff_mask, dtype=torch.float32)

        # Compute target progress for trajectory A in each comparison
        # For ref_sim: A is ref if ref_sim_order[i] is True, otherwise A is sim
        target_progress_sim_A = []
        target_progress_sim_A_mask = []
        for i, sample in enumerate(similarity_samples):
            if ref_sim_order[i]:
                # Ref is A (first)
                target_progress_sim_A.append(sample.ref_trajectory.target_progress)
                target_progress_sim_A_mask.append(
                    should_compute_progress(
                        sample.ref_trajectory.quality_label,
                        "successful",
                        data_source=sample.ref_trajectory.data_source,
                    )
                )
            else:
                # Sim is A (first)
                target_progress_sim_A.append(sample.sim_trajectory.target_progress)
                target_progress_sim_A_mask.append(
                    should_compute_progress(
                        sample.sim_trajectory.quality_label,
                        sample.data_gen_strategy,
                        data_source=sample.sim_trajectory.data_source,
                    )
                )

        # For ref_diff: A is ref if ref_diff_order[i] is True, otherwise A is diff
        target_progress_diff_A = []
        target_progress_diff_A_mask = []
        for i, sample in enumerate(similarity_samples):
            if ref_diff_order[i]:
                # Ref is A (first)
                target_progress_diff_A.append(sample.ref_trajectory.target_progress)
                target_progress_diff_A_mask.append(
                    should_compute_progress(
                        sample.ref_trajectory.quality_label,
                        "successful",
                        data_source=sample.ref_trajectory.data_source,
                    )
                )
            else:
                # Diff is A (first)
                target_progress_diff_A.append(sample.diff_trajectory.target_progress)
                target_progress_diff_A_mask.append(
                    should_compute_progress(
                        sample.diff_trajectory.quality_label,
                        "different_task",
                        data_source=sample.diff_trajectory.data_source,
                    )
                )

        batch_inputs["target_progress_sim_A"] = pad_list_to_max(target_progress_sim_A)
        batch_inputs["target_progress_sim_A_mask"] = torch.tensor(target_progress_sim_A_mask, dtype=torch.float32)
        batch_inputs["target_progress_diff_A"] = pad_list_to_max(target_progress_diff_A)
        batch_inputs["target_progress_diff_A_mask"] = torch.tensor(target_progress_diff_A_mask, dtype=torch.float32)

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

        success_label_ref_list = [sample.ref_trajectory.success_label for sample in similarity_samples]
        success_label_sim_list = [sample.sim_trajectory.success_label for sample in similarity_samples]
        success_label_diff_list = [sample.diff_trajectory.success_label for sample in similarity_samples]
        batch_inputs["success_labels_ref"] = pad_list_to_max(success_label_ref_list)
        batch_inputs["success_labels_sim"] = pad_list_to_max(success_label_sim_list)
        batch_inputs["success_labels_diff"] = pad_list_to_max(success_label_diff_list)

        # Compute success labels for trajectory A in each comparison
        # For ref_sim: A is ref if ref_sim_order[i] is True, otherwise A is sim
        success_labels_sim_A = []
        for i, sample in enumerate(similarity_samples):
            if ref_sim_order[i]:
                # Ref is A (first)
                success_labels_sim_A.append(sample.ref_trajectory.success_label)
            else:
                # Sim is A (first)
                success_labels_sim_A.append(sample.sim_trajectory.success_label)

        # For ref_diff: A is ref if ref_diff_order[i] is True, otherwise A is diff
        success_labels_diff_A = []
        for i, sample in enumerate(similarity_samples):
            if ref_diff_order[i]:
                # Ref is A (first)
                success_labels_diff_A.append(sample.ref_trajectory.success_label)
            else:
                # Diff is A (first)
                success_labels_diff_A.append(sample.diff_trajectory.success_label)

        batch_inputs["success_labels_sim_A"] = pad_list_to_max(success_labels_sim_A)
        batch_inputs["success_labels_diff_A"] = pad_list_to_max(success_labels_diff_A)

        batch_inputs["metadata"] = [
            sample.ref_trajectory.metadata if sample.ref_trajectory.metadata else {} for sample in similarity_samples
        ]

        return batch_inputs
