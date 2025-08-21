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

from rfm.data.batch_collator import BaseSample, PreferenceSample, SimilaritySample

@dataclass
class ProgressSample:
    """Sample structure for progress evaluation."""
    frames: Optional[Union[List[str], np.ndarray]] = None
    frames_shape: Optional[tuple] = None
    task: Optional[str] = None
    target_progress: Optional[List[float]] = None
    quality_label: Optional[str] = None
    sample_type: str = "progress"


class VQABatchCollator:
    """Batch collator that processes Sample objects through the processor for VQA-based reward modeling."""

    def __init__(
        self,
        processor: AutoProcessor,
        max_length: int = 1024,
        resized_height: int = 128,
        resized_width: int = 128,
    ):
        """
        Initialize the VQA batch collator.

        Args:
            processor: HuggingFace processor for text and vision processing
            max_length: Maximum sequence length for text
            resized_height: Height to resize images/videos to (default: 128)
            resized_width: Width to resize images/videos to (default: 128)
        """
        self.processor = processor
        self.max_length = max_length
        self.resized_height = resized_height
        self.resized_width = resized_width

    def _pad_target_progress(self, progress_list):
        """Helper function to pad target progress sequences to max length."""
        if not progress_list:
            return None

        max_length = max(len(progress) for progress in progress_list)
        padded_list = []
        for progress in progress_list:
            if len(progress) < max_length:
                # Pad with zeros at the end
                padded_progress = progress + [0.0] * (max_length - len(progress))
            else:
                padded_progress = progress
            padded_list.append(padded_progress)
        return torch.tensor(padded_list, dtype=torch.float32)

    def _convert_frames_to_pil_images(self, frames, frames_shape=None):
        """Convert frames to PIL images if they are numpy arrays or serialized bytes."""
        if frames is None:
            return None

        # If frames are already paths (strings), return as is
        if isinstance(frames, str) or (isinstance(frames, list) and all(isinstance(f, str) for f in frames)):
            return frames

        # If frames are serialized bytes, deserialize first
        if isinstance(frames, bytes):
            # Deserialize bytes to numpy array (TxHxWxC) using provided shape
            if frames_shape is not None:
                # Convert to tuple if it's a list
                if isinstance(frames_shape, list):
                    frames_shape = tuple(frames_shape)
                try:
                    frames = np.frombuffer(frames, dtype=np.uint8).reshape(frames_shape)
                except Exception as e:
                    print(f"Warning: Failed to reshape with provided shape {frames_shape}: {e}")
                    # Fall back to 1D array
                    frames = np.frombuffer(frames, dtype=np.uint8)
            else:
                # No shape provided, try to infer
                frames = np.frombuffer(frames, dtype=np.uint8)

        # If frames are numpy array (TxHxWxC), convert to list of PIL images
        if isinstance(frames, np.ndarray):
            from PIL import Image

            pil_images = []

            # Handle different array shapes
            if len(frames.shape) == 4:  # TxHxWxC
                for i in range(frames.shape[0]):  # Iterate over time dimension
                    frame = frames[i]  # HxWxC
                    # Convert to PIL Image (already in HxWxC format)
                    pil_image = Image.fromarray(frame.astype(np.uint8))
                    pil_images.append(pil_image)
            elif len(frames.shape) == 3:  # HxWxC (single frame)
                pil_image = Image.fromarray(frames.astype(np.uint8))
                pil_images.append(pil_image)
            else:
                # Try to reshape as 1D array (backward compatibility)
                print(f"Warning: Unexpected frames shape {frames.shape}, treating as 1D array")
                return frames

            return pil_images

        # If frames are list of numpy arrays, convert each to PIL
        if isinstance(frames, list) and all(isinstance(f, np.ndarray) for f in frames):
            from PIL import Image

            pil_images = []
            for frame in frames:
                # Convert to PIL Image (assuming HxWxC format)
                pil_image = Image.fromarray(frame.astype(np.uint8))
                pil_images.append(pil_image)
            return pil_images

        return frames

    def __call__(
        self,
        samples: Union[List[BaseSample], List[PreferenceSample], List[SimilaritySample], List[ProgressSample], List[dict]],
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a list of samples into separate batches for preferences, progress, and similarities.
        For VQA-based reward modeling, everything goes through language generation.

        Args:
            samples: List of Sample objects or dictionaries that can be converted to Sample objects

        Returns:
            Dictionary containing separate batches for preferences, progress, and similarities
        """
        # Convert dictionaries to Sample objects if needed
        sample_objects = []
        for sample in samples:
            if isinstance(sample, dict):
                # Convert dict to appropriate Sample object based on sample_type
                sample_type = sample.get("sample_type", "unknown")
                if sample_type == "preference":
                    sample_obj = PreferenceSample(**sample)
                elif sample_type == "similarity":
                    sample_obj = SimilaritySample(**sample)
                elif sample_type == "progress":
                    sample_obj = ProgressSample(**sample)
                else:
                    raise ValueError(
                        f"Unknown sample_type: {sample_type}. Must be 'preference', 'similarity', or 'progress'"
                    )
                sample_objects.append(sample_obj)
            elif isinstance(sample, (BaseSample, PreferenceSample, SimilaritySample, ProgressSample)):
                sample_objects.append(sample)
            else:
                raise ValueError(f"Expected Sample object or dict, got {type(sample)}")

        # Separate samples by sample type
        preference_samples = [s for s in sample_objects if s.sample_type == "preference"]
        similarity_samples = [s for s in sample_objects if s.sample_type == "similarity"]
        progress_samples = [s for s in sample_objects if s.sample_type == "progress"]

        # Process preferences
        preference_inputs = {}
        if preference_samples:
            preference_inputs = self._process_preference_batch(preference_samples)

        # Process similarities
        similarity_inputs = {}
        if similarity_samples:
            similarity_inputs = self._process_similarity_batch(similarity_samples)

        # Process progress
        progress_inputs = {}
        if progress_samples:
            progress_inputs = self._process_progress_batch(progress_samples)

        # Return all batches
        return {
            "preference_inputs": preference_inputs,
            "similarity_inputs": similarity_inputs,
            "progress_inputs": progress_inputs,
            "num_preferences": len(preference_samples),
            "num_similarities": len(similarity_samples),
            "num_progress": len(progress_samples),
        }

    def _process_preference_batch(self, preference_samples: List[PreferenceSample]) -> Dict[str, torch.Tensor]:
        """Process a batch of preference samples with VQA-style question."""
        # Collect all messages for batch processing
        all_messages = []

        # Randomly decide whether chosen trajectory goes first or second
        preference_labels = np.random.randint(0, 2, len(preference_samples))

        for i, sample in enumerate(preference_samples):
            # Convert frames to appropriate format using stored shapes
            chosen_frames = self._convert_frames_to_pil_images(sample.chosen_frames, sample.chosen_frames_shape)
            rejected_frames = self._convert_frames_to_pil_images(sample.rejected_frames, sample.rejected_frames_shape)

            if preference_labels[i] == 1.0:
                # Chosen trajectory first: Trajectory A (chosen) + Trajectory B (rejected)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Given these two trajectories for the task '{sample.chosen_task}', which one do you prefer? Trajectory A or B?"},
                            {
                                "type": "video",
                                "video": chosen_frames,
                                "resized_height": self.resized_height,
                                "resized_width": self.resized_width,
                            },
                            {"type": "text", "text": "Trajectory A"},
                            {
                                "type": "video",
                                "video": rejected_frames,
                                "resized_height": self.resized_height,
                                "resized_width": self.resized_width,
                            },
                            {"type": "text", "text": "Trajectory B"},
                        ],
                    }
                ]
            else:
                # Chosen trajectory second: Trajectory A (rejected) + Trajectory B (chosen)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Given these two trajectories for the task '{sample.chosen_task}', which one do you prefer? Trajectory A or B?"},
                            {
                                "type": "video",
                                "video": rejected_frames,
                                "resized_height": self.resized_height,
                                "resized_width": self.resized_width,
                            },
                            {"type": "text", "text": "Trajectory A"},
                            {
                                "type": "video",
                                "video": chosen_frames,
                                "resized_height": self.resized_height,
                                "resized_width": self.resized_width,
                            },
                            {"type": "text", "text": "Trajectory B"},
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
        batch_inputs["sample_type"] = ["preference"] * len(preference_samples)
        # Use the dynamically generated preference labels based on trajectory order
        batch_inputs["preference_labels"] = torch.tensor(preference_labels, dtype=torch.float32)
        # Convert preference labels to text
        preference_labels_text = ["I prefer trajectory A" if label == 0 else "I prefer trajectory B" for label in preference_labels]
        batch_inputs["labels"] = self.processor.tokenizer(preference_labels_text, return_tensors="pt", padding=True, truncation=True).input_ids

        # Add target progress for both trajectories based on conversation order
        target_progress_A_list = []
        target_progress_B_list = []

        for i, sample in enumerate(preference_samples):
            # Get the preference label to determine which trajectory went first
            if preference_labels[i] == 1.0:
                # First trajectory is chosen (chosen_frames), second is rejected (rejected_frames)
                if sample.target_progress_A is not None:
                    target_progress_A_list.append(sample.target_progress_A)  # chosen progress
                if sample.target_progress_B is not None:
                    target_progress_B_list.append(sample.target_progress_B)  # rejected progress
            else:
                # First trajectory is rejected (rejected_frames), second is chosen (chosen_frames)
                if sample.target_progress_B is not None:
                    target_progress_A_list.append(sample.target_progress_B)  # rejected progress (now first)
                if sample.target_progress_A is not None:
                    target_progress_B_list.append(sample.target_progress_A)  # chosen progress (now second)

        # Pad target progress tensors to max length in last dimension
        batch_inputs["target_progress_A"] = self._pad_target_progress(target_progress_A_list)
        batch_inputs["target_progress_B"] = self._pad_target_progress(target_progress_B_list)

        # Also add the frame_shapes
        batch_inputs["chosen_frames_shape"] = torch.tensor(
            [sample.chosen_frames_shape for sample in preference_samples], dtype=torch.int32
        )
        batch_inputs["rejected_frames_shape"] = torch.tensor(
            [sample.rejected_frames_shape for sample in preference_samples], dtype=torch.int32
        )

        return batch_inputs

    def _process_similarity_batch(self, similarity_samples: List[SimilaritySample]) -> Dict[str, torch.Tensor]:
        """Process a batch of similarity samples with VQA-style question."""
        # Collect all messages for batch processing
        all_messages = []

        for sample in similarity_samples:
            # Convert frames to appropriate format using stored shapes
            reference_frames = self._convert_frames_to_pil_images(
                sample.reference_frames, sample.reference_frames_shape
            )
            traj_sim_frames = self._convert_frames_to_pil_images(sample.traj_sim_frames, sample.traj_sim_frames_shape)
            traj_diff_frames = self._convert_frames_to_pil_images(
                sample.traj_diff_frames, sample.traj_diff_frames_shape
            )

            # Create conversation for similarity comparison
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Given the following reference trajectory for the task '{sample.task_ref}', which one of the two trajectories are more similar to it? Trajectory A or B?"},
                        {
                            "type": "video",
                            "video": reference_frames,
                            "resized_height": self.resized_height,
                            "resized_width": self.resized_width,
                        },
                        {"type": "text", "text": "Reference Trajectory"},
                        {
                            "type": "video",
                            "video": traj_sim_frames,
                            "resized_height": self.resized_height,
                            "resized_width": self.resized_width,
                        },
                        {"type": "text", "text": "Trajectory A"},
                        {
                            "type": "video",
                            "video": traj_diff_frames,
                            "resized_height": self.resized_height,
                            "resized_width": self.resized_width,
                        },
                        {"type": "text", "text": "Trajectory B"},
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
            if sample.target_progress_ref is not None:
                target_progress_ref_list.append(sample.target_progress_ref)
            if sample.target_progress_sim is not None:
                target_progress_sim_list.append(sample.target_progress_sim)
            if sample.target_progress_diff is not None:
                target_progress_diff_list.append(sample.target_progress_diff)

        # Pad target progress tensors to max length in last dimension
        batch_inputs["target_progress_ref"] = self._pad_target_progress(target_progress_ref_list)
        batch_inputs["target_progress_sim"] = self._pad_target_progress(target_progress_sim_list)
        batch_inputs["target_progress_diff"] = self._pad_target_progress(target_progress_diff_list)

        # Also add the frame_shapes
        batch_inputs["ref_frames_shape"] = torch.tensor(
            [sample.reference_frames_shape for sample in similarity_samples], dtype=torch.int32
        )
        batch_inputs["traj_sim_frames_shape"] = torch.tensor(
            [sample.traj_sim_frames_shape for sample in similarity_samples], dtype=torch.int32
        )
        batch_inputs["traj_diff_frames_shape"] = torch.tensor(
            [sample.traj_diff_frames_shape for sample in similarity_samples], dtype=torch.int32
        )
        return batch_inputs

    def _process_progress_batch(self, progress_samples: List[ProgressSample]) -> Dict[str, torch.Tensor]:
        """Process a batch of progress samples with VQA-style question."""
        # Collect all messages for batch processing
        all_messages = []

        for sample in progress_samples:
            # Convert frames to appropriate format using stored shapes
            frames = self._convert_frames_to_pil_images(sample.frames, sample.frames_shape)

            # Create conversation for progress evaluation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"How close is the following trajectory in terms of the task progress for '{sample.task}'? Give a number between 0 and 1 where 0 means no progress and 1 means successful completion of the task."},
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
        batch_inputs["sample_type"] = ["progress"] * len(progress_samples)

        # Add target progress and quality labels
        target_progress_list = []
        quality_labels = []

        for sample in progress_samples:
            if sample.target_progress is not None:
                target_progress_list.append(sample.target_progress)
            quality_labels.append(1.0 if sample.quality_label == 'successful' else 0.0)

        # Pad target progress tensors to max length in last dimension
        batch_inputs["target_progress"] = self._pad_target_progress(target_progress_list)
        # Convert progress labels to text
        progress_labels_text = [f"The task is {int(progress * 100)}% complete" for progress in target_progress_list]
        batch_inputs["labels"] = self.processor.tokenizer(progress_labels_text, return_tensors="pt", padding=True, truncation=True).input_ids
        batch_inputs["quality_labels"] = torch.tensor(quality_labels, dtype=torch.float32)

        return batch_inputs

    def collate_fn(self, batch: List[Union[BaseSample, PreferenceSample, SimilaritySample, ProgressSample]]) -> Dict[str, torch.Tensor]:
        """
        Alternative method name for compatibility with PyTorch DataLoader.

        Args:
            batch: List of Sample objects

        Returns:
            Dictionary containing the processed tensors
        """
        return self(batch)
