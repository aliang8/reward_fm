#!/usr/bin/env python3
"""
Batch collator for processing Sample objects through the processor and returning processed tensors.
This collator handles the conversion from PreferenceSample and SimilaritySample objects to processed tensors.
"""

import torch
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, field
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np
import random


@dataclass
class BaseSample:
    """Base sample structure with common fields for all prediction types."""

    # Core HF dataset fields
    id: str
    task: str
    lang_vector: np.ndarray
    data_source: str
    quality_label: str
    is_robot: bool

    # Frame shape information for deserialization
    frames_shape: Optional[tuple] = None  # Shape of frames (T, H, W, C)

    # Progress fields
    target_progress_A: Optional[List[float]] = None  # Progress values for trajectory A
    target_progress_B: Optional[List[float]] = None  # Progress values for trajectory B
    sample_type: Optional[str] = None  # how this sample was generated
    num_frames_rewound: Optional[int] = None  # number of frames rewound (for rewound trajectories)

    data_gen_strategy: Optional[str] = None  # how this sample was generated (e.g. rewinding, random, etc.)


@dataclass
class PreferenceSample:
    """Sample structure for preference prediction: chosen vs rejected where chosen is preferred."""

    # Preference-specific fields using chosen/rejected naming

    # chosen metadata
    chosen_frames: Optional[Union[List[str], np.ndarray]] = None
    chosen_frames_shape: Optional[tuple] = None
    chosen_id: Optional[str] = None
    chosen_task: Optional[str] = None
    chosen_lang_vector: Optional[np.ndarray] = None
    chosen_data_source: Optional[str] = None
    chosen_quality_label: Optional[str] = None
    chosen_is_robot: Optional[bool] = None

    # rejected metadata
    rejected_frames: Optional[Union[List[str], np.ndarray]] = None
    rejected_frames_shape: Optional[tuple] = None  # Shape of rejected trajectory frames
    preferred_trajectory: Optional[str] = None  # "chosen" or "rejected" (should always be "chosen")
    rejected_id: Optional[str] = None
    rejected_task: Optional[str] = None
    rejected_lang_vector: Optional[np.ndarray] = None
    rejected_data_source: Optional[str] = None
    rejected_quality_label: Optional[str] = None
    rejected_is_robot: Optional[bool] = None

    target_progress_chosen: Optional[List[float]] = None
    target_progress_rejected: Optional[List[float]] = None
    data_gen_strategy: Optional[str] = None
    num_frames_rewound: Optional[int] = None

    sample_type = "preference"

    # extra stuff for eval
    bin_idx_chosen: Optional[int] = None
    bin_idx_rejected: Optional[int] = None
    video_path: Optional[str] = None    
    chosen_start_end: Optional[List[int]] = None
    rejected_start_end: Optional[List[int]] = None
    fps: Optional[int] = None
    
    # Consolidated metadata for all additional information
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SimilaritySample:
    """Sample structure for similarity scoring: traj_sim and traj_diff ranked against o^ref."""

    # Similarity-specific fields using traj_sim/traj_diff naming
    reference_frames: Optional[Union[List[str], np.ndarray]] = None  # o^ref
    traj_sim_frames: Optional[Union[List[str], np.ndarray]] = None  # Similar trajectory
    traj_diff_frames: Optional[Union[List[str], np.ndarray]] = None  # Different trajectory
    reference_frames_shape: Optional[tuple] = None  # Shape of reference frames
    traj_sim_frames_shape: Optional[tuple] = None  # Shape of similar trajectory frames
    traj_diff_frames_shape: Optional[tuple] = None  # Shape of different trajectory frames
    task_ref: Optional[str] = None
    task_sim: Optional[str] = None
    task_diff: Optional[str] = None
    ref_trajectory_id: Optional[str] = None
    traj_sim_id: Optional[str] = None
    traj_diff_id: Optional[str] = None
    traj_sim_task: Optional[str] = None
    traj_sim_lang_vector: Optional[np.ndarray] = None
    traj_sim_data_source: Optional[str] = None
    traj_sim_quality_label: Optional[str] = None
    traj_sim_is_robot: Optional[bool] = None
    traj_diff_task: Optional[str] = None
    traj_diff_lang_vector: Optional[np.ndarray] = None
    traj_diff_data_source: Optional[str] = None
    traj_diff_quality_label: Optional[str] = None
    traj_diff_is_robot: Optional[bool] = None

    target_progress_sim: Optional[List[float]] = None
    target_progress_diff: Optional[List[float]] = None
    target_progress_ref: Optional[List[float]] = None
    data_gen_strategy: Optional[str] = None
    num_frames_rewound: Optional[int] = None

    sample_type = "similarity"


class BatchCollator:
    """Batch collator that processes Sample objects through the processor."""

    def __init__(
        self,
        processor: AutoProcessor,
        max_length: int = 1024,
        resized_height: int = 128,
        resized_width: int = 128,
    ):
        """
        Initialize the batch collator.

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

    def __call__(
        self,
        samples: Union[List[BaseSample], List[PreferenceSample], List[SimilaritySample], List[dict]],
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a list of samples into separate batches for preferences and similarities.

        Args:
            samples: List of Sample objects or dictionaries that can be converted to Sample objects

        Returns:
            Dictionary containing separate batches for preferences and similarities
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
                else:
                    raise ValueError(
                        f"Unknown sample_type: {sample_type}. Must be 'preference', 'similarity', or 'paired_video'"
                    )
                sample_objects.append(sample_obj)
            elif isinstance(sample, (BaseSample, PreferenceSample, SimilaritySample)):
                sample_objects.append(sample)
            else:
                raise ValueError(f"Expected Sample object or dict, got {type(sample)}")

        # Separate samples by sample type
        preference_samples = [s for s in sample_objects if s.sample_type == "preference"]
        similarity_samples = [s for s in sample_objects if s.sample_type == "similarity"]
        paired_video_samples = [s for s in sample_objects if s.sample_type == "paired_video"]

        # Process preferences
        preference_inputs = {}
        if preference_samples:
            preference_inputs = self._process_preference_batch(preference_samples)

        # Process similarities
        similarity_inputs = {}
        if similarity_samples:
            similarity_inputs = self._process_similarity_batch(similarity_samples)

        # Return both batches
        return {
            "preference_inputs": preference_inputs,
            "similarity_inputs": similarity_inputs,
            "num_preferences": len(preference_samples),
            "num_similarities": len(similarity_samples),
        }

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

    def _process_preference_batch(self, preference_samples: List[PreferenceSample]) -> Dict[str, torch.Tensor]:
        """Process a batch of preference samples."""
        # Collect all messages for batch processing
        all_messages = []

        # Randomly decide whether chosen trajectory goes first or second
        preference_labels = np.random.randint(0, 2, len(preference_samples))

        for i, sample in enumerate(preference_samples):
            # Convert frames to appropriate format using stored shapes
            chosen_frames = self._convert_frames_to_pil_images(sample.chosen_frames, sample.chosen_frames_shape)
            rejected_frames = self._convert_frames_to_pil_images(sample.rejected_frames, sample.rejected_frames_shape)

            if preference_labels[i] == 1.0:
                # Chosen trajectory first: task + video A (chosen) + <|split_token|> + video B (rejected) + <|pref_token|>
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Task: {sample.chosen_task}"},
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
                            {"type": "text", "text": f"Task: {sample.chosen_task}"},
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

        # Add metadata
        batch_inputs["sample_type"] = ["preference"] * len(preference_samples)
        # Use the dynamically generated preference labels based on trajectory order
        batch_inputs["preference_labels"] = torch.tensor(preference_labels, dtype=torch.float32)

        # Add target progress for both trajectories based on conversation order
        target_progress_chosen = [sample.target_progress_chosen for sample in preference_samples]
        target_progress_rejected = [sample.target_progress_rejected for sample in preference_samples]
        target_progress_chosen_mask = [
            1.0
            if sample.chosen_quality_label == "successful" or sample.data_gen_strategy == "rewind_same_task"
            else 0.0
            for sample in preference_samples
        ]
        target_progress_rejected_mask = [
            1.0
            if sample.rejected_quality_label == "successful" or sample.data_gen_strategy == "rewind_same_task"
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
            [sample.chosen_frames_shape for sample in preference_samples], dtype=torch.int32
        )
        batch_inputs["rejected_frames_shape"] = torch.tensor(
            [sample.rejected_frames_shape for sample in preference_samples], dtype=torch.int32
        )

        # Add some rewind metrics for logging
        rewind_lengths = [
            sample.num_frames_rewound if sample.num_frames_rewound is not None else 0 for sample in preference_samples
        ]
        batch_inputs["rewind_lengths"] = torch.tensor(rewind_lengths, dtype=torch.int32)
        
        # Add video-binned metadata if available
        video_binned_metadata = []
        for sample in preference_samples:
            if hasattr(sample, 'data_gen_strategy') and sample.data_gen_strategy == "video_binned":
                metadata = sample.metadata or {}
                video_binned_metadata.append({
                    "chosen_bin_idx": metadata.get("chosen_bin_idx"),
                    "rejected_bin_idx": metadata.get("rejected_bin_idx"),
                    "original_traj_id": metadata.get("original_traj_id"),
                    "num_bins": metadata.get("num_bins"),
                    "bin_size": metadata.get("bin_size"),
                    "chosen_bin_frames": metadata.get("chosen_bin_frames"),
                    "rejected_bin_frames": metadata.get("rejected_bin_frames"),
                    "chosen_bin_progress": metadata.get("chosen_bin_progress"),
                    "rejected_bin_progress": metadata.get("rejected_bin_progress"),
                })
            else:
                video_binned_metadata.append(None)
        
        batch_inputs["video_binned_metadata"] = video_binned_metadata
        return batch_inputs

    def _process_similarity_batch(self, similarity_samples: List[SimilaritySample]) -> Dict[str, torch.Tensor]:
        """Process a batch of similarity samples."""
        # Collect all messages for batch processing (ref and traj_sim for each sample)
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

            # Process reference vs trajectory sim
            conversation_ref_sim = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Reference task: {sample.task_ref}"},
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
                        {"type": "text", "text": f"Reference task: {sample.task_ref}"},
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
            if sample.target_progress_A is not None:
                target_progress_sim_list.append(sample.target_progress_A)

            if sample.target_progress_B is not None:
                target_progress_diff_list.append(sample.target_progress_B)

            if sample.target_progress_ref is not None:
                target_progress_ref_list.append(sample.target_progress_ref)

        # Pad target progress tensors to max length in last dimension
        combined_inputs["target_progress_A"] = self._pad_target_progress(target_progress_sim_list)
        combined_inputs["target_progress_B"] = self._pad_target_progress(target_progress_diff_list)
        combined_inputs["target_progress_ref"] = self._pad_target_progress(target_progress_ref_list)

        # Also add the frame_shapes
        combined_inputs["ref_frames_shape"] = torch.tensor(
            [sample.reference_frames_shape for sample in similarity_samples], dtype=torch.int32
        )
        combined_inputs["traj_sim_frames_shape"] = torch.tensor(
            [sample.traj_sim_frames_shape for sample in similarity_samples], dtype=torch.int32
        )
        combined_inputs["traj_diff_frames_shape"] = torch.tensor(
            [sample.traj_diff_frames_shape for sample in similarity_samples], dtype=torch.int32
        )
        return combined_inputs

    def collate_fn(self, batch: List[Union[BaseSample, PreferenceSample, SimilaritySample]]) -> Dict[str, torch.Tensor]:
        """
        Alternative method name for compatibility with PyTorch DataLoader.

        Args:
            batch: List of Sample objects

        Returns:
            Dictionary containing the processed tensors
        """
        return self(batch)
