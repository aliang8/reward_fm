#!/usr/bin/env python3
"""
Batch collator for processing Sample objects through the processor and returning Batch objects.
This collator handles the conversion from Sample objects to processed tensors in a Batch format.
"""

import torch
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Sample:
    """A unified sample structure that can handle all prediction types."""
    
    # Core HF dataset fields
    prediction_type: str
    id: str
    task: str
    lang_vector: np.ndarray
    data_source: str
    frames: List[str]
    optimal: bool
    is_robot: bool
    
    # Preference-specific fields
    trajectory_A_frames: Optional[List[str]] = None
    trajectory_B_frames: Optional[List[str]] = None
    preferred_trajectory: Optional[str] = None  # "A" or "B"
    trajectory_A_id: Optional[str] = None
    trajectory_B_id: Optional[str] = None
    trajectory_B_task: Optional[str] = None
    trajectory_B_lang_vector: Optional[np.ndarray] = None
    trajectory_B_data_source: Optional[str] = None
    trajectory_B_optimal: Optional[bool] = None
    trajectory_B_is_robot: Optional[bool] = None
    
    # Comparative-specific fields
    reference_frames: Optional[List[str]] = None  # o^ref
    trajectory_A_frames: Optional[List[str]] = None  # o^1
    trajectory_B_frames: Optional[List[str]] = None  # o^2
    task_ref: Optional[str] = None
    task_A: Optional[str] = None
    task_B: Optional[str] = None
    ref_trajectory_id: Optional[str] = None
    trajectory_A_id: Optional[str] = None
    trajectory_B_id: Optional[str] = None
    trajectory_A_task: Optional[str] = None
    trajectory_A_lang_vector: Optional[np.ndarray] = None
    trajectory_A_data_source: Optional[str] = None
    trajectory_A_optimal: Optional[bool] = None
    trajectory_A_is_robot: Optional[bool] = None
    trajectory_B_task: Optional[str] = None
    trajectory_B_lang_vector: Optional[np.ndarray] = None
    trajectory_B_data_source: Optional[str] = None
    trajectory_B_optimal: Optional[bool] = None
    trajectory_B_is_robot: Optional[bool] = None
    
    # Progress fields
    target_progress_A: Optional[List[float]] = None  # Progress values for trajectory A
    target_progress_B: Optional[List[float]] = None  # Progress values for trajectory B
    
    # Metadata field
    metadata: Optional[Dict] = None


@dataclass
class Batch:
    """A batch of samples with all prediction types mixed together."""
    
    samples: List[Sample] = field(default_factory=list)
    
    def __len__(self) -> int:
        """Return the number of samples in the batch."""
        return len(self.samples)
    
    def __iter__(self):
        """Iterate over samples."""
        return iter(self.samples)


class BatchCollator:
    """Batch collator that processes Sample objects through the processor."""
    
    def __init__(self, processor: AutoProcessor, max_length: int = 1024):
        """
        Initialize the batch collator.
        
        Args:
            processor: HuggingFace processor for text and vision processing
            max_length: Maximum sequence length for text
        """
        self.processor = processor
        self.max_length = max_length
    
    def _process_sample(self, sample: Sample) -> Dict[str, torch.Tensor]:
        """
        Process a single Sample through the processor.
        
        Args:
            sample: Sample object containing trajectory information
            
        Returns:
            Dictionary with processed inputs
        """
        if sample.prediction_type == "preference":
            # For preference samples, process trajectory A and B in a single conversation
            if sample.trajectory_A_frames is None or sample.trajectory_B_frames is None:
                raise ValueError("Preference sample must have trajectory_A_frames and trajectory_B_frames")
            
            # Single conversation with both videos: task + video A + <|split_token|> + video B + <|pref_token|>
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Task: {sample.task}"},
                        {"type": "video", "video": sample.trajectory_A_frames},
                        {"type": "text", "text": "<|split_token|>"},
                        {"type": "video", "video": sample.trajectory_B_frames},
                        {"type": "text", "text": "<|pref_token|>"}
                    ]
                }
            ]
            
            # Process the conversation through the processor
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False, add_vision_id=True
            )
            
            # Process vision information
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                conversation, return_video_kwargs=True
            )

            # Process through the processor
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                **video_kwargs,
            )

            print("video_inputs")
            print(inputs["pixel_values_videos"].shape)
            # Add prediction type and preference labels
            inputs["prediction_type"] = sample.prediction_type
            inputs["preference_labels"] = torch.tensor([1.0], dtype=torch.float32)  # A is preferred
            
            # Add target progress for both trajectories
            if sample.target_progress_A is not None:
                inputs["target_progress_A"] = torch.tensor([sample.target_progress_A], dtype=torch.float32)
            if sample.target_progress_B is not None:
                inputs["target_progress_B"] = torch.tensor([sample.target_progress_B], dtype=torch.float32)
            
            return inputs
            
        elif sample.prediction_type == "similarity":
            # For similarity samples, process reference vs trajectory A and reference vs trajectory B separately
            if sample.reference_frames is None or sample.trajectory_A_frames is None or sample.trajectory_B_frames is None:
                raise ValueError("Similarity sample must have reference_frames, trajectory_A_frames, and trajectory_B_frames")
            
            # Process reference vs trajectory A
            conversation_ref_A = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Reference task: {sample.task_ref}"},
                        {"type": "video", "video": sample.reference_frames},
                        {"type": "text", "text": "<|split_token|>"},
                        {"type": "text", "text": f"Candidate A task: {sample.task_A}"},
                        {"type": "video", "video": sample.trajectory_A_frames},
                        {"type": "text", "text": "<|reward_token|>"}
                    ]
                }
            ]
            
            # Process reference vs trajectory B
            conversation_ref_B = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Reference task: {sample.task_ref}"},
                        {"type": "video", "video": sample.reference_frames},
                        {"type": "text", "text": "<|split_token|>"},
                        {"type": "text", "text": f"Candidate B task: {sample.task_B}"},
                        {"type": "video", "video": sample.trajectory_B_frames},
                        {"type": "text", "text": "<|reward_token|>"}
                    ]
                }
            ]

            # Process both conversations
            text_ref_A = self.processor.apply_chat_template(
                conversation_ref_A, tokenize=False, add_generation_prompt=False, add_vision_id=True
            )
            text_ref_B = self.processor.apply_chat_template(
                conversation_ref_B, tokenize=False, add_generation_prompt=False, add_vision_id=True
            )
            
            # Process vision information for both
            image_inputs_ref_A, video_inputs_ref_A, video_kwargs_ref_A = process_vision_info(
                conversation_ref_A, return_video_kwargs=True
            )
            image_inputs_ref_B, video_inputs_ref_B, video_kwargs_ref_B = process_vision_info(
                conversation_ref_B, return_video_kwargs=True
            )
            
            # Process through the processor for both
            inputs_ref_A = self.processor(
                text=[text_ref_A],
                images=image_inputs_ref_A,
                videos=video_inputs_ref_A,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                **video_kwargs_ref_A,
            )
            
            inputs_ref_B = self.processor(
                text=[text_ref_B],
                images=image_inputs_ref_B,
                videos=video_inputs_ref_B,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                **video_kwargs_ref_B,
            )

            print(inputs_ref_B["pixel_values_videos"].shape)
            
            # Combine into single inputs dict with ref_A and ref_B suffixes
            inputs = {
                "prediction_type": sample.prediction_type,
            }
            
            # Add target progress for both trajectories
            if sample.target_progress_A is not None:
                inputs["target_progress_A"] = torch.tensor([sample.target_progress_A], dtype=torch.float32)
            if sample.target_progress_B is not None:
                inputs["target_progress_B"] = torch.tensor([sample.target_progress_B], dtype=torch.float32)
            
            # Add reference vs A inputs
            for key, value in inputs_ref_A.items():
                inputs[f"{key}_ref_A"] = value
            
            # Add reference vs B inputs
            for key, value in inputs_ref_B.items():
                inputs[f"{key}_ref_B"] = value
            
            return inputs
            
        else:
            raise ValueError(f"Unknown prediction type: {sample.prediction_type}")
    
    def __call__(self, samples: Union[List[Sample], List[dict]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of samples into a dictionary of processed tensors.
        
        Args:
            samples: List of Sample objects or dictionaries that can be converted to Sample objects
            
        Returns:
            Dictionary containing the processed tensors
        """
        # Convert dictionaries to Sample objects if needed
        sample_objects = []
        for sample in samples:
            if isinstance(sample, dict):
                # Convert dict to Sample object
                sample_obj = Sample(**sample)
                sample_objects.append(sample_obj)
            elif isinstance(sample, Sample):
                sample_objects.append(sample)
            else:
                raise ValueError(f"Expected Sample object or dict, got {type(sample)}")
        
        # Process each sample individually
        processed_inputs = []
        for sample in sample_objects:
            inputs = self._process_sample(sample)
            processed_inputs.append(inputs)
        
        # Initialize batched inputs
        batched_inputs = {}
        
        # Get all possible keys from all processed inputs
        all_keys = set()
        for inputs in processed_inputs:
            all_keys.update(inputs.keys())

        # Batch each key
        for key in all_keys:
            if key == "prediction_type":
                # prediction_type is a string, collect as list
                batched_inputs[key] = [inputs[key] for inputs in processed_inputs]
            else:
                # Everything else is a tensor, handle specially for pixel_values_videos
                tensors = [inputs[key] for inputs in processed_inputs if key in inputs]
                
                if len(tensors) > 0:
                    if key == "pixel_values_videos":
                        # N x D where N is the number of patches and D is the dimension of the patch
                        # some videos may have less than max frames
                        # Special handling for pixel_values_videos - pad to max first dimension
                        # Find max first dimension
                        max_first_dim = max(tensor.shape[0] for tensor in tensors)
                        
                        # Pad all tensors to max first dimension
                        padded_tensors = []
                        for i, tensor in enumerate(tensors):
                            current_first_dim = tensor.shape[0]
                            if current_first_dim < max_first_dim:
                                # Calculate padding needed
                                padding_needed = max_first_dim - current_first_dim
                                # Pad with zeros at the end (assuming first dim is time/frames)
                                padding = torch.zeros(padding_needed, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
                                padded_tensor = torch.cat([tensor, padding], dim=0)
                            else:
                                padded_tensor = tensor
                            padded_tensors.append(padded_tensor)
                        
                        # Stack the padded tensors
                        batched_inputs[key] = torch.stack(padded_tensors, dim=0)
                    elif key in ["target_progress_A", "target_progress_B"]:
                        # 1 x D where D is the number of frames
                        # Special handling for target_progress - pad to max last dimension
                        # Find max last dimension
                        max_last_dim = max(tensor.shape[-1] for tensor in tensors)
                        
                        # Pad all tensors to max last dimension
                        padded_tensors = []
                        for i, tensor in enumerate(tensors):
                            current_last_dim = tensor.shape[-1]
                            if current_last_dim < max_last_dim:
                                # Calculate padding needed
                                padding_needed = max_last_dim - current_last_dim
                                # Pad with zeros at the end (last dim is progress values)
                                padding = torch.zeros(*tensor.shape[:-1], padding_needed, dtype=tensor.dtype, device=tensor.device)
                                padded_tensor = torch.cat([tensor, padding], dim=-1)
                            else:
                                padded_tensor = tensor
                            padded_tensors.append(padded_tensor)
                        
                        # Stack the padded tensors
                        batched_inputs[key] = torch.stack(padded_tensors, dim=0)
                    else:
                        # Regular tensor stacking for other keys
                        batched_inputs[key] = torch.stack(tensors, dim=0)
                else:
                    # No tensors found for this key, create empty tensor
                    batched_inputs[key] = torch.empty(0)
        return batched_inputs
    
    def collate_fn(self, batch: List[Sample]) -> Dict[str, torch.Tensor]:
        """
        Alternative method name for compatibility with PyTorch DataLoader.
        
        Args:
            batch: List of Sample objects
            
        Returns:
            Dictionary containing the processed tensors
        """
        return self(batch) 