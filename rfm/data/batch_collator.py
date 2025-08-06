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
    
    def __call__(self, samples: Union[List[Sample], List[dict]]) -> Dict[str, torch.Tensor]:
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
                # Convert dict to Sample object
                sample_obj = Sample(**sample)
                sample_objects.append(sample_obj)
            elif isinstance(sample, Sample):
                sample_objects.append(sample)
            else:
                raise ValueError(f"Expected Sample object or dict, got {type(sample)}")
        
        # Separate samples by prediction type
        preference_samples = [s for s in sample_objects if s.prediction_type == "preference"]
        similarity_samples = [s for s in sample_objects if s.prediction_type == "similarity"]
        
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
            "num_similarities": len(similarity_samples)
        }
    
    def _process_preference_batch(self, preference_samples: List[Sample]) -> Dict[str, torch.Tensor]:
        """Process a batch of preference samples."""
        # Collect all messages for batch processing
        all_messages = []
        
        for sample in preference_samples:
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
            all_messages.append(conversation)
        
        # Process all messages in one batch
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False, add_vision_id=True)
            for msg in all_messages
        ]
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(all_messages, return_video_kwargs=True)
        
        import ipdb; ipdb.set_trace()

        # Process through the processor in one batch
        batch_inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **video_kwargs,
        )
        
        # Add metadata
        batch_inputs["prediction_type"] = ["preference"] * len(preference_samples)
        batch_inputs["preference_labels"] = torch.ones(len(preference_samples), dtype=torch.float32)  # A is preferred
        
        # Add target progress for both trajectories
        target_progress_A_list = []
        target_progress_B_list = []
        
        for sample in preference_samples:
            if sample.target_progress_A is not None:
                target_progress_A_list.append(sample.target_progress_A)
            if sample.target_progress_B is not None:
                target_progress_B_list.append(sample.target_progress_B)
        
        # Pad target progress tensors to max length in last dimension
        batch_inputs["target_progress_A"] = self._pad_target_progress(target_progress_A_list)
        batch_inputs["target_progress_B"] = self._pad_target_progress(target_progress_B_list)
        
        return batch_inputs
    
    def _process_similarity_batch(self, similarity_samples: List[Sample]) -> Dict[str, torch.Tensor]:
        """Process a batch of similarity samples."""
        # Collect all messages for batch processing (ref_A and ref_B for each sample)
        all_messages = []
        
        for sample in similarity_samples:
            # Process reference vs trajectory A
            conversation_ref_A = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Reference task: {sample.task_ref}"},
                        {"type": "video", "video": sample.reference_frames},
                        {"type": "text", "text": "<|split_token|>"},
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
                        {"type": "video", "video": sample.trajectory_B_frames},
                        {"type": "text", "text": "<|reward_token|>"}
                    ]
                }
            ]
            
            all_messages.extend([conversation_ref_A, conversation_ref_B])
        
        # Process all messages in one batch
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False, add_vision_id=True)
            for msg in all_messages
        ]
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(all_messages, return_video_kwargs=True)
        
        # Process through the processor in one batch
        batch_inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **video_kwargs,
        )
        
        # Split the batch inputs back into ref_A and ref_B
        num_samples = len(similarity_samples)
        ref_A_inputs = {}
        ref_B_inputs = {}
        
        for key, value in batch_inputs.items():
            if isinstance(value, torch.Tensor):
                # Split into ref_A and ref_B (alternating)
                ref_A_inputs[key] = value[::2]  # Even indices (0, 2, 4, ...)
                ref_B_inputs[key] = value[1::2]  # Odd indices (1, 3, 5, ...)
            else:
                ref_A_inputs[key] = value
                ref_B_inputs[key] = value
        
        # Combine into single batch with ref_A and ref_B suffixes
        combined_inputs = {"prediction_type": ["similarity"] * num_samples}
        
        # Add ref_A inputs
        for key, value in ref_A_inputs.items():
            combined_inputs[f"{key}_ref_A"] = value
        
        # Add ref_B inputs
        for key, value in ref_B_inputs.items():
            combined_inputs[f"{key}_ref_B"] = value
        
        # Add target progress for both trajectories
        target_progress_A_list = []
        target_progress_B_list = []
        
        for sample in similarity_samples:
            if sample.target_progress_A is not None:
                target_progress_A_list.append(sample.target_progress_A)
                
            if sample.target_progress_B is not None:
                target_progress_B_list.append(sample.target_progress_B)
        
        # Pad target progress tensors to max length in last dimension
        combined_inputs["target_progress_A"] = self._pad_target_progress(target_progress_A_list)
        combined_inputs["target_progress_B"] = self._pad_target_progress(target_progress_B_list)
        
        return combined_inputs
    
    def collate_fn(self, batch: List[Sample]) -> Dict[str, torch.Tensor]:
        """
        Alternative method name for compatibility with PyTorch DataLoader.
        
        Args:
            batch: List of Sample objects
            
        Returns:
            Dictionary containing the processed tensors
        """
        return self(batch) 