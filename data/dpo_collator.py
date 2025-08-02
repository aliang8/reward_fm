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
from .data_generator import Sample, Batch


@dataclass
class ProcessedBatch:
    """A batch of processed samples with model inputs."""
    
    # Text inputs
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    
    # Vision inputs (if present)
    pixel_values: Optional[torch.Tensor] = None
    pixel_values_videos: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    video_grid_thw: Optional[torch.Tensor] = None
    
    # Sample metadata
    samples: List[Sample] = None


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
        # Create conversation format based on sample type
        if sample.prediction_type == "preference":
            # For preference samples, we have trajectory_A and trajectory_B
            if sample.trajectory_A_frames is None or sample.trajectory_B_frames is None:
                raise ValueError("Preference sample must have trajectory_A_frames and trajectory_B_frames")
            
            # Create a conversation with both trajectories
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
        elif sample.prediction_type == "progress":
            # For progress samples, we have a single trajectory
            if sample.trajectory_frames is None:
                raise ValueError("Progress sample must have trajectory_frames")
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Task: {sample.task}"},
                        {"type": "video", "video": sample.trajectory_frames},
                        {"type": "text", "text": "<|progress_token|>"}
                    ]
                }
            ]
        elif sample.prediction_type == "comparative":
            # For comparative samples, we have reference, trajectory_A, and trajectory_B
            if sample.reference_frames is None or sample.trajectory_A_frames is None or sample.trajectory_B_frames is None:
                raise ValueError("Comparative sample must have reference_frames, trajectory_A_frames, and trajectory_B_frames")
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Reference task: {sample.task_ref}"},
                        {"type": "video", "video": sample.reference_frames},
                        {"type": "text", "text": "<|split_token|>"},
                        {"type": "text", "text": f"Candidate A task: {sample.task_A}"},
                        {"type": "video", "video": sample.trajectory_A_frames},
                        {"type": "text", "text": "<|split_token|>"},
                        {"type": "text", "text": f"Candidate B task: {sample.task_B}"},
                        {"type": "video", "video": sample.trajectory_B_frames},
                        {"type": "text", "text": "<|reward_token|>"}
                    ]
                }
            ]
        else:
            raise ValueError(f"Unknown prediction type: {sample.prediction_type}")
        
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
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **video_kwargs,
        )
        
        # Add prediction type and additional fields based on sample type
        inputs["prediction_type"] = sample.prediction_type
        
        if sample.prediction_type == "progress":
            # Add target progress (frame index / total frames)
            total_frames = len(sample.trajectory_frames)
            # For now, assume we're predicting progress for the first frame
            target_progress = torch.tensor([1.0 / total_frames], dtype=torch.float32)
            inputs["target_progress"] = target_progress
        elif sample.prediction_type == "preference":
            # Add preference labels (1 if A is preferred, 0 if B is preferred)
            # For now, assume A is preferred (this should come from the sample)
            preference_labels = torch.tensor([1.0], dtype=torch.float32)
            inputs["preference_labels"] = preference_labels
        
        return inputs
    
    def __call__(self, samples: Union[List[Sample], List[dict]]) -> ProcessedBatch:
        """
        Collate a list of samples into a ProcessedBatch object.
        
        Args:
            samples: List of Sample objects or dictionaries that can be converted to Sample objects
            
        Returns:
            ProcessedBatch object containing the processed tensors
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
        
        # Batch the processed inputs
        batched_inputs = {
            "input_ids": torch.cat([inputs["input_ids"] for inputs in processed_inputs], dim=0),
            "attention_mask": torch.cat([inputs["attention_mask"] for inputs in processed_inputs], dim=0),
        }
        
        # Add vision inputs if they exist
        if "pixel_values" in processed_inputs[0]:
            batched_inputs["pixel_values"] = torch.cat([inputs["pixel_values"] for inputs in processed_inputs], dim=0)
        if "pixel_values_videos" in processed_inputs[0]:
            batched_inputs["pixel_values_videos"] = torch.cat([inputs["pixel_values_videos"] for inputs in processed_inputs], dim=0)
        if "image_grid_thw" in processed_inputs[0]:
            batched_inputs["image_grid_thw"] = torch.cat([inputs["image_grid_thw"] for inputs in processed_inputs], dim=0)
        if "video_grid_thw" in processed_inputs[0]:
            batched_inputs["video_grid_thw"] = torch.cat([inputs["video_grid_thw"] for inputs in processed_inputs], dim=0)
        
        # Create and return ProcessedBatch
        return ProcessedBatch(
            input_ids=batched_inputs["input_ids"],
            attention_mask=batched_inputs["attention_mask"],
            pixel_values=batched_inputs.get("pixel_values"),
            pixel_values_videos=batched_inputs.get("pixel_values_videos"),
            image_grid_thw=batched_inputs.get("image_grid_thw"),
            video_grid_thw=batched_inputs.get("video_grid_thw"),
            samples=sample_objects
        )
    
    def collate_fn(self, batch: List[Sample]) -> ProcessedBatch:
        """
        Alternative method name for compatibility with PyTorch DataLoader.
        
        Args:
            batch: List of Sample objects
            
        Returns:
            ProcessedBatch object containing the processed tensors
        """
        return self(batch) 