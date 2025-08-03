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
    
    # Trajectory A inputs (for preference/comparative samples)
    input_ids_A: Optional[torch.Tensor] = None
    attention_mask_A: Optional[torch.Tensor] = None
    pixel_values_A: Optional[torch.Tensor] = None
    pixel_values_videos_A: Optional[torch.Tensor] = None
    image_grid_thw_A: Optional[torch.Tensor] = None
    video_grid_thw_A: Optional[torch.Tensor] = None
    
    # Trajectory B inputs (for preference/comparative samples)
    input_ids_B: Optional[torch.Tensor] = None
    attention_mask_B: Optional[torch.Tensor] = None
    pixel_values_B: Optional[torch.Tensor] = None
    pixel_values_videos_B: Optional[torch.Tensor] = None
    image_grid_thw_B: Optional[torch.Tensor] = None
    video_grid_thw_B: Optional[torch.Tensor] = None
    
    # Reference inputs (for comparative samples)
    input_ids_ref: Optional[torch.Tensor] = None
    attention_mask_ref: Optional[torch.Tensor] = None
    pixel_values_ref: Optional[torch.Tensor] = None
    pixel_values_videos_ref: Optional[torch.Tensor] = None
    image_grid_thw_ref: Optional[torch.Tensor] = None
    video_grid_thw_ref: Optional[torch.Tensor] = None
    
    # Progress prediction targets
    target_progress_A: Optional[torch.Tensor] = None  # Progress values for trajectory A
    target_progress_B: Optional[torch.Tensor] = None  # Progress values for trajectory B
    
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
        if sample.prediction_type == "preference":
            # For preference samples, process trajectory A and B separately
            if sample.trajectory_A_frames is None or sample.trajectory_B_frames is None:
                raise ValueError("Preference sample must have trajectory_A_frames and trajectory_B_frames")
            
            # Process trajectory A
            conversation_A = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Task: {sample.task}"},
                        {"type": "video", "video": sample.trajectory_A_frames},
                        {"type": "text", "text": "<|pref_token|>"}
                    ]
                }
            ]
            
            # Process trajectory B
            conversation_B = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Task: {sample.task}"},
                        {"type": "video", "video": sample.trajectory_B_frames},
                        {"type": "text", "text": "<|pref_token|>"}
                    ]
                }
            ]
            
            # Process both conversations
            text_A = self.processor.apply_chat_template(
                conversation_A, tokenize=False, add_generation_prompt=False, add_vision_id=True
            )
            text_B = self.processor.apply_chat_template(
                conversation_B, tokenize=False, add_generation_prompt=False, add_vision_id=True
            )
            
            # Process vision information for both
            image_inputs_A, video_inputs_A, video_kwargs_A = process_vision_info(
                conversation_A, return_video_kwargs=True
            )
            image_inputs_B, video_inputs_B, video_kwargs_B = process_vision_info(
                conversation_B, return_video_kwargs=True
            )
            
            # Process through the processor for both
            inputs_A = self.processor(
                text=[text_A],
                images=image_inputs_A,
                videos=video_inputs_A,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                **video_kwargs_A,
            )
            
            inputs_B = self.processor(
                text=[text_B],
                images=image_inputs_B,
                videos=video_inputs_B,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                **video_kwargs_B,
            )
            
            # Combine into single inputs dict with A and B suffixes
            inputs = {
                "prediction_type": sample.prediction_type,
                "preference_labels": torch.tensor([1.0], dtype=torch.float32),  # A is preferred
            }
            
            # Add target progress for both trajectories
            if sample.target_progress_A is not None:
                inputs["target_progress_A"] = torch.tensor([sample.target_progress_A], dtype=torch.float32)
            if sample.target_progress_B is not None:
                inputs["target_progress_B"] = torch.tensor([sample.target_progress_B], dtype=torch.float32)
            
            # Add trajectory A inputs
            for key, value in inputs_A.items():
                inputs[f"{key}_A"] = value
            
            # Add trajectory B inputs
            for key, value in inputs_B.items():
                inputs[f"{key}_B"] = value
            
            return inputs
            
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
            
            # Add prediction type and target progress
            inputs["prediction_type"] = sample.prediction_type
            total_frames = len(sample.trajectory_frames)
            target_progress = torch.tensor([1.0 / total_frames], dtype=torch.float32)
            inputs["target_progress"] = target_progress
            
            return inputs
            
        elif sample.prediction_type == "comparative":
            # For comparative samples, process reference, trajectory A, and trajectory B separately
            if sample.reference_frames is None or sample.trajectory_A_frames is None or sample.trajectory_B_frames is None:
                raise ValueError("Comparative sample must have reference_frames, trajectory_A_frames, and trajectory_B_frames")
            
            # Process reference trajectory
            conversation_ref = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Reference task: {sample.task_ref}"},
                        {"type": "video", "video": sample.reference_frames},
                        {"type": "text", "text": "<|reward_token|>"}
                    ]
                }
            ]
            
            # Process trajectory A
            conversation_A = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Candidate A task: {sample.task_A}"},
                        {"type": "video", "video": sample.trajectory_A_frames},
                        {"type": "text", "text": "<|reward_token|>"}
                    ]
                }
            ]
            
            # Process trajectory B
            conversation_B = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Candidate B task: {sample.task_B}"},
                        {"type": "video", "video": sample.trajectory_B_frames},
                        {"type": "text", "text": "<|reward_token|>"}
                    ]
                }
            ]
            
            # Process all conversations
            text_ref = self.processor.apply_chat_template(
                conversation_ref, tokenize=False, add_generation_prompt=False, add_vision_id=True
            )
            text_A = self.processor.apply_chat_template(
                conversation_A, tokenize=False, add_generation_prompt=False, add_vision_id=True
            )
            text_B = self.processor.apply_chat_template(
                conversation_B, tokenize=False, add_generation_prompt=False, add_vision_id=True
            )
            
            # Process vision information for all
            image_inputs_ref, video_inputs_ref, video_kwargs_ref = process_vision_info(
                conversation_ref, return_video_kwargs=True
            )
            image_inputs_A, video_inputs_A, video_kwargs_A = process_vision_info(
                conversation_A, return_video_kwargs=True
            )
            image_inputs_B, video_inputs_B, video_kwargs_B = process_vision_info(
                conversation_B, return_video_kwargs=True
            )
            
            # Process through the processor for all
            inputs_ref = self.processor(
                text=[text_ref],
                images=image_inputs_ref,
                videos=video_inputs_ref,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                **video_kwargs_ref,
            )
            
            inputs_A = self.processor(
                text=[text_A],
                images=image_inputs_A,
                videos=video_inputs_A,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                **video_kwargs_A,
            )
            
            inputs_B = self.processor(
                text=[text_B],
                images=image_inputs_B,
                videos=video_inputs_B,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                **video_kwargs_B,
            )
            
            # Combine into single inputs dict with ref, A, and B suffixes
            inputs = {
                "prediction_type": sample.prediction_type,
            }
            
            # Add target progress for both trajectories
            if sample.target_progress_A is not None:
                inputs["target_progress_A"] = torch.tensor([sample.target_progress_A], dtype=torch.float32)
            if sample.target_progress_B is not None:
                inputs["target_progress_B"] = torch.tensor([sample.target_progress_B], dtype=torch.float32)
            
            # Add reference inputs
            for key, value in inputs_ref.items():
                inputs[f"{key}_ref"] = value
            
            # Add trajectory A inputs
            for key, value in inputs_A.items():
                inputs[f"{key}_A"] = value
            
            # Add trajectory B inputs
            for key, value in inputs_B.items():
                inputs[f"{key}_B"] = value
            
            return inputs
            
        else:
            raise ValueError(f"Unknown prediction type: {sample.prediction_type}")
    
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
        
        # Initialize batched inputs
        batched_inputs = {}
        
        # Get all possible keys from all processed inputs
        all_keys = set()
        for inputs in processed_inputs:
            all_keys.update(inputs.keys())
        
        # Batch each key
        for key in all_keys:
            if key in processed_inputs[0]:
                # Check if all samples have this key
                if all(key in inputs for inputs in processed_inputs):
                    batched_inputs[key] = torch.cat([inputs[key] for inputs in processed_inputs], dim=0)
        
        # Create and return ProcessedBatch
        return ProcessedBatch(
            # Main inputs (for progress samples or combined inputs)
            input_ids=batched_inputs.get("input_ids"),
            attention_mask=batched_inputs.get("attention_mask"),
            pixel_values=batched_inputs.get("pixel_values"),
            pixel_values_videos=batched_inputs.get("pixel_values_videos"),
            image_grid_thw=batched_inputs.get("image_grid_thw"),
            video_grid_thw=batched_inputs.get("video_grid_thw"),
            
            # Trajectory A inputs (for preference/comparative samples)
            input_ids_A=batched_inputs.get("input_ids_A"),
            attention_mask_A=batched_inputs.get("attention_mask_A"),
            pixel_values_A=batched_inputs.get("pixel_values_A"),
            pixel_values_videos_A=batched_inputs.get("pixel_values_videos_A"),
            image_grid_thw_A=batched_inputs.get("image_grid_thw_A"),
            video_grid_thw_A=batched_inputs.get("video_grid_thw_A"),
            
            # Trajectory B inputs (for preference/comparative samples)
            input_ids_B=batched_inputs.get("input_ids_B"),
            attention_mask_B=batched_inputs.get("attention_mask_B"),
            pixel_values_B=batched_inputs.get("pixel_values_B"),
            pixel_values_videos_B=batched_inputs.get("pixel_values_videos_B"),
            image_grid_thw_B=batched_inputs.get("image_grid_thw_B"),
            video_grid_thw_B=batched_inputs.get("video_grid_thw_B"),
            
            # Reference inputs (for comparative samples)
            input_ids_ref=batched_inputs.get("input_ids_ref"),
            attention_mask_ref=batched_inputs.get("attention_mask_ref"),
            pixel_values_ref=batched_inputs.get("pixel_values_ref"),
            pixel_values_videos_ref=batched_inputs.get("pixel_values_videos_ref"),
            image_grid_thw_ref=batched_inputs.get("image_grid_thw_ref"),
            video_grid_thw_ref=batched_inputs.get("video_grid_thw_ref"),
            
            # Progress prediction targets
            target_progress_A=batched_inputs.get("target_progress_A"),
            target_progress_B=batched_inputs.get("target_progress_B"),
            
            # Sample metadata
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