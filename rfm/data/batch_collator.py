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

@dataclass
class BaseSample:
    """Base sample structure with common fields for all prediction types."""
    
    # Core HF dataset fields
    id: str
    task: str
    lang_vector: np.ndarray
    data_source: str
    frames: Union[List[str], np.ndarray]
    optimal: bool
    is_robot: bool
    
    # Frame shape information for deserialization
    frames_shape: Optional[tuple] = None  # Shape of frames (T, H, W, C)
    
    # Progress fields
    target_progress_A: Optional[List[float]] = None  # Progress values for trajectory A
    target_progress_B: Optional[List[float]] = None  # Progress values for trajectory B
    prediction_type: Optional[str] = None

    # Metadata field
    metadata: Optional[Dict] = None


@dataclass
class PreferenceSample(BaseSample):
    """Sample structure for preference prediction: chosen vs rejected where chosen is preferred."""
    
    # Preference-specific fields using chosen/rejected naming
    chosen_frames: Optional[Union[List[str], np.ndarray]] = None
    rejected_frames: Optional[Union[List[str], np.ndarray]] = None
    chosen_frames_shape: Optional[tuple] = None  # Shape of chosen trajectory frames
    rejected_frames_shape: Optional[tuple] = None  # Shape of rejected trajectory frames
    preferred_trajectory: Optional[str] = None  # "chosen" or "rejected" (should always be "chosen")
    chosen_id: Optional[str] = None
    rejected_id: Optional[str] = None
    rejected_task: Optional[str] = None
    rejected_lang_vector: Optional[np.ndarray] = None
    rejected_data_source: Optional[str] = None
    rejected_optimal: Optional[bool] = None
    rejected_is_robot: Optional[bool] = None
    
    def __post_init__(self):
        """Set the prediction type after initialization and handle field mapping."""
        self.prediction_type = "preference"
        

@dataclass
class SimilaritySample(BaseSample):
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
    traj_sim_optimal: Optional[bool] = None
    traj_sim_is_robot: Optional[bool] = None
    traj_diff_task: Optional[str] = None
    traj_diff_lang_vector: Optional[np.ndarray] = None
    traj_diff_data_source: Optional[str] = None
    traj_diff_optimal: Optional[bool] = None
    traj_diff_is_robot: Optional[bool] = None
   
    def __post_init__(self):
        """Set the prediction type after initialization and handle field mapping."""
        self.prediction_type = "similarity"

class BatchCollator:
    """Batch collator that processes Sample objects through the processor."""
    
    def __init__(self, processor: AutoProcessor, max_length: int = 1024, resized_height: int = 128, resized_width: int = 128):
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
    
    def __call__(self, samples: Union[List[BaseSample], List[PreferenceSample], List[SimilaritySample], List[dict]]) -> Dict[str, torch.Tensor]:
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
                # Convert dict to appropriate Sample object based on prediction_type
                prediction_type = sample.get('prediction_type', 'unknown')
                if prediction_type == "preference":
                    sample_obj = PreferenceSample(**sample)
                elif prediction_type == "similarity":
                    sample_obj = SimilaritySample(**sample)
                else:
                    raise ValueError(f"Unknown prediction_type: {prediction_type}. Must be 'preference' or 'similarity'")
                sample_objects.append(sample_obj)
            elif isinstance(sample, (BaseSample, PreferenceSample, SimilaritySample)):
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
        
        for sample in preference_samples:
            # Convert frames to appropriate format using stored shapes
            chosen_frames = self._convert_frames_to_pil_images(sample.chosen_frames, sample.chosen_frames_shape)
            rejected_frames = self._convert_frames_to_pil_images(sample.rejected_frames, sample.rejected_frames_shape)
            
            # Single conversation with both videos: task + video A + <|split_token|> + video B + <|pref_token|>
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Task: {sample.task}"},
                        {"type": "video", "video": chosen_frames, "resized_height": self.resized_height, "resized_width": self.resized_width},
                        {"type": "text", "text": "<|split_token|>"},
                        {"type": "video", "video": rejected_frames, "resized_height": self.resized_height, "resized_width": self.resized_width},
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
        
        # Process through the processor in one batch
        batch_inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=False,
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
    
    def _process_similarity_batch(self, similarity_samples: List[SimilaritySample]) -> Dict[str, torch.Tensor]:
        """Process a batch of similarity samples."""
        # Collect all messages for batch processing (ref_A and ref_B for each sample)
        all_messages = []
        
        for sample in similarity_samples:
            # Convert frames to appropriate format using stored shapes
            reference_frames = self._convert_frames_to_pil_images(sample.reference_frames, sample.reference_frames_shape)
            traj_sim_frames = self._convert_frames_to_pil_images(sample.traj_sim_frames, sample.traj_sim_frames_shape)
            traj_diff_frames = self._convert_frames_to_pil_images(sample.traj_diff_frames, sample.traj_diff_frames_shape)
            
            # Process reference vs trajectory A
            conversation_ref_A = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Reference task: {sample.task_ref}"},
                        {"type": "video", "video": reference_frames, "resized_height": self.resized_height, "resized_width": self.resized_width},
                        {"type": "text", "text": "<|split_token|>"},
                        {"type": "video", "video": traj_sim_frames, "resized_height": self.resized_height, "resized_width": self.resized_width},
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
                        {"type": "video", "video": reference_frames, "resized_height": self.resized_height, "resized_width": self.resized_width},
                        {"type": "text", "text": "<|split_token|>"},
                        {"type": "video", "video": traj_diff_frames, "resized_height": self.resized_height, "resized_width": self.resized_width},
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
            padding=True,
            truncation=False,
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
    
    def collate_fn(self, batch: List[Union[BaseSample, PreferenceSample, SimilaritySample]]) -> Dict[str, torch.Tensor]:
        """
        Alternative method name for compatibility with PyTorch DataLoader.
        
        Args:
            batch: List of Sample objects
            
        Returns:
            Dictionary containing the processed tensors
        """
        return self(batch)


 