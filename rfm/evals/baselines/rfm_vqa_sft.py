#!/usr/bin/env python3
"""
RFM VQA SFT baseline trained with scripts/train_vqa_sft.py

RFM VQA SFT predicts scores from 0 to 100 for task completion.
"""

from multiprocessing import Value
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Union, Any, Dict
import numpy as np
from PIL import Image
from scripts.train_vqa_sft import extract_answer_from_generation, process_progress_answer, linspace_subsample_frames, convert_frames_to_pil
import torch
import shutil

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    from unsloth import FastVisionModel

    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

from rfm.utils.logger import get_logger
from scripts.generate_vqa_dataset import PROGRESS_PROMPT_TEMPLATE, PREFERENCE_PROMPT_TEMPLATE, RESPONSE_PREFIX

logger = get_logger()


def prepare_frames_for_conversation(
    frames: List[Image.Image], 
) -> tuple[Any, Dict[str, Any]]:
    """
    Prepare frames for conversation format (multi-image or video).
    
    Args:
        frames: List of PIL Images
        
    Returns:
        Tuple of (frames_or_video, content_extras)
    """
    content_extras = {"nframes": len(frames)}
    return frames
    

def add_vision_content_to_list(
    content_list: List[Dict], frames_or_video: Any, use_multi_image: bool,
) -> None:
    """
    Add vision content (images or video) to the conversation content list.
    
    Args:
        content_list: List to append vision content to
        frames_or_video: Frames (list of PIL Images) or video path
        content_extras: Extra content information (e.g., nframes)
    """
    if use_multi_image:
        # Multi-image mode: add each image separately
        for img in frames_or_video:
            content_list.append({"type": "image", "image": img})
    else:
        # Video mode: add as video with proper metadata
        # Qwen3VL needs video_metadata to avoid FPS warnings
        num_frames = len(frames_or_video)
        fps = 1.0  # Set a reasonable FPS (2 fps for robotics videos)
        
        # Validate frame count
        assert num_frames > 0, f"Video must have at least 1 frame, got {num_frames}"
        
        content_list.append({
            "type": "video",
            "video": frames_or_video,
            "sample_fps": fps,
            "video_metadata": {
                "fps": fps,
                "total_frames": num_frames,
            }
        })

class RFMVQASFT:
    """RFM VQA SFT Model"""

    def __init__(
        self,
        model_path: str, 
        max_new_tokens: int = 10,
        use_unsloth: bool = True,
        use_multi_image: bool = False,
        batch_size: int = 8,
        max_frames=None,
    ):
        """
        Initialize RoboReward model.

        Args:
            model_path: HuggingFace model path (e.g., "teetone/RoboReward-8B" or "teetone/RoboReward-4B")
            max_new_tokens: Maximum number of tokens to generate
            use_unsloth: Whether to use unsloth for faster inference (default: True)
            use_multi_image: Whether to pass multiple images instead of video
            batch_size: max batch size
        """
        logger.info(f"Loading RoboReward model: {model_path}")

        # Use unsloth for faster inference if available and requested
        if use_unsloth and HAS_UNSLOTH:
            print("Using Unsloth for faster inference")
            # Load model with unsloth's FastVisionModel
            try:
                self.model, _ = FastVisionModel.from_pretrained(
                    model_path,
                    dtype=torch.bfloat16,
                    device_map="auto",
                    full_finetuning=False,  # Inference only
                    trust_remote_code=True,
                )
            except Exception as e:
                print(f"FAILED TO LOAD UNSLOTH: {e}")
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto"
                )
        else:
            # Standard loading
            if use_unsloth and not HAS_UNSLOTH:
                print("Warning: Unsloth requested but not available, using standard loading")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",  # Auto device placement is best practice
                trust_remote_code=True,
            )
        print("HARDCODED PROCESSOR FOR QWEN VL RN, PLEASE MODIFY TRAIN_VQA_SFT TO SAVE PROCESSOR")
        if "4b" in model_path:
            processor_model_path = "Qwen/Qwen3-VL-4B-Instruct"
        elif "8b" in model_path:
            processor_model_path = "Qwen/Qwen3-VL-8B-Instruct"
        else:
            raise ValueError(f"COULDN'T FIND HARDCODED PROCESSOR MAP FROM {model_path}")
        self.processor = AutoProcessor.from_pretrained(processor_model_path, trust_remote_code=True, do_sample_frames=False, fps=1)
        self.max_new_tokens = max_new_tokens
        self.model_path = model_path
        self.use_multi_image=use_multi_image
        self.batch_size=batch_size
        self.max_frames = max_frames

        print(f"RoboReward model loaded on device: {self.model.device}")

    def _build_prompt(self, task_description: str, type: str) -> str:
        """Build the prompt for RoboReward inference.

        Args:
            task_description: Task instruction text
            type: Preference or Progress

        Returns:
            Formatted prompt string
        """
        if type.lower() == "preference":
            return PREFERENCE_PROMPT_TEMPLATE.format(response_prefix=RESPONSE_PREFIX, task=task_description)
        elif type.lower() == "progress":
            return PROGRESS_PROMPT_TEMPLATE.format(task=task_description)
        else:
            raise ValueError(f"Prompt type {type} not supported")

    def compute_progress(self, frames_array: np.ndarray, task_description: str = "") -> List[Optional[float]]:
        """
        Compute progress prediction for a frame sequence using RFMVQASFT.

        Args:
            frames_array: (N, H, W, 3) uint8 array from trajectory frames (already a subsequence), or (T, N, W, 3)
            task_description: Task description text

        Returns:
            List of discrete scores (1.0-5.0) for each frame.
            All frames get the same discrete score (end-of-episode score for this subsequence).
        """
        if frames_array is None or frames_array.size == 0:
            return []

        # Convert frames to PIL Images
        frames_pil = convert_frames_to_pil(frames_array)

        logger.info(f"RoboReward: Converted {len(frames_pil)} frames to PIL Images")

        if not frames_pil:
            return []

        num_frames = len(frames_pil)

        # Ensure at least 2 frames for video processing (qwen_vl_utils requires minimum 2 frames)
        if num_frames == 1:
            # Duplicate the single frame to make it 2 frames
            frames_pil = [frames_pil[0], frames_pil[0]]
            num_frames = 2

        # Build prompt
        prompt = self._build_prompt(task_description, type="progress")



        # Prepare frames for conversation
        video, extras = prepare_frames_for_conversation(frames_pil, prefix="video")
        
        # Build content list
        content_list = []
        add_vision_content_to_list(content_list, video, extras)
        content_list.append({"type": "text", "text": prompt})

        conversation = [
            {
                "role": "user",
                "content": content_list,
            }
        ]
        # Apply chat template
        processed_conversation = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        texts = []
        text = self.processor.apply_chat_template(
            processed_conversation, tokenize=False, add_generation_prompt=True, fps=1
        )
        texts.append(text)

        # Process vision info (qwen-vl-utils handles resizing)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            texts,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        # Split videos and metadata (video_inputs is list of (video, video_metadata) tuples)
        if video_inputs is not None:
            videos, video_metadatas = zip(*video_inputs)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            videos = None
            video_metadatas = None

        # Prepare processor kwargs
        processor_kwargs = {
            "text": texts,
            "padding": True,
            "truncation": False,
            "return_tensors": "pt",
            "video_metadata": video_metadatas,
            "return_tensors": "pt",
        }
        
        # Add images if present
        if image_inputs:
            processor_kwargs["images"] = image_inputs
        
        # Add videos if present
        if video_inputs:
            processor_kwargs["videos"] = video_inputs
        
        # Add video_kwargs if present
        if video_kwargs:
            processor_kwargs.update(video_kwargs)

        # Process inputs (do_resize=False since qwen-vl-utils already resized)
        inputs = self.processor(
            **processor_kwargs
        )


        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Deterministic
            )

        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Parse score
        output_text = output_texts[0]
        try:
            progress = process_progress_answer(extract_answer_from_generation(output_text))
            result = [float(progress)] * len(frames_array)
        except Exception as e:
            print(f"ERROR PROCESSING PROGRESS: {e}")
            result = [0] * len(frames_array)

        return result

    def compute_preference(
        self, chosen_images: List, rejected_images: List, task_description: str = ""
    ) -> Dict[str, Any]:
        """Compute preference prediction between two trajectories.

        Args:
            chosen_images: List of images/frames for the chosen trajectory
            rejected_images: List of images/frames for the rejected trajectory
            task_description: Task description text

        Returns:
            Dictionary containing:
            - prediction_prob: Probability that chosen is preferred (0.0 to 1.0)
            - is_correct: True if prediction matches ground truth (always True for chosen)
            - preference_pred: Binary prediction (1.0 if chosen is preferred, 0.0 otherwise)
            - Other metadata
        """
        # TODO: fill this in and remember to subsample frames by 2
        start_time = time.time()

        # Create trajectories
        chosen_traj = create_trajectory_from_dict({
            "frames": chosen_images,
            "task": task_description,
            "num_frames": len(chosen_images),
        })
        rejected_traj = create_trajectory_from_dict({
            "frames": rejected_images,
            "task": task_description,
            "num_frames": len(rejected_images),
        })

        # Create PreferenceSample
        sample = PreferenceSample(
            chosen_trajectory=chosen_traj,
            rejected_trajectory=rejected_traj,
        )

        # Collate into batch
        batch_inputs = self.batch_collator([sample])

        # Extract preference_inputs from batch_inputs (batch_collator returns nested structure)
        preference_inputs = batch_inputs["preference_inputs"]

        # Move to device
        preference_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in preference_inputs.items()
        }

        # Forward pass with inference mode for additional optimization
        with torch.inference_mode():  # Faster than torch.no_grad() for inference-only code
            model_output, _ = forward_model(self.model, preference_inputs, sample_type="preference")

        # Extract preference logits
        pref_logits = model_output.pref_logits
        if pref_logits is None:
            raise ValueError("No preference logits returned from model")

        # Convert logits to probability
        pref_probs = torch.sigmoid(pref_logits)
        prediction_prob = pref_probs.item()
        preference_pred = 1.0 if prediction_prob > 0.5 else 0.0

        processing_time = time.time() - start_time

        # Build result dict (matching RLVLMF format)
        result = {
            "is_correct": True,  # Chosen is always preferred by construction
            "prediction_prob": float(prediction_prob),
            "preference_pred": float(preference_pred),
            "preference_logits": float(pref_logits.item()) if pref_logits is not None else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": task_description,
            "num_chosen_frames": len(chosen_images),
            "num_rejected_frames": len(rejected_images),
            "processing_time_seconds": processing_time,
        }

        return result