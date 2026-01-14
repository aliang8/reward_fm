#!/usr/bin/env python3
"""
RoboReward baseline for discrete end-of-episode progress reward prediction.

RoboReward predicts discrete scores (1-5) for task completion:
- 1: No success
- 2: Minimal progress
- 3: Partial completion
- 4: Near completion
- 5: Perfect completion

Based on: https://huggingface.co/teetone/RoboReward-8B
"""

import re
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from PIL import Image
import torch
import shutil

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    from unsloth import FastVisionModel

    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

from rfm.data.collators.utils import convert_frames_to_pil_images, write_mp4
from rfm.utils.logger import get_logger

logger = get_logger()

class RoboReward:
    """RoboReward baseline for discrete end-of-episode progress reward prediction."""

    def __init__(
        self,
        model_path: str = "teetone/RoboReward-8B",
        max_new_tokens: int = 128,
        use_unsloth: bool = True,
    ):
        """
        Initialize RoboReward model.

        Args:
            model_path: HuggingFace model path (e.g., "teetone/RoboReward-8B" or "teetone/RoboReward-4B")
            max_new_tokens: Maximum number of tokens to generate
            use_unsloth: Whether to use unsloth for faster inference (default: True)
        """
        logger.info(f"Loading RoboReward model: {model_path}")

        # Use unsloth for faster inference if available and requested
        if use_unsloth and HAS_UNSLOTH:
            print("Using Unsloth for faster inference")
            # Load model with unsloth's FastVisionModel
            self.model, _ = FastVisionModel.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map="auto",
                full_finetuning=False,  # Inference only
            )
        else:
            # Standard loading
            if use_unsloth and not HAS_UNSLOTH:
                print("Warning: Unsloth requested but not available, using standard loading")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",  # Auto device placement is best practice
            )

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, do_sample_frames=False, fps=1)
        self.max_new_tokens = max_new_tokens
        self.model_path = model_path

        print(f"RoboReward model loaded on device: {self.model.device}")

    def _build_prompt(self, task_description: str) -> str:
        """Build the prompt for RoboReward inference.

        Args:
            task_description: Task instruction text

        Returns:
            Formatted prompt string
        """
        prompt = """Given the task, assign a discrete progress score reward (1,2,3,4,5) for the robot in the video in the format: ANSWER: <score>
Rubric for end-of-episode progress (judge only the final state without time limits):
1 - No Success: Final state shows no goal-relevant change for the command.
2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.
3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement.
4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.
5 - Perfect Completion: Final state satisfies all requirements.

Task: {task}""".format(task=task_description)
        return prompt

    def _parse_score(self, output_text: str) -> Optional[int]:
        """Parse discrete score (1-5) from model output.

        Args:
            output_text: Model output text

        Returns:
            Discrete score (1-5) or None if parsing fails
        """
        # Look for "ANSWER: <number>" pattern
        pattern = r"ANSWER:\s*(\d+)"
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score

        # Fallback: look for any single digit 1-5 in the text
        pattern = r"\b([1-5])\b"
        matches = re.findall(pattern, output_text)
        if matches:
            # Take the last occurrence (most likely the answer)
            score = int(matches[-1])
            if 1 <= score <= 5:
                return score

        return None

    def compute_progress(self, frames_array: np.ndarray, task_description: str = "") -> List[Optional[float]]:
        """
        Compute progress prediction for a frame sequence using RoboReward baseline.

        RoboReward predicts a discrete score (1-5) for the end-of-episode state.
        Since the sampler already uses use_frame_steps to create progressively longer
        sequences, we just process the single sequence provided here.

        Args:
            frames_array: (N, H, W, 3) uint8 array from trajectory frames (already a subsequence)
            task_description: Task description text

        Returns:
            List of discrete scores (1.0-5.0) for each frame.
            All frames get the same discrete score (end-of-episode score for this subsequence).
        """
        if frames_array is None or frames_array.size == 0:
            return []

        # Convert frames to PIL Images
        frames_pil = convert_frames_to_pil_images(frames_array)

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
        prompt = self._build_prompt(task_description)

        # Create temporary directory for frame files
        # Use individual frame files instead of video to avoid torchcodec memory issues
        # According to qwen-vl-utils docs, we can pass frames as a list of file paths
        tmpdir = tempfile.mkdtemp()
        try:
            unique_id = uuid.uuid4().hex
            
            # Save frames as individual JPEG files (much smaller than video, avoids torchcodec overhead)
            frame_paths = []
            for i, frame_pil in enumerate(frames_pil):
                frame_path = Path(tmpdir) / f"roboreward_{unique_id}_frame_{i:04d}.jpg"
                # Save as JPEG with reasonable quality to reduce file size
                frame_pil.save(frame_path, "JPEG", quality=85, optimize=True)
                frame_paths.append(f"file://{frame_path}")
            
            logger.info(f"RoboReward: Saved {len(frame_paths)} frames as JPEG files in {tmpdir}")

            # Build message with frames as list of file paths (following Qwen3-VL pattern)
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frame_paths, "sample_fps": 1.0},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False) 

            # Process vision info (qwen-vl-utils handles resizing)
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                [message],
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

            # Process inputs (do_resize=False since qwen-vl-utils already resized)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=videos,
                video_metadata=video_metadatas,
                padding=True,
                return_tensors="pt",
                do_resize=False,  # qwen-vl-utils already resized
                **video_kwargs,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

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
            discrete_score = self._parse_score(output_text)
            logger.info(f"RoboReward: Discrete score: {discrete_score}")

            if discrete_score is None:
                print(f"[!] Failed to parse score from output: {output_text}")
                discrete_score = 1  # Default to minimum score if parsing fails

            # Return same discrete score for all frames in this subsequence
            # Use original num_frames from frames_array (before duplication)
            original_num_frames = len(convert_frames_to_pil_images(frames_array))

            # because RoboReward returns a score between 1 and 5, we need to normalize it to 0-1
            result = [float(discrete_score) / 5.0] * original_num_frames
        finally:
            # Clean up temporary directory and files after all processing is complete
            # This ensures the video file exists during process_vision_info and processor calls
            shutil.rmtree(tmpdir, ignore_errors=True)

        return result
