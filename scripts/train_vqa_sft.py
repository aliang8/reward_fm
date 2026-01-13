#!/usr/bin/env python3
"""
Self-contained VQA training script for Qwen3-VL models.

This script trains Qwen3-VL models on a pre-generated VQA dataset using standard
HuggingFace Trainer. The dataset should be generated using generate_vqa_dataset.py.

Usage:
    python scripts/train_vqa_sft.py \\
        --dataset_path /path/to/generated/dataset \\
        --model_name Qwen/Qwen3-VL-4B-Instruct \\
        --output_dir ./outputs/vqa_training \\
        --per_device_train_batch_size 4 \\
        --num_train_epochs 3
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.trainer_utils import EvalPrediction

# Try to import unsloth
try:
    from unsloth import FastVisionModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    FastVisionModel = None

# Add project root to path for utility imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rfm.data.datasets.helpers import load_frames_from_npz

# Constants
RESPONSE_PREFIX = "ANS:"
IGNORE_INDEX = -100


def extract_answer_from_generation(text: str) -> Optional[str]:
    """
    Extract the answer from generated text.
    
    Looks for "ANS: X" pattern and extracts X.
    
    Args:
        text: Generated text
        
    Returns:
        Extracted answer or None if not found
    """
    # Look for "ANS: X" pattern
    match = re.search(r'ANS:\s*(\S+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: look for the last word after "ANS:"
    if "ANS:" in text.upper():
        parts = text.upper().split("ANS:")
        if len(parts) > 1:
            answer_part = parts[-1].strip()
            # Extract first word/number
            tokens = answer_part.split()
            if tokens:
                return tokens[0].strip()
    
    return None


class VQAEvaluationCallback(TrainerCallback):
    """
    Custom callback to perform VQA evaluation during training.
    
    This callback generates answers on the eval set and computes:
    - Preference accuracy (for preference samples)
    - Progress MAE and RMSE (for progress samples)
    """
    
    def __init__(
        self,
        eval_dataset,
        processor,
        collator,
        max_new_tokens: int = 10,
        eval_batch_size: int = 4,
    ):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.collator = collator
        self.max_new_tokens = max_new_tokens
        self.eval_batch_size = eval_batch_size
        
    def on_evaluate(self, args, state, control, model, **kwargs):
        """Run VQA evaluation after standard evaluation."""
        print("\n" + "="*80)
        print("Running VQA Evaluation (Generation-based)")
        print("="*80)
        
        model.eval()
        device = model.device
        
        # Separate by sample type
        pref_samples = [s for s in self.eval_dataset if s['sample_type'] == 'preference']
        prog_samples = [s for s in self.eval_dataset if s['sample_type'] == 'progress']
        
        metrics = {}
        
        # Evaluate preference samples
        if pref_samples:
            pref_correct = 0
            pref_total = 0
            
            with torch.no_grad():
                for i in range(0, len(pref_samples), self.eval_batch_size):
                    batch = pref_samples[i:i+self.eval_batch_size]
                    
                    try:
                        # Create inference collator
                        inference_collator = VQADataCollator(
                            processor=self.processor,
                            use_multi_image=False,
                            inference=True,
                        )
                        
                        inputs = inference_collator(batch)
                        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in inputs.items()}
                        
                        # Generate
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                        )
                        
                        # Decode
                        generated_texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
                        
                        # Check correctness
                        for sample, gen_text in zip(batch, generated_texts):
                            predicted = extract_answer_from_generation(gen_text)
                            ground_truth = sample['answer']
                            
                            if predicted == ground_truth:
                                pref_correct += 1
                            pref_total += 1
                            
                    except Exception as e:
                        print(f"Error in preference eval batch {i}: {e}")
                        continue
            
            if pref_total > 0:
                pref_accuracy = pref_correct / pref_total
                metrics['eval_preference_accuracy'] = pref_accuracy
                print(f"Preference Accuracy: {pref_accuracy:.4f} ({pref_correct}/{pref_total})")
        
        # Evaluate progress samples
        if prog_samples:
            prog_errors = []
            
            with torch.no_grad():
                for i in range(0, len(prog_samples), self.eval_batch_size):
                    batch = prog_samples[i:i+self.eval_batch_size]
                    
                    try:
                        # Create inference collator
                        inference_collator = VQADataCollator(
                            processor=self.processor,
                            use_multi_image=False,
                            inference=True,
                        )
                        
                        inputs = inference_collator(batch)
                        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in inputs.items()}
                        
                        # Generate
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                        )
                        
                        # Decode
                        generated_texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
                        
                        # Compute errors
                        for sample, gen_text in zip(batch, generated_texts):
                            predicted_str = extract_answer_from_generation(gen_text)
                            ground_truth_str = sample['answer']
                            
                            try:
                                predicted = int(predicted_str) if predicted_str else None
                                ground_truth = int(ground_truth_str)
                                
                                if predicted is not None:
                                    error = abs(predicted - ground_truth)
                                    prog_errors.append(error)
                                else:
                                    prog_errors.append(100)  # Max error
                            except (ValueError, TypeError):
                                prog_errors.append(100)  # Max error
                                
                    except Exception as e:
                        print(f"Error in progress eval batch {i}: {e}")
                        continue
            
            if prog_errors:
                mae = np.mean(prog_errors)
                rmse = np.sqrt(np.mean(np.array(prog_errors) ** 2))
                metrics['eval_progress_mae'] = mae
                metrics['eval_progress_rmse'] = rmse
                print(f"Progress MAE: {mae:.2f}")
                print(f"Progress RMSE: {rmse:.2f}")
        
        print("="*80 + "\n")
        
        # Log metrics
        if hasattr(state, 'log_history'):
            # Add to the last log entry
            if state.log_history:
                state.log_history[-1].update(metrics)
        
        return control


def load_and_subsample_frames(npz_path: str, frame_indices: List[int]) -> np.ndarray:
    """
    Load frames from npz file and subsample based on frame indices.
    
    Args:
        npz_path: Path to .npz file containing frames
        frame_indices: List of frame indices to extract
        
    Returns:
        Numpy array of shape (T, H, W, C) with selected frames
    """
    # Load all frames from npz
    frames = load_frames_from_npz(npz_path)
    
    # Subsample based on indices
    if frame_indices:
        # Ensure indices are within bounds
        max_idx = frames.shape[0] - 1
        valid_indices = [min(idx, max_idx) for idx in frame_indices]
        subsampled = frames[valid_indices]
    else:
        subsampled = frames
    
    return subsampled


def convert_frames_to_pil(frames: np.ndarray) -> List[Image.Image]:
    """
    Convert numpy array frames to list of PIL Images.
    
    Args:
        frames: Numpy array of shape (T, H, W, C)
        
    Returns:
        List of PIL Images
    """
    pil_images = []
    
    if len(frames.shape) == 4:  # TxHxWxC
        for i in range(frames.shape[0]):
            frame = frames[i]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            pil_images.append(Image.fromarray(frame))
    elif len(frames.shape) == 3:  # HxWxC (single frame)
        frame = frames
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        pil_images.append(Image.fromarray(frame))
    else:
        raise ValueError(f"Unexpected frames shape: {frames.shape}")
    
    return pil_images


class VQADataCollator:
    """
    Data collator for VQA training that loads frames on-the-fly from .npz files.
    
    This collator:
    1. Loads frames from .npz files based on stored paths
    2. Subsamples frames using stored frame indices
    3. Converts to PIL images
    4. Formats as Qwen3-VL conversation
    5. Tokenizes with processor
    6. Masks prompt tokens (only train on answer)
    """
    
    def __init__(
        self,
        processor: AutoProcessor,
        use_multi_image: bool = False,
        inference: bool = False,
    ):
        """
        Initialize VQA data collator.
        
        Args:
            processor: Qwen3-VL processor for tokenization and image processing
            use_multi_image: If True, use multi-image mode (list of images), else use video mode
            inference: If True, don't add assistant response (for inference)
        """
        self.processor = processor
        self.use_multi_image = use_multi_image
        self.inference = inference
        self.base_model_id = "Qwen"  # For compatibility checks

    def _mask_labels(self, labels: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Mask out the prompt - only train on assistant response after 'ANS:'.
        
        Args:
            labels: List of label tensors for each sample in batch
            
        Returns:
            List of masked label tensors
        """
        masked_labels = []
        
        for label_tensor in labels:
            seq_len = label_tensor.shape[0]
            max_window = 8  # Number of tokens to decode when searching for ANS:
            ans_token_positions = []

            # Find positions where "ANS:" appears
            for idx in range(seq_len):
                end_idx = min(seq_len, idx + max_window)
                window_tokens = label_tensor[idx:end_idx]
                window_text = self.processor.tokenizer.decode(window_tokens, skip_special_tokens=False)
                if window_text.lstrip().startswith(RESPONSE_PREFIX):
                    ans_token_positions.append(idx)

            # Mask everything before the last occurrence of "ANS:"
            if ans_token_positions:
                start_idx = ans_token_positions[-1]
                label_tensor[:start_idx] = IGNORE_INDEX
            else:
                # If "ANS:" not found, mask everything (shouldn't happen in training)
                label_tensor[:] = IGNORE_INDEX
            
            masked_labels.append(label_tensor)
        
        return masked_labels

    def _prepare_frames_for_conversation(
        self, frames: List[Image.Image], prefix: str = "video"
    ) -> tuple[Any, Dict[str, Any]]:
        """
        Prepare frames for conversation format (multi-image or video).
        
        Args:
            frames: List of PIL Images
            prefix: Prefix for temporary file naming
            
        Returns:
            Tuple of (frames_or_video, content_extras)
        """
        content_extras = {"nframes": len(frames)}
        
        if self.use_multi_image:
            # Multi-image mode: return list of PIL images directly
            return frames, content_extras
        else:
            # Video mode: return list of PIL images (Qwen processor handles video conversion)
            return frames, content_extras

    def _add_vision_content_to_list(
        self, content_list: List[Dict], frames_or_video: Any, content_extras: Dict
    ) -> None:
        """
        Add vision content (images or video) to the conversation content list.
        
        Args:
            content_list: List to append vision content to
            frames_or_video: Frames (list of PIL Images) or video path
            content_extras: Extra content information (e.g., nframes)
        """
        if self.use_multi_image:
            # Multi-image mode: add each image separately
            for img in frames_or_video:
                content_list.append({"type": "image", "image": img})
        else:
            # Video mode: add as video with proper metadata
            # Qwen3VL needs video_metadata to avoid FPS warnings
            num_frames = len(frames_or_video)
            fps = 1.0  # We're using pre-sampled frames, so effective FPS is 1
            
            content_list.append({
                "type": "video",
                "video": frames_or_video,
                "sample_fps": fps,
                "video_metadata": {
                    "fps": fps,
                    "total_frames": num_frames,
                }
            })

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of sample dictionaries from the dataset
            
        Returns:
            Dictionary with tokenized inputs ready for training
        """
        all_messages = []
        
        for sample in batch:
            sample_type = sample["sample_type"]
            prompt = sample["prompt"]
            answer = sample["answer"]
            
            if sample_type == "preference":
                # Debug: Check what we're getting
                first_npz = sample.get("first_npz_path")
                second_npz = sample.get("second_npz_path")
                if first_npz is None or first_npz == "":
                    print(f"ERROR: first_npz_path is None or empty!")
                    print(f"Sample keys: {sample.keys()}")
                    print(f"Sample: {sample}")
                    raise ValueError(f"first_npz_path is None or empty. Sample type: {sample_type}")
                if second_npz is None or second_npz == "":
                    print(f"ERROR: second_npz_path is None or empty!")
                    print(f"Sample keys: {sample.keys()}")
                    print(f"Sample: {sample}")
                    raise ValueError(f"second_npz_path is None or empty. Sample type: {sample_type}")
                
                # Load frames for both trajectories
                first_frames = load_and_subsample_frames(
                    first_npz,
                    sample["first_frame_indices"]
                )
                second_frames = load_and_subsample_frames(
                    second_npz,
                    sample["second_frame_indices"]
                )
                
                # Convert to PIL
                first_pil = convert_frames_to_pil(first_frames)
                second_pil = convert_frames_to_pil(second_frames)
                
                # Prepare frames for conversation
                first_video, first_extras = self._prepare_frames_for_conversation(first_pil, prefix="first")
                second_video, second_extras = self._prepare_frames_for_conversation(second_pil, prefix="second")
                
                # Build content list
                content_list = []
                self._add_vision_content_to_list(content_list, first_video, first_extras)
                self._add_vision_content_to_list(content_list, second_video, second_extras)
                content_list.append({"type": "text", "text": prompt})
                
            elif sample_type == "progress":
                # Debug: Check what we're getting
                npz = sample.get("npz_path")
                if npz is None or npz == "":
                    print(f"ERROR: npz_path is None or empty for progress sample!")
                    print(f"Sample keys: {sample.keys()}")
                    print(f"Sample: {sample}")
                    raise ValueError(f"npz_path is None or empty. Sample type: {sample_type}")
                
                # Load frames
                frames = load_and_subsample_frames(
                    npz,
                    sample["frame_indices"]
                )
                
                # Convert to PIL
                pil_frames = convert_frames_to_pil(frames)
                
                # Prepare frames for conversation
                video, extras = self._prepare_frames_for_conversation(pil_frames, prefix="video")
                
                # Build content list
                content_list = []
                self._add_vision_content_to_list(content_list, video, extras)
                content_list.append({"type": "text", "text": prompt})
            else:
                raise ValueError(f"Unknown sample type: {sample_type}")
            
            # Create conversation
            conversation = [
                {
                    "role": "user",
                    "content": content_list,
                }
            ]
            
            # Add assistant response (only during training, not inference)
            if not self.inference:
                conversation.append({
                    "role": "assistant",
                    "content": f"{RESPONSE_PREFIX} {answer}"
                })
            
            all_messages.append(conversation)
        
        # Process through processor
        texts = [
            self.processor.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=self.inference,
                fps=1,
            )
            for msg in all_messages
        ]
        
        # Prepare processor inputs
        # Note: Qwen processor handles both multi-image and video modes
        try:
            # Try to import Qwen3's process_vision_info
            from qwen_vl_utils import process_vision_info
            
            # Extract vision information
            vision_result = process_vision_info(all_messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
            
            # Handle both Qwen2.5 (2 values) and Qwen3 (3 values)
            if len(vision_result) == 2:
                # Qwen2.5-VL format
                image_inputs, video_inputs = vision_result
                video_kwargs = {}
            else:
                # Qwen3-VL format
                image_inputs, video_inputs, video_kwargs = vision_result
            
            # split the videos and according metadatas
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
            else:
                video_metadatas = None
            
            # Prepare processor kwargs
            processor_kwargs = {
                "text": texts,
                "padding": True,
                "truncation": False,
                "return_tensors": "pt",
                "video_metadata": video_metadatas,
                "do_resize": False,
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
            
            # Process
            batch_inputs = self.processor(**processor_kwargs)

            
        except ImportError:
            # Fallback: use processor directly (for Qwen2.5-VL without qwen_vl_utils)
            # Extract images/videos manually
            image_inputs = []
            video_inputs = []
            
            for msg in all_messages:
                for turn in msg:
                    if "content" in turn and isinstance(turn["content"], list):
                        for content_item in turn["content"]:
                            if content_item.get("type") == "image":
                                image_inputs.append(content_item["image"])
                            elif content_item.get("type") == "video":
                                video_inputs.append(content_item["video"])
            
            processor_kwargs = {
                "text": texts,
                "padding": True,
                "truncation": False,
                "return_tensors": "pt",
            }
            
            if image_inputs:
                processor_kwargs["images"] = image_inputs
            
            if video_inputs:
                processor_kwargs["videos"] = video_inputs
            
            batch_inputs = self.processor(**processor_kwargs)
        
        # Create labels by masking prompt tokens
        if not self.inference:
            labels = batch_inputs["input_ids"].clone()
            masked_labels = self._mask_labels([labels[i] for i in range(labels.shape[0])])
            batch_inputs["labels"] = torch.stack(masked_labels)
        
        return batch_inputs


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL on VQA dataset")
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the generated HuggingFace dataset",
    )
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        default=None,
        help="Path to evaluation dataset (optional)",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Model name or path (default: Qwen/Qwen3-VL-4B-Instruct)",
    )
    parser.add_argument(
        "--use_multi_image",
        action="store_true",
        help="Use multi-image mode instead of video mode",
    )
    parser.add_argument(
        "--use_unsloth",
        action="store_true",
        help="Use unsloth for faster training (if available)",
    )
    parser.add_argument(
        "--quantization",
        action="store_true",
        help="Use 4-bit quantization (requires unsloth)",
    )
    parser.add_argument(
        "--freeze_vision_tower",
        action="store_true",
        help="Freeze vision encoder (only train LLM + projector, saves memory)",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=0,
        help="LoRA rank for adapter layers (only used with unsloth, set to 0 for full finetuning)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha for adapter layers (only used with unsloth)",
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/vqa_training",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size per device for evaluation",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Save checkpoint strategy",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Evaluation strategy",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=10.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 training",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb", "none"],
        help="Reporting tool (default: tensorboard)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="rfm",
        help="Weights & Biases project name (default: vqa-training)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="clvr",
        help="Weights & Biases entity/team name (optional)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Run name for logging (required)",
    )
    
    args = parser.parse_args()

    # Print configuration
    print("=" * 100)
    print("VQA Training Configuration")
    print("=" * 100)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Model: {args.model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.per_device_train_batch_size}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Use multi-image: {args.use_multi_image}")
    print("=" * 100)

    # Load dataset
    print("Loading dataset...")
    train_dataset = load_from_disk(args.dataset_path)
    print(f"Loaded {len(train_dataset)} training samples")
    
    eval_dataset = None
    if args.eval_dataset_path:
        eval_dataset = load_from_disk(args.eval_dataset_path)
        print(f"Loaded {len(eval_dataset)} evaluation samples")

    # Check unsloth availability
    use_unsloth = args.use_unsloth and HAS_UNSLOTH and "Qwen" in args.model_name
    
    if args.use_unsloth and not HAS_UNSLOTH:
        print("⚠️  Warning: unsloth requested but not installed. Using standard loading.")
        use_unsloth = False
    elif args.use_unsloth and "Qwen" not in args.model_name:
        print("⚠️  Warning: unsloth only supports Qwen models. Using standard loading.")
        use_unsloth = False
    
    # Load model and processor
    print(f"Loading model: {args.model_name}")
    print(f"Using unsloth: {use_unsloth}")
    print(f"Using quantization: {args.quantization}")
    print(f"Freeze vision tower: {args.freeze_vision_tower}")
    
    if use_unsloth:
        # Load with unsloth for faster training
        print("Loading model with unsloth...")
        
        # Determine if we're doing full finetuning or LoRA
        # If lora_rank is set and not freezing vision, we can use LoRA on vision tower
        use_lora = args.lora_rank > 0
        
        if use_lora:
            print(f"Using LoRA with rank={args.lora_rank}, alpha={args.lora_alpha}")
            # Load model for LoRA training
            model, tokenizer = FastVisionModel.from_pretrained(
                args.model_name,
                load_in_4bit=args.quantization,
                use_gradient_checkpointing="unsloth",
                dtype=torch.bfloat16 if args.bf16 else torch.float32,
                device_map=None,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            
            # Apply LoRA to model
            # Target modules: LLM layers + optionally vision tower
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
            
            # Add vision tower modules if not freezing
            if not args.freeze_vision_tower:
                print("Applying LoRA to vision tower as well...")
                target_modules.extend([
                    "visual.transformer.resblocks.*.attn.in_proj_weight",
                    "visual.transformer.resblocks.*.attn.out_proj",
                    "visual.transformer.resblocks.*.mlp.c_fc",
                    "visual.transformer.resblocks.*.mlp.c_proj",
                ])
            
            model = FastVisionModel.get_peft_model(
                model,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.0,
                target_modules=target_modules,
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
        else:
            print("Using full finetuning (no LoRA)")
            # Full finetuning
            model, tokenizer = FastVisionModel.from_pretrained(
                args.model_name,
                load_in_4bit=args.quantization,
                use_gradient_checkpointing="unsloth",
                dtype=torch.bfloat16 if args.bf16 else torch.float32,
                full_finetuning=True,
                device_map=None,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        
        # Load processor separately
        processor = AutoProcessor.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            do_sample_frames=False,
            padding_side="left",
        )
        
        print(f"Model loaded with unsloth: {model.__class__.__name__}")
    else:
        # Standard loading
        processor = AutoProcessor.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            do_sample_frames=False,
            padding_side="left",
        )
        
        # Set pad token if not set
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
        # Load model
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        
        tokenizer = processor.tokenizer
        print(f"Model loaded: {model.__class__.__name__}")
    
    # Set pad token if not set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # Optionally freeze vision tower
    if args.freeze_vision_tower:
        print("Freezing vision tower (visual encoder)...")
        frozen_params = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            # Freeze visual/vision tower parameters
            if any(keyword in name.lower() for keyword in ['visual', 'vision', 'image', 'vit']):
                param.requires_grad = False
                frozen_params += param.numel()
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Frozen {frozen_params / 1e9:.2f}B parameters in vision tower")
        print(f"Trainable parameters: {trainable_params / 1e9:.2f}B ({100 * trainable_params / total_params:.1f}%)")
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Training all layers (vision + LLM)")
        print(f"Trainable parameters: {trainable_params / 1e9:.2f}B ({100 * trainable_params / total_params:.1f}%)")

    # Create data collator
    print("Creating data collator...")
    collator = VQADataCollator(
        processor=processor,
        use_multi_image=args.use_multi_image,
        inference=False,
    )

    # Determine run name
    run_name = args.run_name
    if run_name is None:
        # Auto-generate run name
        model_short = args.model_name.split("/")[-1]
        run_name = f"{model_short}_bs{args.per_device_train_batch_size * args.gradient_accumulation_steps}_lr{args.learning_rate}"
        if args.use_unsloth:
            run_name += "_unsloth"
        if args.quantization:
            run_name += "_4bit"
    
    # Setup reporting
    report_to_list = []
    if args.report_to != "none":
        report_to_list = [args.report_to]
    
    # Initialize wandb if requested
    if args.report_to == "wandb":
        import wandb
        print(f"Initializing Weights & Biases:")
        print(f"  Project: {args.wandb_project}")
        print(f"  Entity: {args.wandb_entity or 'default'}")
        print(f"  Run name: {run_name}")
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "model": args.model_name,
                "batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "lr_scheduler_type": args.lr_scheduler_type,
                "warmup_ratio": args.warmup_ratio,
                "num_epochs": args.num_train_epochs,
                "use_unsloth": args.use_unsloth,
                "quantization": args.quantization,
                "freeze_vision_tower": args.freeze_vision_tower,
                "lora_rank": args.lora_rank if args.lora_rank > 0 else None,
                "lora_alpha": args.lora_alpha if args.lora_rank > 0 else None,
            }
        )
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=run_name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy if eval_dataset else "no",
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to=report_to_list,
        save_total_limit=3,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Create custom callbacks
    callbacks = []
    
    # Add VQA evaluation callback if eval dataset exists
    if eval_dataset:
        print("Adding VQA evaluation callback...")
        vqa_eval_callback = VQAEvaluationCallback(
            eval_dataset=eval_dataset,
            processor=processor,
            collator=collator,
            max_new_tokens=10,
            eval_batch_size=args.per_device_eval_batch_size,
        )
        callbacks.append(vqa_eval_callback)
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    # Train
    print("=" * 100)
    print("Starting training...")
    print("=" * 100)
    trainer.train()

    # Save final model
    print("=" * 100)
    print(f"Saving final model to {args.output_dir}/final")
    print("=" * 100)
    trainer.save_model(os.path.join(args.output_dir, "final"))
    processor.save_pretrained(os.path.join(args.output_dir, "final"))

    print("Training complete!")


if __name__ == "__main__":
    main()
