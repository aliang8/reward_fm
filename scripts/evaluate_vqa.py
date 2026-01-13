#!/usr/bin/env python3
"""
Evaluation script for VQA-trained models.

This script evaluates trained Qwen3-VL models on a VQA dataset,
computing metrics for both preference and progress samples.

Usage:
    python scripts/evaluate_vqa.py \\
        --model_path /path/to/trained/model \\
        --dataset_path /path/to/test/dataset \\
        --output_path /path/to/results.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the collator
sys.path.insert(0, str(Path(__file__).parent))
from train_vqa_sft import VQADataCollator

# Try to import unsloth
try:
    from unsloth import FastVisionModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    FastVisionModel = None

RESPONSE_PREFIX = "ANS:"


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


def evaluate_preference_samples(
    model,
    processor,
    dataset,
    batch_size: int = 1,
    max_new_tokens: int = 10,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Evaluate preference samples (binary classification: 1 or 2).
    
    Args:
        model: Trained model
        processor: Processor
        dataset: Dataset with preference samples
        batch_size: Batch size for inference
        max_new_tokens: Max tokens to generate
        device: Device to use
        
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    # Filter preference samples
    pref_samples = [s for s in dataset if s['sample_type'] == 'preference']
    
    if not pref_samples:
        return {"count": 0, "accuracy": 0.0}
    
    print(f"\nEvaluating {len(pref_samples)} preference samples...")
    
    # Create collator for inference
    collator = VQADataCollator(
        processor=processor,
        use_multi_image=False,
        inference=True,  # Important: inference mode
    )
    
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(pref_samples), batch_size), desc="Preference eval"):
            batch = pref_samples[i:i+batch_size]
            
            # Process batch
            try:
                inputs = collator(batch)
                
                # Move to device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                # Generate
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding
                    temperature=None,
                    top_p=None,
                )
                
                # Decode
                generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
                
                # Extract answers and check correctness
                for j, (sample, gen_text) in enumerate(zip(batch, generated_texts)):
                    predicted = extract_answer_from_generation(gen_text)
                    ground_truth = sample['answer']
                    
                    is_correct = (predicted == ground_truth) if predicted else False
                    
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    predictions.append({
                        'sample_idx': i + j,
                        'task': sample['task'],
                        'ground_truth': ground_truth,
                        'predicted': predicted,
                        'correct': is_correct,
                        'generated_text': gen_text,
                    })
                    
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "count": total,
        "correct": correct,
        "accuracy": accuracy,
        "predictions": predictions,
    }


def evaluate_progress_samples(
    model,
    processor,
    dataset,
    batch_size: int = 1,
    max_new_tokens: int = 10,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Evaluate progress samples (regression: 0-100).
    
    Args:
        model: Trained model
        processor: Processor
        dataset: Dataset with progress samples
        batch_size: Batch size for inference
        max_new_tokens: Max tokens to generate
        device: Device to use
        
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    # Filter progress samples
    prog_samples = [s for s in dataset if s['sample_type'] == 'progress']
    
    if not prog_samples:
        return {"count": 0, "mae": 0.0, "rmse": 0.0}
    
    print(f"\nEvaluating {len(prog_samples)} progress samples...")
    
    # Create collator for inference
    collator = VQADataCollator(
        processor=processor,
        use_multi_image=False,
        inference=True,  # Important: inference mode
    )
    
    errors = []
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prog_samples), batch_size), desc="Progress eval"):
            batch = prog_samples[i:i+batch_size]
            
            # Process batch
            try:
                inputs = collator(batch)
                
                # Move to device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                # Generate
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding
                    temperature=None,
                    top_p=None,
                )
                
                # Decode
                generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
                
                # Extract answers and compute error
                for j, (sample, gen_text) in enumerate(zip(batch, generated_texts)):
                    predicted_str = extract_answer_from_generation(gen_text)
                    ground_truth_str = sample['answer']
                    
                    # Parse as integers
                    try:
                        predicted = int(predicted_str) if predicted_str else None
                        ground_truth = int(ground_truth_str)
                        
                        if predicted is not None:
                            error = abs(predicted - ground_truth)
                            errors.append(error)
                        else:
                            # Failed to parse prediction
                            errors.append(100)  # Max error
                            predicted = None
                        
                        predictions.append({
                            'sample_idx': i + j,
                            'task': sample['task'],
                            'ground_truth': ground_truth,
                            'predicted': predicted,
                            'error': error if predicted is not None else 100,
                            'generated_text': gen_text,
                        })
                    except (ValueError, TypeError) as e:
                        print(f"Error parsing answer for sample {i+j}: {e}")
                        errors.append(100)  # Max error
                        predictions.append({
                            'sample_idx': i + j,
                            'task': sample['task'],
                            'ground_truth': ground_truth_str,
                            'predicted': predicted_str,
                            'error': 100,
                            'generated_text': gen_text,
                        })
                    
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if errors:
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    else:
        mae = 0.0
        rmse = 0.0
    
    return {
        "count": len(errors),
        "mae": float(mae),
        "rmse": float(rmse),
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA-trained model")
    
    # Model and data
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to test dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./eval_results.json",
        help="Path to save evaluation results",
    )
    
    # Inference settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Max new tokens to generate (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to evaluate (for testing, default: all)",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save individual predictions to output file",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VQA Model Evaluation")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output path: {args.output_path}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk(args.dataset_path)
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Count sample types
    sample_types = {}
    for sample in dataset:
        sample_type = sample['sample_type']
        sample_types[sample_type] = sample_types.get(sample_type, 0) + 1
    
    print(f"Sample types: {sample_types}")
    
    # Load model and processor
    print(f"\nLoading model from: {args.model_path}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        do_sample_frames=False,
        padding_side="left",
    )
    
    # Set pad token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # Evaluate
    results = {
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "total_samples": len(dataset),
        "sample_types": sample_types,
    }
    
    # Evaluate preference samples
    if sample_types.get('preference', 0) > 0:
        pref_results = evaluate_preference_samples(
            model=model,
            processor=processor,
            dataset=dataset,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        
        results['preference'] = {
            "count": pref_results["count"],
            "correct": pref_results["correct"],
            "accuracy": pref_results["accuracy"],
        }
        
        if args.save_predictions:
            results['preference']['predictions'] = pref_results['predictions']
        
        print("\n" + "=" * 80)
        print("Preference Results:")
        print("=" * 80)
        print(f"Samples: {pref_results['count']}")
        print(f"Correct: {pref_results['correct']}")
        print(f"Accuracy: {pref_results['accuracy']:.4f} ({pref_results['accuracy']*100:.2f}%)")
    
    # Evaluate progress samples
    if sample_types.get('progress', 0) > 0:
        prog_results = evaluate_progress_samples(
            model=model,
            processor=processor,
            dataset=dataset,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        
        results['progress'] = {
            "count": prog_results["count"],
            "mae": prog_results["mae"],
            "rmse": prog_results["rmse"],
        }
        
        if args.save_predictions:
            results['progress']['predictions'] = prog_results['predictions']
        
        print("\n" + "=" * 80)
        print("Progress Results:")
        print("=" * 80)
        print(f"Samples: {prog_results['count']}")
        print(f"MAE: {prog_results['mae']:.2f}")
        print(f"RMSE: {prog_results['rmse']:.2f}")
    
    # Save results
    print("\n" + "=" * 80)
    print(f"Saving results to: {args.output_path}")
    print("=" * 80)
    
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Evaluation complete!")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
