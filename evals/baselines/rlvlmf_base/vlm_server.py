#!/usr/bin/env python3
"""
VLM evaluation server - drop-in replacement for server.py using Gemini.
Same API, different backend.
"""

import base64
import io
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from vlm_baseline import VLMPreferenceBaseline


# Same payload models as main server.py
class SamplePayload(BaseModel):
    task: str = ""
    prediction_type: str
    chosen_frames_b64: List[str]
    rejected_frames_b64: List[str]
    target_progress_A: List[float] = None
    target_progress_B: List[float] = None


class BatchPayload(BaseModel):
    samples: List[SamplePayload]


def create_app(task_description: str = "", use_temporal_prompts: bool = False) -> FastAPI:
    """Create FastAPI app with VLM backend."""
    
    app = FastAPI(title="VLM Evaluation Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize VLM - fail fast if API key missing
    print("Initializing VLM baseline...")
    print("Frame handling: Using all frames provided by client")
    if use_temporal_prompts:
        print("Prompting: Temporal-aware for >4 total frames (EXPERIMENTAL)")
    else:
        print("Prompting: RL-VLM-F baseline (DEFAULT)")
    
    vlm = VLMPreferenceBaseline(
        vlm_provider="gemini", 
        verbose=True,
        use_temporal_prompts=use_temporal_prompts,
        log_dir="vlm_eval_logs"  # Enable detailed logging
    )
    print("VLM ready!")
    
    # Add shutdown handler to finalize logs
    @app.on_event("shutdown")
    def shutdown_event():
        print("🔄 Finalizing evaluation logs...")
        vlm.finalize_log()
    
    def decode_b64_image(b64_str: str) -> Image.Image:
        """Decode base64 to PIL Image."""
        img_data = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(img_data))
    
    def compute_metrics(samples: List[SamplePayload]) -> Dict[str, Any]:
        """Compute evaluation metrics using VLM - matching RL-VLM-F capabilities."""
        
        results = []
        for sample in samples:
            if sample.prediction_type != "preference":
                continue
            
            # Decode images from base64
            chosen_images = [decode_b64_image(b64) for b64 in sample.chosen_frames_b64]
            rejected_images = [decode_b64_image(b64) for b64 in sample.rejected_frames_b64]
            
            # Query VLM for preference  
            result = vlm.query_preference(
                chosen_images,
                rejected_images,
                task_description or sample.task
            )
            
            # Convert VLM preference to match run_model_eval expectations
            # run_model_eval expects: 1 = correct (chosen preferred), 0 = incorrect, -1 = tie
            # VLM baseline now handles randomization internally and returns is_correct
            if result["is_correct"]:
                prediction = 1  # VLM correctly chose chosen trajectory
            elif result["vlm_preference"] == "tie":
                prediction = -1  # Tie or uncertain
            else:
                prediction = 0  # VLM incorrectly chose rejected trajectory
            
            predictions.append(prediction)
            
            # VLM doesn't provide per-frame rewards, return empty lists
            reward_chosen.append([])
            reward_rejected.append([])
        
        return {
            "predictions": predictions,
            "reward_chosen": reward_chosen, 
            "reward_rejected": reward_rejected
        }
    
    def _empty_metrics():
        return {
            "eval_loss": 0.0,                      # No samples to evaluate
            "eval_accuracy": 0.0,                  # No samples to evaluate
            "eval_reward_diff": 0.0,               # Placeholder
            "eval_avg_reward_chosen": 0.0,         # Placeholder
            "eval_avg_reward_rejected": 0.0,       # Placeholder
            "demo_reward_alignment": []            # Placeholder
        }
    
    @app.post("/evaluate_batch")
    def evaluate_batch(batch: BatchPayload):
        """Evaluate batch using VLM - same API as main server."""
        pref_samples = [s for s in batch.samples if s.prediction_type == "preference"]
        
        if not pref_samples:
            return _empty_metrics()
        
        return compute_metrics(pref_samples)
    
    return app


def main():
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", default="", help="Task description for VLM queries")
    parser.add_argument("--temporal", action="store_true", 
                       help="Enable temporal-aware prompting for multi-frame trajectories (experimental)")
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print(f"VLM Evaluation Server")
    if args.task:
        print(f"Task: {args.task}")
    print(f"Frame handling: Client-driven (uses all frames sent)")
    if args.temporal:
        print(f"Prompting: Temporal-aware (EXPERIMENTAL)")
    else:
        print(f"Prompting: RL-VLM-F baseline (DEFAULT)")
    print(f"Same API as main server.py")
    print(f"{'='*50}\n")
    
    app = create_app(args.task, args.temporal)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main() 