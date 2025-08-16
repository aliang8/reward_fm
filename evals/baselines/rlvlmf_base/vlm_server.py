#!/usr/bin/env python3
"""VLM evaluation server - following server.py."""

import base64
import io
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from vlm_baseline import VLMPreferenceBaseline


# API payload models
#TODO: temp for import issue, consilidate
class SamplePayload(BaseModel):
    task: str = ""
    prediction_type: str
    chosen_frames_b64: List[str]
    rejected_frames_b64: List[str]
    target_progress_A: List[float] = None
    target_progress_B: List[float] = None


class BatchPayload(BaseModel):
    samples: List[SamplePayload]


def create_app(task_description: str = "", debug: bool = False) -> FastAPI:
    """Create FastAPI app with VLM backend."""
    
    app = FastAPI(title="VLM Evaluation Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize VLM
    print("🚀 Initializing VLM baseline...")
    print("📍 Smart frame selection: last frame from each trajectory")
    print("🔍 RL-VLM-F prompting: 2-frame comparison")
    if debug:
        print("🐛 Debug mode enabled")
    
    vlm = VLMPreferenceBaseline(
        vlm_provider="gemini", 
        verbose=True,
        debug=debug,
        log_dir="vlm_eval_logs"
    )
    print("VLM ready!")
    
    # Add shutdown handler to finalize logs
    @app.on_event("shutdown")
    def shutdown_event():
        print("🔄 Finalizing logs...")
        vlm.finalize_log()
    
    def decode_b64_image(b64_str: str) -> Image.Image:
        """Decode base64 to PIL Image."""
        img_data = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(img_data))
    
    def compute_batch_outputs(samples: List[SamplePayload]) -> Dict[str, Any]:
        """Process samples and return raw VLM predictions matching qwen server format."""
        
        predictions = []
        reward_chosen = []
        reward_rejected = []
        
        for sample in samples:
            if sample.prediction_type != "preference":
                continue
            
            # DEBUG: Print task info from multiple sources
            print(f"🔍 Sample {len(predictions)} Task Debug:")
            print(f"  - sample.task: '{sample.task}'")
            print(f"  - task_description param: '{task_description}'")
            print(f"  - Using sample task: '{sample.task}'")
            
            # Decode images from base64
            chosen_images = [decode_b64_image(b64) for b64 in sample.chosen_frames_b64]
            rejected_images = [decode_b64_image(b64) for b64 in sample.rejected_frames_b64]
            
            # Query VLM for preference - always use the sample's task, not server override
            task_to_use = sample.task  # Use actual task from evaluation data
            result = vlm.query_preference(
                chosen_images,
                rejected_images,
                task_to_use
            )
            
            # Convert VLM preference to match run_model_eval expectations
            # run_model_eval expects: 1 = correct (chosen preferred), 0 = incorrect, -1 = tie
            # VLM returns: "A" (chosen better), "B" (rejected better), "tie" 
            if result["vlm_preference"] == "A":
                prediction = 1  # Chosen is better = CORRECT
            elif result["vlm_preference"] == "B":
                prediction = 0  # Rejected is better = INCORRECT
            else:  # tie or error
                prediction = -1  # Tie or uncertain
            
            predictions.append(prediction)
            
            # VLM doesn't provide per-frame rewards, return empty lists
            reward_chosen.append([])
            reward_rejected.append([])
        
        return {
            "predictions": predictions,
            "reward_chosen": reward_chosen, 
            "reward_rejected": reward_rejected
        }
    
    @app.post("/evaluate_batch")
    def evaluate_batch(batch: BatchPayload):
        """Evaluate batch using VLM - matching qwen server API."""
        pref_samples = [s for s in batch.samples if s.prediction_type == "preference"]
        
        if not pref_samples:
            return {
                "predictions": [],
                "reward_chosen": [],
                "reward_rejected": [],
            }
        
        return compute_batch_outputs(pref_samples)
    
    return app


def main():
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", default="", help="Task description")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print(f"VLM Evaluation Server")
    if args.task:
        print(f"Task: {args.task}")
    print(f"API: Same as main server.py")
    print(f"{'='*50}\n")
    
    app = create_app(args.task, args.debug)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main() 