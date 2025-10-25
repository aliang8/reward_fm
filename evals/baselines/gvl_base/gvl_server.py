#!/usr/bin/env python3
"""GVL evaluation server - following vlm_server.py structure."""

import base64
import io
import os
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from gvl_baseline import GVLPreferenceBaseline


# API payload models
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
    
    app = FastAPI(title="GVL Evaluation Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize GVL
    print("🚀 Initializing GVL baseline...")
    print("📍 Task completion percentage comparison")
    print("🔍 GVL prompting: trajectory-based completion analysis")
    if debug:
        print("🐛 Debug mode enabled")
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    gvl = GVLPreferenceBaseline(
        api_key=api_key,
        verbose=True,
        debug=debug,
        log_dir="gvl_eval_logs"
    )
    print("GVL ready!")
    
    # Add shutdown handler to finalize logs
    @app.on_event("shutdown")
    def shutdown_event():
        print("🔄 Finalizing logs...")
        gvl.finalize_log()
    
    def decode_b64_image(b64_str: str) -> Image.Image:
        """Decode base64 to PIL Image."""
        img_data = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(img_data))
    
    def compute_batch_outputs(samples: List[SamplePayload]) -> Dict[str, Any]:
        """Process samples and return raw GVL predictions matching run_model_eval format."""
        
        predictions = []
        reward_chosen = []
        reward_rejected = []
        
        for sample in samples:
            if sample.prediction_type != "preference":
                continue
            
            # DEBUG: Print task info
            print(f"🔍 Sample {len(predictions)} Task Debug:")
            print(f"  - sample.task: '{sample.task}'")
            print(f"  - task_description param: '{task_description}'")
            print(f"  - Using sample task: '{sample.task}'")
            
            # Decode images from base64
            chosen_images = [decode_b64_image(b64) for b64 in sample.chosen_frames_b64]
            rejected_images = [decode_b64_image(b64) for b64 in sample.rejected_frames_b64]
            
            # Query GVL for preference - always use the sample's task
            task_to_use = sample.task
            result = gvl.query_preference(
                chosen_images,
                rejected_images,
                task_to_use
            )
            
            # Convert GVL preference to match run_model_eval expectations
            # run_model_eval expects: 1 = correct (chosen preferred), 0 = incorrect, -1 = tie
            # GVL baseline now handles randomization internally and returns is_correct
            if result["is_correct"]:
                prediction = 1  # GVL correctly determined chosen has higher completion
            elif result["vlm_preference"] == "tie":
                prediction = -1  # Tie or uncertain
            else:
                prediction = 0  # GVL incorrectly chose rejected trajectory
            
            predictions.append(prediction)
            
            # Extract completion percentages as reward predictions
            # GVL provides task completion percentages (0-100), normalize to 0-1 for rewards
            chosen_completions = []
            rejected_completions = []
            
            # Get completion data from result
            if "chosen_completions" in result and result["chosen_completions"]:
                chosen_completions = [
                    c / 100.0 if c is not None else 0.0 
                    for c in result["chosen_completions"]
                ]
            
            if "rejected_completions" in result and result["rejected_completions"]:
                rejected_completions = [
                    c / 100.0 if c is not None else 0.0 
                    for c in result["rejected_completions"]
                ]
            
            # If no completions available, use empty lists
            reward_chosen.append(chosen_completions)
            reward_rejected.append(rejected_completions)
            
            if len(predictions) <= 3:  # Debug first few samples
                print(f"🔍 Sample {len(predictions)} GVL completions:")
                print(f"   Chosen: {chosen_completions}")
                print(f"   Rejected: {rejected_completions}")
        
        return {
            "predictions": predictions,
            "reward_chosen": reward_chosen,
            "reward_rejected": reward_rejected,
        }
    
    @app.post("/evaluate_batch")
    async def evaluate_batch(payload: BatchPayload) -> Dict[str, Any]:
        """Main evaluation endpoint."""
        try:
            print(f"📦 Received batch with {len(payload.samples)} samples")
            
            # Process all samples
            outputs = compute_batch_outputs(payload.samples)
            
            print(f"✅ Processed {len(outputs['predictions'])} predictions")
            return outputs
            
        except Exception as e:
            print(f"❌ Error processing batch: {e}")
            # Return empty responses to avoid breaking the client
            return {
                "predictions": [],
                "reward_chosen": [],
                "reward_rejected": [],
            }
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "model": "GVL"}
    
    return app


def main():
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="GVL Evaluation Server")
    parser.add_argument("--port", type=int, default=8003, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--task", type=str, default="", help="Default task description (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    print(f"🌟 Starting GVL evaluation server...")
    print(f"🔧 Configuration:")
    print(f"   - Host: {args.host}")
    print(f"   - Port: {args.port}")
    print(f"   - Task: '{args.task}' (optional)")
    print(f"   - Debug: {args.debug}")
    print(f"   - API Key: {'✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Missing'}")
    print()
    
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable is required")
        print("   Please set it with: export GEMINI_API_KEY='your-key'")
        return
    
    app = create_app(task_description=args.task, debug=args.debug)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
