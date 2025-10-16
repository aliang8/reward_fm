#!/usr/bin/env python3
"""Direct VLM baseline for preference comparison."""

import os
import time
import json
import base64
import random
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from PIL import Image
import numpy as np

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class VLMPreferenceBaseline:
    """Direct VLM preference queries following RL-VLM-F."""
    
    def __init__(
        self,
        vlm_provider: str = "gemini",
        temperature: float = 0.0,
        verbose: bool = False,
        debug: bool = False,
        log_dir: str = None
    ):
        self.vlm_provider = vlm_provider
        self.temperature = temperature
        self.verbose = verbose
        self.debug = debug
        
        # Debug failure tracking
        self.failure_count = 0
        self.max_failures_to_debug = 3
        self.debug_failures_dir = None
        
        # Setup logging
        self._setup_logging(log_dir)
        
        # Setup VLM
        self._setup_vlm()
    
    def _setup_logging(self, log_dir: str):
        """Initialize logging infrastructure."""
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "vlm_eval_logs")
        
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"vlm_eval_{timestamp}.json")
        
        if self.debug:
            self.sample_dir = os.path.join(log_dir, f"samples_{timestamp}")
            os.makedirs(self.sample_dir, exist_ok=True)
            # Create debug failures directory
            self.debug_failures_dir = os.path.join(self.sample_dir, "debug_failures")
            os.makedirs(self.debug_failures_dir, exist_ok=True)
        else:
            self.sample_dir = None
        
        self.eval_log = {
            "start_time": datetime.now().isoformat(),
            "vlm_provider": self.vlm_provider,
            "temperature": self.temperature,
            "debug_mode": self.debug,
            "samples": [],
            "summary": {}
        }
        
        if self.verbose:
            print(f"📝 Logging to: {self.log_file}")
            if self.debug:
                print(f"🐛 Debug mode: saving frames to {self.sample_dir}")
    
    def _setup_vlm(self):
        """Initialize VLM provider."""
        if self.vlm_provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("pip install google-generativeai")
            
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Set GEMINI_API_KEY environment variable")
            
            genai.configure(api_key=api_key)
            # Note: Original RL-VLM-F uses 'gemini-pro-vision' which is deprecated
            # Using 'gemini-1.5-pro' for better reasoning and accuracy
            self.model = genai.GenerativeModel('gemini-1.5-flash') # gemini-1.5-flash
            
        elif self.vlm_provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("pip install openai")
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Set OPENAI_API_KEY environment variable")
            
            openai.api_key = api_key
            self.client = openai.OpenAI()
        else:
            raise ValueError(f"Unknown provider: {self.vlm_provider}")
    
    def query_preference(
        self,
        chosen_images: List[Image.Image],
        rejected_images: List[Image.Image],
        task_description: str = ""
    ) -> Dict[str, Any]:
        """Query VLM for preference between trajectories."""
        start_time = time.time()
        sample_id = len(self.eval_log["samples"])
        
        # DEBUG: Print task that reaches VLM baseline
        print(f"🎯 VLM Baseline received task: '{task_description}'")
        
        # Select last frame from each trajectory
        chosen_frame = chosen_images[-1]
        rejected_frame = rejected_images[-1]
        
        # Randomize image order to avoid position bias
        # chosen_is_first = True means chosen goes to Image A, False means chosen goes to Image B
        chosen_is_first = random.choice([True, False])
        
        if chosen_is_first:
            image_a, image_b = chosen_frame, rejected_frame
            image_a_label, image_b_label = "chosen", "rejected"
        else:
            image_a, image_b = rejected_frame, chosen_frame
            image_a_label, image_b_label = "rejected", "chosen"
        
        if self.verbose:
            print(f"🎯 Frame selection: {len(chosen_images)} → 1, {len(rejected_images)} → 1")
            print(f"🎲 Randomized order: Image A = {image_a_label}, Image B = {image_b_label}")
        
        # Initialize log entry
        sample_log = self._init_sample_log(
            sample_id, task_description, 
            len(chosen_images), len(rejected_images)
        )
        
        try:
            # Save frames if debug mode
            if self.debug:
                self._save_frames(sample_id, chosen_images, rejected_images, 
                                chosen_frame, rejected_frame)
            
            # Query VLM with randomized images
            prompt = self._build_prompt(task_description)
            sample_log["prompt"] = prompt if self.debug else prompt[:200] + "..."
            sample_log["chosen_is_first"] = chosen_is_first  # Track randomization for debugging
            
            if self.vlm_provider == "gemini":
                preference, raw_response = self._query_gemini(prompt, image_a, image_b)
            else:
                preference, raw_response = self._query_openai(prompt, image_a, image_b)
            
            # Process result - adjust for randomized order
            # preference is "A"/"B"/"tie" based on VLM's choice of Image A vs Image B
            # We need to map this back to chosen vs rejected based on randomization
            if preference == "A":
                # VLM chose Image A
                vlm_chose_chosen = chosen_is_first  # If chosen is first, A=chosen, else A=rejected
            elif preference == "B":
                # VLM chose Image B  
                vlm_chose_chosen = not chosen_is_first  # If chosen is first, B=rejected, else B=chosen
            else:  # preference == "tie"
                vlm_chose_chosen = None  # Tie case
            
            # Determine correctness
            if vlm_chose_chosen is True:
                is_correct = True  # VLM correctly chose the chosen trajectory
            elif vlm_chose_chosen is None:
                # For ties, treating as incorrect since we expect clear preferences
                is_correct = False
            else:  # vlm_chose_chosen is False
                is_correct = False  # VLM incorrectly chose the rejected trajectory
            
            # Update log
            sample_log.update({
                "success": True,
                "vlm_response": raw_response if self.debug else raw_response[:500] + "...",
                "preference": preference,
                "vlm_chose_chosen": vlm_chose_chosen,
                "is_correct": is_correct,
                "processing_time_seconds": time.time() - start_time
            })
            
            # Debug failures: save detailed info for first 3 incorrect predictions
            if not is_correct and self.debug and self.failure_count < self.max_failures_to_debug:
                self._save_failure_debug(
                    sample_id, task_description, chosen_frame, rejected_frame, 
                    preference, raw_response, sample_log, chosen_is_first
                )
                self.failure_count += 1
            
            if self.verbose:
                print(f"✅ Sample {sample_id}: {preference} (correct: {is_correct})")
            
            result = {"is_correct": is_correct, "vlm_preference": preference}
            
        except Exception as e:
            sample_log.update({
                "error": str(e),
                "processing_time_seconds": time.time() - start_time
            })
            
            if self.verbose:
                print(f"❌ Sample {sample_id} failed: {e}")
            
            result = {"is_correct": False, "vlm_preference": "error"}
        
        self.eval_log["samples"].append(sample_log)
        self._save_log()
        
        return result
    
    def _init_sample_log(self, sample_id: int, task: str, 
                         chosen_count: int, rejected_count: int) -> Dict:
        """Initialize sample log entry."""
        return {
            "sample_id": sample_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": task,
            "num_chosen_frames": chosen_count,
            "num_rejected_frames": rejected_count,
            "selected_frames": 1,
            "strategy": "last_frame",
            "success": False,
            "error": None,
            "vlm_response": None,
            "preference": None,
            "is_correct": None,
            "processing_time_seconds": None
        }
    
    def _build_prompt(self, task: str) -> str:
        """Build RL-VLM-F prompt - exact match to original paper."""
        base = """ Each frame comes from a robot trajectory. (Think causally and use image comparison to verify any confusion between the base of the robot and the end effector)
1. What is shown in the first image (Image A)?
2. What is shown in the second image (Image B)?
3. For this question here is the Goal Text: {goal_text}  
Is the goal being better achieved in Image A or Image B? 
Reply a single line of 0 if the goal is better achieved in Image A, or 1 if it is better achieved in Image B.
Reply -1 if the text is really unsure about either making any progress towards the goal or there is absolutely no difference. (only use this if there is no discernible signs of progress in either image or no discernible difference between the progress in the two images for the given task)
"""
        
        if task:
            goal_text = f"The goal is {task}."
        else:
            goal_text = "Which image shows better task execution?"
        
        return base.format(goal_text=goal_text)
    
    def _query_gemini(self, prompt: str, chosen: Image.Image, 
                     rejected: Image.Image) -> Tuple[str, str]:
        """Query Gemini for preference."""
        query = [
            "Consider the following two images:\nImage 1:",
            chosen,
            "Image 2:",
            rejected,
            prompt
        ]
        
        response = self.model.generate_content(
            query,
            generation_config=genai.types.GenerationConfig(temperature=self.temperature),
            safety_settings=[
                {"category": cat, "threshold": "BLOCK_NONE"}
                for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                           "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS"]
            ]
        )
        
        try:
            # DEBUG: Check response status and safety ratings
            if self.verbose:
                print(f"🔍 DEBUG Response candidates: {len(response.candidates) if response.candidates else 0}")
                if response.candidates:
                    candidate = response.candidates[0]
                    print(f"🔍 DEBUG Finish reason: {candidate.finish_reason}")
                    print(f"🔍 DEBUG Safety ratings: {candidate.safety_ratings}")
            
            full_response = response.text
            # Fix parsing: strip whitespace first, then split and take last non-empty line
            result = full_response.strip()
            if "\n" in result:
                lines = [line.strip() for line in result.split("\n") if line.strip()]
                result = lines[-1] if lines else ""
            
            # DEBUG: Print full response to understand what Gemini is returning
            if self.verbose:
                print(f"🔍 DEBUG Gemini Full Response: '{full_response}'")
                print(f"🔍 DEBUG Parsed Result: '{result}'")
                print(f"🔍 DEBUG Response Length: {len(full_response)}")
            
            # Parse response
            if "-1" in result:
                return "tie", full_response
            elif "0" in result:
                return "A", full_response  # Image 1 (chosen)
            elif "1" in result:
                return "B", full_response  # Image 2 (rejected)
            else:
                if self.verbose:
                    print(f"⚠️  Unexpected response format: '{result}', defaulting to tie")
                return "tie", full_response
                
        except Exception as e:
            if self.verbose:
                print(f"Gemini parsing failed: {e}")
            return "tie", f"Error: {str(e)}"
    
    def _query_openai(self, prompt: str, chosen: Image.Image,
                     rejected: Image.Image) -> Tuple[str, str]:
        """Query GPT-4V for preference."""
        def to_base64(img: Image.Image) -> str:
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode()
        
        content = [
            {"type": "text", "text": "Consider the following two images:\nImage 1:"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{to_base64(chosen)}", "detail": "high"
            }},
            {"type": "text", "text": "Image 2:"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{to_base64(rejected)}", "detail": "high"
            }},
            {"type": "text", "text": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=1000
        )
        
        try:
            full_response = response.choices[0].message.content.strip()
            result = full_response.split("\n")[-1].strip()
            
            if self.verbose:
                print(f"GPT-4V: {result}")
            
            # Parse response
            if "-1" in result:
                return "tie", full_response
            elif "0" in result:
                return "A", full_response
            elif "1" in result:
                return "B", full_response
            else:
                return "tie", full_response
                
        except Exception as e:
            if self.verbose:
                print(f"GPT-4V parsing failed: {e}")
            return "tie", f"Error: {str(e)}"
    
    def _save_frames(self, sample_id: int, orig_chosen: List[Image.Image],
                    orig_rejected: List[Image.Image], sel_chosen: Image.Image,
                    sel_rejected: Image.Image):
        """Save frames for debugging."""
        if not self.debug or not self.sample_dir:
            return
        
        sample_folder = os.path.join(self.sample_dir, f"sample_{sample_id:03d}")
        
        # Save original trajectories only for every 20th sample (0, 20, 40, ...)
        if sample_id % 20 == 0:
            for traj_type, images in [("original_chosen", orig_chosen), 
                                      ("original_rejected", orig_rejected)]:
                folder = os.path.join(sample_folder, traj_type)
                os.makedirs(folder, exist_ok=True)
                for i, img in enumerate(images):
                    img.save(os.path.join(folder, f"frame_{i:02d}.jpg"))
            
            if self.verbose:
                print(f"💾 Saved full trajectories for sample {sample_id} (every 20th sample)")
        
        # Always save selected frames for all samples
        for traj_type, img in [("selected_chosen", sel_chosen),
                               ("selected_rejected", sel_rejected)]:
            folder = os.path.join(sample_folder, traj_type)
            os.makedirs(folder, exist_ok=True)
            img.save(os.path.join(folder, "frame_00.jpg"))
    
    def _save_log(self):
        """Save evaluation log."""
        samples = self.eval_log["samples"]
        if samples:
            successful = [s for s in samples if s["success"]]
            self.eval_log["summary"] = {
                "total_samples": len(samples),
                "successful_samples": len(successful),
                "failed_samples": len(samples) - len(successful),
                "accuracy": sum(s["is_correct"] for s in successful) / len(successful) if successful else 0.0,
                "avg_time": sum(s["processing_time_seconds"] for s in samples if s["processing_time_seconds"]) / len(samples),
                "last_updated": datetime.now().isoformat()
            }
        
        with open(self.log_file, 'w') as f:
            json.dump(self.eval_log, f, indent=2)
        
        if self.verbose and samples:
            s = self.eval_log["summary"]
            print(f"📝 {s['successful_samples']}/{s['total_samples']} successful, accuracy: {s['accuracy']:.2%}")
    
    def _save_failure_debug(self, sample_id: int, task: str, chosen_frame: Image.Image, 
                           rejected_frame: Image.Image, preference: str, raw_response: str,
                           sample_log: Dict, chosen_is_first: bool) -> None:
        """Save detailed debug information for a failed prediction."""
        if not self.debug_failures_dir:
            return
            
        failure_dir = os.path.join(self.debug_failures_dir, f"failure_{self.failure_count + 1:03d}")
        os.makedirs(failure_dir, exist_ok=True)
        
        # Save images
        chosen_frame.save(os.path.join(failure_dir, "chosen_image1.jpg"))
        rejected_frame.save(os.path.join(failure_dir, "rejected_image2.jpg"))
        
        # Determine which image the VLM actually chose
        if chosen_is_first:
            image_a_type, image_b_type = "chosen", "rejected"
        else:
            image_a_type, image_b_type = "rejected", "chosen"
            
        vlm_chose_desc = f"Image A ({image_a_type})" if preference == "A" else f"Image B ({image_b_type})" if preference == "B" else "tie"
        
        # Create detailed analysis
        analysis = {
            "sample_id": sample_id,
            "task": task,
            "vlm_prediction": preference,
            "chosen_is_first": chosen_is_first,
            "vlm_chose": vlm_chose_desc,
            "correct_answer": f"Image {'A' if chosen_is_first else 'B'} (chosen trajectory)",
            "randomization": {
                "image_a_contains": image_a_type + "_trajectory",
                "image_b_contains": image_b_type + "_trajectory",
                "should_choose": "A" if chosen_is_first else "B"
            },
            "why_wrong": f"VLM chose {preference} but should have chosen {'A' if chosen_is_first else 'B'} (chosen trajectory)",
            "full_vlm_response": raw_response,
            "prompt_used": self._build_prompt(task),
            "image_mapping": {
                "chosen_image1.jpg": "chosen_trajectory (should be preferred)",
                "rejected_image2.jpg": "rejected_trajectory (should not be preferred)",
                "note": f"In this query: Image A = {image_a_type}, Image B = {image_b_type}"
            },
            "analysis_notes": [
                "Look at chosen_image1.jpg vs rejected_image2.jpg",
                "Check if VLM's reasoning makes sense given the task",
                "Consider if ground truth labels might be wrong",
                f"VLM said: {preference} - check if this is reasonable"
            ],
            "sample_log": sample_log
        }
        
        # Save analysis as JSON
        with open(os.path.join(failure_dir, "failure_analysis.json"), 'w') as f:
            json.dump(analysis, f, indent=2)
            
        # Save human-readable summary
        summary_text = f"""FAILURE DEBUG #{self.failure_count + 1}
==============================

TASK: {task}

RANDOMIZATION: chosen_is_first = {chosen_is_first}
- Image A contains: {image_a_type} trajectory  
- Image B contains: {image_b_type} trajectory
- VLM should choose: {'A' if chosen_is_first else 'B'} (chosen trajectory)

VLM PREDICTION: {preference}
- VLM chose: {vlm_chose_desc}
- Should have chosen: {analysis['correct_answer']}

IMAGES (always same files):
- chosen_image1.jpg = chosen trajectory (CORRECT choice)
- rejected_image2.jpg = rejected trajectory (WRONG choice)
- Note: Image A = {image_a_type}, Image B = {image_b_type} in this query

VLM REASONING:
{raw_response}

PROMPT USED:
{self._build_prompt(task)}

WHY THIS IS WRONG:
{analysis['why_wrong']}

QUESTIONS TO INVESTIGATE:
1. Does the VLM's visual reasoning make sense?
2. Are the ground truth labels correct for this sample?
3. Is the task description clear enough?
4. Which image actually shows better progress?
"""
        
        with open(os.path.join(failure_dir, "README.txt"), 'w') as f:
            f.write(summary_text)
            
        print(f"🐛 DEBUG: Saved failure #{self.failure_count + 1} to {failure_dir}")
        should_choose = "A" if chosen_is_first else "B"
        print(f"   VLM chose {preference} (should be {should_choose}), task: {task[:50]}...")
    
    def finalize_log(self):
        """Finalize and save log."""
        self.eval_log["end_time"] = datetime.now().isoformat()
        self._save_log()
        
        if self.verbose:
            print(f"🏁 Log saved to: {self.log_file}")
            if self.debug and self.sample_dir:
                print(f"🖼️ Frames saved to: {self.sample_dir}") 