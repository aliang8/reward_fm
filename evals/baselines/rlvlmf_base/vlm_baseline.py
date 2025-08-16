#!/usr/bin/env python3
"""
Direct VLM Baseline for Preference Comparison
Queries Gemini/GPT-4V directly for preferences without training a reward model
"""

import os
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np

# VLM imports (will be installed separately)
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
    """Direct VLM preference queries following RL-VLM-F approach"""
    
    def __init__(
        self,
        vlm_provider: str = "gemini",
        temperature: float = 0.0,  # deterministic 
        verbose: bool = False,
        use_temporal_prompts: bool = False,  # Enable smart temporal prompts for >2 frames
        log_dir: str = None  # Directory to save evaluation logs
    ):
        self.vlm_provider = vlm_provider
        self.temperature = temperature
        self.verbose = verbose
        self.use_temporal_prompts = use_temporal_prompts
        
        # Setup logging
        import os
        from datetime import datetime
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "vlm_eval_logs")
        
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"vlm_eval_{timestamp}.json")
        self.sample_dir = os.path.join(log_dir, f"samples_{timestamp}")
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # Initialize log data
        self.eval_log = {
            "start_time": datetime.now().isoformat(),
            "vlm_provider": vlm_provider,
            "temperature": temperature,
            "use_temporal_prompts": use_temporal_prompts,
            "samples": [],
            "summary": {}
        }
        
        if self.verbose:
            print(f"📝 Logging to: {self.log_file}")
            print(f"🖼️ Sample frames will be saved to: {self.sample_dir}")

        # Setup VLM - fail fast if misconfigured
        if vlm_provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("Install google-generativeai: pip install google-generativeai")
            
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Set GEMINI_API_KEY environment variable")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            
        elif vlm_provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("Install openai: pip install openai")
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Set OPENAI_API_KEY environment variable")
            
            openai.api_key = api_key
            self.client = openai.OpenAI()
        else:
            raise ValueError(f"Unknown VLM provider: {vlm_provider}")
    
    def query_preference(
        self,
        chosen_images: List[Image.Image],
        rejected_images: List[Image.Image],
        task_description: str = ""
    ) -> Dict[str, Any]:
        """
        Query VLM for preference between two trajectories.
        Returns metrics matching RFM API format.
        """
        
        import time
        sample_start_time = time.time()
        sample_id = len(self.eval_log["samples"])
        
        # Use all frames provided by client - warn if many frames
        self._warn_if_many_frames(chosen_images, rejected_images)
        
        # Choose prompting strategy based on frame count and configuration
        total_frames = len(chosen_images) + len(rejected_images)
        use_temporal = self.use_temporal_prompts and total_frames > 4  # More than 2 per trajectory
        
        if use_temporal and self.verbose:
            print(f"📺 Using temporal prompting for {len(chosen_images)} vs {len(rejected_images)} frames")
        elif self.verbose:
            print(f"🔍 Using RL-VLM-F baseline prompting")
        
        # Initialize sample log
        sample_log = {
            "sample_id": sample_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": task_description,
            "num_chosen_frames": len(chosen_images),
            "num_rejected_frames": len(rejected_images),
            "prompting_strategy": "temporal" if use_temporal else "rlvlmf_baseline",
            "success": False,
            "error": None,
            "vlm_response": None,
            "preference": None,
            "is_correct": None,
            "processing_time_seconds": None
        }
        
        try:
            # Save sample frames for debugging
            self._save_sample_frames(sample_id, chosen_images, rejected_images)
            
            # Build prompt based on strategy
            if use_temporal:
                analysis_prompt = self._build_temporal_prompt(task_description, chosen_images, rejected_images)
            else:
                analysis_prompt = self._build_rlvlmf_prompt(task_description)
            
            sample_log["prompt"] = analysis_prompt[:200] + "..." if len(analysis_prompt) > 200 else analysis_prompt
            
            # Query VLM
            if self.vlm_provider == "gemini":
                if use_temporal:
                    preference, raw_response = self._query_gemini_temporal(analysis_prompt, chosen_images, rejected_images)
                else:
                    preference, raw_response = self._query_gemini_rlvlmf(analysis_prompt, chosen_images, rejected_images)
            else:
                if use_temporal:
                    preference, raw_response = self._query_openai_temporal(analysis_prompt, chosen_images, rejected_images)
                else:
                    preference, raw_response = self._query_openai_rlvlmf(analysis_prompt, chosen_images, rejected_images)
            
            # Convert A/B preference to numerical scores
            result = self._preference_output(preference)
            
            # Update sample log with success
            sample_log["success"] = True
            sample_log["vlm_response"] = raw_response[:500] + "..." if len(raw_response) > 500 else raw_response
            sample_log["preference"] = preference
            sample_log["is_correct"] = result["is_correct"]
            sample_log["processing_time_seconds"] = time.time() - sample_start_time
            
            if self.verbose:
                print(f"✅ Sample {sample_id}: {preference} (correct: {result['is_correct']})")
            
        except Exception as e:
            # Log error
            sample_log["error"] = str(e)
            sample_log["processing_time_seconds"] = time.time() - sample_start_time
            
            if self.verbose:
                print(f"❌ Sample {sample_id} failed: {e}")
            
            # Return default result for failed queries
            result = {
                "is_correct": False,
                "vlm_preference": "error"
            }
        
        # Add sample to log
        self.eval_log["samples"].append(sample_log)
        
        # Save log after every sample for immediate visibility
        self._save_log()
        
        return result
    
    def _warn_if_many_frames(self, chosen_images: List[Image.Image], rejected_images: List[Image.Image]):
        """Warn if using many frames (potential cost/performance issue)."""
        total_frames = len(chosen_images) + len(rejected_images)
        
        if total_frames > 2:  # More than 3 per trajectory
            print(f"⚠️  WARNING: Using more than 2 frames!")
            print(f"   This may result in high API costs and slower processing.")
            print(f"   Consider reducing frames in your data pipeline if not needed.")
        # elif total_frames > 2 and self.verbose:
        #     print(f"📊 Using {total_frames} frames for comparison ({len(chosen_images)} vs {len(rejected_images)})")
    
    def _build_rlvlmf_prompt(self, task: str) -> str:
        """Build evaluation prompt matching RL-VLM-F exactly (BASELINE)."""
        if task:
            # RL-VLM-F single prompt format
            prompt = f"""1. What is shown in Image 1?
2. What is shown in Image 2?
3. The goal is {task}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?

Is the goal better achieved in Image 1 or Image 2?
Reply a single line of 0 if the goal is better achieved in Image 1, or 1 if it is better achieved in Image 2.
Reply -1 if the text is unsure or there is no difference."""
        else:
            # Generic version when no specific task provided
            prompt = """1. What is shown in Image 1?
2. What is shown in Image 2?
3. Which image shows better task execution?

Is the goal better achieved in Image 1 or Image 2?
Reply a single line of 0 if the goal is better achieved in Image 1, or 1 if it is better achieved in Image 2.
Reply -1 if the text is unsure or there is no difference."""
        
        return prompt
    
    def _build_temporal_prompt(self, task: str, chosen_images: List[Image.Image], rejected_images: List[Image.Image]) -> str:
        """Build temporal-aware prompt for multi-frame trajectories (EXPERIMENTAL)."""
        
        chosen_count = len(chosen_images)
        rejected_count = len(rejected_images)
        
        if task:
            prompt = f"""You are comparing two robot trajectories for the task: {task}

TRAJECTORY A: A sequence of {chosen_count} images showing the robot's actions over time
TRAJECTORY B: A sequence of {rejected_count} images showing the robot's actions over time

Analyze each trajectory considering:
1. INITIAL STATE: How does each trajectory start?
2. PROGRESS: How does the robot's progress toward the goal evolve over time?
3. FINAL OUTCOME: What is the end result of each trajectory?
4. EFFICIENCY: Which trajectory accomplishes the task more directly?
5. TEMPORAL CONSISTENCY: Which shows smoother, more coherent action sequences?

The goal is {task}. Considering the full temporal sequence, which trajectory better accomplishes this goal?

Reply a single line of 0 if Trajectory A is better, or 1 if Trajectory B is better.
Reply -1 if there is no clear difference."""
        else:
            prompt = f"""You are comparing two robot trajectories:

TRAJECTORY A: A sequence of {chosen_count} images showing robot actions over time  
TRAJECTORY B: A sequence of {rejected_count} images showing robot actions over time

Analyze each trajectory considering:
1. INITIAL STATE: Starting position and setup
2. PROGRESS: How actions unfold over time  
3. FINAL OUTCOME: End result achieved
4. EFFICIENCY: Directness of approach
5. TEMPORAL CONSISTENCY: Smoothness of action sequence

Which trajectory shows better overall task execution considering the full temporal sequence?

Reply a single line of 0 if Trajectory A is better, or 1 if Trajectory B is better.
Reply -1 if there is no clear difference."""
        
        return prompt
    
    def _query_gemini(
        self,
        prompt: str,
        chosen_frames: List[Image.Image],
        rejected_frames: List[Image.Image]
    ) -> str:
        """Query Gemini API for preference - exact RL-VLM-F format."""
        
        # RL-VLM-F exact structure: "Consider..." + Image 1: + images + Image 2: + images + prompt
        query_list = [
            "Consider the following two images:\nImage 1:",
            *chosen_frames,
            "Image 2:",
            *rejected_frames,
            prompt
        ]
        
        # Use same generation config as RL-VLM-F
        response = self.model.generate_content(
            query_list,
            generation_config=genai.types.GenerationConfig(temperature=self.temperature),
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
            ]
        )
        
        # Parse response exactly like RL-VLM-F
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
            
            # Parse exactly like RL-VLM-F
            if "-1" in result:
                return "tie"  # Will map to random choice
            elif "0" in result:
                return "A"  # Image 1 (chosen) is better
            elif "1" in result:
                return "B"  # Image 2 (rejected) is better
            else:
                return "tie"  # Default to tie if unclear
                
        except:
            if self.verbose:
                print("Gemini response parsing failed")
            return "tie"
    
    def _query_openai_rlvlmf(
        self,
        prompt: str,
        chosen_frames: List[Image.Image],
        rejected_frames: List[Image.Image]
    ) -> tuple[str, str]:
        """Query GPT-4V API for preference - exact RL-VLM-F format (BASELINE)."""
        
        # Convert images to base64 for API
        def img_to_base64(img: Image.Image) -> str:
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode()
        
        # RL-VLM-F exact format: "Consider..." + Image 1 + images + Image 2 + images + prompt
        content = [
            {
                "type": "text",
                "text": "Consider the following two images:\nImage 1:"
            }
        ]
        
        # Add chosen frames (Image 1)
        for frame in chosen_frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_to_base64(frame)}",
                    "detail": "high"
                }
            })
        
        content.append({
            "type": "text", 
            "text": "Image 2:"
        })
        
        # Add rejected frames (Image 2)
        for frame in rejected_frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_to_base64(frame)}",
                    "detail": "high"
                }
            })
        
        content.append({
            "type": "text",
            "text": prompt
        })
        
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=1000  # Same as RL-VLM-F
        )
        
        # Parse response exactly like RL-VLM-F
        try:
            full_response = response.choices[0].message.content.strip()
            result = full_response.split("\n")[-1].strip()
            if self.verbose:
                print(f"GPT-4V (RL-VLM-F): {result}")
            
            # Parse exactly like RL-VLM-F (Pattern 1)
            if "-1" in result:
                preference = "tie"
            elif "0" in result:
                preference = "A"  # 0 = Image 1 (chosen) is better
            elif "1" in result:
                preference = "B"  # 1 = Image 2 (rejected) is better
            else:
                preference = "tie"  # DEFAULT to -1 if none found (like RL-VLM-F)
            
            return preference, full_response
                
        except Exception as e:
            if self.verbose:
                print("GPT-4V (RL-VLM-F) response parsing failed")
            return "tie", f"Error: {str(e)}"  # Exception also defaults to -1
    
    def _query_openai_temporal(
        self,
        prompt: str,
        chosen_frames: List[Image.Image],
        rejected_frames: List[Image.Image]
    ) -> tuple[str, str]:
        """Query GPT-4V API with temporal-aware prompting (EXPERIMENTAL)."""
        
        # Convert images to base64 for API
        def img_to_base64(img: Image.Image) -> str:
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode()
        
        # Temporal format: prompt + TRAJECTORY A + images + TRAJECTORY B + images
        content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "\n\nTRAJECTORY A:"}
        ]
        
        # Add chosen frames 
        for frame in chosen_frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_to_base64(frame)}",
                    "detail": "high"
                }
            })
        
        content.append({
            "type": "text", 
            "text": "\nTRAJECTORY B:"
        })
        
        # Add rejected frames
        for frame in rejected_frames:
            content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_to_base64(frame)}",
                    "detail": "high"
                }
            })
        
        content.append({
            "type": "text",
            "text": "\nBased on the full sequences above, which trajectory is better?"
        })
        
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=1000
        )
        
        # Parse response
        try:
            full_response = response.choices[0].message.content.strip()
            result = full_response.split("\n")[-1].strip()
            if self.verbose:
                print(f"GPT-4V (Temporal): {result}")
            
            # Parse for 0/1/-1 responses
            if "-1" in result:
                preference = "tie"
            elif "0" in result:
                preference = "A"  # Trajectory A (chosen) is better  
            elif "1" in result:
                preference = "B"  # Trajectory B (rejected) is better
            else:
                preference = "tie"  # DEFAULT to -1 if none found (like RL-VLM-F)
            
            return preference, full_response
                
        except Exception as e:
            if self.verbose:
                print("GPT-4V (Temporal) response parsing failed")
            return "tie", f"Error: {str(e)}"  # Exception also defaults to -1
    
    def _query_gemini_rlvlmf(
        self,
        prompt: str,
        chosen_frames: List[Image.Image],
        rejected_frames: List[Image.Image]
    ) -> tuple[str, str]:
        """Query Gemini API for preference - exact RL-VLM-F format (BASELINE)."""
        
        # RL-VLM-F exact structure: "Consider..." + Image 1: + images + Image 2: + images + prompt
        query_list = [
            "Consider the following two images:\nImage 1:",
            *chosen_frames,
            "Image 2:",
            *rejected_frames,
            prompt
        ]
        
        # Use same generation config as RL-VLM-F
        response = self.model.generate_content(
            query_list,
            generation_config=genai.types.GenerationConfig(temperature=self.temperature),
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
            ]
        )
        
        # Parse response exactly like RL-VLM-F
        try:
            full_response = response.text
            result = full_response.split("\n")[-1].strip().lstrip()
            if self.verbose:
                print(f"Gemini (RL-VLM-F): {result}")
            
            # Parse exactly like RL-VLM-F (Pattern 1)
            if "-1" in result:
                preference = "tie"  # -1 response
            elif "0" in result:
                preference = "A"  # 0 = Image 1 (chosen) is better
            elif "1" in result:
                preference = "B"  # 1 = Image 2 (rejected) is better
            else:
                preference = "tie"  # DEFAULT to -1 if none found (like RL-VLM-F)
            
            return preference, full_response
                
        except Exception as e:
            if self.verbose:
                print("Gemini (RL-VLM-F) response parsing failed")
            return "tie", f"Error: {str(e)}"  # Exception also defaults to -1
    
    def _query_gemini_temporal(
        self,
        prompt: str,
        chosen_frames: List[Image.Image],
        rejected_frames: List[Image.Image]
    ) -> tuple[str, str]:
        """Query Gemini API with temporal-aware prompting (EXPERIMENTAL)."""
        
        # Temporal structure: prompt + "TRAJECTORY A:" + images + "TRAJECTORY B:" + images
        query_list = [
            prompt,
            "\n\nTRAJECTORY A:",
            *chosen_frames,
            "\nTRAJECTORY B:",
            *rejected_frames,
            "\nBased on the full sequences above, which trajectory is better?"
        ]
        
        response = self.model.generate_content(
            query_list,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=50  # Slightly more tokens for temporal reasoning
            ),
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
            ]
        )
        
        # Parse response 
        try:
            full_response = response.text
            result = full_response.split("\n")[-1].strip().lstrip()
            if self.verbose:
                print(f"Gemini (Temporal): {result}")
            
            # Parse for 0/1/-1 responses (same logic as RL-VLM-F)
            if "-1" in result:
                preference = "tie"
            elif "0" in result:
                preference = "A"  # Trajectory A (chosen) is better
            elif "1" in result:
                preference = "B"  # Trajectory B (rejected) is better
            else:
                preference = "tie"  # DEFAULT to -1 if none found (like RL-VLM-F)
            
            return preference, full_response
                
        except Exception as e:
            if self.verbose:
                print("Gemini (Temporal) response parsing failed")
            return "tie", f"Error: {str(e)}"  # Exception also defaults to -1
    
    def _preference_output(self, preference: str) -> Dict[str, Any]:
        """Convert VLM preference to metrics - matching RL-VLM-F exactly."""
        
        # RL-VLM-F only gets discrete responses (0, 1, -1), no probabilities
        if preference == "A":
            # VLM said "0" - chosen trajectory is better (correct)
            is_correct = True
        elif preference == "B":
            # VLM said "1" - rejected trajectory is better (incorrect)
            is_correct = False
        else:  # preference == "tie"
            # VLM said "-1" or unclear - treat as incorrect (failed to choose correctly)
            is_correct = False
        
        return {
            "is_correct": is_correct,            # Only thing we can compute
            "vlm_preference": preference         # For debugging
        } 

    def _save_sample_frames(self, sample_id: int, chosen_images: List[Image.Image], rejected_images: List[Image.Image]):
        """Save sample frames for debugging and analysis."""
        import os
        
        sample_folder = os.path.join(self.sample_dir, f"sample_{sample_id:03d}")
        os.makedirs(sample_folder, exist_ok=True)
        
        # Save chosen frames
        chosen_folder = os.path.join(sample_folder, "chosen")
        os.makedirs(chosen_folder, exist_ok=True)
        for i, img in enumerate(chosen_images):
            img.save(os.path.join(chosen_folder, f"frame_{i:02d}.jpg"))
        
        # Save rejected frames  
        rejected_folder = os.path.join(sample_folder, "rejected")
        os.makedirs(rejected_folder, exist_ok=True)
        for i, img in enumerate(rejected_images):
            img.save(os.path.join(rejected_folder, f"frame_{i:02d}.jpg"))
    
    def _save_log(self):
        """Save evaluation log to JSON file."""
        import json
        from datetime import datetime
        
        # Update summary stats
        samples = self.eval_log["samples"]
        if samples:
            successful_samples = [s for s in samples if s["success"]]
            self.eval_log["summary"] = {
                "total_samples": len(samples),
                "successful_samples": len(successful_samples),
                "failed_samples": len(samples) - len(successful_samples),
                "accuracy": sum(s["is_correct"] for s in successful_samples) / len(successful_samples) if successful_samples else 0.0,
                "avg_processing_time": sum(s["processing_time_seconds"] for s in samples if s["processing_time_seconds"]) / len(samples),
                "last_updated": datetime.now().isoformat()
            }
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.eval_log, f, indent=2)
        
        if self.verbose:
            summary = self.eval_log["summary"]
            print(f"📝 Log saved: {summary['successful_samples']}/{summary['total_samples']} successful, accuracy: {summary['accuracy']:.2%}")
    
    def finalize_log(self):
        """Finalize and save the evaluation log."""
        from datetime import datetime
        
        self.eval_log["end_time"] = datetime.now().isoformat()
        self._save_log()
        
        if self.verbose:
            print(f"🏁 Evaluation complete. Full log saved to: {self.log_file}")
            print(f"🖼️ Sample frames saved to: {self.sample_dir}") 