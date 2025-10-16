#!/usr/bin/env python3
"""GVL baseline for preference comparison using task completion percentages."""

import os
import time
import json
import base64
import random
import cv2
import re
import requests
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from PIL import Image
import numpy as np


class GVLPreferenceBaseline:
    """GVL preference queries using task completion percentages."""
    
    def __init__(
        self,
        api_key: str,
        temperature: float = 0.0,
        verbose: bool = False,
        debug: bool = False,
        log_dir: str = None,
        max_frames: int = 15,
        offset: float = 0.5
    ):
        self.api_key = api_key
        self.temperature = temperature
        self.verbose = verbose
        self.debug = debug
        self.max_frames = max_frames
        self.offset = offset
        
        # Debug failure tracking
        self.failure_count = 0
        self.max_failures_to_debug = 3
        self.debug_failures_dir = None
        
        # Setup logging
        self._setup_logging(log_dir)
    
    def _setup_logging(self, log_dir: str):
        """Initialize logging infrastructure."""
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "gvl_eval_logs")
        
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"gvl_eval_{timestamp}.json")
        
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
            "api_key_set": bool(self.api_key),
            "temperature": self.temperature,
            "debug_mode": self.debug,
            "max_frames": self.max_frames,
            "offset": self.offset,
            "samples": [],
            "summary": {}
        }
    
    def query_preference(
        self,
        chosen_images: List[Image.Image],
        rejected_images: List[Image.Image],
        task_description: str = ""
    ) -> Dict[str, Any]:
        """Query GVL for preference between trajectories using task completion percentages."""
        start_time = time.time()
        sample_id = len(self.eval_log["samples"])
        
        # DEBUG: Print task that reaches GVL baseline
        print(f"🎯 GVL Baseline received task: '{task_description}'")
        
        if self.verbose:
            print(f"🎯 Frame selection: {len(chosen_images)} chosen, {len(rejected_images)} rejected")
        
        # Initialize log entry
        sample_log = self._init_sample_log(
            sample_id, task_description,
            len(chosen_images), len(rejected_images)
        )
        
        try:
            # Save frames if debug mode
            if self.debug:
                self._save_frames(sample_id, chosen_images, rejected_images)
            
            # Convert PIL images to numpy arrays for GVL processing
            chosen_frames = self._pil_to_numpy_array(chosen_images)
            rejected_frames = self._pil_to_numpy_array(rejected_images)
            
            # Get task completion percentages for both trajectories
            chosen_completions = self._get_task_completion(chosen_frames, task_description)
            rejected_completions = self._get_task_completion(rejected_frames, task_description)
            
            if self.verbose:
                print(f"🔍 Chosen trajectory completions: {chosen_completions}")
                print(f"🔍 Rejected trajectory completions: {rejected_completions}")
            
            # Determine preference based on final completion percentages
            preference, comparison_result = self._compare_trajectories(
                chosen_completions, rejected_completions
            )
            
            # Randomize image order to avoid position bias
            chosen_is_first = random.choice([True, False])
            
            if self.verbose:
                print(f"🎲 Randomized order: {'chosen first' if chosen_is_first else 'rejected first'}")
            
            # Determine correctness based on randomization
            if preference == "chosen":
                is_correct = True
            elif preference == "tie":
                is_correct = False  # Treating ties as incorrect for now
            else:  # preference == "rejected"
                is_correct = False
            
            # Update log
            sample_log.update({
                "success": True,
                "chosen_completions": chosen_completions,
                "rejected_completions": rejected_completions,
                "preference": preference,
                "comparison_result": comparison_result,
                "chosen_is_first": chosen_is_first,
                "is_correct": is_correct,
                "processing_time_seconds": time.time() - start_time
            })
            
            # Debug failures: save detailed info for first 3 incorrect predictions
            if not is_correct and self.debug and self.failure_count < self.max_failures_to_debug:
                self._save_failure_debug(
                    sample_id, task_description, chosen_images[-1], rejected_images[-1],
                    preference, comparison_result, sample_log, chosen_is_first
                )
                self.failure_count += 1
            
            if self.verbose:
                print(f"✅ Sample {sample_id}: {preference} (correct: {is_correct})")
            
            result = {
                "is_correct": is_correct, 
                "vlm_preference": preference,
                "chosen_completions": chosen_completions,
                "rejected_completions": rejected_completions
            }
            
        except Exception as e:
            sample_log.update({
                "error": str(e),
                "processing_time_seconds": time.time() - start_time
            })
            
            if self.verbose:
                print(f"❌ Sample {sample_id} failed: {e}")
            
            result = {
                "is_correct": False, 
                "vlm_preference": "error",
                "chosen_completions": [],
                "rejected_completions": []
            }
        
        self.eval_log["samples"].append(sample_log)
        self._save_log()
        
        return result
    
    def _pil_to_numpy_array(self, pil_images: List[Image.Image]) -> np.ndarray:
        """Convert list of PIL images to numpy array (N, H, W, 3)."""
        if not pil_images:
            return np.array([])
        
        # Convert PIL to numpy and stack
        numpy_frames = []
        for pil_img in pil_images:
            # Convert to RGB if needed and then to numpy
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            numpy_frame = np.array(pil_img)
            numpy_frames.append(numpy_frame)
        
        return np.array(numpy_frames)
    
    def _get_task_completion(self, frames_array: np.ndarray, task_description: str) -> List[Optional[float]]:
        """Get task completion percentages for a trajectory using GVL."""
        if frames_array.size == 0:
            return []
        
        try:
            # Create GVL analyzer
            analyzer = GVLAnalyzer(
                api_key=self.api_key,
                frames_array=frames_array,
                task_description=task_description,
                max_frames=self.max_frames,
                offset=self.offset
            )
            
            # Run analysis
            completion_list = analyzer.run_analysis()
            return completion_list
            
        except Exception as e:
            if self.verbose:
                print(f"❌ GVL analysis failed: {e}")
            return [None] * len(frames_array)
    
    def _compare_trajectories(self, chosen_completions: List[Optional[float]], 
                            rejected_completions: List[Optional[float]]) -> Tuple[str, Dict]:
        """Compare trajectories based on task completion percentages."""
        # Get final completion percentages (last valid value)
        chosen_final = self._get_final_completion(chosen_completions)
        rejected_final = self._get_final_completion(rejected_completions)
        
        comparison_result = {
            "chosen_final": chosen_final,
            "rejected_final": rejected_final,
            "chosen_trajectory": chosen_completions,
            "rejected_trajectory": rejected_completions
        }
        
        # Handle cases where we couldn't get completion percentages
        if chosen_final is None and rejected_final is None:
            return "tie", comparison_result
        elif chosen_final is None:
            return "rejected", comparison_result
        elif rejected_final is None:
            return "chosen", comparison_result
        
        # Compare final completion percentages
        completion_diff = chosen_final - rejected_final
        
        # Use a small threshold to handle noise
        threshold = 5.0  # 5% difference threshold
        
        if abs(completion_diff) < threshold:
            preference = "tie"
        elif completion_diff > 0:
            preference = "chosen"
        else:
            preference = "rejected"
        
        comparison_result["completion_diff"] = completion_diff
        comparison_result["threshold"] = threshold
        
        return preference, comparison_result
    
    def _get_final_completion(self, completions: List[Optional[float]]) -> Optional[float]:
        """Get the final (last valid) completion percentage from a list."""
        # Return the last non-None value
        for completion in reversed(completions):
            if completion is not None:
                return completion
        return None
    
    def _init_sample_log(self, sample_id: int, task: str,
                        chosen_count: int, rejected_count: int) -> Dict:
        """Initialize sample log entry."""
        return {
            "sample_id": sample_id,
            "task": task,
            "num_chosen_frames": chosen_count,
            "num_rejected_frames": rejected_count,
            "prompting_strategy": "gvl_task_completion",
            "success": False,
            "error": None,
            "chosen_completions": None,
            "rejected_completions": None,
            "preference": None,
            "comparison_result": None,
            "chosen_is_first": None,
            "is_correct": None,
            "processing_time_seconds": None
        }
    
    def _save_frames(self, sample_id: int, chosen_images: List[Image.Image],
                    rejected_images: List[Image.Image]) -> None:
        """Save frames if debug mode is enabled."""
        if not self.sample_dir:
            return
        
        sample_folder = os.path.join(self.sample_dir, f"sample_{sample_id:03d}")
        
        # Save selected frames (last frame from each trajectory)
        selected_folder = os.path.join(sample_folder, "selected_chosen")
        os.makedirs(selected_folder, exist_ok=True)
        if chosen_images:
            chosen_images[-1].save(os.path.join(selected_folder, "frame_00.jpg"))
        
        selected_folder = os.path.join(sample_folder, "selected_rejected")
        os.makedirs(selected_folder, exist_ok=True)
        if rejected_images:
            rejected_images[-1].save(os.path.join(selected_folder, "frame_00.jpg"))
        
        # Save full trajectories occasionally (every 20th sample)
        if sample_id % 20 == 0:
            # Save all chosen frames
            original_chosen_folder = os.path.join(sample_folder, "original_chosen")
            os.makedirs(original_chosen_folder, exist_ok=True)
            for i, img in enumerate(chosen_images):
                img.save(os.path.join(original_chosen_folder, f"frame_{i:02d}.jpg"))
            
            # Save all rejected frames
            original_rejected_folder = os.path.join(sample_folder, "original_rejected")
            os.makedirs(original_rejected_folder, exist_ok=True)
            for i, img in enumerate(rejected_images):
                img.save(os.path.join(original_rejected_folder, f"frame_{i:02d}.jpg"))
    
    def _save_log(self):
        """Save current log to file."""
        # Update summary statistics
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
                           rejected_frame: Image.Image, preference: str, comparison_result: Dict,
                           sample_log: Dict, chosen_is_first: bool) -> None:
        """Save detailed debug information for a failed prediction."""
        if not self.debug_failures_dir:
            return
            
        failure_dir = os.path.join(self.debug_failures_dir, f"failure_{self.failure_count + 1:03d}")
        os.makedirs(failure_dir, exist_ok=True)
        
        # Save images
        chosen_frame.save(os.path.join(failure_dir, "chosen_image1.jpg"))
        rejected_frame.save(os.path.join(failure_dir, "rejected_image2.jpg"))
        
        # Create detailed analysis
        analysis = {
            "sample_id": sample_id,
            "task": task,
            "gvl_prediction": preference,
            "chosen_is_first": chosen_is_first,
            "correct_answer": "chosen trajectory should have higher completion percentage",
            "comparison_result": comparison_result,
            "why_wrong": f"GVL preferred {preference} trajectory over chosen trajectory",
            "sample_log": sample_log,
            "image_mapping": {
                "chosen_image1.jpg": "chosen_trajectory (should have higher completion %)",
                "rejected_image2.jpg": "rejected_trajectory (should have lower completion %)"
            },
            "analysis_notes": [
                "Look at chosen_image1.jpg vs rejected_image2.jpg",
                "Check completion percentages in comparison_result",
                "Consider if ground truth labels might be wrong",
                f"GVL said: {preference} - check if this is reasonable"
            ]
        }
        
        # Save analysis as JSON
        with open(os.path.join(failure_dir, "failure_analysis.json"), 'w') as f:
            json.dump(analysis, f, indent=2)
            
        # Save human-readable summary
        summary_text = f"""FAILURE DEBUG #{self.failure_count + 1}
==============================

TASK: {task}

GVL PREDICTION: {preference}
- GVL preferred: {preference} trajectory
- Should have preferred: chosen trajectory

COMPLETION PERCENTAGES:
- Chosen final: {comparison_result.get('chosen_final', 'N/A')}%
- Rejected final: {comparison_result.get('rejected_final', 'N/A')}%
- Difference: {comparison_result.get('completion_diff', 'N/A')}%

IMAGES:
- chosen_image1.jpg = chosen trajectory (should have higher completion %)
- rejected_image2.jpg = rejected trajectory (should have lower completion %)

COMPARISON DETAILS:
{json.dumps(comparison_result, indent=2)}

QUESTIONS TO INVESTIGATE:
1. Do the completion percentages make sense given the task?
2. Are the ground truth labels correct for this sample?
3. Is the task description clear enough for GVL?
4. Which trajectory actually shows better progress?
"""
        
        with open(os.path.join(failure_dir, "README.txt"), 'w') as f:
            f.write(summary_text)
            
        print(f"🐛 DEBUG: Saved failure #{self.failure_count + 1} to {failure_dir}")
        print(f"   GVL chose {preference} (should be chosen), task: {task[:50]}...")
    
    def finalize_log(self):
        """Finalize and save log."""
        self.eval_log["end_time"] = datetime.now().isoformat()
        self._save_log()
        
        if self.verbose:
            print(f"🏁 Log saved to: {self.log_file}")
            if self.debug and self.sample_dir:
                print(f"🖼️  Sample frames saved to: {self.sample_dir}")


class GVLAnalyzer:
    """Adapted GVL analyzer for task completion percentage estimation."""
    
    def __init__(
        self,
        api_key: str,
        frames_array: np.ndarray,
        task_description: str,
        max_frames: int = 15,
        offset: float = 0.5
    ):
        self.api_key = api_key
        self.frames_array = frames_array
        self.task_description = task_description
        self.max_frames = max_frames
        self.offset = offset
        self.frames_info: List[Dict] = []

    def extract_frames_from_memory(self) -> None:
        """Extract and encode frames from memory array."""
        total_frames = self.frames_array.shape[0]
        if total_frames == 0:
            self.frames_info = []
            return

        # Sampling logic similar to original GVL
        if total_frames <= self.max_frames:
            frame_count = total_frames
            frame_interval = 1.0
            temp_indices = []
            for i in range(frame_count):
                sample_time = self.offset + i * frame_interval
                frame_index = int(sample_time)
                if 0 <= frame_index < total_frames:
                    temp_indices.append(frame_index)
        else:
            # Always include first and last frames
            temp_indices = [0, total_frames - 1]
            inner_count = self.max_frames - 2
            if total_frames > 2 and inner_count > 0:
                frame_interval = (total_frames - 2) / float(inner_count)
                for i in range(inner_count):
                    sample_time = self.offset + i * frame_interval
                    frame_index = int(1 + sample_time)
                    if 1 <= frame_index < (total_frames - 1):
                        temp_indices.append(frame_index)
            
            temp_indices = sorted(set(temp_indices))

        # Convert to JPEG + base64
        temp_frames_info = []
        for idx in temp_indices:
            frame = self.frames_array[idx]  # (H, W, 3)
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            temp_frames_info.append({
                "gt_index": len(temp_frames_info) + 1,
                "base64": frame_b64
            })

        self.frames_info = temp_frames_info

    def shuffle_frames(self) -> None:
        """Randomly shuffle frames."""
        indices = list(range(1, len(self.frames_info) + 1))
        random.shuffle(indices)
        for frame, new_idx in zip(self.frames_info, indices):
            frame["shuffled_index"] = new_idx

    def build_prompt_parts(self) -> List[Dict]:
        """Build prompt parts for GVL analysis."""
        initial_frame = next((f for f in self.frames_info if f["gt_index"] == 1), None)
        if not initial_frame:
            if not self.frames_info:
                return []
            initial_frame = self.frames_info[0]

        prompt1 = (
            f"You are an expert roboticist tasked to predict task completion percentages "
            f"for frames of a robot for the task of {self.task_description}. "
            f"The task completion percentages are between 0 and 100, where 100 corresponds to full task completion. "
            f"Note that these frames are in random order, so please pay attention to the individual frames. "
            f"\nInitial robot scene:\nThis frame:"
        )

        prompt2 = (
            f" shows the initial robot scene, where the task completion percentage is 0.\n\n"
            f"Now, for the task of *{self.task_description}*, output the task completion percentage "
            f"for the following frames that are presented in random order. "
            f"Format your response in JSON as follows, making sure to include all frames:\n\n"
            f"[\n"
            f'  {{"frame_number": i, "frame_description": "...", "task_completion_percentage": 0-100}}\n'
            f"]\n"
        )

        parts = []
        parts.append({"text": prompt1})
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": initial_frame["base64"]
            }
        })
        parts.append({"text": prompt2})

        # Add frames sorted by shuffled index
        frames_sorted = sorted(self.frames_info, key=lambda f: f["shuffled_index"])
        for i, frame in enumerate(frames_sorted, start=1):
            parts.append({"text": f"Frame {i}:"})
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": frame["base64"]
                }
            })

        return parts

    def stream_inference(self, parts: List[Dict]) -> str:
        """Call Gemini streaming API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?alt=sse&key={self.api_key}"
        body = {
            "contents": [{"parts": parts}]
        }
        headers = {"Content-Type": "application/json"}

        full_text = ""
        with requests.post(url, headers=headers, json=body, stream=True) as resp:
            resp.raise_for_status()
            
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                    
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    break
                    
                try:
                    data_json = json.loads(data_str)
                    candidates = data_json.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts_list = content.get("parts", [])
                        if parts_list:
                            text_piece = parts_list[0].get("text", "")
                            full_text += text_piece
                except json.JSONDecodeError:
                    continue
                    
        return full_text

    @staticmethod
    def extract_json_from_response(text: str) -> str:
        """Extract JSON from response text."""
        # Try fenced code block first
        code_block_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")
        match = code_block_pattern.search(text)
        if match:
            return match.group(1).strip()

        # Try array pattern
        array_pattern = re.compile(r"\[\s*\{[\s\S]*?\}\s*\]")
        match = array_pattern.search(text)
        if match:
            return match.group(0).strip()

        return ""

    @staticmethod
    def parse_model_output(model_text: str) -> Optional[List[Dict]]:
        """Parse model output to extract completion data."""
        json_str = GVLAnalyzer.extract_json_from_response(model_text)
        if not json_str:
            return None
            
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
            return None
        except (json.JSONDecodeError, TypeError):
            return None

    def run_analysis(self) -> List[Optional[float]]:
        """Run complete GVL analysis pipeline."""
        # Extract frames
        self.extract_frames_from_memory()
        if not self.frames_info:
            return []

        # Shuffle frames
        self.shuffle_frames()

        # Build prompt
        parts = self.build_prompt_parts()
        if not parts:
            return []

        # Run inference
        model_output_text = self.stream_inference(parts)

        # Parse results
        result_data = self.parse_model_output(model_output_text)
        if result_data is None:
            return []

        # Map results back to original frame order
        mapped_by_shuffled = {}
        for item in result_data:
            sidx = item.get("frame_number")
            if isinstance(sidx, int):
                mapped_by_shuffled[sidx] = item

        # Update frames_info with model results
        for frame in self.frames_info:
            sidx = frame.get("shuffled_index")
            if sidx in mapped_by_shuffled:
                frame["model_output"] = mapped_by_shuffled[sidx]
            else:
                frame["model_output"] = None

        # Extract completion percentages in original order
        frames_in_gt_order = sorted(self.frames_info, key=lambda f: f["gt_index"])
        task_completion_list = []
        for f in frames_in_gt_order:
            if f["model_output"] is not None:
                completion = f["model_output"].get("task_completion_percentage")
                task_completion_list.append(completion)
            else:
                task_completion_list.append(None)

        return task_completion_list
