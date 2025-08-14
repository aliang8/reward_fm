#!/usr/bin/env python3
"""Quick VLM baseline setup test."""

import os
from vlm_baseline import VLMPreferenceBaseline


def test_setup():
    """Verify VLM baseline is properly configured."""
    
    print("🧪 Testing VLM Setup")
    print("=" * 30)
    
    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Missing GEMINI_API_KEY")
        print("Get key: https://makersuite.google.com/app/apikey")
        return False
    
    print("✅ API key found")
    
    # Test initialization
    try:
        # Test RL-VLM-F baseline
        vlm_baseline = VLMPreferenceBaseline(verbose=False, use_temporal_prompts=False)
        print("✅ VLM initialized (RL-VLM-F baseline)")
        
        # Test temporal prompting
        vlm_temporal = VLMPreferenceBaseline(verbose=False, use_temporal_prompts=True)
        print("✅ VLM initialized (temporal prompting)")
        
    except Exception as e:
        print(f"❌ Init failed: {e}")
        return False
    
    print("\n🎉 Setup looks good!")
    print("Usage:")
    print("  # RL-VLM-F baseline (DEFAULT):")
    print("  python vlm_server.py --task 'robot manipulation'")
    print("  # Temporal-aware prompting (EXPERIMENTAL):")
    print("  python vlm_server.py --task 'robot manipulation' --temporal")
    print("\nNote: VLM will use all frames sent by client.")
    print("      Warnings will appear if >2 total frames used.")
    print("      Temporal prompting activates when >4 total frames.")
    return True


def test_vlm_logging():
    """Test VLM logging functionality with dummy data."""
    from PIL import Image
    import numpy as np
    import os
    import json
    import time
    from datetime import datetime
    
    print("\n🧪 Testing VLM logging...")
    
    try:
        # Test logging without initializing VLM
        log_dir = "test_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"vlm_eval_{timestamp}.json")
        sample_dir = os.path.join(log_dir, f"samples_{timestamp}")
        os.makedirs(sample_dir, exist_ok=True)
        
        print(f"📝 Testing log file creation: {log_file}")
        
        # Create dummy log data
        eval_log = {
            "start_time": datetime.now().isoformat(),
            "vlm_provider": "gemini",
            "temperature": 0.0,
            "use_temporal_prompts": False,
            "samples": [],
            "summary": {}
        }
        
        # Add a test sample
        sample_log = {
            "sample_id": 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": "test robot manipulation task",
            "num_chosen_frames": 1,
            "num_rejected_frames": 1,
            "prompting_strategy": "rlvlmf_baseline",
            "success": True,
            "error": None,
            "vlm_response": "Looking at the two images:\n\nImage 1 shows a robot arm reaching toward a red block on a table. The robot appears to be in the initial phase of attempting to grasp the object.\n\nImage 2 shows the robot arm has successfully grasped the red block and is lifting it off the table surface.\n\nThe goal is robot manipulation. There is a clear difference between Image 1 and Image 2 in terms of achieving the goal.\n\n1",
            "preference": "B",
            "is_correct": False,
            "processing_time_seconds": 2.1
        }
        
        eval_log["samples"].append(sample_log)
        
        # Test summary calculation
        successful_samples = [s for s in eval_log["samples"] if s["success"]]
        eval_log["summary"] = {
            "total_samples": len(eval_log["samples"]),
            "successful_samples": len(successful_samples),
            "failed_samples": len(eval_log["samples"]) - len(successful_samples),
            "accuracy": sum(s["is_correct"] for s in successful_samples) / len(successful_samples) if successful_samples else 0.0,
            "avg_processing_time": sum(s["processing_time_seconds"] for s in eval_log["samples"] if s["processing_time_seconds"]) / len(eval_log["samples"]),
            "last_updated": datetime.now().isoformat()
        }
        
        # Save to file
        with open(log_file, 'w') as f:
            json.dump(eval_log, f, indent=2)
        
        # Test frame saving
        dummy_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        sample_folder = os.path.join(sample_dir, "sample_000")
        os.makedirs(sample_folder, exist_ok=True)
        
        chosen_folder = os.path.join(sample_folder, "chosen")
        rejected_folder = os.path.join(sample_folder, "rejected")
        os.makedirs(chosen_folder, exist_ok=True)
        os.makedirs(rejected_folder, exist_ok=True)
        
        dummy_img.save(os.path.join(chosen_folder, "frame_00.jpg"))
        dummy_img.save(os.path.join(rejected_folder, "frame_00.jpg"))
        
        # Verify everything was created
        if os.path.exists(log_file):
            print("✅ JSON log file created")
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                print(f"✅ Log contains {len(log_data['samples'])} samples")
                print(f"✅ Sample VLM response preview: {log_data['samples'][0]['vlm_response'][:100]}...")
                print(f"✅ Accuracy: {log_data['summary']['accuracy']:.1%}")
        else:
            print("❌ JSON log file missing")
            return False
            
        if os.path.exists(sample_folder):
            print("✅ Sample frames directory created")
            print(f"✅ Frame files: {len(os.listdir(chosen_folder))} chosen, {len(os.listdir(rejected_folder))} rejected")
        else:
            print("❌ Sample frames directory missing")
            return False
            
        print("✅ Logging test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔧 Testing VLM Baseline Setup")
    print("=" * 40)
    
    # Test basic setup
    setup_ok = test_setup()
    
    # Test logging functionality
    logging_ok = test_vlm_logging()
    
    if setup_ok and logging_ok:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed") 