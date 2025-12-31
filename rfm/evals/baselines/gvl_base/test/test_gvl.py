#!/usr/bin/env python3
"""Test script for GVL baseline setup verification."""

import os
import sys
import tempfile
import numpy as np
from PIL import Image

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gvl_baseline import GVLPreferenceBaseline


def test_gvl_setup():
    """Test GVL baseline initialization and basic functionality."""
    print("🧪 Testing GVL baseline setup...")
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        print("   Please set it with: export GEMINI_API_KEY='your-key'")
        return False
    else:
        print(f"✅ API key set (length: {len(api_key)})")
    
    # Test initialization
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            gvl_baseline = GVLPreferenceBaseline(
                api_key=api_key,
                verbose=False,
                debug=False,
                log_dir=temp_dir
            )
            print("✅ GVL baseline initialized")
            
            # Test with dummy images
            print("🖼️  Testing with dummy images...")
            
            # Create dummy trajectories
            chosen_images = []
            rejected_images = []
            
            for i in range(3):
                # Create simple colored images
                chosen_img = Image.fromarray(
                    np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                )
                rejected_img = Image.fromarray(
                    np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                )
                
                chosen_images.append(chosen_img)
                rejected_images.append(rejected_img)
            
            print(f"✅ Created {len(chosen_images)} chosen and {len(rejected_images)} rejected dummy images")
            
            # Test preference query (without actually calling API)
            print("🔧 Testing preference query structure...")
            
            # Mock the API call to avoid using quota during testing
            original_get_completion = gvl_baseline._get_task_completion
            def mock_get_completion(frames_array, task_description):
                # Return mock completion percentages
                return [i * 20.0 for i in range(len(frames_array))]
            
            gvl_baseline._get_task_completion = mock_get_completion
            
            result = gvl_baseline.query_preference(
                chosen_images,
                rejected_images,
                "test robot manipulation task"
            )
            
            print(f"✅ Query completed successfully")
            print(f"   Result: {result}")
            
            # Check result structure
            required_keys = ["is_correct", "vlm_preference"]
            for key in required_keys:
                if key not in result:
                    print(f"❌ Missing key in result: {key}")
                    return False
            
            print("✅ Result structure correct")
            
            # Test log finalization
            gvl_baseline.finalize_log()
            print("✅ Log finalization successful")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    
    print("\n🎉 GVL baseline setup looks good!")
    print("Usage:")
    print("  # Start GVL server:")
    print("  python gvl_server.py --task 'robot manipulation' --port 8003")
    print("  # Run evaluation:")
    print("  python evals/run_model_eval.py --server_url http://localhost:8003")
    
    return True


def test_gvl_components():
    """Test individual GVL components."""
    print("\n🔧 Testing GVL components...")
    
    # Test frame conversion
    print("📸 Testing PIL to numpy conversion...")
    test_images = []
    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        test_images.append(img)
    
    # Mock baseline for component testing
    api_key = os.getenv("GEMINI_API_KEY", "dummy_key")
    with tempfile.TemporaryDirectory() as temp_dir:
        baseline = GVLPreferenceBaseline(
            api_key=api_key,
            verbose=False,
            log_dir=temp_dir
        )
        
        # Test conversion
        numpy_array = baseline._pil_to_numpy_array(test_images)
        expected_shape = (3, 64, 64, 3)
        
        if numpy_array.shape == expected_shape:
            print(f"✅ PIL to numpy conversion: {numpy_array.shape}")
        else:
            print(f"❌ PIL to numpy conversion failed: got {numpy_array.shape}, expected {expected_shape}")
            return False
        
        # Test completion comparison
        print("📊 Testing completion comparison...")
        
        chosen_completions = [10.0, 30.0, 60.0, 80.0]
        rejected_completions = [5.0, 20.0, 45.0, 70.0]
        
        preference, comparison = baseline._compare_trajectories(chosen_completions, rejected_completions)
        
        if preference == "chosen":
            print(f"✅ Comparison logic: {preference} (chosen: 80% vs rejected: 70%)")
        else:
            print(f"❌ Comparison logic failed: got {preference}")
            return False
        
        # Test tie detection
        chosen_completions_tie = [50.0, 60.0, 75.0]
        rejected_completions_tie = [48.0, 62.0, 77.0]  # Within 5% threshold
        
        preference_tie, _ = baseline._compare_trajectories(chosen_completions_tie, rejected_completions_tie)
        
        if preference_tie == "tie":
            print(f"✅ Tie detection: {preference_tie} (difference within threshold)")
        else:
            print(f"❌ Tie detection failed: got {preference_tie}")
            return False
    
    print("✅ All component tests passed!")
    return True


if __name__ == "__main__":
    print("🚀 GVL Baseline Test Suite")
    print("=" * 40)
    
    setup_ok = test_gvl_setup()
    components_ok = test_gvl_components()
    
    if setup_ok and components_ok:
        print("\n🎉 All tests passed! GVL baseline is ready to use.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the setup.")
        sys.exit(1)
