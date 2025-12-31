#!/usr/bin/env python3
"""
Simple test of VLM server without dataset dependencies
"""

import requests
import base64
import io
from PIL import Image
import numpy as np

def create_test_image(color='red'):
    """Create a simple test image"""
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    if color == 'red':
        img[:, :, 0] = 200  # More red
    elif color == 'blue':
        img[:, :, 2] = 200  # More blue
    return Image.fromarray(img)

def image_to_base64(img):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

def test_vlm_server():
    """Test VLM server with simple synthetic data"""
    
    print("🧪 Testing VLM Server")
    print("=" * 40)
    
    # Create test images
    chosen_img = create_test_image('red')
    rejected_img = create_test_image('blue')
    
    # Convert to base64
    chosen_b64 = image_to_base64(chosen_img)
    rejected_b64 = image_to_base64(rejected_img)
    
    # Create test payload
    payload = {
        "samples": [
            {
                "task": "pick up red block",
                "prediction_type": "preference",
                "chosen_frames_b64": [chosen_b64],
                "rejected_frames_b64": [rejected_b64]
            }
        ]
    }
    
    # Test server
    try:
        print("Sending test request to VLM server...")
        response = requests.post(
            "http://localhost:8002/evaluate_batch",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ VLM server responded successfully!")
            print(f"📊 Results:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print(f"❌ Server error: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            try:
                error_detail = response.json()
                print(f"Error detail: {error_detail}")
            except:
                print(f"Raw response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to VLM server at http://localhost:8002")
        print("Make sure the server is running!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_vlm_server() 