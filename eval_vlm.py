# Requirements: pip install torch transformers pillow
# eval_vlm.py

import sys
from typing import List
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

def call_qwen25vl(images: List[str], prompt: str):
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
    except ImportError as e:
        print("Required packages not found. Please install transformers, torch, and pillow.")
        raise e

    pil_images = [Image.open(img_path).convert("RGB") for img_path in images]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": images[0]},
                {"type": "image", "image": images[1]},
                {"type": "image", "image": images[2]},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    # Use add_vision_id=True to label images as Picture 1, 2, 3
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
    inputs = processor(text=[text], images=pil_images, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    # Only decode the newly generated tokens
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts[0]

def main():
    # task = "pick_up_green_block"
    task = "square"
    
    tests = [
        # ["expert_A.png", "expert_B.png", "subopt_C.png"],
        # ["expert_A.png", "subopt_C.png", "expert_B.png"],
        # ["expert_A.png", "expert_B.png", "expert_subopt_2.png"],
        # ["expert_A.png", "expert_hand.png", "subopt_C.png"],
        # ["expert_A.png", "subopt_C.png", "expert_subopt_2.png"],
        # ["expert_A.png", "expert_subopt_2.png", "subopt_C.png"],
        # ["expert_A.png", "expert_camera_angle.png", "subopt_C.png"],
        # ["expert_A.png", "expert_camera_angle.png", "expert_subopt_2.png"],
    ]

    tests = [
        ["square_peg_pick.png"]
    ]

    prompt_discriminator = (
        "There are three images. Each image depicts a trajectory for the robot arm. "
        "The first image is the reference policy. Of the two other images, which one is demonstrating similar behavior and is likely from the same policy?"
        "Report the index of the image that is similar to the reference policy either 1 or 2."
    )

    prompt_description = (
        "Describe the trajectory of the robot arm in the image."
    )

    verify_prompt = (
        "The task is to pick up the square peg. Is the trajectory in the image correct?"
    )

    for test in tests:
        for index, path in enumerate(test):
            test[index] = f"{task}/{path}"

        response = call_qwen25vl(test, prompt_discriminator)
        # response = call_qwen25vl(test, verify_prompt)
        print(f"Test: {test}")
        print(f"Qwen2.5-VL Response: {response}")
        print("-" * 100)

if __name__ == "__main__":
    main()
