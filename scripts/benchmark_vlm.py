#!/usr/bin/env python3
f"""
Benchmark VLM inference latency.

- Loads a VLM Image-Text-to-Text model and processor
- Runs optional warmup, then timed generate calls
- Reports p50/p90/p95/mean latencies

Example:
    PYTHONPATH=. python scripts/benchmark_vlm.py \
        --model-id HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
        --prompt "Describe this video." \
        --runs 20 --warmup 5 --max-new-tokens 64

    PYTHONPATH=. python scripts/benchmark_vlm.py \
        --model-id Qwen/Qwen2.5-VL-3B-Instruct \
        --prompt "Describe this video." \
        --runs 20 --warmup 5 --max-new-tokens 64
"""

import argparse
import statistics
import time
import os
import tempfile
from pathlib import Path


import torch
import numpy as np

try:
    import psutil  # optional, for CPU RSS
except Exception:
    psutil = None
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from rfm.data.collators.utils import write_mp4
from qwen_vl_utils import process_vision_info


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def bytes_to_gb(num_bytes: int) -> float:
    return float(num_bytes) / (1024**3)


def get_cpu_memory_gb() -> float:
    # Prefer psutil if available for current RSS
    if psutil is not None:
        proc = psutil.Process(os.getpid())
        return bytes_to_gb(proc.memory_info().rss)
    # Fallback: read from /proc/self/statm on Linux
    try:
        with open("/proc/self/statm", "r") as f:
            contents = f.read().strip().split()
            if len(contents) >= 2:
                rss_pages = int(contents[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                rss_bytes = rss_pages * page_size
                return bytes_to_gb(rss_bytes)
    except Exception:
        pass
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SmolVLM inference time")
    parser.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt", type=str, default="Describe the video.")
    parser.add_argument("--do-sampling", action="store_true", help="Use sampling instead of greedy decoding")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inputs")
    parser.add_argument("--dummy", action="store_true", help="Use randomly generated video frames")
    parser.add_argument("--height", type=int, default=128, help="Dummy video frame height")
    parser.add_argument("--width", type=int, default=128, help="Dummy video frame width")
    parser.add_argument("--num-frames", type=int, default=32, help="Number of frames per generated video")
    parser.add_argument("--fps", type=int, default=4, help="FPS for generated videos")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for the model")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode",
    )
    parser.add_argument(
        "--compile-backend",
        type=str,
        default="inductor",
        help="torch.compile backend (e.g., inductor)",
    )
    args = parser.parse_args()

    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    device = torch.device(args.device)

    print(f"Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        padding_side="left",
        size={"longest_edge": 512},
        max_image_size={"longest_edge": 512},
    )

    print(f"Loading model: {args.model_id} on {device} ({torch_dtype})")
    from transformers import Qwen2_5_VLModel, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.to(device)
    model.eval()

    # Optional torch.compile
    if args.compile:
        try:
            model = torch.compile(model, mode=args.compile_mode, backend=args.compile_backend)
            print(f"Model compiled with torch.compile (backend={args.compile_backend}, mode={args.compile_mode})")
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}); continuing without compile.")

    # Build batch of inputs
    batch_size = max(1, args.batch_size)
    temp_video_paths: list[Path] = []

    try:
        # Always construct per-item inputs with random frames; for Qwen use frames,
        # for SmolVLM write MP4 and pass path
        conversations: list[list[dict]] = []
        for _ in range(batch_size):
            frames = []
            for _f in range(args.num_frames):
                arr = np.random.randint(0, 256, size=(args.height, args.width, 3), dtype=np.uint8)
                frames.append(Image.fromarray(arr, mode="RGB"))

            if "Qwen" in args.model_id:
                video_field = frames
                content_extras = {"resized_height": args.height, "resized_width": args.width}
            else:
                fd, tmp_path = tempfile.mkstemp(prefix="smolvlm_bench_", suffix=".mp4")
                os.close(fd)
                tmp = Path(tmp_path)
                temp_video_paths.append(tmp)
                write_mp4(frames, tmp, fps=args.fps)
                video_field = str(tmp)
                content_extras = {"resized_height": args.height, "resized_width": args.width}

            conversations.append([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_field,
                            **content_extras,
                        },
                        {"type": "text", "text": args.prompt},
                    ],
                }
            ])

        # Let the processor handle tokenization and video ingestion directly
        if "Qwen" in args.model_id:
            texts = [
                processor.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=False,
                    add_vision_id=True,
                    fps=1,
                )
                for msg in conversations
            ]

            is_qwen3 = "Qwen3" in args.model_id
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                conversations, 
                return_video_kwargs=True,
                return_video_metadata=is_qwen3
            )

            # For Qwen3, video_inputs is a list of (video, video_metadata) tuples
            # that need to be split before passing to processor
            if is_qwen3 and video_inputs is not None:
                videos, video_metadatas = zip(*video_inputs)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                videos = video_inputs
                video_metadatas = None

            processor_kwargs = {
                "text": texts,
                "images": image_inputs,
                "videos": videos,
                "padding": True,
                "truncation": False,
                "return_tensors": "pt",
            }
            
            # Add video_metadata and video_kwargs for Qwen3
            if is_qwen3:
                if video_metadatas is not None:
                    processor_kwargs["video_metadata"] = video_metadatas
                if video_kwargs:
                    processor_kwargs.update(video_kwargs)
            
            batch_inputs = processor(**processor_kwargs)
        else:
            batch_inputs = processor.apply_chat_template(
                conversations,
                add_generation_prompt=False,
                tokenize=True,
                padding=True,
                truncation=False,
                return_dict=True,
                return_tensors="pt",
                fps=args.fps,
            )

        inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        # Ensure input tensor dtypes match model dtype (e.g., bf16 on CUDA)
        model_dtype = next(model.parameters()).dtype
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                inputs[key] = value.to(dtype=model_dtype)

        # Reset CUDA peak stats before measuring
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Warmup
        warmup_runs = max(0, args.warmup)
        if warmup_runs > 0:
            print(f"Warmup: {warmup_runs} runs ...")
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sampling,
                    )
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Timed runs
        print(f"Benchmarking: {args.runs} runs ...")
        latencies_s: list[float] = []
        with torch.no_grad():
            for i in range(args.runs):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sampling,
                    )
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                else:
                    start = time.perf_counter()
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sampling,
                    )
                    end = time.perf_counter()
                lat_s = end - start
                latencies_s.append(lat_s)
                print(f"Run {i + 1}/{args.runs}: {lat_s:.3f} s")

        mean_s = statistics.mean(latencies_s) if latencies_s else float("nan")
        p50_s = percentile(latencies_s, 50)
        p90_s = percentile(latencies_s, 90)
        p95_s = percentile(latencies_s, 95)

        print("\nResults:")
        print(f"  mean: {mean_s:.3f} s")
        print(f"  p50 : {p50_s:.3f} s")
        print(f"  p90 : {p90_s:.3f} s")
        print(f"  p95 : {p95_s:.3f} s")

        # Memory usage
        print("\nMemory usage:")
        cpu_gb = get_cpu_memory_gb()
        print(f"  CPU RSS: {cpu_gb:.2f} GB")
        if device.type == "cuda":
            alloc = bytes_to_gb(torch.cuda.memory_allocated(device))
            reserv = bytes_to_gb(torch.cuda.memory_reserved(device))
            peak_alloc = bytes_to_gb(torch.cuda.max_memory_allocated(device))
            peak_reserv = bytes_to_gb(torch.cuda.max_memory_reserved(device))
            print(f"  CUDA current allocated: {alloc:.2f} GB")
            print(f"  CUDA current reserved:  {reserv:.2f} GB")
            print(f"  CUDA peak allocated:    {peak_alloc:.2f} GB")
            print(f"  CUDA peak reserved:     {peak_reserv:.2f} GB")
    finally:
        for p in temp_video_paths:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
