#!/usr/bin/env python3
"""
Send a demo batch to the local evaluation server.

The payload mirrors the dataclasses in `rfm/data/dataset_types.py`.
Replace the randomly generated tensors with real embeddings / trajectories
when running against a trained checkpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import numpy as np

from evals.eval_utils import (
    build_payload,
    post_batch,
    post_batch_npy,
    post_batch_npy_async,
)
from rfm.data.dataset_types import PreferenceSample, Trajectory


def build_preference_sample(seed: int, embedding_dim: int = 8) -> PreferenceSample:
    """Create a toy preference sample with reproducible random embeddings."""
    rng = np.random.default_rng(seed)

    def make_traj(tag: str) -> Trajectory:
        video_embeddings = rng.normal(size=(4, 768))
        text_embedding = rng.normal(size=(384,))
        target_progress = np.linspace(0.0, 1.0, num=4)
        return Trajectory(
            task=f"demo_task_{tag}",
            video_embeddings=video_embeddings,
            text_embedding=text_embedding,
            target_progress=target_progress.tolist(),
            frames_shape=tuple(video_embeddings.shape),
            metadata={"source": "sample_eval_request.py"},
        )

    return PreferenceSample(
        chosen_trajectory=make_traj("chosen"),
        rejected_trajectory=make_traj("rejected"),
        data_gen_strategy="demo",
    )


def numpy_to_builtin(value: Any) -> Any:
    """Recursively convert numpy arrays to Python lists for JSON payloads."""
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, dict):
        return {k: numpy_to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [numpy_to_builtin(v) for v in value]
    return value


def create_samples(num_samples: int, embedding_dim: int, seed_offset: int = 0) -> list[PreferenceSample]:
    return [build_preference_sample(seed=seed_offset + i, embedding_dim=embedding_dim) for i in range(num_samples)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a sample batch to the eval server.")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--num-samples", type=int, default=1, help="How many demo samples to include")
    parser.add_argument("--embedding-dim", type=int, default=8, help="Size of the toy embedding vectors")
    parser.add_argument("--timeout", type=float, default=120.0, help="Requests timeout in seconds")
    parser.add_argument(
        "--use-npy",
        action="store_true",
        help="Send payload through /evaluate_batch_npy using helpers from eval_utils",
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async client when hitting /evaluate_batch_npy",
    )
    parser.add_argument(
        "--async-requests",
        type=int,
        default=1,
        help="Number of concurrent requests to send when --async is used",
    )
    args = parser.parse_args()

    if args.use_async and not args.use_npy:
        raise ValueError("--async requires --use-npy")

    samples = create_samples(args.num_samples, args.embedding_dim)

    if args.use_npy:
        if args.use_async:
            import aiohttp

            async def _send_async_requests():
                async with aiohttp.ClientSession() as session:
                    tasks = []
                    for request_idx in range(args.async_requests):
                        req_samples = create_samples(
                            args.num_samples,
                            args.embedding_dim,
                            seed_offset=request_idx * args.num_samples,
                        )
                        files_req, sample_data_req = build_payload(req_samples)
                        tasks.append(
                            post_batch_npy_async(
                                session,
                                args.base_url,
                                files_req,
                                sample_data_req,
                                timeout_s=args.timeout,
                            )
                        )
                    return await asyncio.gather(*tasks)

            print(
                f"Sending {args.async_requests} async request(s) "
                f"({args.num_samples} samples each) to {args.base_url} via /evaluate_batch_npy"
            )
            response = asyncio.run(_send_async_requests())
        else:
            files, sample_data = build_payload(samples)
            print(f"Sending {len(samples)} sample(s) to {args.base_url} via /evaluate_batch_npy")
            response = post_batch_npy(args.base_url, files, sample_data, timeout_s=args.timeout)
    else:
        payload = [numpy_to_builtin(sample.model_dump()) for sample in samples]
        print(f"Sending {len(samples)} sample(s) to {args.base_url} via /evaluate_batch")
        response = post_batch(args.base_url, payload, timeout_s=args.timeout)

    if isinstance(response, list):
        for idx, resp in enumerate(response):
            print(f"Response {idx}:\n{json.dumps(resp, indent=2)[:2000]}")
    else:
        print(json.dumps(response, indent=2)[:2000])


if __name__ == "__main__":
    main()
