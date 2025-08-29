#!/usr/bin/env python3
import os
import sys
import argparse
import time
from collections import deque


def count_episodes(dataset_root: str):
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not os.path.isdir(dataset_root):
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    total = 0
    per_shard = {}

    try:
        shard_names = sorted(os.listdir(dataset_root))
    except Exception as e:
        raise RuntimeError(f"Failed to list dataset root '{dataset_root}': {e}") from e

    for shard_name in shard_names:
        shard_path = os.path.join(dataset_root, shard_name)
        if not os.path.isdir(shard_path):
            continue
        if not shard_name.startswith("shard_"):
            continue

        try:
            children = os.listdir(shard_path)
        except Exception as e:
            print(f"Warning: could not list shard '{shard_path}': {e}")
            continue

        shard_count = 0
        for child in children:
            if not child.startswith("episode_"):
                continue
            ep_path = os.path.join(shard_path, child)
            if os.path.isdir(ep_path):
                shard_count += 1
        per_shard[shard_name] = shard_count
        total += shard_count

    return total, per_shard


def main():
    parser = argparse.ArgumentParser(description="Count AgiBotWorld episodes in a local dataset directory.")
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default="datasets/agibotworld_dataset/agibotworld",
        help="Path to the agibotworld dataset root (default: datasets/agibotworld_dataset/agibotworld)",
    )
    parser.add_argument(
        "--per-shard",
        action="store_true",
        help="Print per-shard episode counts",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Continuously monitor counts and estimate episodes/sec and ETA to target",
    )
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=5.0,
        help="Seconds between indexing scans when --monitor is enabled (default: 5.0)",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=100_000,
        help="Target total number of episodes for ETA estimation (default: 100000)",
    )
    parser.add_argument(
        "--rate-window",
        type=int,
        default=6,
        help="Number of recent intervals to average for rate smoothing (default: 6)",
    )
    args = parser.parse_args()

    try:
        total, per_shard = count_episodes(args.dataset_root)
    except Exception as e:
        print(str(e))
        sys.exit(1)

    print(f"Dataset root: {args.dataset_root}")
    print(f"Shards found: {len(per_shard)}")
    print(f"Total episodes: {total}")

    if args.per_shard and not args.monitor:
        for shard_name in sorted(per_shard.keys()):
            print(f"  {shard_name}: {per_shard[shard_name]}")

    if not args.monitor:
        return

    def fmt_eta(seconds: float) -> str:
        if seconds <= 0 or seconds == float("inf"):
            return "unknown"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        if m > 0:
            return f"{m}m {s}s"
        return f"{s}s"

    # Monitoring loop with periodic re-indexing
    print("\nMonitoring enabled. Press Ctrl+C to stop.")
    target_total = max(1, int(args.target_total))
    start_time = time.perf_counter()
    last_time = start_time
    last_total = total
    rate_window = deque(maxlen=max(1, args.rate_window))

    try:
        while True:
            time.sleep(max(0.1, args.scan_interval))
            try:
                current_total, current_per_shard = count_episodes(args.dataset_root)
            except Exception as e:
                print(f"Scan error: {e}")
                continue

            now = time.perf_counter()
            dt = max(1e-6, now - last_time)
            dcount = max(0, current_total - last_total)
            inst_rate = dcount / dt
            rate_window.append(inst_rate)
            smooth_rate = sum(rate_window) / len(rate_window)

            elapsed = now - start_time
            overall_rate = (current_total - total) / max(1e-6, elapsed)

            # Prefer smoothed recent rate for ETA if available, fallback to overall
            rate_for_eta = smooth_rate if smooth_rate > 0 else overall_rate
            remaining = max(0, target_total - current_total)
            eta_sec = (remaining / rate_for_eta) if rate_for_eta > 0 else float("inf")

            status = (
                f"episodes={current_total} | recent_rate={smooth_rate:.2f}/s | "
                f"overall_rate={overall_rate:.2f}/s | target={target_total} | "
                f"remaining={remaining} | eta={fmt_eta(eta_sec)}"
            )
            print(status)

            if args.per_shard:
                # Print a brief per-shard summary (non-verbose): top 3 shards by count
                if current_per_shard:
                    top = sorted(current_per_shard.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    top_str = ", ".join([f"{k}:{v}" for k, v in top])
                    print(f"  top_shards: {top_str}")

            last_time = now
            last_total = current_total

            if current_total >= target_total:
                print("Target reached. Exiting.")
                break
    except KeyboardInterrupt:
        print("Stopped by user.")


if __name__ == "__main__":
    main()


