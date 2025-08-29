#!/usr/bin/env python3
import os
import sys
import argparse
from multiprocessing import Pool, cpu_count
from datasets import load_dataset


def worker(dataset_name: str, split: str, num_shards: int, shard_index: int, max_items: int) -> list[str]:
    ds = load_dataset(dataset_name, streaming=True, split=split)
    # Apply sharding at the dataset iterator level
    ds = ds.shard(num_shards=num_shards, index=shard_index)
    ids: list[str] = []
    it = iter(ds)
    for _ in range(max_items):
        try:
            ex = next(it)
        except StopIteration:
            break
        except Exception:
            break
        key = ex.get("__key__") or ex.get("id") or ex.get("_id") or ex.get("guid")
        if not isinstance(key, str):
            # Fallback to a stable textual representation
            key = str(ex.get("__key__", ""))
        if key:
            ids.append(key)
    return ids


def main():
    parser = argparse.ArgumentParser(description="Demonstrate non-overlapping ds.shard subsets")
    parser.add_argument("dataset", type=str, help="HF dataset name, e.g. webdataset or agibot-world/AgiBotWorld-Alpha")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument("--max_items", type=int, default=200)
    parser.add_argument("--parallel", type=int, default=4, help="number of shard workers to run in parallel")
    args = parser.parse_args()

    shard_indices = list(range(args.num_shards))
    par = max(1, min(args.parallel, cpu_count()))

    with Pool(processes=par) as pool:
        results = pool.starmap(
            worker,
            [
                (args.dataset, args.split, args.num_shards, i, args.max_items)
                for i in shard_indices
            ],
        )

    # Verify non-overlap and print summary
    all_ids = set()
    collisions = []
    for i, ids in enumerate(results):
        print(f"Shard {i} collected {len(ids)} ids. Sample: {ids[:5]}")
        for id_ in ids:
            if id_ in all_ids:
                collisions.append((i, id_))
            else:
                all_ids.add(id_)

    if collisions:
        print(f"\nFound {len(collisions)} overlapping ids across shards (unexpected):")
        for i, id_ in collisions[:20]:
            print(f"  - shard {i}: {id_}")
    else:
        print("\nNo overlaps detected across shard id samples.")


if __name__ == "__main__":
    main()


