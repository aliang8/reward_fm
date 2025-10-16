#!/usr/bin/env python3
"""Quick script to check dataset availability and get sample counts."""

import os
from pathlib import Path

def check_datasets():
    """Check what datasets are available locally."""
    print("🔍 Checking dataset availability...")
    print("=" * 60)
    
    # Check local datasets
    local_path = Path("datasets/libero_rfm")
    if local_path.exists():
        print(f"✅ Local datasets found at: {local_path}")
        datasets = [d.name for d in local_path.iterdir() if d.is_dir()]
        for dataset in sorted(datasets):
            print(f"  📁 {dataset}")
            # Try to get sample count
            try:
                from datasets import load_from_disk
                ds = load_from_disk(local_path / dataset)
                print(f"     Samples: {len(ds)}")
            except Exception as e:
                print(f"     Error loading: {e}")
    else:
        print(f"❌ Local datasets not found at: {local_path}")
    
    print("\n🎯 Target datasets for evaluation:")
    target_datasets = ["libero_goal", "libero_spatial", "libero_object", "libero256_10"]
    for dataset in target_datasets:
        if local_path.exists() and (local_path / dataset.replace("256_10", "")).exists():
            print(f"  ✅ {dataset}")
        else:
            print(f"  ❌ {dataset}")
    
    print("\n📊 Current evaluation logs:")
    log_path = Path("evals/logs")
    if log_path.exists():
        recent_logs = sorted(log_path.glob("*vlmf_*.log"), key=os.path.getmtime, reverse=True)[:5]
        for log in recent_logs:
            print(f"  📄 {log.name}")

if __name__ == "__main__":
    check_datasets()
