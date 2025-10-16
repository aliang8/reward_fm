#!/usr/bin/env python3
"""
Script to consolidate all LIBERO datasets into a single local directory.
Downloads HuggingFace datasets and saves them locally alongside the processed ones.
"""

import os
from pathlib import Path
from datasets import load_dataset

def download_and_save_dataset(hf_repo: str, subset: str, local_dir: str):
    """Download a HuggingFace dataset and save it locally."""
    print(f"📥 Downloading {hf_repo}/{subset}...")
    
    try:
        # Load from HuggingFace
        dataset = load_dataset(hf_repo, subset)
        
        # Save locally
        local_path = Path(local_dir) / subset
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Save the dataset
        if 'train' in dataset:
            dataset['train'].save_to_disk(str(local_path))
            print(f"✅ Saved {len(dataset['train'])} samples to {local_path}")
        else:
            # Handle datasets without train split
            for split_name, split_data in dataset.items():
                split_path = local_path / split_name
                split_data.save_to_disk(str(split_path))
                print(f"✅ Saved {len(split_data)} samples to {split_path}")
                
    except Exception as e:
        print(f"❌ Error downloading {hf_repo}/{subset}: {e}")

def main():
    """Consolidate all datasets into rfm_dataset."""
    
    # Target directory
    local_dataset_dir = "rfm_dataset"
    
    print("🚀 Consolidating LIBERO datasets...")
    print("=" * 60)
    
    # Create the directory
    Path(local_dataset_dir).mkdir(exist_ok=True)
    
    # HuggingFace datasets to download
    hf_datasets = [
        ("abraranwar/libero_rfm", "libero256_10"),
        ("ykorkmaz/libero_failure_rfm", "libero_10_failure"),
    ]
    
    # Download HuggingFace datasets
    for repo, subset in hf_datasets:
        download_and_save_dataset(repo, subset, local_dataset_dir)
    
    # Copy existing local datasets
    print("\n📂 Copying existing local datasets...")
    existing_local_dir = "datasets/libero_rfm"
    
    if Path(existing_local_dir).exists():
        import shutil
        
        for dataset_dir in Path(existing_local_dir).iterdir():
            if dataset_dir.is_dir():
                target_path = Path(local_dataset_dir) / dataset_dir.name
                if not target_path.exists():
                    print(f"📋 Copying {dataset_dir.name}...")
                    shutil.copytree(dataset_dir, target_path)
                    print(f"✅ Copied {dataset_dir.name}")
                else:
                    print(f"⏭️  {dataset_dir.name} already exists, skipping")
    else:
        print(f"⚠️  Local dataset directory {existing_local_dir} not found")
    
    # Show final status
    print("\n📊 Final dataset status:")
    print("=" * 60)
    
    if Path(local_dataset_dir).exists():
        datasets = list(Path(local_dataset_dir).iterdir())
        datasets.sort()
        
        for dataset_path in datasets:
            if dataset_path.is_dir():
                try:
                    from datasets import load_from_disk
                    ds = load_from_disk(str(dataset_path))
                    print(f"✅ {dataset_path.name:<20}: {len(ds):>5} samples")
                except Exception as e:
                    print(f"❓ {dataset_path.name:<20}: Error loading ({e})")
    
    print(f"\n🎉 All datasets consolidated in: {local_dataset_dir}/")
    print("🔧 You can now update your config to use this single directory!")

if __name__ == "__main__":
    main()
