# Adding New Datasets to RFM

This guide explains how to add new datasets (like DROID, Bridge, or any custom dataset) to the Reward Foundation Model (RFM) training pipeline.

## Overview

Each dataset type has its own loader module. The main converter (`generate_hf_dataset.py`) is dataset-agnostic and works with any dataset-specific loader that follows the established interface.

## Dataset Structure Requirements

Your dataset loader must produce trajectories in the following format:

```python
{
    'frames': List[Union[str, bytes]], # List of file paths (frame images) or video bytes (MP4 data)
    'actions': np.ndarray,             # Actions 
    'is_robot': bool,                  # Whether this is robot data (True) or human data (False)
    'task': str,                       # Human-readable task description
    'optimal': str                     # Whether this trajectory is optimal
}
```

## Step-by-Step Guide

### 1. Create Your Dataset Loader

Create a new Python file in the `data/` directory following the naming convention: `{dataset_name}_loader.py`

Example: `droid_loader.py` or `bridge_loader.py`

```python
#!/usr/bin/env python3
"""
DROID dataset loader for the generic dataset converter for RFM model training.
This module contains DROID-specific logic for loading and processing data.
"""

import os
import numpy as np
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

def load_droid_dataset(base_path: str) -> Dict[str, List[Dict]]:
    """Load DROID dataset and organize by task.
    
    Args:
        base_path: Path to the DROID dataset directory
        
    Returns:
        Dictionary mapping task names to lists of trajectory dictionaries
    """
    
    print(f"Loading DROID dataset from: {base_path}")
    
    task_data = {}
    
    # Your dataset-specific loading logic here
    # This is where you'll implement the logic to:
    # 1. Find and read your data files
    # 2. Extract frames, actions, rewards, etc.
    # 3. Organize by task
    # 4. Convert to the required format
    
    # Example structure (adapt to your dataset):
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"DROID dataset path not found: {base_path}")
    
    # Find your data files (adapt this to your dataset structure)
    data_files = list(base_path.glob("**/*.hdf5"))  # or whatever format you use
    
    for file_path in tqdm(data_files, desc="Processing DROID dataset"):
        task_name = file_path.stem
        
        # Load your data file
        # This is where you'll implement your specific loading logic
        trajectories = load_trajectories_from_file(file_path)
        
        task_data[task_name] = trajectories
    
    print(f"Loaded {sum(len(trajectories) for trajectories in task_data.values())} trajectories from {len(task_data)} tasks")
    return task_data

def load_trajectories_from_file(file_path: Path) -> List[Dict]:
    """Load trajectories from a single DROID data file."""
    trajectories = []
    
    # Implement your file loading logic here
    # This will depend on your dataset format (HDF5, JSON, pickle, etc.)
    
    # Example for HDF5 format:
    import h5py
    with h5py.File(file_path, 'r') as f:
        # Navigate your HDF5 structure and extract data
        # Convert to the required format
        pass
    
    return trajectories
```

### 2. Update the Main Converter

Add your dataset type to the main converter in `generate_hf_dataset.py`:

```python
# In the main() function, add your dataset type:
elif cfg.dataset.dataset_type == "droid":
    from droid_loader import load_droid_dataset
    # Load the trajectories using your loader
    task_data = load_droid_dataset(cfg.dataset.dataset_path)
    trajectories = flatten_task_data(task_data)
```

### 3. Test Your Loader

Create a simple test script to verify your loader works:

```python
# test_droid_loader.py
from droid_loader import load_droid_dataset
from helpers import flatten_task_data

# Test your loader
task_data = load_droid_dataset("/path/to/your/droid/dataset")
trajectories = flatten_task_data(task_data)

print(f"Loaded {len(trajectories)} trajectories")
print(f"Sample trajectory keys: {list(trajectories[0].keys())}")
print(f"Sample task: {trajectories[0].get('task', 'No task found')}")
```

### 4. Run Dataset Conversion

Use the main converter with your new dataset:

```bash
python data/generate_hf_dataset.py \
    --config_path=configs/data_gen.yaml \
    --dataset.dataset_type=droid \
    --dataset.dataset_path=/path/to/your/droid/dataset \
    --dataset.dataset_name=DROID \
    --output.output_dir=rfm_dataset \
    --output.max_trajectories=1000 \
    --output.max_frames=-1 \
    --output.use_video=true \
    --output.fps=10
```