#!/usr/bin/env python3
"""
Simple validation script for the RFM dataset format.
Checks fields and data types only.
"""

import os
import json
import numpy as np
from datasets import load_from_disk, Dataset
from typing import Dict, List, Any
import argparse
from pathlib import Path


def validate_dataset_fields_and_types(dataset: Dataset, sample_size: int = 10) -> Dict[str, Any]:
    """Validate dataset fields and data types."""
    
    print(f"Validating dataset fields and data types on {sample_size} sample entries...")
    
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {
            "dataset_size": len(dataset),
            "samples_checked": 0
        }
    }
    
    # Expected schema for the new format
    expected_schema = {
        "id": str,
        "task": str,
        "lang_vector": np.ndarray,
        "data_source": str,
        "frames": list,
        "optimal": bool,
        "ranking": int,
        "preference_embedding": np.ndarray,
        "is_robot": bool,
        "metadata": dict
    }
    
    # Check if dataset has features
    if not hasattr(dataset, 'features') or dataset.features is None:
        validation_results["valid"] = False
        validation_results["errors"].append("Dataset has no features defined")
        return validation_results
    
    print(f"Dataset size: {len(dataset)} entries")
    print(f"Dataset features: {list(dataset.features.keys())}")
    
    # Check required fields
    for field_name, expected_type in expected_schema.items():
        if field_name not in dataset.features:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Missing required field: {field_name}")
        else:
            print(f"✓ Field '{field_name}' present")
    
    # Sample entries for validation
    sample_indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)
    validation_results["stats"]["samples_checked"] = len(sample_indices)
    
    for idx in sample_indices:
        trajectory = dataset[idx]
        
        try:
            # Validate each field
            if not isinstance(trajectory["id"], str):
                validation_results["errors"].append(f"Trajectory {idx}: 'id' is not a string")
            
            if not isinstance(trajectory["task"], str):
                validation_results["errors"].append(f"Trajectory {idx}: 'task' is not a string")
            
            if not isinstance(trajectory["lang_vector"], np.ndarray):
                validation_results["errors"].append(f"Trajectory {idx}: 'lang_vector' is not a numpy array")
            elif trajectory["lang_vector"].shape != (384,):
                validation_results["errors"].append(f"Trajectory {idx}: 'lang_vector' shape is {trajectory['lang_vector'].shape}, expected (384,)")
            
            if not isinstance(trajectory["data_source"], str):
                validation_results["errors"].append(f"Trajectory {idx}: 'data_source' is not a string")
            
            if not isinstance(trajectory["frames"], list):
                validation_results["errors"].append(f"Trajectory {idx}: 'frames' is not a list")
            else:
                for frame_path in trajectory["frames"]:
                    if not isinstance(frame_path, str):
                        validation_results["errors"].append(f"Trajectory {idx}: frame path is not a string")
            
            if not isinstance(trajectory["optimal"], bool):
                validation_results["errors"].append(f"Trajectory {idx}: 'optimal' is not a boolean")
            
            if not isinstance(trajectory["ranking"], int):
                validation_results["errors"].append(f"Trajectory {idx}: 'ranking' is not an integer")
            
            if not isinstance(trajectory["preference_embedding"], np.ndarray):
                validation_results["errors"].append(f"Trajectory {idx}: 'preference_embedding' is not a numpy array")
            elif trajectory["preference_embedding"].shape != (384,):
                validation_results["errors"].append(f"Trajectory {idx}: 'preference_embedding' shape is {trajectory['preference_embedding'].shape}, expected (384,)")
            
            if not isinstance(trajectory["is_robot"], bool):
                validation_results["errors"].append(f"Trajectory {idx}: 'is_robot' is not a boolean")
            
            if not isinstance(trajectory["metadata"], dict):
                validation_results["errors"].append(f"Trajectory {idx}: 'metadata' is not a dictionary")
            else:
                # Check metadata fields
                expected_metadata_fields = ["original_file", "scene", "demo_id", "trajectory_info", "trajectory_length", "file_path"]
                for field in expected_metadata_fields:
                    if field not in trajectory["metadata"]:
                        validation_results["warnings"].append(f"Trajectory {idx}: metadata missing field '{field}'")
                
                # Print sample metadata for first trajectory
                if idx == sample_indices[0]:
                    print(f"\nSample metadata from first trajectory:")
                    for key, value in trajectory["metadata"].items():
                        print(f"  {key}: {value}")
            
            # Print sample task for first trajectory
            if idx == sample_indices[0]:
                print(f"\nSample task from first trajectory:")
                print(f"  Task: {trajectory['task']}")
                print(f"  ID: {trajectory['id']}")
                
        except Exception as e:
            validation_results["errors"].append(f"Trajectory {idx}: Error during validation: {e}")
    
    if validation_results["errors"]:
        validation_results["valid"] = False
    
    return validation_results


def print_validation_summary(validation_results: Dict[str, Any]):
    """Print validation summary."""
    
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    status = "✅ PASS" if validation_results["valid"] else "❌ FAIL"
    print(f"Status: {status}")
    
    print(f"Dataset size: {validation_results['stats']['dataset_size']}")
    print(f"Samples checked: {validation_results['stats']['samples_checked']}")
    
    if validation_results.get("errors"):
        print(f"\nErrors ({len(validation_results['errors'])}):")
        for error in validation_results["errors"][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(validation_results["errors"]) > 10:
            print(f"  ... and {len(validation_results['errors']) - 10} more errors")
    
    if validation_results.get("warnings"):
        print(f"\nWarnings ({len(validation_results['warnings'])}):")
        for warning in validation_results["warnings"][:5]:  # Show first 5 warnings
            print(f"  - {warning}")
        if len(validation_results["warnings"]) > 5:
            print(f"  ... and {len(validation_results['warnings']) - 5} more warnings")
    
    print("="*50)


def main():
    """Main validation function."""
    
    parser = argparse.ArgumentParser(description="Validate dataset fields and data types")
    parser.add_argument("dataset_path", help="Path to the HuggingFace dataset")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of samples to check")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    try:
        dataset = load_from_disk(args.dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"Dataset loaded successfully.")
    
    # Run validation
    validation_results = validate_dataset_fields_and_types(dataset, args.sample_size)
    
    # Print summary
    print_validation_summary(validation_results)
    
    # Exit with error code if validation failed
    if not validation_results["valid"]:
        exit(1)


if __name__ == "__main__":
    main() 