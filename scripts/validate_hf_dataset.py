#!/usr/bin/env python3
"""
Script to validate HuggingFace dataset format for robot trajectory data.
Checks the schema and data integrity of the dataset.
"""

import os
import json
import numpy as np
from datasets import load_from_disk, Dataset
from sentence_transformers import SentenceTransformer
from PIL import Image
from typing import Dict, List, Any, Optional
import argparse
from pathlib import Path


def validate_dataset_schema(dataset: Dataset) -> Dict[str, Any]:
    """Validate the dataset schema and return validation results."""
    
    print("Validating dataset schema...")
    
    validation_results = {
        "schema_valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # Expected schema
    expected_schema = {
        "id": str,
        "task": str,
        "lang_vector": np.ndarray,
        "data_source": str,
        "frames": list,
        "optimal": bool,
        "ranking": int,
        "preference_embedding": np.ndarray,
        "is_robot": bool
    }
    
    # Check if dataset has features
    if not hasattr(dataset, 'features') or dataset.features is None:
        validation_results["schema_valid"] = False
        validation_results["errors"].append("Dataset has no features defined")
        return validation_results
    
    # Check required fields
    for field_name, expected_type in expected_schema.items():
        if field_name not in dataset.features:
            validation_results["schema_valid"] = False
            validation_results["errors"].append(f"Missing required field: {field_name}")
        else:
            actual_type = dataset.features[field_name]
            print(f"Field '{field_name}': {actual_type}")
    
    # Check dataset size
    validation_results["stats"]["dataset_size"] = len(dataset)
    print(f"Dataset size: {len(dataset)} entries")
    
    return validation_results


def validate_data_types(dataset: Dataset, sample_size: int = 10) -> Dict[str, Any]:
    """Validate data types and formats of a sample of entries."""
    
    print(f"Validating data types on {sample_size} sample entries...")
    
    validation_results = {
        "types_valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # Sample entries for validation
    sample_indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)
    
    for idx in sample_indices:
        entry = dataset[idx]
        
        # Validate each field
        try:
            # Check id
            if not isinstance(entry["id"], str):
                validation_results["errors"].append(f"Entry {idx}: 'id' is not a string")
            
            # Check task
            if not isinstance(entry["task"], str):
                validation_results["errors"].append(f"Entry {idx}: 'task' is not a string")
            
            # Check lang_vector
            if not isinstance(entry["lang_vector"], np.ndarray):
                validation_results["errors"].append(f"Entry {idx}: 'lang_vector' is not a numpy array")
            elif entry["lang_vector"].shape != (384,):
                validation_results["errors"].append(f"Entry {idx}: 'lang_vector' shape is {entry['lang_vector'].shape}, expected (384,)")
            
            # Check data_source
            if not isinstance(entry["data_source"], str):
                validation_results["errors"].append(f"Entry {idx}: 'data_source' is not a string")
            
            # Check frames
            if not isinstance(entry["frames"], list):
                validation_results["errors"].append(f"Entry {idx}: 'frames' is not a list")
            else:
                for frame_path in entry["frames"]:
                    if not isinstance(frame_path, str):
                        validation_results["errors"].append(f"Entry {idx}: frame path is not a string")
                    elif not os.path.exists(frame_path):
                        validation_results["warnings"].append(f"Entry {idx}: frame path does not exist: {frame_path}")
            
            # Check optimal
            if not isinstance(entry["optimal"], bool):
                validation_results["errors"].append(f"Entry {idx}: 'optimal' is not a boolean")
            
            # Check ranking
            if not isinstance(entry["ranking"], int):
                validation_results["errors"].append(f"Entry {idx}: 'ranking' is not an integer")
            elif entry["ranking"] < 1:
                validation_results["warnings"].append(f"Entry {idx}: 'ranking' is {entry['ranking']}, should be >= 1")
            
            # Check preference_embedding
            if not isinstance(entry["preference_embedding"], np.ndarray):
                validation_results["errors"].append(f"Entry {idx}: 'preference_embedding' is not a numpy array")
            elif entry["preference_embedding"].shape != (384,):
                validation_results["errors"].append(f"Entry {idx}: 'preference_embedding' shape is {entry['preference_embedding'].shape}, expected (384,)")
            
            # Check is_robot
            if not isinstance(entry["is_robot"], bool):
                validation_results["errors"].append(f"Entry {idx}: 'is_robot' is not a boolean")
                
        except Exception as e:
            validation_results["errors"].append(f"Entry {idx}: Error during validation: {e}")
    
    if validation_results["errors"]:
        validation_results["types_valid"] = False
    
    validation_results["stats"]["samples_checked"] = len(sample_indices)
    
    return validation_results


def validate_ranking_consistency(dataset: Dataset) -> Dict[str, Any]:
    """Validate ranking consistency within preference groups."""
    
    print("Validating ranking consistency...")
    
    validation_results = {
        "ranking_valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # Group entries by preference_embedding
    preference_groups = {}
    
    for i, entry in enumerate(dataset):
        # Convert numpy array to tuple for hashing
        pref_emb = tuple(entry["preference_embedding"].flatten())
        if pref_emb not in preference_groups:
            preference_groups[pref_emb] = []
        preference_groups[pref_emb].append(entry)
    
    validation_results["stats"]["num_preference_groups"] = len(preference_groups)
    
    # Check each preference group
    for group_idx, (pref_emb, group_entries) in enumerate(preference_groups.items()):
        rankings = [entry["ranking"] for entry in group_entries]
        
        # Check for duplicate rankings
        if len(rankings) != len(set(rankings)):
            validation_results["errors"].append(f"Group {group_idx}: Duplicate rankings found: {rankings}")
        
        # Check ranking range
        min_rank = min(rankings)
        max_rank = max(rankings)
        expected_max = len(group_entries)
        
        if min_rank != 1:
            validation_results["warnings"].append(f"Group {group_idx}: Minimum ranking is {min_rank}, expected 1")
        
        if max_rank != expected_max:
            validation_results["warnings"].append(f"Group {group_idx}: Maximum ranking is {max_rank}, expected {expected_max}")
        
        # Check for gaps in rankings
        expected_rankings = set(range(1, expected_max + 1))
        actual_rankings = set(rankings)
        missing_rankings = expected_rankings - actual_rankings
        
        if missing_rankings:
            validation_results["errors"].append(f"Group {group_idx}: Missing rankings: {missing_rankings}")
    
    if validation_results["errors"]:
        validation_results["ranking_valid"] = False
    
    return validation_results


def validate_frame_paths(dataset: Dataset, sample_size: int = 20) -> Dict[str, Any]:
    """Validate that frame paths exist and are valid images."""
    
    print(f"Validating frame paths on {sample_size} sample entries...")
    
    validation_results = {
        "frames_valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # Sample entries for validation
    sample_indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)
    
    total_frames_checked = 0
    frames_exist = 0
    frames_valid = 0
    
    for idx in sample_indices:
        entry = dataset[idx]
        frame_paths = entry["frames"]
        
        for frame_path in frame_paths:
            total_frames_checked += 1
            
            # Check if file exists
            if os.path.exists(frame_path):
                frames_exist += 1
                
                # Try to load image
                try:
                    with Image.open(frame_path) as img:
                        img.verify()  # Verify image integrity
                    frames_valid += 1
                except Exception as e:
                    validation_results["errors"].append(f"Entry {idx}: Invalid image at {frame_path}: {e}")
            else:
                validation_results["warnings"].append(f"Entry {idx}: Frame path does not exist: {frame_path}")
    
    validation_results["stats"]["total_frames_checked"] = total_frames_checked
    validation_results["stats"]["frames_exist"] = frames_exist
    validation_results["stats"]["frames_valid"] = frames_valid
    validation_results["stats"]["frame_existence_rate"] = frames_exist / total_frames_checked if total_frames_checked > 0 else 0
    validation_results["stats"]["frame_validity_rate"] = frames_valid / total_frames_checked if total_frames_checked > 0 else 0
    
    if validation_results["errors"]:
        validation_results["frames_valid"] = False
    
    return validation_results


def validate_language_embeddings(dataset: Dataset, sample_size: int = 10) -> Dict[str, Any]:
    """Validate language embeddings using sentence transformers."""
    
    print(f"Validating language embeddings on {sample_size} sample entries...")
    
    validation_results = {
        "embeddings_valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # Load sentence transformer model
    try:
        lang_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        validation_results["errors"].append(f"Failed to load sentence transformer model: {e}")
        return validation_results
    
    # Sample entries for validation
    sample_indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)
    
    embedding_similarities = []
    
    for idx in sample_indices:
        entry = dataset[idx]
        
        try:
            # Generate embedding from task text
            task_text = entry["task"]
            computed_embedding = lang_model.encode(task_text)
            
            # Get stored embedding
            stored_embedding = entry["lang_vector"]
            
            # Compute cosine similarity
            similarity = np.dot(computed_embedding, stored_embedding) / (
                np.linalg.norm(computed_embedding) * np.linalg.norm(stored_embedding)
            )
            
            embedding_similarities.append(similarity)
            
            # Check if embeddings are very different (similarity < 0.9)
            if similarity < 0.9:
                validation_results["warnings"].append(
                    f"Entry {idx}: Low embedding similarity ({similarity:.3f}) between computed and stored embeddings"
                )
                
        except Exception as e:
            validation_results["errors"].append(f"Entry {idx}: Error validating embeddings: {e}")
    
    if embedding_similarities:
        validation_results["stats"]["avg_embedding_similarity"] = np.mean(embedding_similarities)
        validation_results["stats"]["min_embedding_similarity"] = np.min(embedding_similarities)
        validation_results["stats"]["max_embedding_similarity"] = np.max(embedding_similarities)
    
    if validation_results["errors"]:
        validation_results["embeddings_valid"] = False
    
    return validation_results


def print_validation_summary(validation_results: Dict[str, Dict[str, Any]]):
    """Print a summary of validation results."""
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_valid = True
    
    for validation_name, results in validation_results.items():
        print(f"\n{validation_name.upper()}:")
        print("-" * 40)
        
        # Check if validation passed
        valid_key = f"{validation_name.split('_')[0]}_valid"
        if valid_key in results:
            status = "✅ PASS" if results[valid_key] else "❌ FAIL"
            print(f"Status: {status}")
            all_valid = all_valid and results[valid_key]
        
        # Print stats
        if "stats" in results:
            print("Statistics:")
            for key, value in results["stats"].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        # Print errors
        if results.get("errors"):
            print(f"Errors ({len(results['errors'])}):")
            for error in results["errors"][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(results["errors"]) > 5:
                print(f"  ... and {len(results['errors']) - 5} more errors")
        
        # Print warnings
        if results.get("warnings"):
            print(f"Warnings ({len(results['warnings'])}):")
            for warning in results["warnings"][:5]:  # Show first 5 warnings
                print(f"  - {warning}")
            if len(results["warnings"]) > 5:
                print(f"  ... and {len(results['warnings']) - 5} more warnings")
    
    print("\n" + "="*60)
    overall_status = "✅ ALL VALIDATIONS PASSED" if all_valid else "❌ SOME VALIDATIONS FAILED"
    print(f"OVERALL STATUS: {overall_status}")
    print("="*60)


def main():
    """Main validation function."""
    
    parser = argparse.ArgumentParser(description="Validate HuggingFace dataset format")
    parser.add_argument("dataset_path", help="Path to the HuggingFace dataset")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of samples to check for detailed validation")
    parser.add_argument("--frame-sample-size", type=int, default=20, help="Number of samples to check for frame validation")
    parser.add_argument("--embedding-sample-size", type=int, default=10, help="Number of samples to check for embedding validation")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    try:
        dataset = load_from_disk(args.dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"Dataset loaded successfully. Size: {len(dataset)} entries")
    
    # Run all validations
    validation_results = {}
    
    # Schema validation
    validation_results["schema"] = validate_dataset_schema(dataset)
    
    # Data type validation
    validation_results["data_types"] = validate_data_types(dataset, args.sample_size)
    
    # Ranking consistency validation
    validation_results["ranking_consistency"] = validate_ranking_consistency(dataset)
    
    # Frame path validation
    validation_results["frame_paths"] = validate_frame_paths(dataset, args.frame_sample_size)
    
    # Language embedding validation
    validation_results["language_embeddings"] = validate_language_embeddings(dataset, args.embedding_sample_size)
    
    # Print summary
    print_validation_summary(validation_results)
    
    # Save detailed results
    output_file = f"validation_results_{Path(args.dataset_path).name}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(validation_results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nDetailed validation results saved to: {output_file}")


if __name__ == "__main__":
    main() 