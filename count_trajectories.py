#!/usr/bin/env python3
"""
Script to count trajectories in each processed dataset.

This script reads the preprocess.yaml configuration and counts the number of
trajectories in each dataset/subset combination by loading the processed datasets.
"""

import os
import json
import yaml
from pathlib import Path
from datasets import Dataset
from rich.console import Console
from rich.table import Table
from rich import print as rprint


def load_preprocess_config(config_path: str = "rfm/configs/preprocess.yaml") -> dict:
    """Load the preprocess configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_cache_dir_for_dataset(base_cache_dir: str, dataset_path: str, subset: str) -> str:
    """
    Construct the cache directory path for a dataset/subset pair.
    
    Following the logic from rfm/data/scripts/preprocess_datasets.py:
    cache_key = f"{dataset_path}/{subset}"
    individual_cache_dir = os.path.join(cache_dir, cache_key.replace("/", "_").replace(":", "_"))
    """
    cache_key = f"{dataset_path}/{subset}"
    individual_cache_dir = os.path.join(base_cache_dir, cache_key.replace("/", "_").replace(":", "_"))
    return individual_cache_dir


def count_trajectories_in_dataset(cache_dir: str) -> dict:
    """
    Count trajectories in a processed dataset directory.
    
    Returns a dictionary with:
    - 'count': number of trajectories
    - 'exists': whether the dataset exists
    - 'error': any error message
    """
    result = {
        'count': 0,
        'exists': False,
        'error': None
    }
    
    # Check if cache directory exists
    if not os.path.exists(cache_dir):
        result['error'] = "Cache directory not found"
        return result
    
    # Check for dataset_info.json
    info_file = os.path.join(cache_dir, "dataset_info.json")
    if not os.path.exists(info_file):
        result['error'] = "dataset_info.json not found"
        return result
    
    # Check for processed_dataset directory
    dataset_cache_dir = os.path.join(cache_dir, "processed_dataset")
    if not os.path.exists(dataset_cache_dir):
        result['error'] = "processed_dataset directory not found"
        return result
    
    result['exists'] = True
    
    try:
        # Load the dataset to count trajectories
        dataset = Dataset.load_from_disk(dataset_cache_dir, keep_in_memory=False)
        result['count'] = len(dataset)
        
        # Also read dataset_info.json for additional metadata
        with open(info_file, 'r') as f:
            info = json.load(f)
            result['info'] = info
            
    except Exception as e:
        result['error'] = f"Error loading dataset: {str(e)}"
    
    return result


def main():
    """Main function to count trajectories in all processed datasets."""
    console = Console()
    
    # Get the processed datasets path from environment variable
    cache_dir = os.environ.get("RFM_PROCESSED_DATASETS_PATH")
    
    if not cache_dir:
        console.print("[bold red]Error:[/bold red] RFM_PROCESSED_DATASETS_PATH environment variable is not set!")
        console.print("Please set it to the directory containing your processed datasets.")
        console.print("Example: export RFM_PROCESSED_DATASETS_PATH=/scr/shared/reward_fm/processed_datasets")
        return
    
    console.print(f"[bold]Using cache directory:[/bold] {cache_dir}")
    
    # Load preprocess configuration
    config_path = "rfm/configs/preprocess.yaml"
    if not os.path.exists(config_path):
        console.print(f"[bold red]Error:[/bold red] Config file not found: {config_path}")
        return
    
    config = load_preprocess_config(config_path)
    console.print(f"[bold]Loaded config from:[/bold] {config_path}\n")
    
    # Process training datasets
    train_datasets = config.get('train_datasets', [])
    train_subsets = config.get('train_subsets', [])
    
    # Process evaluation datasets
    eval_datasets = config.get('eval_datasets', [])
    eval_subsets = config.get('eval_subsets', [])
    
    # Create tables for display
    train_table = Table(title="Training Datasets - Trajectory Counts", show_header=True, header_style="bold magenta")
    train_table.add_column("Dataset", style="cyan", width=40)
    train_table.add_column("Subset", style="green", width=35)
    train_table.add_column("Trajectories", justify="right", style="yellow")
    train_table.add_column("Status", style="white")
    
    eval_table = Table(title="Evaluation Datasets - Trajectory Counts", show_header=True, header_style="bold magenta")
    eval_table.add_column("Dataset", style="cyan", width=40)
    eval_table.add_column("Subset", style="green", width=35)
    eval_table.add_column("Trajectories", justify="right", style="yellow")
    eval_table.add_column("Status", style="white")
    
    # Count training datasets
    console.print("[bold blue]Processing Training Datasets...[/bold blue]")
    train_total = 0
    train_found = 0
    
    for dataset_path, dataset_subsets in zip(train_datasets, train_subsets):
        for subset in dataset_subsets:
            individual_cache_dir = get_cache_dir_for_dataset(cache_dir, dataset_path, subset)
            result = count_trajectories_in_dataset(individual_cache_dir)
            
            if result['exists'] and result['error'] is None:
                count_str = f"{result['count']:,}"
                status = "✓ Found"
                status_style = "green"
                train_total += result['count']
                train_found += 1
            else:
                count_str = "N/A"
                status = f"✗ {result['error']}"
                status_style = "red"
            
            train_table.add_row(dataset_path, subset, count_str, f"[{status_style}]{status}[/{status_style}]")
    
    # Count evaluation datasets
    console.print("[bold blue]Processing Evaluation Datasets...[/bold blue]")
    eval_total = 0
    eval_found = 0
    
    for dataset_path, dataset_subsets in zip(eval_datasets, eval_subsets):
        for subset in dataset_subsets:
            individual_cache_dir = get_cache_dir_for_dataset(cache_dir, dataset_path, subset)
            result = count_trajectories_in_dataset(individual_cache_dir)
            
            if result['exists'] and result['error'] is None:
                count_str = f"{result['count']:,}"
                status = "✓ Found"
                status_style = "green"
                eval_total += result['count']
                eval_found += 1
            else:
                count_str = "N/A"
                status = f"✗ {result['error']}"
                status_style = "red"
            
            eval_table.add_row(dataset_path, subset, count_str, f"[{status_style}]{status}[/{status_style}]")
    
    # Display results
    console.print()
    console.print(train_table)
    console.print()
    console.print(eval_table)
    console.print()
    
    # Summary
    summary_table = Table(title="Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Category", style="cyan")
    summary_table.add_column("Total Trajectories", justify="right", style="yellow")
    summary_table.add_column("Datasets Found", justify="right", style="green")
    
    summary_table.add_row("Training", f"{train_total:,}", str(train_found))
    summary_table.add_row("Evaluation", f"{eval_total:,}", str(eval_found))
    summary_table.add_row("[bold]Grand Total[/bold]", f"[bold]{train_total + eval_total:,}[/bold]", f"[bold]{train_found + eval_found}[/bold]")
    
    console.print(summary_table)
    console.print()
    
    # Save results to JSON
    output_file = "trajectory_counts.json"
    results = {
        "cache_dir": cache_dir,
        "training_datasets": [],
        "evaluation_datasets": [],
        "summary": {
            "train_total_trajectories": train_total,
            "train_datasets_found": train_found,
            "eval_total_trajectories": eval_total,
            "eval_datasets_found": eval_found,
            "grand_total_trajectories": train_total + eval_total,
            "total_datasets_found": train_found + eval_found
        }
    }
    
    # Add training dataset details
    for dataset_path, dataset_subsets in zip(train_datasets, train_subsets):
        for subset in dataset_subsets:
            individual_cache_dir = get_cache_dir_for_dataset(cache_dir, dataset_path, subset)
            result = count_trajectories_in_dataset(individual_cache_dir)
            results["training_datasets"].append({
                "dataset": dataset_path,
                "subset": subset,
                "cache_dir": individual_cache_dir,
                "trajectory_count": result['count'] if result['exists'] else None,
                "exists": result['exists'],
                "error": result['error']
            })
    
    # Add evaluation dataset details
    for dataset_path, dataset_subsets in zip(eval_datasets, eval_subsets):
        for subset in dataset_subsets:
            individual_cache_dir = get_cache_dir_for_dataset(cache_dir, dataset_path, subset)
            result = count_trajectories_in_dataset(individual_cache_dir)
            results["evaluation_datasets"].append({
                "dataset": dataset_path,
                "subset": subset,
                "cache_dir": individual_cache_dir,
                "trajectory_count": result['count'] if result['exists'] else None,
                "exists": result['exists'],
                "error": result['error']
            })
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"[bold green]Results saved to:[/bold green] {output_file}")


if __name__ == "__main__":
    main()

