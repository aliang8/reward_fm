#!/usr/bin/env python3
"""Extract and summarize evaluation results from log files."""

import os
import re
import glob
from typing import Dict, List, Tuple

def extract_accuracy_from_log(log_file: str) -> Tuple[str, float, Dict]:
    """Extract final accuracy and metrics from a log file."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for the final evaluation summary
        pattern = r'eval_accuracy:\s+([\d.]+)'
        matches = re.findall(pattern, content)
        
        if matches:
            accuracy = float(matches[-1])  # Take the last one (final result)
        else:
            return log_file, -1.0, {}
        
        # Extract other metrics if available
        metrics = {}
        
        # Extract reward difference
        reward_diff_pattern = r'eval_reward_diff:\s+([\d.-]+)'
        reward_diff_matches = re.findall(reward_diff_pattern, content)
        if reward_diff_matches:
            metrics['reward_diff'] = float(reward_diff_matches[-1])
        
        # Extract demo reward alignment (progress prediction accuracy)
        alignment_pattern = r'demo_reward_alignment:\s+([\d.-]+)'
        alignment_matches = re.findall(alignment_pattern, content)
        if alignment_matches:
            metrics['progress_accuracy'] = float(alignment_matches[-1])
        
        # Extract progress prediction MSE
        mse_pattern = r'progress_prediction_mse:\s+([\d.-]+)'
        mse_matches = re.findall(mse_pattern, content)
        if mse_matches:
            metrics['progress_mse'] = float(mse_matches[-1])
        
        return log_file, accuracy, metrics
        
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return log_file, -1.0, {}

def main():
    """Extract and display results from all evaluation logs."""
    print("🔍 Extracting evaluation results from log files...")
    print("=" * 80)
    
    # Find all log files
    log_patterns = [
        "evals/logs/rlvlmf_*.log",
        "evals/logs/gvl_*.log", 
        "evals/logs/libero_regular_*.log",
        "evals/logs/libero_failure_*.log"
    ]
    
    all_logs = []
    for pattern in log_patterns:
        all_logs.extend(glob.glob(pattern))
    
    if not all_logs:
        print("❌ No log files found. Make sure evaluations have been run.")
        return
    
    # Sort logs by modification time (newest first)
    all_logs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print(f"📊 Found {len(all_logs)} log files")
    print()
    
    # Group results by method and dataset
    results = {
        'rlvlmf': {},
        'gvl': {},
        'other': {}
    }
    
    for log_file in all_logs:
        filename = os.path.basename(log_file)
        log_path, accuracy, metrics = extract_accuracy_from_log(log_file)
        
        if accuracy < 0:
            continue
            
        # Categorize by method
        if 'rlvlmf_' in filename:
            # Extract dataset name
            dataset = filename.replace('rlvlmf_', '').split('_')[0]
            results['rlvlmf'][dataset] = {
                'accuracy': accuracy,
                'metrics': metrics,
                'log_file': log_file
            }
        elif 'gvl_' in filename:
            # Extract dataset name  
            dataset = filename.replace('gvl_', '').split('_')[0]
            results['gvl'][dataset] = {
                'accuracy': accuracy,
                'metrics': metrics,
                'log_file': log_file
            }
        else:
            # Other logs (libero_regular, libero_failure, etc.)
            base_name = filename.split('_')[0] + '_' + filename.split('_')[1]
            results['other'][base_name] = {
                'accuracy': accuracy,
                'metrics': metrics,
                'log_file': log_file
            }
    
    # Display results
    print("📈 EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    
    # RL-VLM-F Results
    if results['rlvlmf']:
        print("\n🔍 RL-VLM-F Results:")
        print("-" * 40)
        for dataset, data in sorted(results['rlvlmf'].items()):
            print(f"  {dataset:15}: {data['accuracy']:.1%}")
            if 'reward_diff' in data['metrics']:
                print(f"                  Reward diff: {data['metrics']['reward_diff']:.3f}")
    
    # GVL Results
    if results['gvl']:
        print("\n🎯 GVL Results:")
        print("-" * 40)
        for dataset, data in sorted(results['gvl'].items()):
            print(f"  {dataset:15}: {data['accuracy']:.1%}")
            if 'reward_diff' in data['metrics']:
                print(f"                  Reward diff: {data['metrics']['reward_diff']:.3f}")
            if 'progress_accuracy' in data['metrics']:
                print(f"                  Progress acc: {data['metrics']['progress_accuracy']:.3f}")
            if 'progress_mse' in data['metrics']:
                print(f"                  Progress MSE: {data['metrics']['progress_mse']:.4f}")
    
    # Other Results
    if results['other']:
        print("\n📊 Other Results:")
        print("-" * 40)
        for name, data in sorted(results['other'].items()):
            print(f"  {name:15}: {data['accuracy']:.1%}")
    
    # Comparison table if we have both methods
    if results['rlvlmf'] and results['gvl']:
        print("\n🆚 METHOD COMPARISON:")
        print("-" * 60)
        print(f"{'Dataset':<15} {'RL-VLM-F':<12} {'GVL':<12} {'Difference':<12}")
        print("-" * 60)
        
        all_datasets = set(results['rlvlmf'].keys()) | set(results['gvl'].keys())
        for dataset in sorted(all_datasets):
            rlvlmf_acc = results['rlvlmf'].get(dataset, {}).get('accuracy', None)
            gvl_acc = results['gvl'].get(dataset, {}).get('accuracy', None)
            
            rlvlmf_str = f"{rlvlmf_acc:.1%}" if rlvlmf_acc is not None else "N/A"
            gvl_str = f"{gvl_acc:.1%}" if gvl_acc is not None else "N/A"
            
            if rlvlmf_acc is not None and gvl_acc is not None:
                diff = gvl_acc - rlvlmf_acc
                diff_str = f"{diff:+.1%}"
            else:
                diff_str = "N/A"
            
            print(f"{dataset:<15} {rlvlmf_str:<12} {gvl_str:<12} {diff_str:<12}")
    
    print("\n" + "=" * 80)
    print("✅ Results extraction complete!")
    
    # Show file locations for detailed analysis
    print("\n📁 Log file locations:")
    for method in ['rlvlmf', 'gvl', 'other']:
        if results[method]:
            print(f"\n{method.upper()}:")
            for dataset, data in sorted(results[method].items()):
                print(f"  {dataset}: {data['log_file']}")

if __name__ == "__main__":
    main()
