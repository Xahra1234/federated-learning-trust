#!/usr/bin/env python3
"""Combine results from parallel GPU experiments into CSV files"""
import json
import pandas as pd
from pathlib import Path
import sys

def combine_results(input_dir="results_v2/cifar100"):
    """Combine all JSON results into CSV files"""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Directory not found: {input_dir}")
        sys.exit(1)
    
    # Find all JSON result files
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"❌ No result files found in {input_dir}")
        sys.exit(1)
    
    print(f"📊 Found {len(json_files)} result files")
    
    # Load all results
    results = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"⚠️  Error loading {json_file}: {e}")
    
    if not results:
        print("❌ No valid results loaded")
        sys.exit(1)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by mode, alpha, ratio, attack
    df = df.sort_values(['mode', 'alpha', 'ratio', 'attack'])
    
    # Save all results
    all_results_file = input_path / "all_results.csv"
    df.to_csv(all_results_file, index=False)
    print(f"✅ Saved all results to: {all_results_file}")
    
    # Create summary with aggregated statistics
    group_cols = ['mode', 'alpha', 'ratio', 'attack', 'rounds']
    
    summary = df.groupby(group_cols).agg({
        'clean_acc': ['mean', 'std'],
        'attacked_acc': ['mean', 'std'],
        'accuracy_drop': ['mean', 'std'],
        'tp': ['mean', 'std'],
        'fp': ['mean', 'std'],
        'fn': ['mean', 'std'],
        'tn': ['mean', 'std'],
        'runtime_s': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    
    # Save summary
    summary_file = input_path / "summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"✅ Saved summary to: {summary_file}")
    
    # Print quick statistics
    print(f"\n📈 Quick Statistics:")
    print(f"   Total experiments: {len(df)}")
    print(f"   Modes: {df['mode'].nunique()}")
    print(f"   Alphas: {sorted(df['alpha'].unique())}")
    print(f"   Ratios: {sorted(df['ratio'].unique())}")
    print(f"   Attacks: {sorted(df['attack'].unique())}")
    print(f"   Total runtime: {df['runtime_s'].sum() / 3600:.2f} hours")
    
    # Show best performing methods
    print(f"\n🏆 Best Performing Methods (by attacked accuracy):")
    best = df.nlargest(5, 'attacked_acc')[['mode', 'alpha', 'ratio', 'attack', 'attacked_acc', 'accuracy_drop']]
    print(best.to_string(index=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="results_v2/cifar100", help="Directory with JSON results")
    args = parser.parse_args()
    
    combine_results(args.input_dir)
