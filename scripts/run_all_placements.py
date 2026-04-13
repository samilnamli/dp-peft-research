#!/usr/bin/env python3
import argparse
import subprocess
import pandas as pd
from pathlib import Path
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Run all DP placements')
    parser.add_argument('--model', type=str, required=True, choices=['bert', 'distilbert', 'vit'])
    parser.add_argument('--dataset', type=str, required=True, choices=['agnews', 'sst2', 'cifar10'])
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    placements = ['no_dp', 'full_dp', 'last_layer', 'adapter_only', 'head_adapter', 'partial_backbone']
    
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    for placement in placements:
        print(f"\n{'='*80}")
        print(f"Running placement: {placement}")
        print(f"{'='*80}\n")
        
        cmd = [
            'python', 'scripts/run_experiment.py',
            '--model', args.model,
            '--dataset', args.dataset,
            '--placement', placement,
            '--epsilon', str(args.epsilon),
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--seed', str(args.seed)
        ]
        
        subprocess.run(cmd, check=True)
        
        result_file = results_dir / f"{args.model}_{args.dataset}_{placement}_eps{args.epsilon}.json"
        
        if result_file.exists():
            with open(result_file, 'r') as f:
                result = json.load(f)
                result['placement'] = placement
                all_results.append(result)
    
    df = pd.DataFrame(all_results)
    
    comparison_file = results_dir / f"comparison_{args.model}_{args.dataset}_eps{args.epsilon}.csv"
    df.to_csv(comparison_file, index=False)
    
    print(f"\n{'='*80}")
    print("All placements completed!")
    print(f"Comparison saved to: {comparison_file}")
    print(f"{'='*80}\n")
    
    print("\nSummary:")
    print(df[['placement', 'final_accuracy', 'final_f1', 'final_epsilon', 'avg_epoch_time']])


if __name__ == '__main__':
    main()
