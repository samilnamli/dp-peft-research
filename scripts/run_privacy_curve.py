#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Generate privacy-utility curves')
    parser.add_argument('--model', type=str, required=True, choices=['bert', 'distilbert', 'vit'])
    parser.add_argument('--dataset', type=str, required=True, choices=['agnews', 'sst2', 'cifar10'])
    parser.add_argument('--placement', type=str, required=True,
                        choices=['no_dp', 'full_dp', 'last_layer', 'adapter_only', 'head_adapter', 'partial_backbone'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    epsilon_values = [0.5, 1.0, 2.0, 4.0, 8.0, float('inf')]
    
    for epsilon in epsilon_values:
        print(f"\n{'='*80}")
        print(f"Running with epsilon: {epsilon}")
        print(f"{'='*80}\n")
        
        cmd = [
            'python', 'scripts/run_experiment.py',
            '--model', args.model,
            '--dataset', args.dataset,
            '--placement', args.placement,
            '--epsilon', str(epsilon),
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--seed', str(args.seed)
        ]
        
        subprocess.run(cmd, check=True)
    
    print(f"\n{'='*80}")
    print("Privacy curve generation completed!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
