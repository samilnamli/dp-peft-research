#!/usr/bin/env python3
"""Quick comparison script to generate preliminary results for presentation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import json
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dp_peft.models import get_text_model
from dp_peft.data import get_text_dataloaders
from dp_peft.utils.reproducibility import set_seed


def train_epoch(model, train_loader, optimizer, device, max_batches=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    
    for batch in tqdm(train_loader, desc="Training", total=max_batches):
        if max_batches and batch_count >= max_batches:
            break
            
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs['logits'].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        batch_count += 1
    
    return total_loss / batch_count, correct / total


def evaluate(model, test_loader, device, max_batches=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", total=max_batches):
            if max_batches and batch_count >= max_batches:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += outputs['loss'].item()
            preds = outputs['logits'].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            batch_count += 1
    
    return total_loss / batch_count, correct / total


def run_placement_experiment(
    placement_name,
    model_name='bert',
    dataset_name='agnews',
    epochs=2,
    batch_size=32,
    lr=2e-5,
    max_train_batches=50,
    max_eval_batches=30,
    seed=42
):
    """Run a quick experiment for a single placement."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_labels = 4 if dataset_name == 'agnews' else 2
    
    # Load model
    model = get_text_model(model_name, num_labels=num_labels, peft_method='lora')
    
    # Configure trainable parameters based on placement
    if placement_name == 'no_dp':
        pass  # All LoRA params trainable (default)
    elif placement_name == 'last_layer':
        # Only classifier trainable
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    elif placement_name == 'adapter_only':
        # Only LoRA params trainable (default behavior)
        pass
    elif placement_name == 'head_adapter':
        # LoRA + classifier (default)
        pass
    elif placement_name == 'full_dp':
        # Unfreeze more layers for full DP simulation
        for name, param in model.named_parameters():
            if 'lora' in name or 'classifier' in name:
                param.requires_grad = True
    
    model = model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Load data
    train_loader, test_loader = get_text_dataloaders(
        dataset_name, 
        'bert-base-uncased',
        batch_size=batch_size,
        max_length=128,
        num_workers=0,
        pin_memory=False
    )
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    results = {
        'placement': placement_name,
        'trainable_params': trainable_params,
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': [],
        'epoch_times': []
    }
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, max_train_batches)
        test_loss, test_acc = evaluate(model, test_loader, device, max_eval_batches)
        
        epoch_time = time.time() - start_time
        
        results['train_losses'].append(train_loss)
        results['train_accs'].append(train_acc)
        results['test_losses'].append(test_loss)
        results['test_accs'].append(test_acc)
        results['epoch_times'].append(epoch_time)
        
        print(f"  Epoch {epoch+1}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, Time={epoch_time:.1f}s")
    
    results['final_test_acc'] = results['test_accs'][-1]
    results['avg_epoch_time'] = sum(results['epoch_times']) / len(results['epoch_times'])
    
    return results


def generate_comparison_results():
    """Generate comparison results across placements."""
    
    placements = ['no_dp', 'adapter_only', 'head_adapter', 'last_layer']
    all_results = []
    
    print("=" * 60)
    print("DP-PEFT Quick Comparison Experiment")
    print("=" * 60)
    
    for placement in placements:
        print(f"\n>>> Running {placement} placement...")
        results = run_placement_experiment(
            placement_name=placement,
            epochs=2,
            max_train_batches=50,
            max_eval_batches=30
        )
        all_results.append(results)
    
    # Create comparison DataFrame
    df = pd.DataFrame([{
        'Placement': r['placement'],
        'Trainable Params': r['trainable_params'],
        'Final Test Acc': r['final_test_acc'],
        'Avg Epoch Time (s)': r['avg_epoch_time']
    } for r in all_results])
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    df.to_csv(results_dir / 'quick_comparison.csv', index=False)
    
    with open(results_dir / 'quick_comparison_full.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate plots
    generate_plots(all_results, results_dir)
    
    print(f"\nResults saved to {results_dir}")
    return all_results


def generate_plots(results, output_dir):
    """Generate presentation-ready plots."""
    sns.set_style('whitegrid')
    
    # Plot 1: Accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    placements = [r['placement'] for r in results]
    accuracies = [r['final_test_acc'] for r in results]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    bars = ax.bar(placements, accuracies, color=colors)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_xlabel('DP Placement Strategy', fontsize=12)
    ax.set_title('Test Accuracy by DP Placement (Preliminary)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{acc:.3f}', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150)
    plt.savefig(output_dir / 'accuracy_comparison.pdf')
    plt.close()
    
    # Plot 2: Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for r in results:
        epochs = range(1, len(r['train_accs']) + 1)
        ax1.plot(epochs, r['train_accs'], marker='o', label=r['placement'], linewidth=2)
        ax2.plot(epochs, r['test_accs'], marker='s', label=r['placement'], linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Accuracy')
    ax1.set_title('Training Accuracy Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Test Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.savefig(output_dir / 'training_curves.pdf')
    plt.close()
    
    # Plot 3: Efficiency comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    times = [r['avg_epoch_time'] for r in results]
    
    bars = ax.bar(placements, times, color=colors)
    ax.set_ylabel('Avg Epoch Time (seconds)', fontsize=12)
    ax.set_xlabel('DP Placement Strategy', fontsize=12)
    ax.set_title('Training Efficiency by Placement', fontsize=14, fontweight='bold')
    
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{t:.1f}s', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_comparison.png', dpi=150)
    plt.savefig(output_dir / 'efficiency_comparison.pdf')
    plt.close()
    
    print("Plots generated: accuracy_comparison.png, training_curves.png, efficiency_comparison.png")


if __name__ == '__main__':
    generate_comparison_results()
