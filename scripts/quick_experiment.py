#!/usr/bin/env python3
"""Quick experiment script for generating preliminary results."""

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

from dp_peft.models import get_text_model
from dp_peft.data import get_text_dataloaders
from dp_peft.utils.reproducibility import set_seed


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
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
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    return total_loss / len(train_loader), correct / total


def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += outputs['loss'].item()
            preds = outputs['logits'].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(test_loader), correct / total


def run_quick_experiment(
    model_name='bert',
    dataset_name='agnews',
    epochs=3,
    batch_size=32,
    lr=2e-5,
    max_samples=5000,
    seed=42
):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine num_labels
    num_labels = 4 if dataset_name == 'agnews' else 2
    
    # Load model
    print(f"\nLoading {model_name} model...")
    model = get_text_model(model_name, num_labels=num_labels, peft_method='lora')
    model = model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Load data
    print(f"\nLoading {dataset_name} dataset...")
    train_loader, test_loader = get_text_dataloaders(
        dataset_name, 
        'bert-base-uncased' if model_name == 'bert' else 'distilbert-base-uncased',
        batch_size=batch_size,
        max_length=128,
        num_workers=0,
        pin_memory=False
    )
    
    # Limit samples for quick experiment
    if max_samples:
        train_loader.dataset.labels = train_loader.dataset.labels[:max_samples]
        for key in train_loader.dataset.encodings:
            train_loader.dataset.encodings[key] = train_loader.dataset.encodings[key][:max_samples]
    
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Training
    results = {
        'model': model_name,
        'dataset': dataset_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': [],
        'epoch_times': []
    }
    
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        
        epoch_time = time.time() - start_time
        
        results['train_losses'].append(train_loss)
        results['train_accs'].append(train_acc)
        results['test_losses'].append(test_loss)
        results['test_accs'].append(test_acc)
        results['epoch_times'].append(epoch_time)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
    
    # Final results
    results['final_train_acc'] = results['train_accs'][-1]
    results['final_test_acc'] = results['test_accs'][-1]
    results['avg_epoch_time'] = sum(results['epoch_times']) / len(results['epoch_times'])
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f'quick_{model_name}_{dataset_name}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Final Results ===")
    print(f"Final Test Accuracy: {results['final_test_acc']:.4f}")
    print(f"Results saved to: {results_file}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert', choices=['bert', 'distilbert'])
    parser.add_argument('--dataset', type=str, default='agnews', choices=['agnews', 'sst2'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    run_quick_experiment(
        model_name=args.model,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        seed=args.seed
    )
