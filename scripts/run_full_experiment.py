#!/usr/bin/env python3
"""
Full DP-PEFT Experiment with Real BERT + AG News + Privacy Metrics
Run in tmux for long-running training on GPU
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import json
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import f1_score, roc_auc_score

from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from datasets import load_dataset

from dp_peft.utils.reproducibility import set_seed


class BERTClassifier(nn.Module):
    """BERT with LoRA for text classification, Opacus compatible."""
    def __init__(self, model_name='bert-base-uncased', num_labels=4, lora_r=8):
        super().__init__()
        
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        
        # Add LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value"]
        )
        self.bert = get_peft_model(self.bert, lora_config)
        
        # Freeze base model, only LoRA trainable
        for name, param in self.bert.named_parameters():
            if 'lora' not in name.lower():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, num_labels)
        )
        
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pool: use CLS token
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {'loss': loss, 'logits': logits}


def configure_placement(model, placement):
    """Configure trainable parameters based on placement strategy."""
    # First freeze all
    for param in model.parameters():
        param.requires_grad = False
    
    if placement == 'no_dp':
        # LoRA + classifier trainable (no DP applied)
        for name, param in model.named_parameters():
            if 'lora' in name.lower() or 'classifier' in name:
                param.requires_grad = True
    
    elif placement == 'adapter_only':
        # Only LoRA params trainable (with DP)
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
    
    elif placement == 'head_adapter':
        # LoRA + classifier trainable (with DP)
        for name, param in model.named_parameters():
            if 'lora' in name.lower() or 'classifier' in name:
                param.requires_grad = True
    
    elif placement == 'last_layer':
        # Only classifier trainable (with DP)
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
    
    elif placement == 'full_dp':
        # All params trainable (with DP) - unfreeze everything
        for param in model.parameters():
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    return trainable, total


def load_agnews_data(tokenizer, batch_size=32, max_length=128, max_samples=None):
    """Load AG News dataset."""
    print("Loading AG News dataset...")
    dataset = load_dataset('ag_news')
    
    train_texts = list(dataset['train']['text'])
    train_labels = list(dataset['train']['label'])
    test_texts = list(dataset['test']['text'])
    test_labels = list(dataset['test']['label'])
    
    if max_samples:
        train_texts = train_texts[:max_samples]
        train_labels = train_labels[:max_samples]
        test_texts = test_texts[:min(max_samples//4, len(test_texts))]
        test_labels = test_labels[:min(max_samples//4, len(test_labels))]
    
    print(f"Tokenizing {len(train_texts)} train, {len(test_texts)} test samples...")
    
    train_encodings = tokenizer(
        train_texts, truncation=True, padding='max_length',
        max_length=max_length, return_tensors='pt'
    )
    test_encodings = tokenizer(
        test_texts, truncation=True, padding='max_length',
        max_length=max_length, return_tensors='pt'
    )
    
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        
        def __len__(self):
            return len(self.labels)
    
    train_dataset = TextDataset(train_encodings, train_labels)
    test_dataset = TextDataset(test_encodings, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader


def compute_mia_metrics(model, train_loader, test_loader, device, max_samples=2000):
    """Compute membership inference attack metrics."""
    model.eval()
    
    train_losses = []
    test_losses = []
    
    with torch.no_grad():
        count = 0
        for batch in train_loader:
            if count >= max_samples:
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fn(outputs['logits'], labels)
            train_losses.extend(losses.cpu().numpy())
            count += len(labels)
        
        count = 0
        for batch in test_loader:
            if count >= max_samples:
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fn(outputs['logits'], labels)
            test_losses.extend(losses.cpu().numpy())
            count += len(labels)
    
    # MIA: lower loss = more likely member
    all_losses = np.array(train_losses + test_losses)
    all_labels = np.array([1] * len(train_losses) + [0] * len(test_losses))
    
    membership_scores = -all_losses  # Invert: lower loss = higher score
    
    try:
        auc = roc_auc_score(all_labels, membership_scores)
    except:
        auc = 0.5
    
    # Compute advantage
    thresholds = np.percentile(membership_scores, np.arange(0, 101, 5))
    best_advantage = 0
    for thresh in thresholds:
        preds = (membership_scores > thresh).astype(int)
        tp = np.sum((preds == 1) & (all_labels == 1))
        fp = np.sum((preds == 1) & (all_labels == 0))
        tn = np.sum((preds == 0) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        advantage = abs(tpr - fpr)
        best_advantage = max(best_advantage, advantage)
    
    return {'mia_auc': auc, 'mia_advantage': best_advantage}


def run_experiment(
    placement,
    target_epsilon=8.0,
    epochs=10,
    batch_size=16,
    lr=2e-5,
    max_grad_norm=1.0,
    delta=1e-5,
    max_samples=10000,
    device='cuda'
):
    """Run a single experiment with specified placement."""
    set_seed(42)
    
    print(f"\n{'='*70}")
    print(f"Running: {placement} | ε={target_epsilon} | epochs={epochs}")
    print(f"{'='*70}")
    
    # Load tokenizer and model
    print("Loading BERT model...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = BERTClassifier(model_name='bert-base-uncased', num_labels=4, lora_r=8)
    
    # Configure placement
    trainable_params, total_params = configure_placement(model, placement)
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Make Opacus compatible
    if placement != 'no_dp':
        model = ModuleValidator.fix(model)
    
    model = model.to(device)
    
    # Load data
    train_loader, test_loader = load_agnews_data(
        tokenizer, batch_size=batch_size, max_length=128, max_samples=max_samples
    )
    
    # Optimizer
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    
    # Attach PrivacyEngine for DP placements
    privacy_engine = None
    if placement != 'no_dp':
        print("Attaching PrivacyEngine...")
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=epochs,
            target_epsilon=target_epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
        )
        print(f"Noise multiplier: {optimizer.noise_multiplier:.4f}")
    
    results = {
        'placement': placement,
        'target_epsilon': target_epsilon,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'train_samples': len(train_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'batch_size': batch_size,
        'epochs': epochs,
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': [],
        'test_f1s': [],
        'epsilons': [],
        'epoch_times': [],
        'grad_norms': []
    }
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        epoch_grad_norms = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            loss.backward()
            
            # Track gradient norms
            grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            epoch_grad_norms.append(grad_norm)
            
            optimizer.step()
            
            total_loss += loss.item()
            preds = outputs['logits'].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # Evaluate
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                test_loss += outputs['loss'].item()
                preds = outputs['logits'].argmax(dim=-1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_acc = test_correct / test_total
        test_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Get epsilon
        if privacy_engine:
            epsilon = privacy_engine.get_epsilon(delta)
        else:
            epsilon = float('inf')
        
        epoch_time = time.time() - start_time
        avg_grad_norm = np.mean(epoch_grad_norms)
        
        results['train_losses'].append(train_loss)
        results['train_accs'].append(train_acc)
        results['test_losses'].append(test_loss)
        results['test_accs'].append(test_acc)
        results['test_f1s'].append(test_f1)
        results['epsilons'].append(epsilon if epsilon != float('inf') else -1)
        results['epoch_times'].append(epoch_time)
        results['grad_norms'].append(avg_grad_norm)
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, "
              f"F1={test_f1:.4f}, ε={epsilon:.2f}, Time={epoch_time:.1f}s")
    
    # Compute MIA metrics
    print("Computing MIA metrics...")
    # Get base model for MIA
    if hasattr(model, '_module'):
        mia_model = model._module
    else:
        mia_model = model
    
    # Recreate loaders without DP wrapping
    train_loader_mia, test_loader_mia = load_agnews_data(
        tokenizer, batch_size=batch_size, max_length=128, max_samples=max_samples
    )
    
    mia_metrics = compute_mia_metrics(mia_model, train_loader_mia, test_loader_mia, device)
    results['mia_auc'] = mia_metrics['mia_auc']
    results['mia_advantage'] = mia_metrics['mia_advantage']
    
    # Final metrics
    results['final_train_acc'] = results['train_accs'][-1]
    results['final_test_acc'] = results['test_accs'][-1]
    results['final_f1'] = results['test_f1s'][-1]
    results['final_epsilon'] = results['epsilons'][-1]
    results['avg_epoch_time'] = np.mean(results['epoch_times'])
    results['grad_norm_variance'] = np.var(results['grad_norms'])
    
    print(f"\nFinal Results for {placement}:")
    print(f"  Test Accuracy: {results['final_test_acc']:.4f}")
    print(f"  F1 Score: {results['final_f1']:.4f}")
    print(f"  Epsilon: {results['final_epsilon']:.2f}")
    print(f"  MIA AUC: {results['mia_auc']:.4f}")
    print(f"  MIA Advantage: {results['mia_advantage']:.4f}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--placements', type=str, nargs='+', 
                        default=['no_dp', 'adapter_only', 'head_adapter', 'last_layer'],
                        help='Placements to run')
    parser.add_argument('--epsilon', type=float, default=8.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_samples', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=2e-5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    all_results = []
    
    for placement in args.placements:
        try:
            results = run_experiment(
                placement=placement,
                target_epsilon=args.epsilon,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_samples=args.max_samples,
                device=device
            )
            all_results.append(results)
            
            # Save intermediate results
            output_dir = Path(__file__).parent.parent / 'results'
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / f'bert_agnews_{placement}_eps{args.epsilon}.json', 'w') as f:
                json.dump(results, f, indent=2)
            
        except Exception as e:
            print(f"Error running {placement}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print(f"{'Placement':<18} {'Params':<12} {'Accuracy':<10} {'F1':<10} {'Epsilon':<10} {'MIA AUC':<10} {'MIA Adv':<10}")
    print("-" * 90)
    for r in all_results:
        eps_str = f"{r['final_epsilon']:.2f}" if r['final_epsilon'] > 0 else "∞"
        print(f"{r['placement']:<18} {r['trainable_params']:<12,} {r['final_test_acc']:<10.4f} "
              f"{r['final_f1']:<10.4f} {eps_str:<10} {r['mia_auc']:<10.4f} {r['mia_advantage']:<10.4f}")
    
    # Save all results
    output_dir = Path(__file__).parent.parent / 'results'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(output_dir / f'full_experiment_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    # Generate plots
    generate_plots(all_results, output_dir, timestamp)


def generate_plots(all_results, output_dir, timestamp):
    """Generate presentation plots."""
    import matplotlib.pyplot as plt
    
    if not all_results:
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    placements = [r['placement'] for r in all_results]
    colors = ['#27ae60', '#3498db', '#9b59b6', '#f39c12', '#e74c3c']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Accuracy
    accs = [r['final_test_acc'] for r in all_results]
    bars = axes[0, 0].bar(range(len(placements)), accs, color=colors[:len(placements)])
    axes[0, 0].set_xticks(range(len(placements)))
    axes[0, 0].set_xticklabels(placements, rotation=15)
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].set_title('Accuracy by DP Placement (BERT + AG News)', fontweight='bold')
    for bar, acc in zip(bars, accs):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{acc:.3f}', ha='center', fontsize=9)
    
    # 2. Epsilon
    epsilons = [r['final_epsilon'] if r['final_epsilon'] > 0 else 0 for r in all_results]
    bars = axes[0, 1].bar(range(len(placements)), epsilons, color=colors[:len(placements)])
    axes[0, 1].set_xticks(range(len(placements)))
    axes[0, 1].set_xticklabels(placements, rotation=15)
    axes[0, 1].set_ylabel('Privacy Budget (ε)')
    axes[0, 1].set_title('Privacy Budget Consumed', fontweight='bold')
    
    # 3. MIA AUC
    mia_aucs = [r['mia_auc'] for r in all_results]
    bars = axes[1, 0].bar(range(len(placements)), mia_aucs, color=colors[:len(placements)])
    axes[1, 0].set_xticks(range(len(placements)))
    axes[1, 0].set_xticklabels(placements, rotation=15)
    axes[1, 0].set_ylabel('MIA Attack AUC')
    axes[1, 0].set_title('Membership Inference Attack AUC\n(0.5 = random guess)', fontweight='bold')
    axes[1, 0].axhline(y=0.5, color='green', linestyle='--', alpha=0.7)
    axes[1, 0].set_ylim(0.4, 1.0)
    for bar, auc in zip(bars, mia_aucs):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{auc:.3f}', ha='center', fontsize=9)
    
    # 4. Training curves
    for i, r in enumerate(all_results):
        epochs = range(1, len(r['test_accs']) + 1)
        axes[1, 1].plot(epochs, r['test_accs'], marker='o', color=colors[i], 
                       label=r['placement'], linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Test Accuracy')
    axes[1, 1].set_title('Training Curves', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'bert_agnews_results_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f'bert_agnews_results_{timestamp}.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}")


if __name__ == '__main__':
    main()
