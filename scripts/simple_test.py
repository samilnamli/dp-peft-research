#!/usr/bin/env python3
"""Simple standalone test - no external dependencies except torch."""

import torch
import torch.nn as nn
import json
from pathlib import Path

print('Running minimal DP placement comparison...')

class SimpleModel(nn.Module):
    def __init__(self, hidden=256, num_classes=4):
        super().__init__()
        self.embed = nn.Embedding(1000, hidden)
        self.encoder = nn.Linear(hidden, hidden)
        self.lora_down = nn.Linear(hidden, 8)
        self.lora_up = nn.Linear(8, hidden)
        self.classifier = nn.Linear(hidden, num_classes)
    
    def forward(self, x, labels=None):
        x = self.embed(x).mean(dim=1)
        x = torch.relu(self.encoder(x))
        x = x + 0.1 * self.lora_up(torch.relu(self.lora_down(x)))
        logits = self.classifier(x)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return loss, logits

torch.manual_seed(42)
train_x = torch.randint(0, 1000, (1000, 32))
train_y = torch.randint(0, 4, (1000,))
test_x = torch.randint(0, 1000, (200, 32))
test_y = torch.randint(0, 4, (200,))

results = []

for placement in ['no_dp', 'adapter_only', 'head_adapter', 'last_layer']:
    print(f'  Testing {placement}...')
    model = SimpleModel()
    
    for p in model.parameters():
        p.requires_grad = False
    
    if placement == 'no_dp':
        for p in model.parameters():
            p.requires_grad = True
    elif placement == 'adapter_only':
        for p in list(model.lora_down.parameters()) + list(model.lora_up.parameters()):
            p.requires_grad = True
    elif placement == 'head_adapter':
        for p in list(model.lora_down.parameters()) + list(model.lora_up.parameters()) + list(model.classifier.parameters()):
            p.requires_grad = True
    elif placement == 'last_layer':
        for p in model.classifier.parameters():
            p.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    
    accs = []
    for epoch in range(3):
        model.train()
        for i in range(0, 1000, 64):
            opt.zero_grad()
            loss, _ = model(train_x[i:i+64], train_y[i:i+64])
            loss.backward()
            opt.step()
        
        model.eval()
        with torch.no_grad():
            _, logits = model(test_x, test_y)
            acc = (logits.argmax(1) == test_y).float().mean().item()
            accs.append(acc)
    
    results.append({
        'placement': placement,
        'trainable_params': trainable,
        'final_acc': accs[-1],
        'accs': accs
    })
    print(f'    Params: {trainable}, Final Acc: {accs[-1]:.4f}')

print()
print('=' * 50)
print('RESULTS SUMMARY')
print('=' * 50)
print(f"{'Placement':<15} {'Params':<10} {'Accuracy':<10}")
print('-' * 35)
for r in results:
    print(f"{r['placement']:<15} {r['trainable_params']:<10} {r['final_acc']:<10.4f}")

Path('results').mkdir(exist_ok=True)
with open('results/quick_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to results/quick_results.json')
