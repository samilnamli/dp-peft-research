import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import time
from tqdm import tqdm
import os
from pathlib import Path

from ..privacy.placements import DPPlacement
from .metrics import MetricsTracker
from ..utils.logging import log_metrics, save_results_to_json


class DPPEFTTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        dp_placement: DPPlacement,
        device: str = 'cuda',
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        epochs: int = 20,
        checkpoint_dir: str = './checkpoints',
        results_dir: str = './results'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.dp_placement = dp_placement
        self.device = device
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        
        self.metrics = MetricsTracker()
        
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        self.model = self.dp_placement.prepare_model()
        
        if self.dp_placement.strategy.value != 'no_dp':
            self.optimizer, self.train_loader = self.dp_placement.attach_privacy_engine(
                self.optimizer,
                self.train_loader,
                self.target_epsilon,
                self.target_delta,
                self.epochs
            )
    
    def train_epoch(self) -> tuple:
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        grad_norms = []
        
        start_time = time.time()
        
        for batch in tqdm(self.train_loader, desc="Training"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            outputs = self.model(**batch)
            loss = outputs['loss']
            
            loss.backward()
            
            if hasattr(self.optimizer, 'original_optimizer'):
                for param in self.model.parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs['logits'], dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(self.train_loader)
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        
        return avg_loss, all_predictions, all_labels, epoch_time, avg_grad_norm
    
    def evaluate(self) -> tuple:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs['loss']
                
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs['logits'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        avg_loss = total_loss / len(self.test_loader)
        
        return avg_loss, all_predictions, all_labels
    
    def train(self) -> Dict[str, Any]:
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            
            train_loss, train_preds, train_labels, epoch_time, grad_norm = self.train_epoch()
            
            test_loss, test_preds, test_labels = self.evaluate()
            
            epsilon = self.dp_placement.get_epsilon(self.target_delta)
            
            num_samples = len(self.train_loader.dataset)
            self.metrics.update(
                loss=test_loss,
                predictions=torch.tensor(test_preds),
                labels=torch.tensor(test_labels),
                epoch_time=epoch_time,
                num_samples=num_samples,
                grad_norm=grad_norm,
                epsilon=epsilon
            )
            
            metrics_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_accuracy': self.metrics.epoch_accuracies[-1],
                'test_f1': self.metrics.epoch_f1_scores[-1],
                'epoch_time': epoch_time,
                'throughput': num_samples / epoch_time,
                'epsilon': epsilon,
                'grad_norm': grad_norm
            }
            
            log_metrics(metrics_dict, step=epoch)
            
            print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {self.metrics.epoch_accuracies[-1]:.4f}, Test F1: {self.metrics.epoch_f1_scores[-1]:.4f}")
            print(f"Epsilon: {epsilon:.2f}, Time: {epoch_time:.2f}s")
        
        summary = self.metrics.get_summary()
        return summary
    
    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics.get_summary()
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
