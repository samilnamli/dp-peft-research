import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Tuple, Dict, Any
from torch.utils.data import DataLoader, Subset
import torch.nn as nn


class MembershipInferenceAttack:
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        self.model = model
        self.device = device
    
    def compute_loss_per_sample(
        self,
        dataloader: DataLoader
    ) -> np.ndarray:
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                per_sample_loss = loss_fct(
                    outputs['logits'],
                    batch['labels']
                )
                
                losses.extend(per_sample_loss.cpu().numpy())
        
        return np.array(losses)
    
    def loss_threshold_attack(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        train_losses = self.compute_loss_per_sample(train_loader)
        test_losses = self.compute_loss_per_sample(test_loader)
        
        labels = np.concatenate([
            np.ones(len(train_losses)),
            np.zeros(len(test_losses))
        ])
        
        scores = np.concatenate([
            -train_losses,
            -test_losses
        ])
        
        auc = roc_auc_score(labels, scores)
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        advantage = np.max(tpr - fpr)
        
        return {
            'auc': auc,
            'advantage': advantage,
            'train_loss_mean': np.mean(train_losses),
            'train_loss_std': np.std(train_losses),
            'test_loss_mean': np.mean(test_losses),
            'test_loss_std': np.std(test_losses)
        }
    
    def likelihood_ratio_attack(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        shadow_model: nn.Module
    ) -> Dict[str, float]:
        train_losses_target = self.compute_loss_per_sample(train_loader)
        test_losses_target = self.compute_loss_per_sample(test_loader)
        
        shadow_attack = MembershipInferenceAttack(shadow_model, self.device)
        train_losses_shadow = shadow_attack.compute_loss_per_sample(train_loader)
        test_losses_shadow = shadow_attack.compute_loss_per_sample(test_loader)
        
        train_ratio = train_losses_target / (train_losses_shadow + 1e-10)
        test_ratio = test_losses_target / (test_losses_shadow + 1e-10)
        
        labels = np.concatenate([
            np.ones(len(train_ratio)),
            np.zeros(len(test_ratio))
        ])
        
        scores = np.concatenate([
            -train_ratio,
            -test_ratio
        ])
        
        auc = roc_auc_score(labels, scores)
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        advantage = np.max(tpr - fpr)
        
        return {
            'auc': auc,
            'advantage': advantage
        }
    
    def run_attack(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        shadow_model: nn.Module = None
    ) -> Dict[str, Any]:
        results = {}
        
        threshold_results = self.loss_threshold_attack(train_loader, test_loader)
        results['threshold_attack'] = threshold_results
        
        if shadow_model is not None:
            lr_results = self.likelihood_ratio_attack(train_loader, test_loader, shadow_model)
            results['likelihood_ratio_attack'] = lr_results
        
        return results
