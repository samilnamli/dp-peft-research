import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from typing import List, Dict, Any, Optional
import torch


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.epoch_losses = []
        self.epoch_accuracies = []
        self.epoch_f1_scores = []
        self.epoch_times = []
        self.epoch_throughputs = []
        self.epoch_grad_norms = []
        self.epoch_epsilons = []
        self.all_predictions = []
        self.all_labels = []

    def update(
        self,
        loss: float,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        epoch_time: float,
        num_samples: int,
        grad_norm: Optional[float] = None,
        epsilon: Optional[float] = None
    ):
        self.epoch_losses.append(loss)

        preds = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
        labs = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

        accuracy = accuracy_score(labs, preds)
        f1 = f1_score(labs, preds, average='macro')

        self.epoch_accuracies.append(accuracy)
        self.epoch_f1_scores.append(f1)
        self.epoch_times.append(epoch_time)
        self.epoch_throughputs.append(num_samples / epoch_time)

        if grad_norm is not None:
            self.epoch_grad_norms.append(grad_norm)

        if epsilon is not None:
            self.epoch_epsilons.append(epsilon)

        self.all_predictions.extend(preds.tolist())
        self.all_labels.extend(labs.tolist())

    def get_gradient_norm_variance(self) -> float:
        if len(self.epoch_grad_norms) == 0:
            return 0.0
        return np.var(self.epoch_grad_norms)

    def get_loss_oscillation(self, window: int = 5) -> float:
        if len(self.epoch_losses) < window:
            return 0.0
        recent_losses = self.epoch_losses[-window:]
        return np.std(recent_losses)

    def get_epochs_to_target(self, target_accuracy: float, baseline_accuracy: float) -> Optional[int]:
        target = baseline_accuracy * target_accuracy
        for i, acc in enumerate(self.epoch_accuracies):
            if acc >= target:
                return i + 1
        return None

    def get_time_to_utility(self, target_accuracy: float, baseline_accuracy: float) -> Optional[float]:
        epochs = self.get_epochs_to_target(target_accuracy, baseline_accuracy)
        if epochs is None:
            return None
        return sum(self.epoch_times[:epochs])

    def get_summary(self) -> Dict[str, Any]:
        return {
            'final_accuracy': self.epoch_accuracies[-1] if self.epoch_accuracies else 0.0,
            'final_f1': self.epoch_f1_scores[-1] if self.epoch_f1_scores else 0.0,
            'final_loss': self.epoch_losses[-1] if self.epoch_losses else 0.0,
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0.0,
            'avg_throughput': np.mean(self.epoch_throughputs) if self.epoch_throughputs else 0.0,
            'grad_norm_variance': self.get_gradient_norm_variance(),
            'loss_oscillation': self.get_loss_oscillation(),
            'final_epsilon': self.epoch_epsilons[-1] if self.epoch_epsilons else float('inf'),
            'convergence_curve': self.epoch_losses,
            'accuracy_curve': self.epoch_accuracies
        }
