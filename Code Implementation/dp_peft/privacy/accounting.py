from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from typing import Optional


class PrivacyAccountant:
    def __init__(
        self,
        target_epsilon: float,
        target_delta: float,
        sample_rate: float,
        epochs: int
    ):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.sample_rate = sample_rate
        self.epochs = epochs
        self.steps = None
        self.noise_multiplier = None
        self.accountant = RDPAccountant()

    def compute_noise_multiplier(self, steps_per_epoch: int) -> float:
        self.steps = steps_per_epoch * self.epochs
        if self.target_epsilon == float('inf'):
            self.noise_multiplier = 0.0
        else:
            self.noise_multiplier = get_noise_multiplier(
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                sample_rate=self.sample_rate,
                epochs=self.epochs,
                accountant='rdp'
            )
        return self.noise_multiplier

    def step(self, noise_multiplier: Optional[float] = None):
        if noise_multiplier is None:
            noise_multiplier = self.noise_multiplier
        if noise_multiplier > 0:
            self.accountant.step(
                noise_multiplier=noise_multiplier,
                sample_rate=self.sample_rate
            )

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        if delta is None:
            delta = self.target_delta
        if self.noise_multiplier == 0.0:
            return float('inf')
        return self.accountant.get_epsilon(delta=delta)

    def get_privacy_spent(self) -> tuple:
        if self.noise_multiplier == 0.0:
            return float('inf'), self.target_delta
        epsilon = self.get_epsilon()
        return epsilon, self.target_delta
