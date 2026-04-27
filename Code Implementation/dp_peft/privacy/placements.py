import copy
from enum import Enum
from typing import List, Optional
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils import module_utils as _opacus_module_utils
from opacus.validators import module_validator as _opacus_module_validator

_deepcopy_clone = lambda m: copy.deepcopy(m)
_opacus_module_utils.clone_module = _deepcopy_clone
_opacus_module_validator.clone_module = _deepcopy_clone


class DPPlacementStrategy(Enum):
    NO_DP = "no_dp"
    FULL_DP = "full_dp"
    LAST_LAYER = "last_layer"
    ADAPTER_ONLY = "adapter_only"
    HEAD_ADAPTER = "head_adapter"
    PARTIAL_BACKBONE = "partial_backbone"


class DPPlacement:
    def __init__(
        self,
        model: nn.Module,
        strategy: DPPlacementStrategy,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        top_k_layers: int = 2
    ):
        self.model = model
        self.strategy = strategy
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.top_k_layers = top_k_layers
        self.privacy_engine = None

    def prepare_model(self):
        if self.strategy == DPPlacementStrategy.NO_DP:
            return self._no_dp()
        elif self.strategy == DPPlacementStrategy.FULL_DP:
            return self._full_dp()
        elif self.strategy == DPPlacementStrategy.LAST_LAYER:
            return self._last_layer_dp()
        elif self.strategy == DPPlacementStrategy.ADAPTER_ONLY:
            return self._adapter_only_dp()
        elif self.strategy == DPPlacementStrategy.HEAD_ADAPTER:
            return self._head_adapter_dp()
        elif self.strategy == DPPlacementStrategy.PARTIAL_BACKBONE:
            return self._partial_backbone_dp()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _no_dp(self):
        return self.model

    def _full_dp(self):
        self.model = ModuleValidator.fix(self.model)
        for name, param in self.model.named_parameters():
            if 'embedding' in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True
        return self.model

    def _last_layer_dp(self):
        self.model = ModuleValidator.fix(self.model)
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return self.model

    def _adapter_only_dp(self):
        self.model = ModuleValidator.fix(self.model)
        for name, param in self.model.named_parameters():
            if 'adapter' in name or 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return self.model

    def _head_adapter_dp(self):
        self.model = ModuleValidator.fix(self.model)
        for name, param in self.model.named_parameters():
            if 'classifier' in name or 'adapter' in name or 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return self.model

    @staticmethod
    def _find_transformer_layers(module):
        if hasattr(module, 'base_model') and hasattr(module.base_model, 'model'):
            module = module.base_model.model
        if hasattr(module, 'encoder') and hasattr(module.encoder, 'layer'):
            return list(module.encoder.layer)
        if hasattr(module, 'transformer') and hasattr(module.transformer, 'layer'):
            return list(module.transformer.layer)
        if hasattr(module, 'layer'):
            return list(module.layer)
        return []

    def _partial_backbone_dp(self):
        self.model = ModuleValidator.fix(self.model)
        if hasattr(self.model, 'backbone'):
            layers = self._find_transformer_layers(self.model.backbone)
            total_layers = len(layers)
            trainable_layer_indices = list(range(total_layers - self.top_k_layers, total_layers))
            for idx, layer in enumerate(layers):
                for param in layer.parameters():
                    if idx in trainable_layer_indices:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        for name, param in self.model.named_parameters():
            if 'classifier' in name or 'adapter' in name or 'lora' in name:
                param.requires_grad = True
        return self.model

    def attach_privacy_engine(
        self,
        optimizer,
        data_loader,
        target_epsilon: float,
        target_delta: float,
        epochs: int
    ):
        if self.strategy == DPPlacementStrategy.NO_DP:
            return optimizer, data_loader
        self.privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=data_loader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            epochs=epochs,
            max_grad_norm=self.max_grad_norm
        )
        self.model = model
        return optimizer, data_loader

    def get_epsilon(self, delta: float) -> float:
        if self.strategy == DPPlacementStrategy.NO_DP:
            return float('inf')
        if self.privacy_engine is None:
            return 0.0
        return self.privacy_engine.get_epsilon(delta)


def get_placement_strategy(
    model: nn.Module,
    strategy_name: str,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    top_k_layers: int = 2
) -> DPPlacement:
    strategy = DPPlacementStrategy(strategy_name)
    placement = DPPlacement(
        model=model,
        strategy=strategy,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
        top_k_layers=top_k_layers
    )
    return placement
