import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from typing import Optional, Dict, Any


class AdapterModule(nn.Module):
    def __init__(self, hidden_size: int, adapter_size: int = 64):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_size, hidden_size)

    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual


class ViTWithAdapters(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        adapter_hidden_dim: int = 64
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        config = ViTConfig.from_pretrained(model_name)
        self.vit = ViTModel.from_pretrained(model_name, config=config)

        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers

        for param in self.vit.parameters():
            param.requires_grad = False

        self.adapters = nn.ModuleList([
            AdapterModule(hidden_size, adapter_hidden_dim)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(hidden_size, num_labels)
        self._register_adapter_hooks()

    def _register_adapter_hooks(self):
        def make_adapter_hook(adapter_module):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                adapted = adapter_module(hidden_states)
                if isinstance(output, tuple):
                    return (adapted,) + output[1:]
                else:
                    return adapted
            return hook

        for i, layer in enumerate(self.vit.encoder.layer):
            layer.attention.output.register_forward_hook(
                make_adapter_hook(self.adapters[i])
            )

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {'loss': loss, 'logits': logits}

    def get_trainable_params_by_component(self):
        adapter_params = []
        classifier_params = []
        backbone_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'classifier' in name:
                classifier_params.append(param)
            elif 'adapters' in name:
                adapter_params.append(param)
            else:
                backbone_params.append(param)
        return {
            'adapter': adapter_params,
            'classifier': classifier_params,
            'backbone': backbone_params
        }


def get_vision_model(
    model_name: str,
    num_labels: int,
    adapter_hidden_dim: int = 64
) -> ViTWithAdapters:
    return ViTWithAdapters(
        model_name=model_name,
        num_labels=num_labels,
        adapter_hidden_dim=adapter_hidden_dim
    )
