import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional, Dict, Any

import adapters
from adapters import AdapterConfig


class TextModelWithPEFT(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        peft_method: str = "adapter",
        peft_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if peft_config is None:
            peft_config = {}

        self.model_name = model_name
        self.num_labels = num_labels
        self.peft_method = peft_method

        config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size

        if peft_method == "lora":
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=peft_config.get("lora_r", 8),
                lora_alpha=peft_config.get("lora_alpha", 16),
                lora_dropout=peft_config.get("lora_dropout", 0.1),
                target_modules=["query", "value"],
            )
            self.backbone = get_peft_model(self.backbone, lora_config)

        elif peft_method == "adapter":
            adapters.init(self.backbone)
            adapter_config = AdapterConfig.load(
                "pfeiffer",
                reduction_factor=peft_config.get("adapter_reduction_factor", 16),
            )
            self.backbone.add_adapter("task_adapter", config=adapter_config)
            self.backbone.train_adapter("task_adapter")
            self.backbone.set_active_adapters("task_adapter")

        self.classifier = nn.Linear(hidden_size, num_labels)

        if peft_method == "adapter":
            for name, param in self.backbone.named_parameters():
                param.requires_grad = ("adapter" in name.lower())

        elif peft_method == "lora":
            for name, param in self.backbone.named_parameters():
                param.requires_grad = ("lora" in name.lower())

        else:
            for _, param in self.backbone.named_parameters():
                param.requires_grad = True

        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, **kwargs):
        backbone_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }

        if token_type_ids is not None:
            backbone_kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**backbone_kwargs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}

    def get_trainable_params_by_component(self):
        adapter_params = []
        classifier_params = []
        backbone_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "classifier" in name:
                classifier_params.append(param)
            elif "adapter" in name or "lora" in name.lower():
                adapter_params.append(param)
            else:
                backbone_params.append(param)

        return {
            "adapter": adapter_params,
            "classifier": classifier_params,
            "backbone": backbone_params,
        }


MODEL_NAME_MAP = {
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
}


def get_text_model(
    model_name: str,
    num_labels: int,
    peft_method: str = "adapter",
    peft_config: Optional[Dict[str, Any]] = None,
) -> TextModelWithPEFT:
    if peft_config is None:
        peft_config = {}

    hf_model_name = MODEL_NAME_MAP.get(model_name, model_name)

    return TextModelWithPEFT(
        model_name=hf_model_name,
        num_labels=num_labels,
        peft_method=peft_method,
        peft_config=peft_config,
    )
