import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional, Dict, Any


class TextModelWithPEFT(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        peft_method: str = 'adapter',
        peft_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.peft_method = peft_method
        
        config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        
        hidden_size = config.hidden_size
        
        if peft_method == 'lora':
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=peft_config.get('lora_r', 8),
                lora_alpha=peft_config.get('lora_alpha', 16),
                lora_dropout=peft_config.get('lora_dropout', 0.1),
                target_modules=["query", "value"]
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
        elif peft_method == 'adapter':
            from transformers.adapters import AdapterConfig
            adapter_config = AdapterConfig.load(
                "pfeiffer",
                reduction_factor=peft_config.get('adapter_reduction_factor', 16)
            )
            self.backbone.add_adapter("task_adapter", config=adapter_config)
            self.backbone.train_adapter("task_adapter")
            self.backbone.set_active_adapters("task_adapter")
        
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        for param in self.backbone.parameters():
            if peft_method == 'adapter':
                if 'adapter' not in param.name if hasattr(param, 'name') else True:
                    param.requires_grad = False
            elif peft_method == 'lora':
                if 'lora' not in str(param):
                    param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
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
            elif 'adapter' in name or 'lora' in name:
                adapter_params.append(param)
            else:
                backbone_params.append(param)
        
        return {
            'adapter': adapter_params,
            'classifier': classifier_params,
            'backbone': backbone_params
        }


MODEL_NAME_MAP = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
}


def get_text_model(
    model_name: str,
    num_labels: int,
    peft_method: str = 'adapter',
    peft_config: Optional[Dict[str, Any]] = None
) -> TextModelWithPEFT:
    if peft_config is None:
        peft_config = {}
    
    hf_model_name = MODEL_NAME_MAP.get(model_name, model_name)
    
    return TextModelWithPEFT(
        model_name=hf_model_name,
        num_labels=num_labels,
        peft_method=peft_method,
        peft_config=peft_config
    )
