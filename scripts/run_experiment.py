#!/usr/bin/env python3
import argparse
import torch
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from dp_peft.models import get_text_model, get_vision_model
from dp_peft.data import get_text_dataloaders, get_vision_dataloaders
from dp_peft.privacy import get_placement_strategy
from dp_peft.training import DPPEFTTrainer
from dp_peft.utils import set_seed, get_environment_info, setup_logging, save_results_to_json


def parse_args():
    parser = argparse.ArgumentParser(description='Run DP-PEFT experiment')
    parser.add_argument('--model', type=str, required=True, choices=['bert', 'distilbert', 'vit'])
    parser.add_argument('--dataset', type=str, required=True, choices=['agnews', 'sst2', 'cifar10'])
    parser.add_argument('--placement', type=str, required=True,
                        choices=['no_dp', 'full_dp', 'last_layer', 'adapter_only', 'head_adapter', 'partial_backbone'])
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--peft_method', type=str, default='adapter', choices=['adapter', 'lora'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wandb_project', type=str, default='dp_peft_research')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    run_name = f"{args.model}_{args.dataset}_{args.peft_method}_{args.placement}_eps{args.epsilon}"
    
    config = {
        'model': args.model,
        'dataset': args.dataset,
        'placement': args.placement,
        'epsilon': args.epsilon,
        'delta': args.delta,
        'peft_method': args.peft_method,
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'device': args.device
    }
    
    config.update(get_environment_info())
    
    setup_logging(
        project_name=args.wandb_project,
        run_name=run_name,
        config=config
    )
    
    print(f"Running experiment: {run_name}")
    print(f"Configuration: {config}")
    
    if args.model in ['bert', 'distilbert']:
        model_name = 'bert-base-uncased' if args.model == 'bert' else 'distilbert-base-uncased'
        num_labels = 4 if args.dataset == 'agnews' else 2
        
        model = get_text_model(
            model_name=model_name,
            num_labels=num_labels,
            peft_method=args.peft_method,
            peft_config={'adapter_reduction_factor': 16, 'lora_r': 8}
        )
        
        train_loader, test_loader = get_text_dataloaders(
            dataset_name=args.dataset,
            tokenizer_name=model_name,
            batch_size=args.batch_size,
            max_length=128
        )
    else:
        model = get_vision_model(
            model_name='google/vit-base-patch16-224',
            num_labels=10,
            adapter_hidden_dim=64
        )
        
        train_loader, test_loader = get_vision_dataloaders(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            image_size=224
        )
    
    dp_placement = get_placement_strategy(
        model=model,
        strategy_name=args.placement,
        max_grad_norm=1.0
    )
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    
    trainer = DPPEFTTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        dp_placement=dp_placement,
        device=args.device,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir
    )
    
    results = trainer.train()
    
    results_path = Path(args.results_dir) / f"{run_name}.json"
    save_results_to_json(results, str(results_path))
    
    checkpoint_path = Path(args.checkpoint_dir) / f"{run_name}.pt"
    trainer.save_checkpoint(str(checkpoint_path))
    
    print(f"\nExperiment completed!")
    print(f"Final accuracy: {results['final_accuracy']:.4f}")
    print(f"Final epsilon: {results['final_epsilon']:.2f}")
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
