#!/usr/bin/env python3
import argparse
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dp_peft.models import get_text_model, get_vision_model
from dp_peft.data import get_text_dataloaders, get_vision_dataloaders
from dp_peft.attacks import MembershipInferenceAttack
from dp_peft.utils import set_seed, save_results_to_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model', type=str, required=True, choices=['bert', 'distilbert', 'vit'])
    parser.add_argument('--dataset', type=str, required=True, choices=['agnews', 'sst2', 'cifar10'])
    parser.add_argument('--peft_method', type=str, default='adapter', choices=['adapter', 'lora'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.model in ['bert', 'distilbert']:
        model_name = 'bert-base-uncased' if args.model == 'bert' else 'distilbert-base-uncased'
        num_labels = 4 if args.dataset == 'agnews' else 2
        model = get_text_model(model_name=model_name, num_labels=num_labels, peft_method=args.peft_method)
        train_loader, test_loader = get_text_dataloaders(
            dataset_name=args.dataset, tokenizer_name=model_name, batch_size=args.batch_size
        )
    else:
        model = get_vision_model(model_name='google/vit-base-patch16-224', num_labels=10)
        train_loader, test_loader = get_vision_dataloaders(
            dataset_name=args.dataset, batch_size=args.batch_size
        )

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)

    mia = MembershipInferenceAttack(model, device=args.device)
    results = mia.run_attack(train_loader, test_loader)

    print(f"AUC: {results['threshold_attack']['auc']:.4f}")
    print(f"Advantage: {results['threshold_attack']['advantage']:.4f}")

    output_path = Path(args.checkpoint).parent / f"{Path(args.checkpoint).stem}_mia.json"
    save_results_to_json(results, str(output_path))
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    main()
