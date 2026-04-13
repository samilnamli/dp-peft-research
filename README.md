# Where to Privatize? Differential Privacy Placement in PEFT

A comprehensive research project investigating where differential privacy (DP) should be applied in parameter-efficient fine-tuning (PEFT) pipelines.

## Overview

This project systematically compares **6 DP placement strategies** under matched privacy budgets across text and vision tasks, analyzing:
- Privacy-utility tradeoffs
- Optimization stability
- Training efficiency
- Empirical resistance to membership inference attacks

## Research Questions

1. **Where should DP be applied in PEFT?** Does applying DP only to adapters preserve utility better than full-model DP?
2. **Privacy-Utility Tradeoffs**: How do different placements compare at matched privacy budgets (ε=1, 8)?
3. **Stability**: Which placements lead to more stable training dynamics?
4. **Empirical Privacy**: Do theoretical privacy guarantees translate to empirical resistance against membership inference attacks?

## DP Placement Strategies

### 1. No DP (Baseline)
Standard PEFT fine-tuning without privacy constraints. Upper bound on utility.

### 2. Full-Model DP
Opacus DP-SGD applied to ALL trainable parameters. Standard approach, expected worst utility.

### 3. Last-Layer DP
DP-SGD applied ONLY to the final classification head. All other layers frozen or updated without DP.

### 4. Adapter-Only DP ⭐
DP-SGD applied exclusively to adapter/LoRA parameters. **Expected best utility** since fewest params are noised.

### 5. Head + Adapter DP
DP-SGD applied to both adapter modules AND output head. Middle ground strategy.

### 6. Partial Backbone DP
DP-SGD applied to adapters + top-k transformer blocks (k=2,4). Partially unfrozen backbone with DP.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd dp_peft

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Run a Single Experiment

```bash
python scripts/run_experiment.py \
    --model bert \
    --dataset agnews \
    --placement adapter_only \
    --epsilon 1.0 \
    --epochs 20 \
    --batch_size 256
```

### Run All Placements

```bash
python scripts/run_all_placements.py \
    --model bert \
    --dataset agnews \
    --epsilon 1.0 \
    --epochs 20
```

### Generate Privacy-Utility Curves

```bash
python scripts/run_privacy_curve.py \
    --model bert \
    --dataset agnews \
    --placement adapter_only \
    --epochs 20
```

### Run Membership Inference Attack

```bash
python scripts/run_mia.py \
    --checkpoint checkpoints/bert_agnews_adapter_only_eps1.0.pt \
    --model bert \
    --dataset agnews
```

## Experiment Configuration

### Models
- **Text**: BERT-base, DistilBERT-base with adapters or LoRA
- **Vision**: ViT-B/16 with custom adapter modules

### Datasets
- **Text**: AG News (4-class, 120k train), SST-2 (binary sentiment)
- **Vision**: CIFAR-10 (10-class, 50k train)

### Privacy Budgets
- Primary: ε ∈ {1, 8}, δ = 1e-5
- Privacy curves: ε ∈ {0.5, 1, 2, 4, 8, ∞}

## Reproducing Paper Results

### 1. Text Experiments (BERT + AG News)

```bash
# Run all placements at ε=1
python scripts/run_all_placements.py --model bert --dataset agnews --epsilon 1.0 --epochs 20

# Run all placements at ε=8
python scripts/run_all_placements.py --model bert --dataset agnews --epsilon 8.0 --epochs 20

# Generate privacy curves for each placement
for placement in no_dp full_dp last_layer adapter_only head_adapter partial_backbone; do
    python scripts/run_privacy_curve.py --model bert --dataset agnews --placement $placement --epochs 20
done
```

### 2. Vision Experiments (ViT + CIFAR-10)

```bash
# Run all placements at ε=1
python scripts/run_all_placements.py --model vit --dataset cifar10 --epsilon 1.0 --epochs 30

# Run all placements at ε=8
python scripts/run_all_placements.py --model vit --dataset cifar10 --epsilon 8.0 --epochs 30
```

### 3. Membership Inference Attacks

```bash
# Run MIA on all saved checkpoints
for checkpoint in checkpoints/*.pt; do
    python scripts/run_mia.py --checkpoint $checkpoint --model bert --dataset agnews
done
```

### 4. Generate Figures

Open and run `notebooks/results_analysis.ipynb` to generate all paper figures.

## Project Structure

```
dp_peft/
├── configs/              # YAML configuration files
├── dp_peft/             # Main package
│   ├── models/          # Text and vision models with PEFT
│   ├── privacy/         # DP placement strategies
│   ├── attacks/         # Membership inference attacks
│   ├── training/        # Training loop and metrics
│   ├── data/            # Dataset loaders
│   └── utils/           # Logging and reproducibility
├── scripts/             # Experiment runners
├── notebooks/           # Analysis and visualization
└── results/             # Experiment results (JSON, CSV)
```

## Expected Results

### Privacy-Utility Tradeoffs (ε=1)
- **No DP**: ~90% accuracy (baseline)
- **Full DP**: ~70% accuracy (worst)
- **Adapter-Only DP**: ~85% accuracy (best among DP methods)
- **Head+Adapter DP**: ~82% accuracy
- **Last-Layer DP**: ~78% accuracy
- **Partial Backbone DP**: ~80% accuracy

### Key Findings
1. **Adapter-Only DP** achieves the best privacy-utility tradeoff
2. Applying DP to fewer parameters (adapters) preserves more utility
3. All placements provide similar empirical privacy protection (MIA resistance)
4. Training stability varies: adapter-only shows lowest gradient variance

## Metrics Tracked

### Utility
- Classification accuracy
- F1 score (macro)
- Epochs to reach 80%/90% of baseline

### Stability
- Gradient norm variance
- Loss oscillation
- Convergence curves

### Efficiency
- Wall-clock time per epoch
- Throughput (samples/sec)
- Time-to-utility

### Privacy
- Theoretical ε (via Opacus accounting)
- MIA attack AUC
- MIA attack advantage

## Implementation Notes

### Opacus Compatibility
Models are automatically fixed for Opacus compatibility using `ModuleValidator.fix()`. This may replace LayerNorm with GroupNorm.

### Batch Size and Memory
- Logical batch size: 256 (for privacy accounting)
- Physical batch size: 32 (adjustable based on GPU memory)
- Gradient accumulation handles the difference

### Reproducibility
All experiments use fixed seeds (default: 42) for reproducibility. Environment info is logged to W&B.

## Citation

```bibtex
@article{dpplacement2024,
  title={Where to Privatize? Differential Privacy Placement in Parameter-Efficient Fine-Tuning},
  author={Your Name},
  journal={CS 577 Data Privacy},
  year={2024}
}
```

## License

MIT License

## Acknowledgments

- Built with PyTorch, HuggingFace Transformers, PEFT, and Opacus
- Experiment tracking with Weights & Biases
- Inspired by recent work on privacy-preserving fine-tuning
