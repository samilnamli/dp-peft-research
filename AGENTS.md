# Project: Where to Privatize? — Differential Privacy Placement in PEFT

## Overview
This is a research project for CS 577 (Data Privacy) that will be extended into a full paper.
The goal is to systematically study **where** differential privacy (DP) should be applied in
parameter-efficient fine-tuning (PEFT) pipelines, rather than treating DP as a monolithic choice.

We compare multiple DP placement strategies under matched privacy budgets across text and vision tasks,
analyzing privacy-utility tradeoffs, optimization stability, training efficiency, and empirical
resistance to membership inference attacks.

---

## Your Mission
Implement this research project end-to-end: full codebase, experiments, evaluation, logging,
and result visualization. The code must be clean, modular, well-documented, and reproducible.
This will eventually become an academic paper, so code quality and reproducibility are critical.

---

## Tech Stack
- **Language:** Python 3.10+
- **Deep Learning:** PyTorch + HuggingFace Transformers
- **PEFT:** HuggingFace PEFT library (adapters, LoRA)
- **Differential Privacy:** Opacus (Facebook's DP library for PyTorch)
- **Datasets:** HuggingFace Datasets
- **Experiment Tracking:** Weights & Biases (wandb) — log everything
- **Visualization:** matplotlib, seaborn
- **Config Management:** Hydra or simple YAML configs
- **Package Management:** pip + requirements.txt

---

## Project Structure to Create

```
dp_peft/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── base.yaml               # shared defaults
│   ├── text_bert.yaml          # BERT on AG News
│   ├── text_distilbert.yaml    # DistilBERT on AG News
│   └── vision_vit.yaml         # ViT-B/16 on CIFAR-10
├── dp_peft/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── text_model.py       # BERT/DistilBERT with adapters/LoRA
│   │   └── vision_model.py     # ViT with adapter modules
│   ├── privacy/
│   │   ├── __init__.py
│   │   ├── placements.py       # DP placement strategies (the core)
│   │   └── accounting.py       # Privacy budget accounting helpers
│   ├── attacks/
│   │   ├── __init__.py
│   │   └── membership_inference.py  # MIA implementation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Main training loop
│   │   └── metrics.py          # All evaluation metrics
│   ├── data/
│   │   ├── __init__.py
│   │   └── loaders.py          # Dataset loading and preprocessing
│   └── utils/
│       ├── __init__.py
│       ├── logging.py          # wandb + local logging
│       └── reproducibility.py  # Seeds, determinism
├── scripts/
│   ├── run_experiment.py       # Single experiment runner
│   ├── run_all_placements.py   # Run all 6 placements in sequence
│   ├── run_privacy_curve.py    # Sweep epsilon values for privacy-utility curves
│   └── run_mia.py              # Run membership inference attacks
├── notebooks/
│   ├── results_analysis.ipynb  # Load results and generate paper figures
│   └── quick_test.ipynb        # Quick sanity checks
└── results/
    └── .gitkeep
```

---

## Core Component: DP Placements

This is the heart of the project. Implement all 6 placement strategies in `dp_peft/privacy/placements.py`:

### Placement 1: No DP (Upper Bound)
- Standard PEFT fine-tuning, no privacy constraints
- Baseline to measure utility loss from DP

### Placement 2: Full-Model DP
- Opacus DP-SGD applied to ALL trainable parameters
- Standard approach, expected worst utility

### Placement 3: Last-Layer DP
- DP-SGD applied ONLY to the final classification head
- All other layers frozen or updated without DP

### Placement 4: Adapter-Only DP
- Backbone frozen
- DP-SGD applied exclusively to adapter or LoRA parameters
- Expected: best utility since fewest params noised

### Placement 5: Head + Adapter DP
- DP-SGD applied to both adapter modules AND output head
- Middle ground between adapter-only and full DP

### Placement 6: Partial Backbone DP
- DP-SGD applied to adapters + top-k transformer blocks (k=2,4)
- Backbone partially unfrozen with DP

**Implementation note:** Use Opacus `GradSampleModule` and per-sample gradient clipping.
For selective placement, use Opacus's support for per-layer DP or manually zero gradients
for non-DP layers. Privacy accounting must be consistent across all placements at the same (ε, δ).

---

## Models

### Text Models (`dp_peft/models/text_model.py`)
- Load `bert-base-uncased` and `distilbert-base-uncased` from HuggingFace
- Add adapter modules (Houlsby-style bottleneck adapters) using PEFT library
- Add LoRA as an alternative PEFT method
- Classification head on top for AG News (4 classes), SST-2 (2 classes)
- Backbone frozen by default; adapters trainable
- Make it easy to specify which modules are trainable (needed for DP placement)

### Vision Models (`dp_peft/models/vision_model.py`)
- Load `google/vit-base-patch16-224` from HuggingFace
- Add adapter modules after attention layers in each transformer block
- Classification head for CIFAR-10 (10 classes) or CIFAR-100 (100 classes)
- Backbone frozen; adapters trainable

---

## Datasets (`dp_peft/data/loaders.py`)

### Text
- **AG News** (primary): 4-class news classification, 120k train / 7.6k test
- **SST-2** (secondary): binary sentiment, from GLUE benchmark
- Tokenize with appropriate tokenizer, max_length=128
- Return PyTorch DataLoaders

### Vision
- **CIFAR-10**: 50k train / 10k test, 10 classes
- Resize to 224x224 for ViT, standard normalization
- Data augmentation: random crop, horizontal flip (training only)

---

## Training Loop (`dp_peft/training/trainer.py`)

Implement a `Trainer` class that:
- Accepts model, optimizer, dataloader, dp_placement config
- Wraps model with Opacus `PrivacyEngine` based on placement
- Tracks per-epoch: loss, accuracy, gradient norm variance, loss oscillation
- Tracks wall-clock time per epoch and throughput (samples/sec)
- Tracks privacy budget consumed (epsilon spent) each epoch
- Supports early stopping based on target accuracy
- Logs everything to wandb AND saves to local JSON in `results/`
- Saves model checkpoints

**Privacy budget:** Use (ε=1, 8) as primary budgets, δ=1e-5 fixed.
Match noise multiplier across placements so they consume same ε by end of training.

---

## Evaluation Metrics (`dp_peft/training/metrics.py`)

### Utility
- Classification accuracy (primary)
- F1 score (macro)
- Epochs to reach 80% / 90% of no-DP baseline accuracy

### Stability
- Gradient norm variance per epoch
- Loss oscillation (std of loss over last 5 epochs)
- Convergence curve (loss vs epoch)

### Systems Efficiency
- Wall-clock time per epoch
- Throughput (samples/sec)
- Time-to-utility: wall-clock time to reach target accuracy at fixed ε

---

## Membership Inference Attack (`dp_peft/attacks/membership_inference.py`)

Implement a shadow model / likelihood-ratio membership inference attack:
- Train a shadow model on a subset of data
- Use loss values on train vs held-out test samples as attack signal
- Compute attack AUC (ROC curve)
- Compute attack advantage = max(TPR - FPR)
- Run this after each placement's training is complete
- Compare MIA AUC across placements to validate empirical privacy

Use the simple loss-based attack (threshold on per-sample loss) as baseline,
then implement the more powerful likelihood ratio attack.

---

## Experiment Scripts

### `scripts/run_experiment.py`
Single experiment runner. Args:
```
--model [bert|distilbert|vit]
--dataset [agnews|sst2|cifar10]
--placement [no_dp|full_dp|last_layer|adapter_only|head_adapter|partial_backbone]
--epsilon [1|8]
--delta 1e-5
--peft_method [adapter|lora]
--seed 42
--epochs 20
--batch_size 256
--lr 1e-3
--wandb_project dp_peft_research
```

### `scripts/run_all_placements.py`
Loops over all 6 placements for a given model/dataset/epsilon combo.
Saves comparison table to `results/comparison_{model}_{dataset}_eps{epsilon}.csv`

### `scripts/run_privacy_curve.py`
Sweeps epsilon in [0.5, 1, 2, 4, 8, inf] for each placement.
Generates data for privacy-utility trade-off curves.

### `scripts/run_mia.py`
Runs MIA on saved model checkpoints from all placements.

---

## Results & Visualization (`notebooks/results_analysis.ipynb`)

Generate all paper-ready figures:

1. **Privacy-Utility Curves**: Accuracy vs ε for all 6 placements on same plot
2. **Stability Plot**: Gradient norm variance per epoch, all placements
3. **Convergence Plot**: Loss curves per epoch, all placements  
4. **Systems Bar Chart**: Time-to-utility and throughput comparison
5. **MIA Bar Chart**: Attack AUC and advantage per placement
6. **Summary Table**: LaTeX-formatted table of all metrics

Use seaborn with a clean academic style. Save all figures as PDF and PNG.

---

## Reproducibility Requirements
- Set all random seeds (torch, numpy, random, CUDA) via `utils/reproducibility.py`
- Log seed, all hyperparameters, and environment info (GPU, CUDA version, library versions) to wandb
- Save full config YAML alongside each result
- All results must be reproducible with same seed

---

## README.md Must Include
- Project description and research questions
- Installation instructions
- How to run each experiment
- How to reproduce paper results
- Description of each DP placement
- Expected results summary

---

## Implementation Notes & Gotchas

1. **Opacus compatibility**: Not all HuggingFace models work out-of-the-box with Opacus.
   Use `opacus.validators.ModuleValidator.fix()` to fix incompatible layers (e.g., LayerNorm → GroupNorm).
   Document which layers were changed.

2. **Batch size with DP**: DP-SGD requires large logical batch sizes (use virtual batches / gradient accumulation).
   Physical batch size may need to be smaller. Implement gradient accumulation properly.

3. **Per-sample gradients**: Opacus computes per-sample gradients which is memory intensive.
   Use `batch_size=256` with gradient accumulation if needed.

4. **Fair comparison**: All placements must consume the same (ε, δ) by end of training.
   Calibrate noise multiplier for each placement separately using Opacus's `get_noise_multiplier`.

5. **Selective DP with Opacus**: To apply DP only to specific modules, wrap only those modules
   with Opacus, or use hooks to zero out gradients for non-DP parameters before the noise step.

6. **ViT adapters**: Insert adapters after the attention output projection in each transformer block.
   Keep adapter hidden dim small (64) relative to model dim (768).

---

## Deliverables Checklist
- [ ] Full working codebase with all 6 placements
- [ ] All experiments run on text (BERT + AG News) and vision (ViT + CIFAR-10)
- [ ] Privacy-utility curves at ε ∈ {1, 8}
- [ ] Stability and convergence analysis
- [ ] MIA evaluation for all placements
- [ ] Systems efficiency measurements
- [ ] Jupyter notebook with all paper figures
- [ ] Complete README with reproduction instructions
- [ ] wandb project with all runs logged

---

## Start Here
1. Create the project structure
2. Implement `requirements.txt` and verify all installs
3. Implement data loaders and verify datasets load correctly
4. Implement models with PEFT (no DP yet), verify training works
5. Implement DP placements one by one, starting with Full-Model DP as reference
6. Implement metrics and logging
7. Implement MIA
8. Run full experiment suite
9. Generate figures and analysis

Ask clarifying questions before making major architectural decisions.
Always run a quick sanity check (1-2 epochs, small dataset) before full runs.
