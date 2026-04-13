#!/bin/bash
# Quick validation test for RunPod setup
# Run this FIRST to verify everything works before full experiments

set -e

echo "=========================================="
echo "DP-PEFT Quick Validation Test"
echo "=========================================="
echo ""

PROJECT_DIR="/workspace/dp-peft-research"
cd ${PROJECT_DIR}

# Activate environment
source venv/bin/activate

# Test 1: GPU Check
echo "Test 1: GPU Check"
echo "------------------"
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✓ GPU: {torch.cuda.get_device_name(0)}')"
echo ""

# Test 2: Package Imports
echo "Test 2: Package Imports"
echo "-----------------------"
python -c "
import torch
import transformers
import peft
import opacus
import datasets
print('✓ All packages imported successfully')
print(f'  PyTorch: {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  PEFT: {peft.__version__}')
print(f'  Opacus: {opacus.__version__}')
"
echo ""

# Test 3: Quick No-DP Experiment (2 epochs, should take ~5 minutes)
echo "Test 3: Quick No-DP Baseline"
echo "----------------------------"
python scripts/run_experiment.py \
    --model bert \
    --dataset agnews \
    --placement no_dp \
    --epsilon 8.0 \
    --epochs 2 \
    --batch_size 32 \
    --lr 1e-3 \
    --device cuda

echo ""
echo "✓ No-DP test passed!"
echo ""

# Test 4: Quick DP Experiment (2 epochs, should take ~5 minutes)
echo "Test 4: Quick DP Test (Adapter-Only)"
echo "------------------------------------"
python scripts/run_experiment.py \
    --model bert \
    --dataset agnews \
    --placement adapter_only \
    --epsilon 8.0 \
    --epochs 2 \
    --batch_size 32 \
    --lr 5e-4 \
    --device cuda

echo ""
echo "✓ DP test passed!"
echo ""

# Summary
echo "=========================================="
echo "ALL TESTS PASSED! ✓"
echo "=========================================="
echo ""
echo "Your RunPod environment is ready for full experiments!"
echo ""
echo "Next steps:"
echo "1. Review hyperparameters in configs/"
echo "2. Run full experiments: bash scripts/runpod_full_experiments.sh"
echo "3. Or use tmux: tmux new -s experiments"
echo "   then: bash scripts/runpod_full_experiments.sh"
echo ""
echo "Estimated time for full experiments: 5-7 days GPU time"
echo "=========================================="
