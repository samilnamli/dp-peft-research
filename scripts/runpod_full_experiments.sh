#!/bin/bash
# Complete experiment suite for RunPod GPU servers
# Optimized for L40S, A100, or similar high-end GPUs

set -e  # Exit on error

echo "=========================================="
echo "DP-PEFT Full Experiment Suite"
echo "=========================================="
echo ""

# Configuration
PROJECT_DIR="/workspace/dp-peft-research"
RESULTS_DIR="${PROJECT_DIR}/results"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"

# Experiment parameters
MODEL="bert"
DATASET="agnews"
BATCH_SIZE=64
EPOCHS=20
LR=5e-4
DEVICE="cuda"

# Create directories
mkdir -p ${RESULTS_DIR}
mkdir -p ${CHECKPOINT_DIR}

# Activate environment
cd ${PROJECT_DIR}
source venv/bin/activate

# Check GPU
echo "Checking GPU..."
nvidia-smi
echo ""

# Verify CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Function to run experiment
run_experiment() {
    local placement=$1
    local epsilon=$2
    local log_suffix=$3
    
    echo "=========================================="
    echo "Running: ${placement} | ε=${epsilon}"
    echo "=========================================="
    
    python scripts/run_experiment.py \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --placement ${placement} \
        --epsilon ${epsilon} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --device ${DEVICE} \
        2>&1 | tee ${RESULTS_DIR}/${placement}_eps${epsilon}_${log_suffix}.log
    
    echo ""
    echo "✓ Completed: ${placement} | ε=${epsilon}"
    echo ""
}

# Start timestamp
START_TIME=$(date +%s)
echo "Experiment started at: $(date)"
echo ""

# ============================================
# PHASE 1: Epsilon = 8.0 (All Placements)
# ============================================
echo "╔════════════════════════════════════════╗"
echo "║  PHASE 1: ε = 8.0 (All Placements)    ║"
echo "╚════════════════════════════════════════╝"
echo ""

EPSILON=8.0
LOG_SUFFIX="full_$(date +%Y%m%d_%H%M%S)"

# Run all 6 placements
run_experiment "no_dp" ${EPSILON} ${LOG_SUFFIX}
run_experiment "adapter_only" ${EPSILON} ${LOG_SUFFIX}
run_experiment "head_adapter" ${EPSILON} ${LOG_SUFFIX}
run_experiment "last_layer" ${EPSILON} ${LOG_SUFFIX}
run_experiment "full_dp" ${EPSILON} ${LOG_SUFFIX}
run_experiment "partial_backbone" ${EPSILON} ${LOG_SUFFIX}

echo "✓ Phase 1 complete!"
echo ""

# ============================================
# PHASE 2: Epsilon = 1.0 (All Placements)
# ============================================
echo "╔════════════════════════════════════════╗"
echo "║  PHASE 2: ε = 1.0 (All Placements)    ║"
echo "╚════════════════════════════════════════╝"
echo ""

EPSILON=1.0

run_experiment "no_dp" ${EPSILON} ${LOG_SUFFIX}
run_experiment "adapter_only" ${EPSILON} ${LOG_SUFFIX}
run_experiment "head_adapter" ${EPSILON} ${LOG_SUFFIX}
run_experiment "last_layer" ${EPSILON} ${LOG_SUFFIX}
run_experiment "full_dp" ${EPSILON} ${LOG_SUFFIX}
run_experiment "partial_backbone" ${EPSILON} ${LOG_SUFFIX}

echo "✓ Phase 2 complete!"
echo ""

# ============================================
# PHASE 3: Privacy-Utility Curves
# ============================================
echo "╔════════════════════════════════════════╗"
echo "║  PHASE 3: Privacy-Utility Curves      ║"
echo "╚════════════════════════════════════════╝"
echo ""

# Run privacy curves for key placements
for placement in adapter_only full_dp; do
    echo "Running privacy curve for ${placement}..."
    python scripts/run_privacy_curve.py \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --placement ${placement} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        2>&1 | tee ${RESULTS_DIR}/privacy_curve_${placement}_${LOG_SUFFIX}.log
    echo ""
done

echo "✓ Phase 3 complete!"
echo ""

# ============================================
# PHASE 4: Membership Inference Attacks
# ============================================
echo "╔════════════════════════════════════════╗"
echo "║  PHASE 4: Membership Inference Attacks ║"
echo "╚════════════════════════════════════════╝"
echo ""

# Run MIA on all checkpoints
for checkpoint in ${CHECKPOINT_DIR}/*.pt; do
    if [ -f "$checkpoint" ]; then
        echo "Running MIA on $(basename $checkpoint)..."
        python scripts/run_mia.py \
            --checkpoint ${checkpoint} \
            --model ${MODEL} \
            --dataset ${DATASET} \
            2>&1 | tee ${RESULTS_DIR}/mia_$(basename ${checkpoint} .pt)_${LOG_SUFFIX}.log
        echo ""
    fi
done

echo "✓ Phase 4 complete!"
echo ""

# ============================================
# Summary
# ============================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=========================================="
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Results saved in: ${RESULTS_DIR}"
echo "Checkpoints saved in: ${CHECKPOINT_DIR}"
echo ""
echo "Next steps:"
echo "1. Download results: scp -r runpod:${RESULTS_DIR} ./local_results"
echo "2. Run analysis notebook: jupyter notebook notebooks/results_analysis.ipynb"
echo "3. Generate paper figures"
echo ""
echo "Experiment completed at: $(date)"
echo "=========================================="

# Create summary file
cat > ${RESULTS_DIR}/experiment_summary_${LOG_SUFFIX}.txt << EOF
DP-PEFT Experiment Summary
==========================

Date: $(date)
Duration: ${HOURS}h ${MINUTES}m

Configuration:
- Model: ${MODEL}
- Dataset: ${DATASET}
- Batch Size: ${BATCH_SIZE}
- Epochs: ${EPOCHS}
- Learning Rate: ${LR}

Phases Completed:
✓ Phase 1: ε = 8.0 (6 placements)
✓ Phase 2: ε = 1.0 (6 placements)
✓ Phase 3: Privacy-Utility Curves
✓ Phase 4: Membership Inference Attacks

Results Location: ${RESULTS_DIR}
Checkpoints Location: ${CHECKPOINT_DIR}

GPU Info:
$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
EOF

echo "Summary saved to: ${RESULTS_DIR}/experiment_summary_${LOG_SUFFIX}.txt"
