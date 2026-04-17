#!/bin/bash
# Resilient Text Experiments with Intermediate Saving
# Runs text experiments only (BERT + AG News) with MIA
# Saves and reports after EACH placement to handle GPU interruptions

set -e  # Exit on error

echo "=========================================="
echo "DP-PEFT Text Experiments (Resilient Mode)"
echo "=========================================="
echo ""

# Configuration
PROJECT_DIR="/workspace/dp-peft-research"
RESULTS_DIR="${PROJECT_DIR}/results"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"
PROGRESS_FILE="${RESULTS_DIR}/progress.txt"

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

# Initialize or load progress
if [ ! -f "${PROGRESS_FILE}" ]; then
    echo "Initializing progress tracker..."
    cat > ${PROGRESS_FILE} << EOF
# DP-PEFT Experiment Progress
# Format: placement,epsilon,status,timestamp
# Status: pending, running, completed, failed
EOF
    
    # Initialize all experiments as pending
    for epsilon in 8.0 1.0; do
        for placement in no_dp adapter_only head_adapter last_layer full_dp partial_backbone; do
            echo "${placement},${epsilon},pending," >> ${PROGRESS_FILE}
        done
    done
fi

echo "Progress file: ${PROGRESS_FILE}"
echo ""

# Function to check if experiment is completed
is_completed() {
    local placement=$1
    local epsilon=$2
    grep -q "^${placement},${epsilon},completed," ${PROGRESS_FILE}
    return $?
}

# Function to mark experiment as running
mark_running() {
    local placement=$1
    local epsilon=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)
    sed -i "s/^${placement},${epsilon},pending,.*/${placement},${epsilon},running,${timestamp}/" ${PROGRESS_FILE}
}

# Function to mark experiment as completed
mark_completed() {
    local placement=$1
    local epsilon=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)
    sed -i "s/^${placement},${epsilon},running,.*/${placement},${epsilon},completed,${timestamp}/" ${PROGRESS_FILE}
}

# Function to mark experiment as failed
mark_failed() {
    local placement=$1
    local epsilon=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)
    sed -i "s/^${placement},${epsilon},running,.*/${placement},${epsilon},failed,${timestamp}/" ${PROGRESS_FILE}
}

# Function to run single experiment with error handling
run_experiment() {
    local placement=$1
    local epsilon=$2
    local log_suffix=$3
    
    # Check if already completed
    if is_completed ${placement} ${epsilon}; then
        echo "⏭️  Skipping ${placement} | ε=${epsilon} (already completed)"
        echo ""
        return 0
    fi
    
    echo "=========================================="
    echo "Running: ${placement} | ε=${epsilon}"
    echo "Started: $(date)"
    echo "=========================================="
    
    # Mark as running
    mark_running ${placement} ${epsilon}
    
    # Run experiment with error handling
    if python scripts/run_experiment.py \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --placement ${placement} \
        --epsilon ${epsilon} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --device ${DEVICE} \
        2>&1 | tee ${RESULTS_DIR}/${placement}_eps${epsilon}_${log_suffix}.log; then
        
        # Success
        mark_completed ${placement} ${epsilon}
        echo ""
        echo "✅ COMPLETED: ${placement} | ε=${epsilon}"
        echo "Finished: $(date)"
        
        # Generate intermediate report
        generate_report ${placement} ${epsilon}
        
        # Run MIA immediately after training
        run_mia_for_placement ${placement} ${epsilon} ${log_suffix}
        
    else
        # Failure
        mark_failed ${placement} ${epsilon}
        echo ""
        echo "❌ FAILED: ${placement} | ε=${epsilon}"
        echo "Check log: ${RESULTS_DIR}/${placement}_eps${epsilon}_${log_suffix}.log"
        echo ""
        
        # Continue with next experiment instead of exiting
        return 1
    fi
    
    echo ""
}

# Function to run MIA for a specific placement
run_mia_for_placement() {
    local placement=$1
    local epsilon=$2
    local log_suffix=$3
    
    local checkpoint="${CHECKPOINT_DIR}/${MODEL}_${DATASET}_${placement}_eps${epsilon}.pt"
    
    if [ -f "${checkpoint}" ]; then
        echo "=========================================="
        echo "Running MIA: ${placement} | ε=${epsilon}"
        echo "=========================================="
        
        if python scripts/run_mia.py \
            --checkpoint ${checkpoint} \
            --model ${MODEL} \
            --dataset ${DATASET} \
            2>&1 | tee ${RESULTS_DIR}/mia_${placement}_eps${epsilon}_${log_suffix}.log; then
            
            echo "✅ MIA completed for ${placement} | ε=${epsilon}"
        else
            echo "⚠️  MIA failed for ${placement} | ε=${epsilon}"
        fi
        echo ""
    else
        echo "⚠️  Checkpoint not found: ${checkpoint}"
        echo "Skipping MIA for ${placement} | ε=${epsilon}"
        echo ""
    fi
}

# Function to generate intermediate report
generate_report() {
    local placement=$1
    local epsilon=$2
    local report_file="${RESULTS_DIR}/report_${placement}_eps${epsilon}_$(date +%Y%m%d_%H%M%S).txt"
    
    echo "Generating report: ${report_file}"
    
    cat > ${report_file} << EOF
========================================
DP-PEFT Experiment Report
========================================

Placement: ${placement}
Epsilon: ${epsilon}
Completed: $(date)

Configuration:
- Model: ${MODEL}
- Dataset: ${DATASET}
- Batch Size: ${BATCH_SIZE}
- Epochs: ${EPOCHS}
- Learning Rate: ${LR}

GPU Info:
$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader)

Results File:
${RESULTS_DIR}/${MODEL}_${DATASET}_${placement}_eps${epsilon}.json

Checkpoint:
${CHECKPOINT_DIR}/${MODEL}_${DATASET}_${placement}_eps${epsilon}.pt

Overall Progress:
$(grep -c "completed" ${PROGRESS_FILE}) / $(grep -c "," ${PROGRESS_FILE}) experiments completed

========================================
EOF
    
    # Display summary
    if [ -f "${RESULTS_DIR}/${MODEL}_${DATASET}_${placement}_eps${epsilon}.json" ]; then
        echo ""
        echo "📊 Results Summary:"
        python -c "
import json
try:
    with open('${RESULTS_DIR}/${MODEL}_${DATASET}_${placement}_eps${epsilon}.json', 'r') as f:
        results = json.load(f)
    print(f\"  Final Test Accuracy: {results.get('final_test_accuracy', 'N/A'):.4f}\")
    print(f\"  Final Test Loss: {results.get('final_test_loss', 'N/A'):.4f}\")
    print(f\"  Final Epsilon: {results.get('final_epsilon', 'N/A'):.4f}\")
    print(f\"  Training Time: {results.get('total_training_time', 'N/A'):.2f}s\")
except Exception as e:
    print(f\"  Error reading results: {e}\")
"
    fi
    
    echo ""
    echo "Report saved: ${report_file}"
    echo ""
}

# Function to display overall progress
display_progress() {
    echo "=========================================="
    echo "Overall Progress Summary"
    echo "=========================================="
    echo ""
    
    local total=$(grep -c "^[^#]" ${PROGRESS_FILE})
    local completed=$(grep -c "completed" ${PROGRESS_FILE})
    local running=$(grep -c "running" ${PROGRESS_FILE})
    local pending=$(grep -c "pending" ${PROGRESS_FILE})
    local failed=$(grep -c "failed" ${PROGRESS_FILE})
    
    echo "Total Experiments: ${total}"
    echo "✅ Completed: ${completed}"
    echo "🔄 Running: ${running}"
    echo "⏳ Pending: ${pending}"
    echo "❌ Failed: ${failed}"
    echo ""
    
    echo "Progress: ${completed}/${total} ($(( completed * 100 / total ))%)"
    echo ""
    
    echo "Detailed Status:"
    echo "----------------"
    grep "^[^#]" ${PROGRESS_FILE} | while IFS=',' read -r placement epsilon status timestamp; do
        case ${status} in
            completed) icon="✅" ;;
            running)   icon="🔄" ;;
            pending)   icon="⏳" ;;
            failed)    icon="❌" ;;
            *)         icon="❓" ;;
        esac
        printf "%-20s ε=%-4s %s %s\n" "${placement}" "${epsilon}" "${icon}" "${status}"
    done
    echo ""
}

# Start timestamp
START_TIME=$(date +%s)
LOG_SUFFIX="resilient_$(date +%Y%m%d_%H%M%S)"

echo "Experiment suite started at: $(date)"
echo ""

# Display initial progress
display_progress

# ============================================
# PHASE 1: Epsilon = 8.0 (All Placements)
# ============================================
echo "╔════════════════════════════════════════╗"
echo "║  PHASE 1: ε = 8.0 (All Placements)    ║"
echo "╚════════════════════════════════════════╝"
echo ""

EPSILON=8.0

for placement in no_dp adapter_only head_adapter last_layer full_dp partial_backbone; do
    run_experiment ${placement} ${EPSILON} ${LOG_SUFFIX}
    display_progress
done

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

for placement in no_dp adapter_only head_adapter last_layer full_dp partial_backbone; do
    run_experiment ${placement} ${EPSILON} ${LOG_SUFFIX}
    display_progress
done

echo "✓ Phase 2 complete!"
echo ""

# ============================================
# Final Summary
# ============================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo "=========================================="
echo "ALL TEXT EXPERIMENTS COMPLETE!"
echo "=========================================="
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Results saved in: ${RESULTS_DIR}"
echo "Checkpoints saved in: ${CHECKPOINT_DIR}"
echo ""

# Final progress display
display_progress

# Create final summary
cat > ${RESULTS_DIR}/final_summary_${LOG_SUFFIX}.txt << EOF
DP-PEFT Text Experiments - Final Summary
=========================================

Completed: $(date)
Duration: ${HOURS}h ${MINUTES}m

Configuration:
- Model: ${MODEL}
- Dataset: ${DATASET}
- Batch Size: ${BATCH_SIZE}
- Epochs: ${EPOCHS}
- Learning Rate: ${LR}

Experiments Completed:
$(grep "completed" ${PROGRESS_FILE} | wc -l) / $(grep -c "^[^#]" ${PROGRESS_FILE})

Results Location: ${RESULTS_DIR}
Checkpoints Location: ${CHECKPOINT_DIR}
Progress File: ${PROGRESS_FILE}

GPU Info:
$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)

Next Steps:
1. Download results: tar -czf results_text.tar.gz results/ checkpoints/
2. Transfer to local: scp runpod:/workspace/dp-peft-research/results_text.tar.gz ./
3. Run analysis: jupyter notebook notebooks/results_analysis.ipynb

=========================================
EOF

echo "Final summary saved to: ${RESULTS_DIR}/final_summary_${LOG_SUFFIX}.txt"
echo ""
echo "To resume if interrupted, just run this script again!"
echo "Completed experiments will be skipped automatically."
echo ""
