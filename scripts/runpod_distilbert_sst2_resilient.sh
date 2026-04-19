#!/bin/bash
# DistilBERT + SST-2 Resilient Experiments
# Runs all 6 DP placements x 2 epsilon values x 2 PEFT methods (adapter + LoRA)
# Saves intermediate results and resumes from interruption automatically.
# Results: results/distilbert_results/{adapter,lora}/
# Checkpoints: checkpoints/distilbert/{adapter,lora}/

set -e
set -o pipefail

echo "=========================================="
echo " DP-PEFT DistilBERT+SST-2 Experiments"
echo " (Resilient Mode — auto-resume on restart)"
echo "=========================================="
echo ""

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR="/workspace/dp-peft-research"
BASE_RESULTS="${PROJECT_DIR}/results/distilbert_results"
BASE_CHECKPOINTS="${PROJECT_DIR}/checkpoints/distilbert"
PROGRESS_FILE="${BASE_RESULTS}/progress.txt"

# ── Experiment parameters ──────────────────────────────────────────────────
MODEL="distilbert"
DATASET="sst2"
BATCH_SIZE=64
EPOCHS=20
LR=5e-4
DEVICE="cuda"

# ── Experiment matrix ──────────────────────────────────────────────────────
PEFT_METHODS="adapter lora"
DP_PLACEMENTS="adapter_only head_adapter last_layer full_dp partial_backbone"
EPSILONS="8.0 1.0"

# ── Setup ──────────────────────────────────────────────────────────────────
for peft in ${PEFT_METHODS}; do
    mkdir -p "${BASE_RESULTS}/${peft}"
    mkdir -p "${BASE_CHECKPOINTS}/${peft}"
done

cd ${PROJECT_DIR}
source venv/bin/activate

echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

# ── Progress tracker ───────────────────────────────────────────────────────
if [ ! -f "${PROGRESS_FILE}" ]; then
    echo "Initializing progress tracker..."
    cat > ${PROGRESS_FILE} << 'HEADER'
# DistilBERT SST-2 Experiment Progress
# Format: peft_method_placement,epsilon,status,timestamp
# Status: pending, running, completed, failed
HEADER

    for peft in ${PEFT_METHODS}; do
        # no_dp baseline (epsilon-independent, use 8.0 as nominal)
        echo "${peft}_no_dp,8.0,pending," >> ${PROGRESS_FILE}
        # DP placements at both epsilons
        for eps in ${EPSILONS}; do
            for placement in ${DP_PLACEMENTS}; do
                echo "${peft}_${placement},${eps},pending," >> ${PROGRESS_FILE}
            done
        done
    done
fi

echo "Progress file: ${PROGRESS_FILE}"
echo ""

# ── Helper functions ───────────────────────────────────────────────────────
is_completed() {
    grep -q "^${1}_${2},${3},completed," ${PROGRESS_FILE}
}

mark_running() {
    local ts=$(date +%Y%m%d_%H%M%S)
    sed -i "s/^${1}_${2},${3},pending,.*/${1}_${2},${3},running,${ts}/"    ${PROGRESS_FILE}
    sed -i "s/^${1}_${2},${3},failed,.*/${1}_${2},${3},running,${ts}/"    ${PROGRESS_FILE}
}

mark_completed() {
    local ts=$(date +%Y%m%d_%H%M%S)
    sed -i "s/^${1}_${2},${3},running,.*/${1}_${2},${3},completed,${ts}/" ${PROGRESS_FILE}
}

mark_failed() {
    local ts=$(date +%Y%m%d_%H%M%S)
    sed -i "s/^${1}_${2},${3},running,.*/${1}_${2},${3},failed,${ts}/"    ${PROGRESS_FILE}
}

display_progress() {
    echo "--- Progress Summary ---"
    local total=$(grep -c "^[^#]" ${PROGRESS_FILE} || echo 0)
    local done=$(grep -c ",completed," ${PROGRESS_FILE} || echo 0)
    local fail=$(grep -c ",failed,"    ${PROGRESS_FILE} || echo 0)
    local pend=$(grep -c ",pending,"   ${PROGRESS_FILE} || echo 0)
    echo "Total: ${total} | Completed: ${done} | Failed: ${fail} | Pending: ${pend}"
    echo "Progress: ${done}/${total} ($(( done * 100 / total ))%)"
    echo ""
}

run_mia_after_training() {
    local peft=$1
    local placement=$2
    local epsilon=$3
    local log_suffix=$4

    local run_name="${MODEL}_${DATASET}_${peft}_${placement}_eps${epsilon}"
    local checkpoint="${BASE_CHECKPOINTS}/${peft}/${run_name}.pt"
    local log_dir="${BASE_RESULTS}/${peft}"

    if [ -f "${checkpoint}" ]; then
        echo "  Running MIA: ${peft}/${placement} eps=${epsilon}"
        if python scripts/run_mia.py \
            --checkpoint "${checkpoint}" \
            --model ${MODEL} \
            --dataset ${DATASET} \
            --peft_method ${peft} \
            2>&1 | tee "${log_dir}/mia_${placement}_eps${epsilon}_${log_suffix}.log"; then
            echo "  [OK] MIA done: ${peft}/${placement} eps=${epsilon}"
        else
            echo "  [WARN] MIA failed: ${peft}/${placement} eps=${epsilon}"
        fi
    else
        echo "  [WARN] Checkpoint missing, skipping MIA: ${checkpoint}"
    fi
    echo ""
}

run_experiment() {
    local peft=$1
    local placement=$2
    local epsilon=$3
    local log_suffix=$4

    local results_dir="${BASE_RESULTS}/${peft}"
    local checkpoint_dir="${BASE_CHECKPOINTS}/${peft}"
    local log_file="${results_dir}/${placement}_eps${epsilon}_${log_suffix}.log"

    # Skip if already done
    if is_completed ${peft} ${placement} ${epsilon}; then
        echo ">> Skipping (done): ${peft} | ${placement} | eps=${epsilon}"
        return 0
    fi

    echo "=========================================="
    echo "START: ${peft} | ${placement} | eps=${epsilon}"
    echo "Time:  $(date)"
    echo "=========================================="

    mark_running ${peft} ${placement} ${epsilon}

    if python scripts/run_experiment.py \
        --model       ${MODEL} \
        --dataset     ${DATASET} \
        --placement   ${placement} \
        --epsilon     ${epsilon} \
        --epochs      ${EPOCHS} \
        --batch_size  ${BATCH_SIZE} \
        --lr          ${LR} \
        --peft_method ${peft} \
        --device      ${DEVICE} \
        --results_dir    "${results_dir}" \
        --checkpoint_dir "${checkpoint_dir}" \
        2>&1 | tee "${log_file}"; then

        mark_completed ${peft} ${placement} ${epsilon}
        echo "[OK] COMPLETED: ${peft} | ${placement} | eps=${epsilon} | $(date)"

        # Immediate MIA right after training
        run_mia_after_training ${peft} ${placement} ${epsilon} ${log_suffix}

    else
        mark_failed ${peft} ${placement} ${epsilon}
        echo "[FAIL] FAILED: ${peft} | ${placement} | eps=${epsilon}"
        echo "Log: ${log_file}"
        return 1
    fi
    echo ""
}

# ── Main experiment loop ───────────────────────────────────────────────────
START_TIME=$(date +%s)
LOG_SUFFIX="distilbert_$(date +%Y%m%d_%H%M%S)"

echo "Suite started: $(date)"
display_progress
echo ""

for PEFT in ${PEFT_METHODS}; do
    echo "+============================================+"
    echo "|  PEFT method: ${PEFT}"
    echo "+============================================+"
    echo ""

    # ── no_dp baseline (once per peft method, nominal eps=8.0) ──
    echo "--- Phase: no_dp baseline (${PEFT}) ---"
    run_experiment ${PEFT} no_dp 8.0 ${LOG_SUFFIX}
    display_progress

    # ── DP placements at eps=8.0 ──
    echo "--- Phase: eps=8.0 DP placements (${PEFT}) ---"
    for placement in ${DP_PLACEMENTS}; do
        run_experiment ${PEFT} ${placement} 8.0 ${LOG_SUFFIX}
        display_progress
    done

    # ── DP placements at eps=1.0 ──
    echo "--- Phase: eps=1.0 DP placements (${PEFT}) ---"
    for placement in ${DP_PLACEMENTS}; do
        run_experiment ${PEFT} ${placement} 1.0 ${LOG_SUFFIX}
        display_progress
    done

    echo "[OK] All ${PEFT} experiments complete!"
    echo ""
done

# ── Final summary ──────────────────────────────────────────────────────────
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo "=========================================="
echo " ALL DISTILBERT EXPERIMENTS COMPLETE"
echo "=========================================="
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Results:    ${BASE_RESULTS}/"
echo "Checkpoints:${BASE_CHECKPOINTS}/"
echo ""
display_progress

cat > "${BASE_RESULTS}/final_summary_${LOG_SUFFIX}.txt" << EOF
DistilBERT SST-2 DP-PEFT Experiments — Final Summary
=====================================================
Completed: $(date)
Duration: ${HOURS}h ${MINUTES}m

Configuration:
  Model:      ${MODEL}
  Dataset:    ${DATASET}
  PEFT:       ${PEFT_METHODS}
  Placements: no_dp ${DP_PLACEMENTS}
  Epsilons:   ${EPSILONS}
  Batch size: ${BATCH_SIZE}
  Epochs:     ${EPOCHS}
  LR:         ${LR}

GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)

Experiments: $(grep -c ",completed," ${PROGRESS_FILE}) / $(grep -c "^[^#]" ${PROGRESS_FILE}) completed

Results layout:
  ${BASE_RESULTS}/adapter/   — adapter PEFT results + MIA logs
  ${BASE_RESULTS}/lora/      — LoRA PEFT results + MIA logs
  ${BASE_CHECKPOINTS}/adapter/ — adapter checkpoints
  ${BASE_CHECKPOINTS}/lora/    — LoRA checkpoints

To download:
  tar -czf distilbert_results.tar.gz results/distilbert_results/ checkpoints/distilbert/

To resume if interrupted:
  bash scripts/runpod_distilbert_sst2_resilient.sh
  (completed experiments are skipped automatically)
=====================================================
EOF

echo "Summary saved: ${BASE_RESULTS}/final_summary_${LOG_SUFFIX}.txt"
echo ""
echo "To resume if interrupted, just run this script again."
