#!/bin/bash
# DistilBERT + SST-2 — Epsilon Sweep (ε ∈ {0.5, 2.0, 4.0})
# Adds 3 extra epsilon points to the existing distilbert_results/adapter/ folder
# so the notebook can draw full privacy-utility curves.
#
# Only adapter PEFT (LoRA already has ε={1,8}; adapter now gets {0.5,1,2,4,8,∞}).
# no_dp is ε-independent — already exists, not re-run.
# Total new runs: 5 placements × 3 epsilons = 15 runs + 15 MIA.

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR="/workspace/dp-peft-research"
BASE_RESULTS="${PROJECT_DIR}/results/distilbert_results"
RESULTS_DIR="${BASE_RESULTS}/adapter"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/distilbert/adapter"
PROGRESS_FILE="${BASE_RESULTS}/progress_eps_sweep.txt"

# ── Experiment parameters ──────────────────────────────────────────────────
MODEL="distilbert"
DATASET="sst2"
PEFT="adapter"
BATCH_SIZE=64
EPOCHS=20
LR=5e-4
DEVICE="cuda"

# ── New epsilon values only ────────────────────────────────────────────────
NEW_EPSILONS="0.5 2.0 4.0"
DP_PLACEMENTS="adapter_only head_adapter last_layer full_dp partial_backbone"

# ── Setup ──────────────────────────────────────────────────────────────────
mkdir -p "${RESULTS_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
cd ${PROJECT_DIR}
source venv/bin/activate

echo "=========================================="
echo " DistilBERT Epsilon Sweep: ε ∈ {0.5, 2.0, 4.0}"
echo " PEFT: adapter | Output: distilbert_results/adapter/"
echo "=========================================="
echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}')"
echo ""

# ── Progress tracker ───────────────────────────────────────────────────────
if [ ! -f "${PROGRESS_FILE}" ]; then
    echo "# Epsilon sweep progress — DistilBERT adapter SST-2" >  ${PROGRESS_FILE}
    echo "# Format: placement,epsilon,status,timestamp"         >> ${PROGRESS_FILE}
    for eps in ${NEW_EPSILONS}; do
        for placement in ${DP_PLACEMENTS}; do
            echo "${placement},${eps},pending," >> ${PROGRESS_FILE}
        done
    done
    echo "Progress file created: ${PROGRESS_FILE}"
fi

# ── Helper functions ───────────────────────────────────────────────────────
is_completed() {
    grep -q "^${1},${2},completed," ${PROGRESS_FILE}
}

mark_running() {
    local ts=$(date +%Y%m%d_%H%M%S)
    sed -i "s|^${1},${2},pending,.*|${1},${2},running,${ts}|"  ${PROGRESS_FILE}
    sed -i "s|^${1},${2},failed,.*|${1},${2},running,${ts}|"   ${PROGRESS_FILE}
}

mark_completed() {
    local ts=$(date +%Y%m%d_%H%M%S)
    sed -i "s|^${1},${2},running,.*|${1},${2},completed,${ts}|" ${PROGRESS_FILE}
}

mark_failed() {
    local ts=$(date +%Y%m%d_%H%M%S)
    sed -i "s|^${1},${2},running,.*|${1},${2},failed,${ts}|"    ${PROGRESS_FILE}
}

display_progress() {
    local total=$(grep -c "^[^#]" ${PROGRESS_FILE} || echo 0)
    local done=$(grep -c  ",completed," ${PROGRESS_FILE} || echo 0)
    local fail=$(grep -c  ",failed,"    ${PROGRESS_FILE} || echo 0)
    echo "--- Progress: ${done}/${total} done | ${fail} failed ---"
}

run_mia() {
    local placement=$1
    local epsilon=$2
    local log_suffix=$3

    local run_name="${MODEL}_${DATASET}_${PEFT}_${placement}_eps${epsilon}"
    local checkpoint="${CHECKPOINT_DIR}/${run_name}.pt"

    if [ -f "${checkpoint}" ]; then
        echo "  Running MIA: ${placement} eps=${epsilon}"
        python scripts/run_mia.py \
            --checkpoint "${checkpoint}" \
            --model      ${MODEL} \
            --dataset    ${DATASET} \
            --peft_method ${PEFT} \
            2>&1 | tee "${RESULTS_DIR}/mia_${placement}_eps${epsilon}_${log_suffix}.log" \
            && echo "  [OK] MIA done" \
            || echo "  [WARN] MIA failed (non-fatal)"
    else
        echo "  [WARN] Checkpoint not found, skipping MIA: ${checkpoint}"
    fi
    echo ""
}

run_experiment() {
    local placement=$1
    local epsilon=$2
    local log_suffix=$3

    if is_completed ${placement} ${epsilon}; then
        echo ">> Skip (done): ${placement} | eps=${epsilon}"
        return 0
    fi

    echo "=========================================="
    echo "START: ${placement} | eps=${epsilon}"
    echo "Time:  $(date)"
    echo "=========================================="

    mark_running ${placement} ${epsilon}

    if python scripts/run_experiment.py \
        --model          ${MODEL} \
        --dataset        ${DATASET} \
        --placement      ${placement} \
        --epsilon        ${epsilon} \
        --epochs         ${EPOCHS} \
        --batch_size     ${BATCH_SIZE} \
        --lr             ${LR} \
        --peft_method    ${PEFT} \
        --device         ${DEVICE} \
        --results_dir    "${RESULTS_DIR}" \
        --checkpoint_dir "${CHECKPOINT_DIR}" \
        2>&1 | tee "${RESULTS_DIR}/${placement}_eps${epsilon}_${log_suffix}.log"; then

        mark_completed ${placement} ${epsilon}
        echo "[OK] DONE: ${placement} | eps=${epsilon} | $(date)"
        run_mia ${placement} ${epsilon} ${log_suffix}

    else
        mark_failed ${placement} ${epsilon}
        echo "[FAIL] FAILED: ${placement} | eps=${epsilon}"
    fi

    display_progress
    echo ""
}

# ── Main loop ──────────────────────────────────────────────────────────────
START_TIME=$(date +%s)
LOG_SUFFIX="epsweep_$(date +%Y%m%d_%H%M%S)"

echo "Starting epsilon sweep: $(date)"
display_progress
echo ""

# Ordered from weakest-to-strongest DP so if interrupted early
# we still have useful partial results (ε=4 and ε=2 are most useful)
for eps in 4.0 2.0 0.5; do
    echo "+============================================+"
    echo "|  Epsilon = ${eps}"
    echo "+============================================+"
    for placement in ${DP_PLACEMENTS}; do
        run_experiment ${placement} ${eps} ${LOG_SUFFIX}
    done
done

# ── Summary ────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo "=========================================="
echo " EPSILON SWEEP COMPLETE"
echo " Total time: ${HOURS}h ${MINUTES}m"
echo "=========================================="
display_progress
echo ""
echo "All results in: ${RESULTS_DIR}/"
echo "Adapter now has ε ∈ {0.5, 1.0, 2.0, 4.0, 8.0, ∞} for all 6 placements."
echo "Ready to build the privacy-utility curve in the notebook."
echo ""
echo "To resume if interrupted: bash scripts/runpod_epsilon_sweep_distilbert.sh"
