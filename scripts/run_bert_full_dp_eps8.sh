#!/bin/bash
# One-shot script: BERT + AG News, full_dp, ε=8.0
# This is the single missing experiment to complete the cross-model analysis.
# Matches parameters used in the corrected ε=1 run (last runs/, Apr 18-19).
#
# Run from the project root:
#   bash scripts/run_bert_full_dp_eps8.sh
# Results land in:
#   results/bert_agnews_full_dp_eps8.0.json
#   mia_results/bert_agnews_full_dp_eps8.0_mia.json (via MIA script)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

RESULTS_DIR="${PROJECT_ROOT}/results"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
MIA_RESULTS_DIR="${PROJECT_ROOT}/mia_results"

mkdir -p "$RESULTS_DIR" "$CHECKPOINT_DIR" "$MIA_RESULTS_DIR"

RUN_NAME="bert_agnews_full_dp_eps8.0"
CHECKPOINT="${CHECKPOINT_DIR}/${RUN_NAME}.pt"

echo "========================================"
echo " BERT + AG News | full_dp | ε = 8.0"
echo "========================================"
echo "Results  : ${RESULTS_DIR}/${RUN_NAME}.json"
echo "Checkpoint: ${CHECKPOINT}"
echo "MIA out  : ${MIA_RESULTS_DIR}/${RUN_NAME}_mia.json"
echo ""

# ── Training ─────────────────────────────────────────────────────────────────
echo "[1/2] Training..."
python3 "${PROJECT_ROOT}/scripts/run_experiment.py" \
    --model        bert       \
    --dataset      agnews     \
    --placement    full_dp    \
    --epsilon      8.0        \
    --delta        1e-5       \
    --peft_method  adapter    \
    --epochs       20         \
    --batch_size   64         \
    --lr           5e-4       \
    --seed         42         \
    --device       cuda       \
    --results_dir    "$RESULTS_DIR"    \
    --checkpoint_dir "$CHECKPOINT_DIR"

echo ""
echo "[2/2] Running MIA on trained checkpoint..."
# run_mia.py saves output next to the checkpoint as <stem>_mia.json
MIA_CHECKPOINT_OUTPUT="${CHECKPOINT_DIR}/${RUN_NAME}_mia.json"

python3 "${PROJECT_ROOT}/scripts/run_mia.py" \
    --checkpoint   "$CHECKPOINT"  \
    --model        bert           \
    --dataset      agnews         \
    --peft_method  adapter        \
    --batch_size   64             \
    --seed         42             \
    --device       cuda

# Copy to mia_results/ under the canonical name used by the notebook loader
if [ -f "$MIA_CHECKPOINT_OUTPUT" ]; then
    cp "$MIA_CHECKPOINT_OUTPUT" "${MIA_RESULTS_DIR}/${RUN_NAME}_mia.json"
    echo "MIA results copied to: ${MIA_RESULTS_DIR}/${RUN_NAME}_mia.json"
else
    echo "WARNING: MIA output not found at $MIA_CHECKPOINT_OUTPUT"
fi

echo ""
echo "========================================"
echo " Done. Results:"
echo "   $(ls -lh ${RESULTS_DIR}/${RUN_NAME}.json 2>/dev/null || echo 'JSON not found')"
echo "   $(ls -lh ${MIA_RESULTS_DIR}/${RUN_NAME}_mia.json 2>/dev/null || echo 'MIA JSON not found')"
python3 -c "
import json, sys
try:
    d = json.load(open('${RESULTS_DIR}/${RUN_NAME}.json'))
    print(f'   Accuracy : {d[\"final_accuracy\"]:.4f}')
    print(f'   Epoch time: {d[\"avg_epoch_time\"]:.0f}s')
    print(f'   Throughput: {d[\"avg_throughput\"]:.0f}/s')
except Exception as e:
    print(f'   (could not parse JSON: {e})')
try:
    d = json.load(open('${MIA_RESULTS_DIR}/${RUN_NAME}_mia.json'))
    ta = d['threshold_attack']
    print(f'   MIA AUC  : {ta[\"auc\"]:.4f}')
    print(f'   MIA Adv  : {ta[\"advantage\"]:.4f}')
except Exception as e:
    print(f'   (could not parse MIA JSON: {e})')
"
echo "========================================"
