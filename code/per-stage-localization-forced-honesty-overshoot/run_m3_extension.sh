#!/bin/bash
# M3-only extension run (post paperreview.ai feedback 2026-05-19).
# Runs C4, C5, C8, C20, C21, C22, C23 on M3 with the extended target set
# (4 fictional + 3 valence-spanned calibration). train.py is idempotent, so
# this only computes the missing (cell × target) pairs.
#
# Cells:
#   C4  — M3 default (fills in calibration targets only; fictional already done elsewhere)
#   C5  — M3 forced-honesty (fills in calibration targets only)
#   C8  — M3 control-honesty (fills in calibration targets only)
#   C20 — M3 alt-control "be truthful" (all 7 targets)
#   C21 — M3 alt-control "be precise and objective" (all 7 targets)
#   C22 — M3 anti-politeness "be blunt + avoid euphemisms + no hedging" (all 7 targets)
#   C23 — M3 anti-politeness "answer directly, no diplomatic softening" (all 7 targets)
#
# Trial count: ~1,850 (50 prompts each).
set -e
cd "$(dirname "$0")"

PROJECT_ROOT=/root/autodl-tmp/free-choice-project
EXPDIR=$PROJECT_ROOT/experiments/code/per-stage-localization-forced-honesty-overshoot
RESULTS=$PROJECT_ROOT/results/per-stage-localization-forced-honesty-overshoot
LOGS=$PROJECT_ROOT/logs

mkdir -p "$RESULTS" "$LOGS"
source "$PROJECT_ROOT/.venv/bin/activate"

echo "=== M3 extension run starting at $(date) ==="
df -h /root/autodl-tmp | tail -1

# --- M3 (Meta-Instruct) — runs C4, C5, C8 on calibration + C20-C23 on all 7 targets ---
echo ""
echo "=== M3: Llama-3.1-8B-Instruct — cells C4, C5, C8, C20, C21, C22, C23 ==="
M3_LOCAL=/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct
echo "  -> ensuring M3 cache complete..."
python "$EXPDIR/download_model.py" --repo LLM-Research/Meta-Llama-3.1-8B-Instruct

echo "  -> running M3 cells..."
python "$EXPDIR/train.py" --model-id M3 --model-path "$M3_LOCAL" \
    --cells C4,C5,C8,C20,C21,C22,C23 \
    --config "$EXPDIR/config.yaml" --out-dir "$RESULTS"

echo ""
echo "=== M3 extension finished at $(date) ==="
df -h /root/autodl-tmp | tail -1
echo "=== DONE ==="
