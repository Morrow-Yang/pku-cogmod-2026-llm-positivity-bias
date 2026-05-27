#!/bin/bash
# M3 Round-2 extension (post-paperreview.ai 2026-05-19 second review).
# Runs C24, C25, C26 on M3:
#   C24 = chat-template forced-honesty WITHOUT "negative/controversial" wording (Q5 deconfound)
#   C25 = completion-style (prefix-injected) forced-honesty (Q6 chat-template vs RLHF disentanglement)
#   C26 = completion-style (prefix-injected) control-honesty (Q6 paired comparator)
#
# train.py is idempotent (skips existing result JSONs), so this only computes the 3 new cells.
# Trial count: 3 cells × 7 targets × 50 prompts = 1,050 trials. ~5 min compute + model load.
set -e
cd "$(dirname "$0")"

PROJECT_ROOT=/root/autodl-tmp/free-choice-project
EXPDIR=$PROJECT_ROOT/experiments/code/per-stage-localization-forced-honesty-overshoot
RESULTS=$PROJECT_ROOT/results/per-stage-localization-forced-honesty-overshoot
LOGS=$PROJECT_ROOT/logs

mkdir -p "$RESULTS" "$LOGS"
source "$PROJECT_ROOT/.venv/bin/activate"

echo "=== M3 Round-2 extension starting at $(date) ==="
df -h /root/autodl-tmp | tail -1

# --- M3 (Meta-Instruct) ---
echo ""
echo "=== M3: Llama-3.1-8B-Instruct — cells C24, C25, C26 ==="
M3_LOCAL=/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct
echo "  -> ensuring M3 cache complete..."
python "$EXPDIR/download_model.py" --repo LLM-Research/Meta-Llama-3.1-8B-Instruct

echo "  -> running M3 cells..."
python "$EXPDIR/train.py" --model-id M3 --model-path "$M3_LOCAL" \
    --cells C24,C25,C26 \
    --config "$EXPDIR/config.yaml" --out-dir "$RESULTS"

echo ""
echo "=== M3 Round-2 extension finished at $(date) ==="
df -h /root/autodl-tmp | tail -1
echo "=== DONE ==="
