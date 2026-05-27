#!/bin/bash
# Per-stage localization launcher.
# Sequential M0 -> M1 -> M2 pipeline with download-run-cleanup pattern.
# M3 not re-run (its C5/C8 cells inherited from prior experiment).
# Run from /root/autodl-tmp/free-choice-project.
set -e
cd "$(dirname "$0")"

PROJECT_ROOT=/root/autodl-tmp/free-choice-project
EXPDIR=$PROJECT_ROOT/experiments/code/per-stage-localization-forced-honesty-overshoot
RESULTS=$PROJECT_ROOT/results/per-stage-localization-forced-honesty-overshoot
LOGS=$PROJECT_ROOT/logs

mkdir -p "$RESULTS" "$LOGS"

source "$PROJECT_ROOT/.venv/bin/activate"

echo "=== Per-stage localization run starting at $(date) ==="
df -h /root/autodl-tmp | tail -1

# --- M0 (base) — runs C9, C10 (prefix-injected honesty prompts) ---
echo ""
echo "=== M0: Llama-3.1-8B (base) — cells C9, C10 ==="
M0_LOCAL=/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B
# Always call download_model.py — snapshot_download is idempotent (skips complete
# files, resumes/retries failed ones). The dir-exists check used to skip retries
# after a transient ModelScope CDN failure; that's how M0 partial download fell
# through to train.py and crashed. 2026-05-18 fix.
echo "  -> ensuring M0 cache complete..."
python "$EXPDIR/download_model.py" --repo LLM-Research/Meta-Llama-3.1-8B
echo "  -> running M0 cells..."
python "$EXPDIR/train.py" --model-id M0 --model-path "$M0_LOCAL" \
    --config "$EXPDIR/config.yaml" --out-dir "$RESULTS"

# --- delete M0 to free space for M1 ---
echo ""
echo "=== freeing M0 disk ==="
rm -rf "$M0_LOCAL"
df -h /root/autodl-tmp | tail -1

# --- M1 (Tülu-3-SFT) — runs C11, C12 ---
echo ""
echo "=== M1: Llama-3.1-Tulu-3-8B-SFT — cells C11, C12 ==="
M1_LOCAL=/root/autodl-tmp/allenai/Llama-3___1-Tulu-3-8B-SFT
echo "  -> ensuring M1 cache complete..."
python "$EXPDIR/download_model.py" --repo allenai/Llama-3.1-Tulu-3-8B-SFT
echo "  -> running M1 cells..."
python "$EXPDIR/train.py" --model-id M1 --model-path "$M1_LOCAL" \
    --config "$EXPDIR/config.yaml" --out-dir "$RESULTS"

# --- delete M1 to free space for M2 ---
echo ""
echo "=== freeing M1 disk ==="
rm -rf "$M1_LOCAL"
df -h /root/autodl-tmp | tail -1

# --- M2 (Tülu-3-DPO) — runs C13, C14 ---
echo ""
echo "=== M2: Llama-3.1-Tulu-3-8B-DPO — cells C13, C14 ==="
M2_LOCAL=/root/autodl-tmp/LLM-Research/Llama-3___1-Tulu-3-8B-DPO
echo "  -> ensuring M2 cache complete..."
python "$EXPDIR/download_model.py" --repo LLM-Research/Llama-3.1-Tulu-3-8B-DPO
echo "  -> running M2 cells..."
python "$EXPDIR/train.py" --model-id M2 --model-path "$M2_LOCAL" \
    --config "$EXPDIR/config.yaml" --out-dir "$RESULTS"

echo ""
echo "=== Per-stage localization finished at $(date) ==="
df -h /root/autodl-tmp | tail -1

echo "=== DONE ==="
