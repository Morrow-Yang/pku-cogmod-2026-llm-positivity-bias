#!/bin/bash
# Stage 1 launcher — 4-model sequential pipeline with download-run-cleanup pattern.
# Manages disk constraint (50 GB autodl-tmp): only one of M0/M1/M2 lives on disk at a time,
# alongside the cached M3 (Stage A baseline). Run from /root/autodl-tmp/free-choice-project.
set -e
cd "$(dirname "$0")"

PROJECT_ROOT=/root/autodl-tmp/free-choice-project
EXPDIR=$PROJECT_ROOT/experiments/code/stage-base-vs-instruct-positivity-prior
RESULTS=$PROJECT_ROOT/results/stage-base-vs-instruct-positivity-prior
LOGS=$PROJECT_ROOT/logs

mkdir -p "$RESULTS" "$LOGS"

source "$PROJECT_ROOT/.venv/bin/activate"

echo "=== Stage 1 run starting at $(date) ==="

# --- M0 (base) ---
echo ""
echo "=== M0: Llama-3.1-8B (base) ==="
M0_LOCAL=/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B
if [ ! -d "$M0_LOCAL" ]; then
    echo "  -> downloading M0..."
    python "$EXPDIR/download_model.py" --repo LLM-Research/Meta-Llama-3.1-8B
fi
echo "  -> running M0 cells..."
python "$EXPDIR/train.py" --model-id M0 --model-path "$M0_LOCAL" \
    --config "$EXPDIR/config.yaml" --out-dir "$RESULTS"

# --- M3 (Meta-Instruct, already cached from Stage A) ---
echo ""
echo "=== M3: Llama-3.1-8B-Instruct (cached from Stage A) ==="
M3_LOCAL=/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct
python "$EXPDIR/train.py" --model-id M3 --model-path "$M3_LOCAL" \
    --config "$EXPDIR/config.yaml" --out-dir "$RESULTS"

# --- delete M0 to free space for M1 ---
echo ""
echo "=== freeing M0 disk (need space for Tülu-SFT) ==="
rm -rf "$M0_LOCAL"
df -h /root/autodl-tmp | tail -1

# --- M1 (Tülu-3-SFT) ---
echo ""
echo "=== M1: Llama-3.1-Tulu-3-8B-SFT ==="
M1_LOCAL=/root/autodl-tmp/allenai/Llama-3___1-Tulu-3-8B-SFT
if [ ! -d "$M1_LOCAL" ]; then
    echo "  -> downloading M1..."
    python "$EXPDIR/download_model.py" --repo allenai/Llama-3.1-Tulu-3-8B-SFT
fi
echo "  -> running M1 cells..."
python "$EXPDIR/train.py" --model-id M1 --model-path "$M1_LOCAL" \
    --config "$EXPDIR/config.yaml" --out-dir "$RESULTS"

# --- delete M1 to free space for M2 ---
echo ""
echo "=== freeing M1 disk (need space for Tülu-DPO) ==="
rm -rf "$M1_LOCAL"
df -h /root/autodl-tmp | tail -1

# --- M2 (Tülu-3-DPO) ---
echo ""
echo "=== M2: Llama-3.1-Tulu-3-8B-DPO ==="
M2_LOCAL=/root/autodl-tmp/LLM-Research/Llama-3___1-Tulu-3-8B-DPO
if [ ! -d "$M2_LOCAL" ]; then
    echo "  -> downloading M2..."
    python "$EXPDIR/download_model.py" --repo LLM-Research/Llama-3.1-Tulu-3-8B-DPO
fi
echo "  -> running M2 cells..."
python "$EXPDIR/train.py" --model-id M2 --model-path "$M2_LOCAL" \
    --config "$EXPDIR/config.yaml" --out-dir "$RESULTS"

echo ""
echo "=== Stage 1 run finished at $(date) ==="
echo "=== final disk ==="
df -h /root/autodl-tmp | tail -1

# Run analysis
echo ""
echo "=== running analyze.py ==="
cd "$EXPDIR"
python analyze.py --config config.yaml --results-dir "$RESULTS" 2>&1 || echo "(analyze.py failed — run manually)"

echo "=== DONE ==="
