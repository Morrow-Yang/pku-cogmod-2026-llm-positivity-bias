#!/bin/bash
# Launcher for Stage A v2 — Lehr-faithful free-choice design, 50×5 hybrid sample sizes.
set -e
cd "$(dirname "$0")"

if [ -f /root/autodl-tmp/free-choice-project/.venv/bin/activate ]; then
    source /root/autodl-tmp/free-choice-project/.venv/bin/activate
fi

N_SEEDS="${N_SEEDS:-5}"

mkdir -p /root/autodl-tmp/free-choice-project/logs

echo "=== Stage A v2 run starting at $(date) ==="
echo "    N_SEEDS=$N_SEEDS  (n_commanded × 2 + n_free read from config.yaml)"

for seed in $(seq 1 $N_SEEDS); do
    echo ""
    echo "=== seed $seed / $N_SEEDS ==="
    python train.py --config config.yaml --seed "$seed"
done

echo ""
echo "=== Stage A v2 run finished at $(date) ==="

# Auto-analyze
python analyze.py 2>&1 || echo "(analyze.py skipped — run manually)"
