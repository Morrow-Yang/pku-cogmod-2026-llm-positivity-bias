#!/usr/bin/env bash
# Run E4 (M0 chat-template), E5 (M3 2x2x2 factorial), E6 (M3 expanded pseudowords)
# sequentially. Loads M0 first, runs E4; then unloads and loads M3, runs E5+E6.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/free-choice-project}"
RESULTS="${RESULTS:-${REPO_ROOT}/experiments/results/post-review-extensions}"
PY="${PY:-python3}"

cd "${REPO_ROOT}/experiments/code/post-review-extensions"

echo "=== E4: M0 chat-template baseline ==="
"$PY" train.py --config config.yaml --model-id M0 \
  --cells E4_default,E4_forced,E4_control \
  --out-dir "${RESULTS}"

echo ""
echo "=== E5 + E6: M3 2x2x2 factorial + expanded pseudowords ==="
"$PY" train.py --config config.yaml --model-id M3 \
  --cells E5_000,E5_001,E5_010,E5_011,E5_100,E5_101,E5_110,E5_111,E6_default,E6_forced,E6_control \
  --out-dir "${RESULTS}"

echo ""
echo "=== Done. Results in ${RESULTS} ==="
ls -la "${RESULTS}" | wc -l
