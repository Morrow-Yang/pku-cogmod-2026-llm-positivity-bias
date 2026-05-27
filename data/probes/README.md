# Hidden-state probe activations (Study 4)

These files are **not bundled** in this repository — they exceed the course's 100 MB in-repo cap. They are deterministically regenerable from the model + stimuli; this README explains how.

## What the data looks like

660 `.npz` files (~315 MB total), one per `(target × prompt_template)` combination across ~132 targets and 5 templates. Each file contains:

| Key | Shape | Description |
|---|---|---|
| `hidden_targetname` | `(33, 4096)` fp16 | Hidden states at the target-name token position, across all 33 layers of Llama-3.1-8B |
| `hidden_lasttoken` | `(33, 4096)` fp16 | Hidden states at the last-token position, across all 33 layers |
| `meta` | JSON string | `{target_id, condition, template_idx, target_name_token_position, last_token_position, n_layers, hidden_dim, target_name_resolved}` |

File-naming convention: `hs_cond_E5_<num>_target_<target_id>_tpl_<template_idx>.npz`.

These activations are consumed by `../../code/post-review-extensions/probe_analysis.py`, which fits linear probes per layer and produces the layer 14–31 localization figure (paper §4 / Figure 4).

## Regenerating

```bash
# From the repo root:
cd code/post-review-extensions

# 1. Ensure Llama-3.1-8B-Instruct is downloaded:
python download_model.py   # if available in this pipeline; otherwise pull from HF directly

# 2. Run extraction:
python extract_hidden_states.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output ../../data/probes/ \
  --config config.yaml
```

**Runtime:** ~1 hour on a single NVIDIA A100; longer on smaller GPUs. The script is idempotent — it skips any (target, template) for which the output `.npz` already exists.

**Disk:** allocate ~350 MB free.

**Memory:** model loading peaks at ~16 GB VRAM (fp16). CPU RAM peak ~8 GB.

## Why not bundled

Under the course's submission rules, datasets >100 MB are documented via their source rather than packaged with the code. The activations qualify (315 MB). The author's working repository does check them in (where storage constraints are looser); they are not redistributed here.

## If you only want the figure, not the raw activations

The output of `probe_analysis.py` — the layer-wise probe accuracies that produce the localization figure — is small (~kilobytes) and is included alongside the analysis outputs in `../trial_results/post-review-extensions/` if present. Look for `probe_results.json` or similar.
