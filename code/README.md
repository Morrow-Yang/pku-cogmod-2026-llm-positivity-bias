# Source code

All experiment code lives here, organized into five sub-pipelines that mirror the structure of the paper. Each sub-pipeline directory contains:

- `train.py` (or `pilot.py`) — data generation: queries the LLM with the specified prompts and collects responses.
- `analyze.py` — analysis: aggregates, dedups, fits models, produces summary tables.
- `config.yaml` — hyperparameters and stimulus configuration (target list, prompt templates, n replications, etc.).
- `run.sh` — end-to-end invocation in the conventional order.
- (additional `.py` files in some pipelines for specialized analyses — model recovery, bootstrapping, hierarchical Bayes, etc.)

## Pipelines

| Directory | What it does |
|---|---|
| `stage-behavioral-replication-lehr-2025-free/` | Reproduces Lehr et al. 2025's positivity-rating phenomenon on free-form Likert responses, establishing the baseline effect on Llama-3.1-8B-Instruct. |
| `stage-base-vs-instruct-positivity-prior/` | Contrasts the Llama-3.1-8B base model against the Instruct (RLHF-tuned) variant to isolate the RLHF-induced shift. |
| `per-stage-localization-forced-honesty-overshoot/` | Per-stage analysis across M0–M3 prompt formats; introduces the forced-honesty / control-honesty diagnostic that distinguishes representation-level from output-policy explanations. |
| `cognitive-model-comparison-polite-speech-post/` | The 7-model cognitive-model comparison (4 theoretical families + statistical baselines), with a full identifiability battery: parameter recovery, model recovery, hierarchical Bayesian fits, bootstrap CIs. |
| `post-review-extensions/` | Hidden-state probe analysis (Study 4) localizing the rating-shift representation to transformer layers 14–31. Also includes per-template β\_L analysis and the post-external-review additions. |

## Running a pipeline

```bash
cd code/<pipeline>
bash run.sh
```

Most pipelines complete in minutes on CPU. The hidden-state extraction step (`code/post-review-extensions/extract_hidden_states.py`) requires a GPU and ~1 hour on an A100 for the full target set.

Trial-level results land in `../data/trial_results/<pipeline>/`.

## Environment

```
Python 3.10+
torch >= 2.1
transformers >= 4.40
numpy, pandas, scipy, statsmodels, matplotlib
arviz, pymc (for hierarchical Bayesian fits)
```

Llama-3.1-8B model weights must be downloaded separately from Hugging Face (gated; requires Meta license acceptance):
- [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
- [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

`download_model.py` scripts in some pipelines wrap the download once `HF_TOKEN` is set in the environment.

## Notes for readers

- Greedy decoding is used throughout (temperature = 0). This made early-version trial files deterministically identical across replications, which we caught and corrected via dedup (see paper §2.5).
- All target names, prompt templates, and exact replication counts are in each pipeline's `config.yaml`.
- The `__pycache__/` directories and `*.pyc` files are ignored by git and not shipped.
