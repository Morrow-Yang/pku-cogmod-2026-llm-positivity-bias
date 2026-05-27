"""Item 3 (from /review W5): extract M3 hidden states under multiple
conditions, for linear-probe and RSA analysis.

Tests whether M3's representations of target valence shift under forced-
honesty vs default (motivated-cognition-v2) or are preserved (consistent
with either expression-policy OR motivated-cognition-v1).

Per design-review (5/10, major-revision -> all 5 fixes applied here):
1. Don't average templates; extract per-template (110 samples per
   condition = 22 targets x 5 templates).
2. Add valence-irrelevant prompt-variance baseline (VERBOSE_CONTROL).
3. Extract at TWO positions: target-name token + last-token-before-
   prediction.
4. Frame as "evidence against representational shift", not "evidence
   for expression-policy".
5. Report probe ceiling (within-condition train/test split as upper bound).

Conditions extracted (6, total ~660 forward passes on M3):
- E5_000        : bare baseline ("You are a research participant. ...")
- E8_verbose    : valence-irrelevant prompt control ("Please be concise")
- E5_001        : license-only negative
- E5_111        : full forced-honesty (= original C5)
- E7_pos_001    : license-only positive (simple valence mirror)
- E7_pos_111    : full mirror (H + A + simple positive license)

Output: one .npz file per (condition, target, template) under
experiments/results/post-review-extensions-probes/.
Each .npz holds:
  hidden_targetname  : float16 (32 layers, 4096)
  hidden_lasttoken   : float16 (32 layers, 4096)
  meta               : dict with target_id, condition, template_idx,
                        target_name_token_position, last_token_position,
                        n_layers, hidden_dim
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the existing train.py infrastructure
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                        "per-stage-localization-forced-honesty-overshoot"))
from train import (  # noqa: E402
    TARGETS as ORIG_TARGETS,
    PRE_RATING_TEMPLATES,
)
# Now import from post-review-extensions train.py
import importlib.util
_pre_train_spec = importlib.util.spec_from_file_location(
    "pre_train", Path(__file__).resolve().parent / "train.py")
_pre_train = importlib.util.module_from_spec(_pre_train_spec)
_pre_train_spec.loader.exec_module(_pre_train)

EXPANDED_TARGETS = _pre_train.EXPANDED_TARGETS
FACTORIAL_PROMPTS = _pre_train.FACTORIAL_PROMPTS
POSITIVE_LICENSE_PROMPTS = _pre_train.POSITIVE_LICENSE_PROMPTS
VERBOSE_CONTROL_PROMPT = _pre_train.VERBOSE_CONTROL_PROMPT


# Conditions to extract: (label, full system-prompt string)
CONDITIONS = [
    ("E5_000",     FACTORIAL_PROMPTS["e5_000"]),
    ("E8_verbose", VERBOSE_CONTROL_PROMPT),
    ("E5_001",     FACTORIAL_PROMPTS["e5_001"]),
    ("E5_111",     FACTORIAL_PROMPTS["e5_111"]),
    ("E7_pos_001", POSITIVE_LICENSE_PROMPTS["e7_pos_001"]),
    ("E7_pos_111", POSITIVE_LICENSE_PROMPTS["e7_pos_111"]),
]


def build_chat_prompt(tokenizer, system_prompt: str, user_content: str) -> str:
    """Apply Llama-3.1 chat template; same as train.py's chat path."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content + "\n\nAnswer with ONLY the digit 1-7."},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def find_target_name_position(tokenizer, full_text: str, target_name: str) -> int:
    """Find the position (in the tokenized full_text) of the LAST token of
    the target name's first occurrence in the user message.

    Strategy: tokenize full_text; tokenize " {target_name}" (with leading
    space, since target names appear after words in the template); find the
    first occurrence of the target-name token subsequence in the full
    tokens; return the position of the LAST token in that match.

    Returns -1 if not found (caller should fall back to last_token_position).
    """
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    # Try a few variants of how the target name appears
    candidate_strings = [
        f" {target_name}",       # leading space (most common after a word)
        f"{target_name}",        # bare
        f" {target_name.lower()}",
        f"{target_name.lower()}",
    ]
    for cand in candidate_strings:
        name_ids = tokenizer(cand, add_special_tokens=False).input_ids
        if not name_ids:
            continue
        # Search for subsequence
        for i in range(len(full_ids) - len(name_ids) + 1):
            if full_ids[i:i+len(name_ids)] == name_ids:
                return i + len(name_ids) - 1  # last token of the match
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--save-fp16", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, trust_remote_code=True,
        output_hidden_states=True,
    ).to(device).eval()
    print(f"  Loaded. dtype={model.dtype}, device={device}.", flush=True)

    n_layers = model.config.num_hidden_layers + 1  # includes input embeddings layer
    hidden_dim = model.config.hidden_size
    print(f"  Hidden states: {n_layers} layers (incl. embedding layer), dim={hidden_dim}", flush=True)

    # All targets (originals + expanded pseudowords) -- 22 total
    all_targets = {**ORIG_TARGETS, **EXPANDED_TARGETS}
    # Drop the legacy "calibration" aggregate and the bursovet boundary case
    # if present in originals -- they are not part of the standard 7+15.
    for k in ("bursovet", "calibration"):
        all_targets.pop(k, None)
    target_ids = sorted(all_targets.keys())
    print(f"  Targets: {len(target_ids)} ({len(ORIG_TARGETS)-2} originals + {len(EXPANDED_TARGETS)} pseudowords)", flush=True)
    print(f"  Conditions: {len(CONDITIONS)}", flush=True)
    print(f"  Templates: {len(PRE_RATING_TEMPLATES)}", flush=True)
    n_passes = len(CONDITIONS) * len(target_ids) * len(PRE_RATING_TEMPLATES)
    print(f"  Total forward passes planned: {n_passes}\n", flush=True)

    t_start = time.time()
    n_done = 0
    n_target_not_found = 0
    for cond_name, sys_prompt in CONDITIONS:
        for tid in target_ids:
            target = all_targets[tid]
            for template_idx in range(len(PRE_RATING_TEMPLATES)):
                out_path = out_dir / f"hs_cond_{cond_name}_target_{tid}_tpl_{template_idx}.npz"
                if out_path.exists():
                    n_done += 1
                    continue
                tpl = PRE_RATING_TEMPLATES[template_idx]
                user_content = tpl.format(name=target["name"], desc=target["description"])
                full_text = build_chat_prompt(tokenizer, sys_prompt, user_content)
                inputs = tokenizer(full_text, return_tensors="pt",
                                    add_special_tokens=False).to(device)
                last_pos = inputs.input_ids.shape[1] - 1
                name_pos = find_target_name_position(
                    tokenizer, full_text, target["name"])
                if name_pos < 0:
                    n_target_not_found += 1
                    name_pos = last_pos  # fall back

                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True)
                # hidden_states: tuple of length n_layers, each [1, seq, hidden]
                # Stack into [n_layers, hidden] at each position
                hs_name = torch.stack([
                    h[0, name_pos] for h in out.hidden_states
                ]).to(torch.float16).cpu().numpy()
                hs_last = torch.stack([
                    h[0, last_pos] for h in out.hidden_states
                ]).to(torch.float16).cpu().numpy()
                # Save
                meta = {
                    "target_id": tid, "condition": cond_name,
                    "template_idx": template_idx,
                    "target_name_token_position": int(name_pos),
                    "last_token_position": int(last_pos),
                    "n_layers": int(n_layers),
                    "hidden_dim": int(hidden_dim),
                    "target_name_resolved": name_pos != last_pos,
                }
                np.savez_compressed(out_path,
                                     hidden_targetname=hs_name,
                                     hidden_lasttoken=hs_last,
                                     meta=json.dumps(meta))
                n_done += 1
                if n_done % 20 == 0:
                    elapsed = time.time() - t_start
                    eta = (n_passes - n_done) * elapsed / n_done
                    print(f"  [{n_done}/{n_passes}]  elapsed={elapsed:.1f}s  "
                          f"ETA={eta:.0f}s  target-name-not-found={n_target_not_found}",
                          flush=True)

    elapsed = time.time() - t_start
    print(f"\nDone. {n_done} hidden-state files saved in {elapsed:.1f}s.")
    print(f"Target-name-not-found fallbacks: {n_target_not_found} / {n_passes}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
