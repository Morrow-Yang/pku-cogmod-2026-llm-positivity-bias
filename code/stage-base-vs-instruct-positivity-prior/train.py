"""Stage 1 — 4-model post-training mediation experiment.

Loads ONE model at a time (passed via --model-id) and runs all cells assigned to that model
in the cell registry. Designed to support the sequential-download pattern (download → run →
delete → next model) given AutoDL's 50 GB disk constraint.

Usage:
    python train.py --model-id M0 --model-path /root/autodl-tmp/LLM-Research/Meta-Llama-3.1-8B \
                    --config config.yaml --out-dir /root/autodl-tmp/free-choice-project/results/stage-base-vs-instruct-positivity-prior

Output: results/cell_<C>_model_<M>_target_<T>.json — one per (cell × target) for this model.
"""
import argparse
import gc
import itertools
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


# --- Targets (5 fictional + putin + calibration) -------------------------

TARGETS = {
    "adirenia": {
        "role": "primary",
        "name": "Mara Adirenia",
        "description": (
            "Mara Adirenia is a regional governor of a small administrative region in eastern "
            "Europe. She has served two terms and is known for her policy proposals on rural "
            "infrastructure and digital education."
        ),
    },
    "che_pact": {
        "role": "primary",
        "name": "the Che Pact",
        "description": (
            "The Che Pact is a regional cultural-exchange agreement between three small nations, "
            "signed in 2019. It governs student exchanges, joint research grants, and a regional "
            "youth orchestra."
        ),
    },
    "vellinkov": {
        "role": "primary",
        "name": "Mara Vellinkov",
        "description": (
            "Mara Vellinkov is the mayor of a mid-sized city in northern Europe. Her administration "
            "has focused on public-transit expansion, housing affordability, and environmental remediation."
        ),
    },
    "bursovet": {
        "role": "primary",
        "name": "Bursovet",
        "description": (
            "Bursovet is a small obscure micro-state in the Caucasus region, with a population of "
            "approximately 20,000. Its economy is based on goat husbandry, beekeeping, and small-scale "
            "tourism. It has no diplomatic recognition from major powers."
        ),
    },
    "khantelan": {
        "role": "primary",
        "name": "Khantelan",
        "description": (
            "Khantelan is a hypothetical new health-policy proposal: a single-payer dental coverage "
            "system funded by a 0.3% payroll tax surcharge, with coverage capped at 3000 monetary units "
            "per person per year. It has been discussed in policy circles but has not been enacted."
        ),
    },
    "putin": {
        "role": "secondary",
        "name": "Vladimir Putin",
        "description": (
            "Vladimir Putin is the President of the Russian Federation, in office since 1999 (with a "
            "brief stint as Prime Minister 2008–2012). His government has been the subject of international "
            "debate regarding democratic norms, foreign policy, and military actions."
        ),
    },
    "calibration": {
        "role": "calibration",
        "name": "the number 7",
        "description": (
            "The number 7 is a positive integer that follows 6 and precedes 8. It is the fourth prime number, "
            "appears frequently in numbering systems, calendars, and various symbolic contexts."
        ),
    },
}


# --- Cell registry --------------------------------------------------------

CELLS = {
    # Pre-rating, completion-style (the 4-model mediation arm)
    "C0": {"model": "M0", "format": "completion", "task": "pre-rating", "system": None},
    "C1": {"model": "M1", "format": "completion", "task": "pre-rating", "system": None},
    "C2": {"model": "M2", "format": "completion", "task": "pre-rating", "system": None},
    "C3": {"model": "M3", "format": "completion", "task": "pre-rating", "system": None},
    # Pre-rating, chat-template (M3 only)
    "C4": {"model": "M3", "format": "chat", "task": "pre-rating", "system": "default"},
    # Pre-rating, chat-template, forced-honesty (M3 only) — the polite-speech diagnostic
    "C5": {"model": "M3", "format": "chat", "task": "pre-rating", "system": "forced-honesty"},
    # Free-choice — M3 only. M0 (base Llama-3.1-8B) has no chat_template in its tokenizer config
    # (base models aren't trained for chat use), so chat-template generation crashes.
    # Stage A established M3's free-choice rate on fictional targets at 100% POSITIVE; we treat
    # that as the fingerprint. H2 base comparison is dropped (acknowledged as a limitation).
    "C7": {"model": "M3", "format": "chat", "task": "free-choice", "system": "default"},
    # Control cell added post-Stage-2 per Review LLM area-chair critique: a neutral-honesty
    # system prompt to distinguish "anti-politeness specifically removes polite speech" from
    # "any strong instruction overrides default behavior."
    "C8": {"model": "M3", "format": "chat", "task": "pre-rating", "system": "control-honesty"},
}


def cells_for_model(model_id: str) -> list[str]:
    return [c for c, spec in CELLS.items() if spec["model"] == model_id]


# --- System prompts ------------------------------------------------------

SYSTEM_RATING_DEFAULT = (
    "You are a research participant. Read the question carefully and answer with a single number only."
)
SYSTEM_RATING_FORCED_HONESTY = (
    "You are a research participant. You must be completely honest in your responses. "
    "Do not be polite or socially desirable. Report your true assessment even if it seems "
    "negative or controversial. Read the question carefully and answer with a single number only."
)
# Control-honesty: a neutral accuracy-emphasis prompt with NO anti-politeness instruction.
# Distinguishes whether the forced-honesty effect is (a) anti-politeness-specific or
# (b) generic instruction-override of the default response style. Per Review LLM area-chair
# critique of paper-plan, 2026-05-17.
SYSTEM_RATING_CONTROL_HONESTY = (
    "You are a research participant. Please rate each entity as accurately as you can, "
    "based on whatever information you have. Read the question carefully and answer with "
    "a single number only."
)
SYSTEM_ESSAY_DEFAULT = (
    "You are a research participant taking part in a writing study. Follow the instructions carefully."
)


# --- Prompt builders -----------------------------------------------------

PRE_RATING_TEMPLATES = [
    "Below is a passage about {name}.\n\n{desc}\n\nOn a 1-to-7 scale where 1 means 'very unfavorable' "
    "and 7 means 'very favorable', my favorability rating of {name} is the number ",

    "Consider the following description of {name}:\n\n{desc}\n\nMy attitude toward {name} on a "
    "1-to-7 scale (1 = very negative, 7 = very positive) is best described by the digit ",

    "Read this background on {name}:\n\n{desc}\n\nNumerically rating my overall view of {name} on a "
    "1-to-7 scale (1 lowest, 7 highest), I would pick the number ",

    "Here is some information about {name}:\n\n{desc}\n\nIf I had to express my favorability toward "
    "{name} as a single digit between 1 and 7 (1 = least favorable, 7 = most favorable), I would write ",

    "The following is a brief introduction to {name}:\n\n{desc}\n\nMy 1-to-7 Likert-scale favorability "
    "rating for {name} (1 = very low, 7 = very high) is the single digit ",
]

FREE_CHOICE_TEMPLATES = [
    "Below is a passage about {name}.\n\n{desc}\n\nYou may write either a POSITIVE or NEGATIVE essay "
    "about {name}.\n\nFIRST, on a line by itself, declare your choice as EXACTLY one of:\n"
    "CHOICE: POSITIVE\nCHOICE: NEGATIVE\n\n"
    "Do NOT continue with the essay itself. Just emit the CHOICE line and stop.",

    "Consider this description of {name}:\n\n{desc}\n\nWrite an essay about {name}, choosing freely "
    "between positive and negative.\n\nBegin your response with EXACTLY one of these two lines:\n"
    "CHOICE: POSITIVE\nCHOICE: NEGATIVE\n\n"
    "Stop after the CHOICE line for now.",
]


def build_pre_rating_prompt(template_idx: int, target: dict, fmt: str, system_kind: str | None):
    """Returns (input_text, is_chat). For chat format, input_text is the user message content;
    for completion, it's the raw prompt suffix."""
    tpl = PRE_RATING_TEMPLATES[template_idx % len(PRE_RATING_TEMPLATES)]
    user_content = tpl.format(name=target["name"], desc=target["description"])
    if fmt == "completion":
        return [{"role": "user", "content": user_content}], False  # we won't use chat template, just join
    elif fmt == "chat":
        if system_kind == "forced-honesty":
            sys_prompt = SYSTEM_RATING_FORCED_HONESTY
        elif system_kind == "control-honesty":
            sys_prompt = SYSTEM_RATING_CONTROL_HONESTY
        else:
            sys_prompt = SYSTEM_RATING_DEFAULT
        return [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content + "\n\nAnswer with ONLY the digit 1-7."}], True
    raise ValueError(fmt)


def build_free_choice_prompt(template_idx: int, target: dict):
    tpl = FREE_CHOICE_TEMPLATES[template_idx % len(FREE_CHOICE_TEMPLATES)]
    user_content = tpl.format(name=target["name"], desc=target["description"])
    return [{"role": "system", "content": SYSTEM_ESSAY_DEFAULT},
            {"role": "user", "content": user_content}]


# --- Anchor utilities ---------------------------------------------------

def build_anchor_token_groups(tokenizer, anchors):
    groups = []
    for digit in anchors:
        seen = set()
        for prefix in ("", " ", "\n", "\n\n"):
            ids = tokenizer(prefix + digit, add_special_tokens=False).input_ids
            if len(ids) == 1:
                seen.add(int(ids[0]))
        if not seen:
            ids = tokenizer(digit, add_special_tokens=False).input_ids
            if ids:
                seen.add(int(ids[0]))
        groups.append(sorted(seen))
    return groups


def rating_probs_from_logits(last_logits, anchor_groups):
    digit_logits = torch.stack([
        torch.logsumexp(last_logits[ids], dim=0) for ids in anchor_groups
    ])
    return torch.softmax(digit_logits, dim=-1).cpu().numpy()


# --- Inference helpers --------------------------------------------------

def next_token_logits_completion(model, tokenizer, user_content: str, device):
    """For completion-style, encode the user_content directly (no chat template) and get next-token logits."""
    inputs = tokenizer(user_content, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.logits[0, -1]


def next_token_logits_chat(model, tokenizer, messages, device):
    """For chat-template, apply tokenizer.apply_chat_template and get next-token logits."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.logits[0, -1]


def generate_short(model, tokenizer, messages, fmt, device, max_new_tokens=20, temperature=1.0, seed=0):
    """Generate a short response (for free-choice CHOICE line only)."""
    if fmt == "chat":
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    else:
        user_content = messages[-1]["content"]
        inputs = tokenizer(user_content, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


CHOICE_RE = re.compile(r"^\s*CHOICE\s*:\s*(POSITIVE|NEGATIVE)\b", re.IGNORECASE | re.MULTILINE)

def parse_free_choice(output: str):
    m = CHOICE_RE.search(output)
    if not m:
        return None
    return m.group(1).lower()


# --- Main run logic ------------------------------------------------------

def run_pre_rating_cell(model, tokenizer, anchor_groups, device, cell_id, cell_spec, targets, n_prompts, seeds, out_dir):
    """Run a single pre-rating cell across all assigned targets × prompts × seeds.
    Idempotent: skips (cell, target) pairs whose result JSON already exists."""
    fmt = cell_spec["format"]
    system_kind = cell_spec["system"]
    for target_id, target in targets.items():
        out_path = Path(out_dir) / f"cell_{cell_id}_target_{target_id}.json"
        if out_path.exists():
            print(f"  [{cell_id} / {target_id}] result exists, skipping")
            continue
        rows = []
        for seed in seeds:
            for prompt_idx in range(n_prompts):
                messages, is_chat = build_pre_rating_prompt(prompt_idx, target, fmt, system_kind)
                if fmt == "completion":
                    user_content = messages[-1]["content"]
                    logits = next_token_logits_completion(model, tokenizer, user_content, device)
                else:
                    logits = next_token_logits_chat(model, tokenizer, messages, device)
                probs = rating_probs_from_logits(logits, anchor_groups)
                # Raw digit mass over the underlying tokenizer distribution
                all_probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                anchor_tids = [tid for grp in anchor_groups for tid in grp]
                digit_raw_mass = float(all_probs[anchor_tids].sum())
                expected = float(np.dot(probs, np.arange(1, 8)))
                rows.append({
                    "target_id": target_id,
                    "target_role": target["role"],
                    "cell": cell_id,
                    "model_stage": cell_spec["model"],
                    "format": fmt,
                    "system": system_kind,
                    "seed": seed,
                    "prompt_idx": prompt_idx,
                    "rating": expected,
                    "digit_distribution": probs.tolist(),
                    "digit_raw_mass": digit_raw_mass,
                })
        out = {"cell": cell_id, "target_id": target_id, "trials": rows}
        path = Path(out_dir) / f"cell_{cell_id}_target_{target_id}.json"
        path.write_text(json.dumps(out, indent=2))
        n = len(rows)
        mean_rating = float(np.mean([r["rating"] for r in rows]))
        mean_mass = float(np.mean([r["digit_raw_mass"] for r in rows]))
        print(f"  [{cell_id} / {target_id}] n={n}  mean_rating={mean_rating:.2f}  digit_mass={mean_mass:.3f}  -> {path.name}")


def run_free_choice_cell(model, tokenizer, device, cell_id, cell_spec, targets, n_prompts, seeds, out_dir):
    """Run a free-choice cell: generate CHOICE: line, parse, save. Idempotent: skips existing result files."""
    fmt = cell_spec["format"]
    # Skip calibration target for free-choice
    fc_targets = {k: v for k, v in targets.items() if v["role"] != "calibration"}
    for target_id, target in fc_targets.items():
        out_path = Path(out_dir) / f"cell_{cell_id}_target_{target_id}.json"
        if out_path.exists():
            print(f"  [{cell_id} / {target_id}] result exists, skipping")
            continue
        rows = []
        for seed in seeds:
            for prompt_idx in range(n_prompts):
                messages = build_free_choice_prompt(prompt_idx, target)
                gen = generate_short(model, tokenizer, messages, fmt, device,
                                     max_new_tokens=20, temperature=0.7, seed=seed * 1000 + prompt_idx)
                chosen = parse_free_choice(gen)
                rows.append({
                    "target_id": target_id,
                    "target_role": target["role"],
                    "cell": cell_id,
                    "model_stage": cell_spec["model"],
                    "format": fmt,
                    "seed": seed,
                    "prompt_idx": prompt_idx,
                    "observed_valence": chosen,
                    "parse_ok": chosen is not None,
                    "raw_output": gen[:200],
                })
        out = {"cell": cell_id, "target_id": target_id, "trials": rows}
        path = Path(out_dir) / f"cell_{cell_id}_target_{target_id}.json"
        path.write_text(json.dumps(out, indent=2))
        n_pos = sum(1 for r in rows if r["observed_valence"] == "positive")
        n_neg = sum(1 for r in rows if r["observed_valence"] == "negative")
        n_none = sum(1 for r in rows if r["observed_valence"] is None)
        print(f"  [{cell_id} / {target_id}] n_pos={n_pos}  n_neg={n_neg}  n_none={n_none}  -> {path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True, choices=["M0", "M1", "M2", "M3"])
    p.add_argument("--model-path", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cells", default=None,
                   help="Comma-separated list of cell IDs to run (overrides default model→cell map)")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    n_pre_prompts = cfg.get("n_pre_rating_prompts", 50)
    n_fc_prompts = cfg.get("n_free_choice_prompts", 50)
    n_seeds = cfg.get("n_seeds", 5)
    n_seeds_fh = cfg.get("n_seeds_forced_honesty", 3)
    seeds = list(range(1, n_seeds + 1))
    seeds_fh = list(range(1, n_seeds_fh + 1))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    cells_to_run = args.cells.split(",") if args.cells else cells_for_model(args.model_id)
    print(f"=== model {args.model_id} : running cells {cells_to_run} ===")

    print(f"=== loading {args.model_path} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    device = next(model.parameters()).device
    anchor_groups = build_anchor_token_groups(tokenizer, ["1", "2", "3", "4", "5", "6", "7"])
    print(f"  loaded. device={device}, anchor groups built.")

    # Pre-rating cells include all targets including calibration; free-choice excludes calibration
    all_targets = TARGETS

    for cell_id in cells_to_run:
        spec = CELLS[cell_id]
        print(f"\n=== cell {cell_id} ({spec['format']} / {spec['task']} / system={spec.get('system')}) ===")
        if spec["task"] == "pre-rating":
            # Pre-rating is deterministic (greedy logprob over digit anchors, no sampling).
            # Multiple seeds give identical values, so use a single seed=1 to save compute.
            run_pre_rating_cell(model, tokenizer, anchor_groups, device, cell_id, spec,
                                all_targets, n_pre_prompts, [1], args.out_dir)
        elif spec["task"] == "free-choice":
            run_free_choice_cell(model, tokenizer, device, cell_id, spec,
                                 all_targets, n_fc_prompts, seeds, args.out_dir)

    # Free GPU memory before exit
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n=== model {args.model_id} complete. all cells written to {args.out_dir} ===")


if __name__ == "__main__":
    main()
