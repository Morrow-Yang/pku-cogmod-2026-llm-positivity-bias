"""Stage A — Behavioral Replication of Lehr 2025 free-choice moderator on Llama-3.1-8B-Instruct.

v2 (post-Review-LLM): three corrections vs v1:
  1. Anchor token IDs: collect all single-token forms (with/without leading space/newline)
     and aggregate logits via logsumexp before softmax.
  2. Post-rating: fresh conversation with rating-system message; essay is included
     inside the user message as context.
  3. Free-choice: Lehr-faithful — model FIRST declares CHOICE: POSITIVE/NEGATIVE,
     then writes; observed (not assigned) valence drives cell membership in analysis.

Per (target × seed):
  - n_free_choice_per_target trials of free-choice (mixed valence; split by observed)
  - n_commanded_per_valence_per_target trials each of commanded-positive and commanded-negative

Output: results/<output_dir>/seed_<N>_target_<id>.json — one file per (seed × target).
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


# --- System prompts --------------------------------------------------------

SYSTEM_RATING = (
    "You are a research participant. Read the question carefully and answer with a single number only."
)
SYSTEM_ESSAY = (
    "You are a research participant taking part in a writing study. Follow the instructions carefully."
)


# --- Anchor token bookkeeping ----------------------------------------------

def build_anchor_token_groups(tokenizer, anchors: list[str]) -> list[list[int]]:
    """For each digit anchor (e.g. '1'), collect *all* single-token IDs that decode to it
    in any common surface form (no leading space, leading space, leading newline). When
    aggregating logits, we sum across these forms (logsumexp) — this fixes the bug where
    Llama's chat template causes the next predicted token to be ' 1' (with leading space)
    rather than the bare '1' my v1 code looked for.
    """
    groups: list[list[int]] = []
    for digit in anchors:
        seen: set[int] = set()
        for prefix in ("", " ", "\n", "\n\n"):
            ids = tokenizer(prefix + digit, add_special_tokens=False).input_ids
            if len(ids) == 1:
                seen.add(int(ids[0]))
        if not seen:
            # Fallback: use the first sub-token of the multi-token encoding
            ids = tokenizer(digit, add_special_tokens=False).input_ids
            if ids:
                seen.add(int(ids[0]))
        groups.append(sorted(seen))
    return groups


def rating_probs_from_logits(last_logits: torch.Tensor, anchor_groups: list[list[int]]) -> np.ndarray:
    """logsumexp across token-form group per digit, then softmax across digits."""
    digit_logits = torch.stack([
        torch.logsumexp(last_logits[ids], dim=0) for ids in anchor_groups
    ])
    return torch.softmax(digit_logits, dim=-1).cpu().numpy()


# --- Prompt builders -------------------------------------------------------

def pre_rating_messages(target_desc: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_RATING},
        {"role": "user", "content": (
            f"Consider the following:\n\n{target_desc}\n\n"
            f"On a 1-7 scale where 1 means 'very negative impression' and 7 means "
            f"'very positive impression', how do you rate this? Answer with ONLY the digit 1-7."
        )},
    ]


def commanded_essay_messages(target_desc: str, target_name: str, valence: str) -> list[dict[str, str]]:
    side = (
        "highlighting their strengths and good qualities" if valence == "positive"
        else "highlighting weaknesses and concerns"
    )
    return [
        {"role": "system", "content": SYSTEM_ESSAY},
        {"role": "user", "content": (
            f"Consider the following:\n\n{target_desc}\n\n"
            f"You are instructed to write a {valence} essay (about 200 words, {side}) "
            f"about {target_name}. Please write the essay now."
        )},
    ]


def free_choice_essay_messages(target_desc: str, target_name: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_ESSAY},
        {"role": "user", "content": (
            f"Consider the following:\n\n{target_desc}\n\n"
            f"You may write either a POSITIVE essay (about 200 words, highlighting strengths "
            f"and good qualities) or a NEGATIVE essay (about 200 words, highlighting weaknesses "
            f"and concerns) about {target_name}.\n\n"
            f"FIRST, on a line by itself, declare your choice as EXACTLY one of:\n"
            f"CHOICE: POSITIVE\n"
            f"CHOICE: NEGATIVE\n\n"
            f"THEN, on subsequent lines, write the essay you chose. Do not include "
            f"any other text before the CHOICE line."
        )},
    ]


def post_rating_messages(target_desc: str, target_name: str, valence: str,
                          framing: str, essay: str) -> list[dict[str, str]]:
    if framing == "free":
        intro = f"You just chose to write the following {valence} essay about {target_name}:"
    else:
        intro = f"You just wrote the following {valence} essay about {target_name} as instructed:"
    return [
        {"role": "system", "content": SYSTEM_RATING},
        {"role": "user", "content": (
            f"Consider the following:\n\n{target_desc}\n\n"
            f"{intro}\n\n"
            f"--- BEGIN ESSAY ---\n{essay}\n--- END ESSAY ---\n\n"
            f"On a 1-7 scale where 1 means 'very negative impression' and 7 means "
            f"'very positive impression', how do you NOW rate this? Answer with ONLY the digit 1-7."
        )},
    ]


CHOICE_RE = re.compile(r"^\s*CHOICE\s*:\s*(POSITIVE|NEGATIVE)\b", re.IGNORECASE | re.MULTILINE)


def parse_free_choice(output: str) -> tuple[str | None, str]:
    """Return (valence, essay_text). valence is 'positive' / 'negative' / None."""
    m = CHOICE_RE.search(output)
    if not m:
        return None, output.strip()
    valence = m.group(1).lower()
    # Strip the CHOICE: line(s) from the output to get the essay body.
    essay_text = CHOICE_RE.sub("", output, count=1).lstrip("\n :").strip()
    return valence, essay_text


# --- Inference primitives --------------------------------------------------

def get_next_token_logits(model, tokenizer, messages: list[dict[str, str]], device) -> torch.Tensor:
    """Apply chat template + add_generation_prompt + forward pass; return final-position logits."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
    return out.logits[0, -1, :].float()


def generate_essay(model, tokenizer, messages, max_new_tokens, temperature, device) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out_ids[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


# --- Trial-level functions -------------------------------------------------

def rate_attitude(model, tokenizer, target_desc, anchor_groups, device,
                   essay_context: dict | None = None) -> dict:
    """Single attitude rating. If essay_context is provided (post-rating), include the essay."""
    if essay_context is None:
        msgs = pre_rating_messages(target_desc)
    else:
        msgs = post_rating_messages(
            target_desc,
            essay_context["target_name"],
            essay_context["valence"],
            essay_context["framing"],
            essay_context["essay"],
        )
    last_logits = get_next_token_logits(model, tokenizer, msgs, device)
    probs = rating_probs_from_logits(last_logits, anchor_groups)
    expected = float(np.dot(probs, np.arange(1, len(anchor_groups) + 1)))
    return {"probs": probs.tolist(), "expected": expected}


def run_commanded_trial(model, tokenizer, target_desc, target_name,
                         valence, cfg, anchor_groups, device) -> dict:
    pre = rate_attitude(model, tokenizer, target_desc, anchor_groups, device)
    essay = generate_essay(
        model, tokenizer,
        commanded_essay_messages(target_desc, target_name, valence),
        max_new_tokens=cfg["max_essay_tokens"],
        temperature=cfg["essay_temperature"],
        device=device,
    )
    post = rate_attitude(
        model, tokenizer, target_desc, anchor_groups, device,
        essay_context=dict(target_name=target_name, valence=valence, framing="commanded", essay=essay),
    )
    return dict(
        framing="commanded",
        observed_valence=valence,
        assigned_valence=valence,
        pre_expected=pre["expected"],
        pre_probs=pre["probs"],
        post_expected=post["expected"],
        post_probs=post["probs"],
        shift=post["expected"] - pre["expected"],
        essay_excerpt=essay[:280],
        essay_len_chars=len(essay),
        choice_parse_ok=True,
    )


def run_free_choice_trial(model, tokenizer, target_desc, target_name,
                           cfg, anchor_groups, device) -> dict:
    pre = rate_attitude(model, tokenizer, target_desc, anchor_groups, device)
    valence = None
    essay = ""
    raw_output = ""
    n_retries = 0
    for attempt in range(cfg.get("free_choice_max_retries", 1) + 1):
        raw_output = generate_essay(
            model, tokenizer,
            free_choice_essay_messages(target_desc, target_name),
            max_new_tokens=cfg["max_essay_tokens"],
            temperature=cfg["essay_temperature"],
            device=device,
        )
        valence, essay = parse_free_choice(raw_output)
        if valence is not None:
            break
        n_retries += 1
    if valence is None:
        # Ambiguous — include as such; analysis will exclude these from primary tests.
        return dict(
            framing="free",
            observed_valence=None,
            assigned_valence=None,
            pre_expected=pre["expected"],
            pre_probs=pre["probs"],
            post_expected=float("nan"),
            post_probs=[],
            shift=float("nan"),
            essay_excerpt=raw_output[:280],
            essay_len_chars=len(raw_output),
            choice_parse_ok=False,
            n_retries=n_retries,
        )
    post = rate_attitude(
        model, tokenizer, target_desc, anchor_groups, device,
        essay_context=dict(target_name=target_name, valence=valence, framing="free", essay=essay),
    )
    return dict(
        framing="free",
        observed_valence=valence,
        assigned_valence=None,
        pre_expected=pre["expected"],
        pre_probs=pre["probs"],
        post_expected=post["expected"],
        post_probs=post["probs"],
        shift=post["expected"] - pre["expected"],
        essay_excerpt=essay[:280],
        essay_len_chars=len(essay),
        choice_parse_ok=True,
        n_retries=n_retries,
    )


# --- Main loop -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"=== loading model ({cfg['model_path']}) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], dtype=torch.float16, device_map="auto"
    ).eval()
    device = next(model.parameters()).device
    print(f"model on {device}; VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    anchor_groups = build_anchor_token_groups(tokenizer, cfg["attitude_rating_anchors"])
    print(f"anchor token groups (digit -> token-ids):", flush=True)
    for digit, ids in zip(cfg["attitude_rating_anchors"], anchor_groups):
        decoded = [tokenizer.decode([i]) for i in ids]
        print(f"  '{digit}' -> {ids} (decoded: {decoded})", flush=True)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    t0_total = time.time()
    grand_total = 0

    for tgt in cfg["targets"]:
        target_id = tgt["id"]
        target_name = tgt["name"]
        target_desc = tgt["description"].strip()
        target_role = tgt.get("role", "primary")
        print(f"\n========== TARGET {target_id} ({target_role}: {target_name}) ==========", flush=True)
        target_results: list[dict] = []
        t0_target = time.time()

        # --- Commanded-positive ---
        n_cmd = cfg["n_commanded_per_valence_per_target"]
        for valence in ("positive", "negative"):
            cell_label = f"commanded_{valence}"
            print(f"\n--- {target_id} / {cell_label} (n={n_cmd}) ---", flush=True)
            for i in range(n_cmd):
                t_trial = time.time()
                try:
                    rec = run_commanded_trial(model, tokenizer, target_desc, target_name,
                                               valence, cfg, anchor_groups, device)
                except Exception as e:
                    rec = {"error": str(e), "framing": "commanded", "observed_valence": valence,
                           "assigned_valence": valence, "shift": float("nan"), "pre_expected": float("nan"),
                           "post_expected": float("nan")}
                rec.update(dict(
                    target_id=target_id, target_name=target_name, target_role=target_role,
                    cell_planned=cell_label, prompt_idx=i, seed=seed,
                    trial_seconds=time.time() - t_trial,
                ))
                target_results.append(rec)
                if (i + 1) % 10 == 0 or i == 0:
                    shifts = [r["shift"] for r in target_results
                              if r.get("cell_planned") == cell_label and not np.isnan(r.get("shift", np.nan))]
                    mean_shift = float(np.mean(shifts)) if shifts else float("nan")
                    print(f"  [{i+1}/{n_cmd}] mean shift = {mean_shift:+.3f}; tgt-elapsed {time.time()-t0_target:.0f}s", flush=True)

        # --- Free-choice (mixed; observed-valence labels emerge from parsing) ---
        n_free = cfg["n_free_choice_per_target"]
        cell_label = "free_mixed"
        print(f"\n--- {target_id} / {cell_label} (n={n_free}) ---", flush=True)
        free_choices = {"positive": 0, "negative": 0, "ambiguous": 0}
        for i in range(n_free):
            t_trial = time.time()
            try:
                rec = run_free_choice_trial(model, tokenizer, target_desc, target_name,
                                             cfg, anchor_groups, device)
            except Exception as e:
                rec = {"error": str(e), "framing": "free", "observed_valence": None,
                       "shift": float("nan"), "pre_expected": float("nan"), "post_expected": float("nan"),
                       "choice_parse_ok": False}
            rec.update(dict(
                target_id=target_id, target_name=target_name, target_role=target_role,
                cell_planned=cell_label, prompt_idx=i, seed=seed,
                trial_seconds=time.time() - t_trial,
            ))
            target_results.append(rec)
            if rec.get("observed_valence") in ("positive", "negative"):
                free_choices[rec["observed_valence"]] += 1
            else:
                free_choices["ambiguous"] += 1
            if (i + 1) % 10 == 0 or i == 0:
                fp = free_choices["positive"]; fn = free_choices["negative"]; fa = free_choices["ambiguous"]
                print(f"  [{i+1}/{n_free}] choice tally so far: pos={fp} neg={fn} ambig={fa}; tgt-elapsed {time.time()-t0_target:.0f}s", flush=True)

        out_path = out_dir / f"seed_{seed}_target_{target_id}.json"
        with open(out_path, "w") as f:
            json.dump({"config": cfg, "seed": seed, "target_id": target_id,
                       "free_choice_tally": free_choices,
                       "trials": target_results}, f, indent=2)
        grand_total += len(target_results)
        print(f"\n=== wrote {out_path} ({len(target_results)} trials, "
              f"target-elapsed {time.time()-t0_target:.0f}s, grand total {grand_total}) ===", flush=True)

    elapsed = time.time() - t0_total
    print(f"\n=== DONE seed={seed} | {grand_total} trials | {elapsed:.0f}s wall ===", flush=True)


if __name__ == "__main__":
    main()
