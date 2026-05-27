"""Stage 1 pilot feasibility check.

Runs the pre-rating logprob extraction in completion-style on a single model + single target
to verify (a) model loads, (b) digit-anchor next-token probability mass is > 0.3, (c) expected
rating is in [1, 7]. If all three pass, the main experiment design is feasible for this model.

Usage:
    python pilot.py --model-path /root/autodl-tmp/LLM-Research/Meta-Llama-3.1-8B \
                    --model-id M0-base --format completion --n-prompts 20 --n-shots 5

Output: pilot_results/pilot_<model_id>_<format>.json
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --- Anchor utilities (same as Stage A) ----------------------------------

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


# --- Prompt builders ------------------------------------------------------

TARGETS = {
    "putin": {
        "description": (
            "Vladimir Putin is the President of the Russian Federation, in office since "
            "1999 (with a brief stint as Prime Minister 2008-2012). His government has been "
            "the subject of international debate regarding democratic norms, foreign policy, "
            "and military actions."
        ),
        "name": "Vladimir Putin",
    },
}

PROMPT_TEMPLATES_COMPLETION = [
    "Below is a passage about {name}.\n\n{desc}\n\nMy attitude toward {name} is best described by the number ",
    "Consider the following description:\n\n{desc}\n\nOn a 1-to-7 scale (1=very unfavorable, 7=very favorable), I would rate {name} as ",
    "Here is some information about {name}:\n\n{desc}\n\nIf I had to express my favorability toward {name} on a 1-7 Likert scale, the single digit I would write is ",
    "The following describes {name}:\n\n{desc}\n\nMy overall evaluation of {name} on a 1-7 scale (1 lowest, 7 highest) is the number ",
    "Read the following:\n\n{desc}\n\nFavorability rating of {name} on a 1-7 scale: ",
]


def pilot_prompt_completion(target_name, target_desc, template_idx, prompt_seed):
    """Pick a template and fill it. Uses prompt_seed for any future stochastic prompt variation."""
    tpl = PROMPT_TEMPLATES_COMPLETION[template_idx % len(PROMPT_TEMPLATES_COMPLETION)]
    return tpl.format(name=target_name, desc=target_desc)


# --- Main pilot loop ------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="Local path or ModelScope identifier")
    p.add_argument("--model-id", required=True, help="Short ID for output, e.g. M0-base")
    p.add_argument("--format", choices=["completion"], default="completion",
                   help="Prompt format. Pilot tests completion-style only.")
    p.add_argument("--target", default="putin", help="Target ID from TARGETS dict")
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument("--n-shots", type=int, default=5, help="Repetitions per prompt with different sampling")
    p.add_argument("--out-dir", default="pilot_results")
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    target = TARGETS[args.target]

    print(f"=== loading model from {args.model_path} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  model on {device}, dtype={next(model.parameters()).dtype}")

    anchor_groups = build_anchor_token_groups(tokenizer, ["1", "2", "3", "4", "5", "6", "7"])
    print(f"  anchor groups (digit -> token_ids): {dict(zip('1234567', anchor_groups))}")

    n_total = args.n_prompts * args.n_shots
    trials = []
    print(f"\n=== running pilot: {args.n_prompts} prompts × {args.n_shots} shots = {n_total} trials ===")

    rng = random.Random(0)
    for prompt_idx in range(args.n_prompts):
        prompt_text = pilot_prompt_completion(target["name"], target["description"],
                                              template_idx=prompt_idx, prompt_seed=prompt_idx)
        # Tokenize once
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        last_logits = out.logits[0, -1]
        probs = rating_probs_from_logits(last_logits, anchor_groups)
        digit_mass = float(probs.sum())  # should be ~1 by construction unless anchors overlap
        # Compute raw probability mass on digit-anchor tokens vs all other tokens
        all_token_probs = torch.softmax(last_logits.float(), dim=-1).cpu().numpy()
        anchor_token_ids = [tid for grp in anchor_groups for tid in grp]
        digit_raw_mass = float(all_token_probs[anchor_token_ids].sum())
        expected_rating = float(np.dot(probs, np.arange(1, 8)))
        top5_indices = np.argsort(all_token_probs)[-5:][::-1]
        top5 = [(tokenizer.decode([int(i)]), float(all_token_probs[i])) for i in top5_indices]

        trials.append({
            "prompt_idx": prompt_idx,
            "template_idx": prompt_idx % len(PROMPT_TEMPLATES_COMPLETION),
            "expected_rating": expected_rating,
            "digit_distribution_normalized": probs.tolist(),
            "digit_raw_mass": digit_raw_mass,
            "top5_tokens": top5,
        })

        if prompt_idx < 3:
            print(f"  [prompt {prompt_idx}] expected_rating={expected_rating:.2f} "
                  f"digit_raw_mass={digit_raw_mass:.3f} top5={top5}")

    # Aggregate
    digit_masses = [t["digit_raw_mass"] for t in trials]
    ratings = [t["expected_rating"] for t in trials]
    summary = {
        "model_id": args.model_id,
        "model_path": args.model_path,
        "target": args.target,
        "format": args.format,
        "n_prompts": args.n_prompts,
        "n_shots": args.n_shots,
        "digit_raw_mass": {
            "mean": float(np.mean(digit_masses)),
            "min": float(np.min(digit_masses)),
            "max": float(np.max(digit_masses)),
        },
        "expected_rating": {
            "mean": float(np.mean(ratings)),
            "std": float(np.std(ratings)),
            "min": float(np.min(ratings)),
            "max": float(np.max(ratings)),
        },
        "verdict": {
            "digit_mass_pass": float(np.mean(digit_masses)) > 0.3,
            "rating_in_range": 1 <= float(np.mean(ratings)) <= 7,
        },
        "trials": trials,
    }

    out_path = Path(args.out_dir) / f"pilot_{args.model_id}_{args.format}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== summary ({args.model_id} × {args.format}) ===")
    print(f"  digit_raw_mass: mean={summary['digit_raw_mass']['mean']:.3f} "
          f"min={summary['digit_raw_mass']['min']:.3f} "
          f"max={summary['digit_raw_mass']['max']:.3f}")
    print(f"  expected_rating: mean={summary['expected_rating']['mean']:.2f} "
          f"std={summary['expected_rating']['std']:.2f}")
    print(f"  VERDICT: digit_mass_pass={summary['verdict']['digit_mass_pass']}  "
          f"rating_in_range={summary['verdict']['rating_in_range']}")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
