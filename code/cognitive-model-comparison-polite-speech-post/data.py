"""Load trial-level data for the cognitive-model comparison.

Reads cell-target JSONs from two result directories and produces a flat
pandas DataFrame with one row per trial. Designed to be the single data-loader
for fit.py, recovery.py, and identifiability.py.

Trial-level fitting (not cell-level aggregation) is the design choice per
Lu/Yi/Zhang 2019 standard practice and per the user's explicit preference.
"""
import json
from pathlib import Path
import pandas as pd

# Cell-to-(stage, condition) mapping — keep in sync with analyze.py.
# Source: experiments/code/per-stage-localization-forced-honesty-overshoot/analyze.py
CELL_TO_STAGE_COND = {
    "C0":  ("M0", "default"),
    "C1":  ("M1", "default"),
    "C2":  ("M2", "default"),
    "C3":  ("M3", "default-completion"),
    "C4":  ("M3", "default"),
    "C5":  ("M3", "forced"),
    "C8":  ("M3", "control"),
    "C9":  ("M0", "control"),
    "C10": ("M0", "forced"),
    "C11": ("M1", "control"),
    "C12": ("M1", "forced"),
    "C13": ("M2", "control"),
    "C14": ("M2", "forced"),
    "C20": ("M3", "alt-control-truthful"),
    "C21": ("M3", "alt-control-objective"),
    "C22": ("M3", "anti-politeness-strong"),
    "C23": ("M3", "anti-politeness-mild"),
    "C24": ("M3", "anti-politeness-no-neg-words"),
    "C25": ("M3", "forced-completion"),
    "C26": ("M3", "control-completion"),
}

# Format mapping per cell (completion vs chat vs prefix).
# "chat" = uses M3's chat template via apply_chat_template
# "completion" = no template, raw prompt completion
# "prefix" = plain-text "Instruction: <system>\n\n<prompt>" injection (M0 etc.)
CELL_TO_FORMAT = {
    "C0":  "completion",  "C1":  "completion",  "C2":  "completion",  "C3":  "completion",
    "C4":  "chat",        "C5":  "chat",        "C8":  "chat",
    "C9":  "prefix",      "C10": "prefix",
    "C11": "chat",        "C12": "chat",        "C13": "chat",        "C14": "chat",
    "C20": "chat",        "C21": "chat",        "C22": "chat",        "C23": "chat",
    "C24": "chat",
    "C25": "prefix",      "C26": "prefix",      # M3 prefix-injected (Round-2 completion mode)
}

FICTIONAL = {"adirenia", "che_pact", "vellinkov", "khantelan"}
CALIBRATION_POS = {"clean_drinking_water"}
CALIBRATION_NEG = {"tooth_decay"}
CALIBRATION_NEUTRAL = {"standard_brick"}
CALIBRATION = CALIBRATION_POS | CALIBRATION_NEG | CALIBRATION_NEUTRAL


def load_trials(*dirs: Path, min_mass: float = 0.0,
                targets: set | None = None,
                exclude_bursovet: bool = True,
                deduplicate: bool = True) -> pd.DataFrame:
    """Load all trial JSONs from one or more results directories.

    Parameters
    ----------
    dirs : paths to cell_*_target_*.json files
    min_mass : exclude trials with digit_raw_mass below this threshold
    targets : if set, restrict to these target IDs
    exclude_bursovet : drop the bursovet boundary-case target
    deduplicate : if True (default), collapse the 10 deterministic repeats per
        template down to 1 observation per (target, cell, template_idx). The
        prompt-generation code uses tpl = TEMPLATES[prompt_idx % 5] with 50
        prompts per cell, which means prompts 0/5/10/.../45 share template 0
        and produce IDENTICAL logits (greedy decoding, n_seeds=1). Treating
        the 50 as independent observations inflates effective N by 10x. We
        deduplicate by default; pass deduplicate=False to inspect the raw
        repeats.

    Returns
    -------
    pd.DataFrame with columns:
        target_id, target_role, cell, stage, condition, format,
        seed, prompt_idx, template_idx, rating, digit_raw_mass
    """
    rows = []
    for d in dirs:
        for p in sorted(Path(d).glob("cell_*_target_*.json")):
            data = json.load(open(p))
            for t in data["trials"]:
                rows.append(t)
    df = pd.DataFrame(rows)

    # Add stage / condition / format derived columns
    df["stage"] = df["cell"].map(lambda c: CELL_TO_STAGE_COND.get(c, (None, None))[0])
    df["condition"] = df["cell"].map(lambda c: CELL_TO_STAGE_COND.get(c, (None, None))[1])
    df["format"] = df["cell"].map(lambda c: CELL_TO_FORMAT.get(c))

    # Template index: prompt_idx // 10 (5 templates × 10 prompts each, matches analyze.py convention)
    df["template_idx"] = df["prompt_idx"] // 10

    # Filter
    if exclude_bursovet:
        df = df[df["target_id"] != "bursovet"]
    if targets is not None:
        df = df[df["target_id"].isin(targets)]
    if min_mass > 0:
        before = len(df)
        df = df[df["digit_raw_mass"] >= min_mass]
        print(f"[min_mass={min_mass}] kept {len(df)}/{before} trials")

    # Drop trials with missing stage/condition (cells not in registry)
    n_before = len(df)
    df = df.dropna(subset=["stage", "condition"]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"[unknown-cell filter] kept {len(df)}/{n_before} trials")

    # Deduplicate identical-template repeats. With n_seeds=1 and greedy
    # decoding, prompts at the same template_idx within a (target, cell)
    # produce identical logits. Treating them as independent inflates
    # effective N by ~10x.
    if deduplicate:
        before = len(df)
        df = (df.sort_values("prompt_idx")
                .drop_duplicates(subset=["target_id", "cell", "template_idx"], keep="first")
                .reset_index(drop=True))
        print(f"[dedup template repeats] kept {len(df)}/{before} unique (target, cell, template) observations")

    return df


def base_ratings(df: pd.DataFrame) -> dict:
    """Compute the M0-default (C0) per-target base rating, used as the evidence-only anchor.

    Returns
    -------
    dict mapping target_id -> mean rating in C0 (M0 base, completion-style default).
    """
    base = df[df["cell"] == "C0"].groupby("target_id")["rating"].mean().to_dict()
    return base


if __name__ == "__main__":
    # Quick sanity check
    DIR_NEW = Path("experiments/results/per-stage-localization-forced-honesty-overshoot")
    DIR_PRIOR = Path("experiments/results/stage-base-vs-instruct-positivity-prior")
    df = load_trials(DIR_NEW, DIR_PRIOR)
    print(f"Loaded {len(df)} trials, {df['cell'].nunique()} cells, {df['target_id'].nunique()} targets")
    print("\nPer (stage, condition) cell counts:")
    print(df.groupby(["stage", "condition"]).size())
    print("\nBase ratings (M0-default):")
    for tgt, br in sorted(base_ratings(df).items()):
        print(f"  {tgt:25s}: {br:.3f}")
