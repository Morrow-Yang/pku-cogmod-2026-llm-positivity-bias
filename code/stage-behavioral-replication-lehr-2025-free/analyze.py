"""Stage A v2 post-run analysis.

Reads results/seed_*_target_*.json (Lehr-faithful design: free-choice cells are split
by OBSERVED valence rather than assigned). Fits per-target regression on essay-content
valence × choice-framing. Applies Holm-Bonferroni across primary targets. Prints the
decision-gate verdict.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg["output_dir"])
    seed_files = sorted(out_dir.glob("seed_*_target_*.json"))
    if not seed_files:
        print("No result files found in", out_dir)
        return

    rows = []
    for sf in seed_files:
        d = json.load(open(sf))
        for r in d["trials"]:
            if "shift" not in r or r.get("observed_valence") not in ("positive", "negative"):
                continue
            if np.isnan(r.get("shift", float("nan"))):
                continue
            rows.append({
                "target_id": r["target_id"],
                "target_role": r["target_role"],
                "framing": r["framing"],
                "observed_valence": r["observed_valence"],
                "assigned_valence": r.get("assigned_valence"),
                "shift": r["shift"],
                "pre": r["pre_expected"],
                "post": r["post_expected"],
                "essay_chars": r.get("essay_len_chars", 0),
                "prompt_idx": r["prompt_idx"],
                "seed": r["seed"],
                "choice_parse_ok": r.get("choice_parse_ok", True),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid trials in any seed file.")
        return
    print(f"loaded {len(df)} valid trials from {len(seed_files)} result files\n")

    # Free-choice tally
    print("=== free-choice tally (per target × seed) ===")
    fc = df[df["framing"] == "free"].groupby(["target_id", "seed"]).agg(
        n_positive=("observed_valence", lambda s: (s == "positive").sum()),
        n_negative=("observed_valence", lambda s: (s == "negative").sum()),
        n_total=("observed_valence", "size"),
    )
    print(fc, "\n")

    # Cell means
    print("=== cell means (mean shift; n) ===")
    summary = df.groupby(["target_id", "target_role", "framing", "observed_valence"]).agg(
        mean_shift=("shift", "mean"),
        sd_shift=("shift", "std"),
        n=("shift", "size"),
    ).round(3)
    print(summary, "\n")

    df["valence_signed"] = df["observed_valence"].map({"positive": 1, "negative": -1})
    df["framing_signed"] = df["framing"].map({"free": 1, "commanded": -1})

    per_target_results = []
    print("=== per-target regressions: shift ~ valence × framing ===")
    for tid in df["target_id"].unique():
        d = df[df["target_id"] == tid]
        role = d["target_role"].iloc[0]
        try:
            model = smf.ols("shift ~ valence_signed * framing_signed", data=d).fit()
            main = model.params["valence_signed"]
            main_p = model.pvalues["valence_signed"]
            interaction = model.params["valence_signed:framing_signed"]
            interaction_p = model.pvalues["valence_signed:framing_signed"]
        except Exception as e:
            print(f"  target={tid} ERROR: {e}")
            main = main_p = interaction = interaction_p = float("nan")
        per_target_results.append({
            "target_id": tid, "role": role,
            "main_valence_b": main, "main_valence_p": main_p,
            "interaction_b": interaction, "interaction_p": interaction_p,
            "n": len(d),
        })
        print(f"  target={tid:<12s} role={role}  n={len(d)}")
        print(f"     main valence:  b={main:+.3f}, p={main_p:.4f}")
        print(f"     interaction:   b={interaction:+.3f}, p={interaction_p:.4f}")

    primaries = [r for r in per_target_results if r["role"] == "primary"]
    print(f"\n=== Holm-Bonferroni across {len(primaries)} primary targets (interaction term) ===")
    p_sorted = sorted(primaries, key=lambda r: r["interaction_p"] if r["interaction_p"] == r["interaction_p"] else 1)
    m = len(p_sorted)
    any_significant = False
    for i, r in enumerate(p_sorted):
        alpha_adjusted = 0.05 / max(1, m - i)
        sig = "✓ SIGNIFICANT" if r["interaction_p"] < alpha_adjusted else "✗ ns"
        if r["interaction_p"] < alpha_adjusted:
            any_significant = True
        print(f"  rank {i+1}: {r['target_id']:<12s} interaction_p={r['interaction_p']:.4f}  threshold={alpha_adjusted:.4f}  {sig}")

    print()
    print("============================================================")
    print("                     STAGE A VERDICT                         ")
    print("============================================================")
    if any_significant:
        print("✓ DECISION GATE PASSED")
        print("  At least one primary fictional target shows a Holm-Bonferroni-corrected")
        print("  significant essay-valence × choice-framing interaction.")
        print("  → eligible to proceed to Stage B (sufficiency test)")
    else:
        print("✗ DECISION GATE FAILED")
        print("  No primary fictional target shows the Lehr signature.")
        print("  → Llama-3.1-8B-Instruct does not exhibit the GPT-4o dissonance signature")
        print("    on neutral targets. Write up as a publishable model-scale divergence.")
    print("============================================================")

    df.to_csv(out_dir / "all_trials.csv", index=False)
    summary.to_csv(out_dir / "cell_summary.csv")
    print(f"\nwrote {out_dir/'all_trials.csv'} and {out_dir/'cell_summary.csv'}")


if __name__ == "__main__":
    main()
