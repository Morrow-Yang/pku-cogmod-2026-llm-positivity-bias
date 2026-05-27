"""Item 6 from /review weakness W6: report per-template beta_L estimates.

Reviewer concern: within-(cell,target) template SD (0.54) > cell-grand-mean SD
(0.41), so prompt-template wording accounts for more variance than the
manipulation x target combination. The headline beta_L = -1.23 is averaged
across 5 templates; the dominance claim is currently template-conditional.

Fix: re-fit the factorial OLS per template (5 fits, one per template index
0..4) and report beta_L per template. If all 5 templates show beta_L as the
largest negative coefficient, the dominance claim is template-robust. If
2/5 templates show beta_A > beta_L, the claim needs qualification.

Runs on the existing E5 trial JSONs; no new compute.
"""
from __future__ import annotations
import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RESULTS_DIR = Path("experiments/results/post-review-extensions")
OUT_PATH = RESULTS_DIR / "per_template_beta_l.json"

FICTIONAL = ["adirenia", "che_pact", "khantelan", "vellinkov"]


def load_e5():
    rows = []
    for f in sorted(glob.glob(str(RESULTS_DIR / "cell_E5_*.json"))):
        d = json.load(open(f))
        for t in d["trials"]:
            rows.append({
                "cell": t["cell"], "target": t["target_id"],
                "template_idx": t["template_idx"],
                "rating": t["rating"],
                "H": int(t["cell"][3]), "A": int(t["cell"][4]),
                "L": int(t["cell"][5]),
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("Item 6: Per-template factorial decomposition (fictional-4 targets)")
    print("=" * 70)
    df = load_e5()
    fic = df[df["target"].isin(FICTIONAL)].copy()
    n_templates = fic["template_idx"].nunique()
    print(f"N total trials = {len(df)}, fictional-4 = {len(fic)}, "
          f"templates = {n_templates}\n")

    formula = "rating ~ H*A*L + C(target)"
    rows = []
    for tpl in sorted(fic["template_idx"].unique()):
        sub = fic[fic["template_idx"] == tpl].copy()
        # One observation per (target, cell) at this template -> 32 rows
        m = smf.ols(formula, data=sub).fit()
        rows.append({
            "template": int(tpl),
            "n": int(len(sub)),
            "rsq": float(m.rsquared),
            "beta_H": float(m.params.get("H", np.nan)),
            "beta_A": float(m.params.get("A", np.nan)),
            "beta_L": float(m.params.get("L", np.nan)),
            "p_L": float(m.pvalues.get("L", np.nan)),
            "ci_L_lo": float(m.conf_int().loc["L", 0]) if "L" in m.params.index else np.nan,
            "ci_L_hi": float(m.conf_int().loc["L", 1]) if "L" in m.params.index else np.nan,
        })

    print(f"{'tpl':>4} {'n':>4} {'R^2':>6} {'beta_H':>9} {'beta_A':>9} "
          f"{'beta_L':>9} {'95% CI on beta_L':>22} {'p_L':>9}  "
          f"{'L dominant?':>12}")
    n_L_dominant = 0
    for r in rows:
        ranked = sorted(
            [("L", abs(r["beta_L"])), ("A", abs(r["beta_A"])), ("H", abs(r["beta_H"]))],
            key=lambda x: -x[1])
        dominant = ranked[0][0]
        is_L_dominant = dominant == "L"
        if is_L_dominant:
            n_L_dominant += 1
        print(f"{r['template']:>4} {r['n']:>4} {r['rsq']:>6.3f} "
              f"{r['beta_H']:>+9.3f} {r['beta_A']:>+9.3f} "
              f"{r['beta_L']:>+9.3f}  "
              f"[{r['ci_L_lo']:>+.2f},{r['ci_L_hi']:>+.2f}] "
              f"{r['p_L']:>9.4g}  "
              f"{'YES' if is_L_dominant else 'NO ({})'.format(dominant):>12}")

    print()
    print(f"L is the dominant (largest-magnitude) negative coefficient in "
          f"{n_L_dominant}/{len(rows)} templates.")

    # Also report the pooled (across templates) for reference
    pooled = smf.ols(formula, data=fic).fit()
    print()
    print(f"Pooled (all 5 templates aggregated):")
    print(f"  beta_H = {pooled.params['H']:+.3f}, "
          f"beta_A = {pooled.params['A']:+.3f}, "
          f"beta_L = {pooled.params['L']:+.3f}")
    print(f"  Note: aggregation here is over trials, not cell-target means")
    print(f"  (each (cell, target, template) is one observation).")

    out = {
        "n_templates_tested": len(rows),
        "n_templates_with_L_dominant": n_L_dominant,
        "per_template": rows,
        "interpretation": (
            f"beta_L is the largest-magnitude negative coefficient in "
            f"{n_L_dominant}/{len(rows)} templates. The headline beta_L = -1.23 "
            f"(averaged across templates) is therefore "
            f"{'template-robust' if n_L_dominant == len(rows) else 'template-conditional'}."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
