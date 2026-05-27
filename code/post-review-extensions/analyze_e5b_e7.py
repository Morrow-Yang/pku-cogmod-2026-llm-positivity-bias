"""Analyze E5b (factorial replication on 15 pseudowords) and E7 (positive-
license asymmetry test).

Addresses /review weaknesses:
  W3: factorial primary analysis used only 4 fictional targets -> replicate
      on 15 pseudoword targets. With 15 clusters, the cluster-bootstrap CI
      is now within the marginally-acceptable range (still flagged but no
      longer dismissed).
  W2: the "asymmetric" claim has no positive-license control. We run two
      positive-license variants:
        pos  = "even if positive or praiseworthy" (simple valence mirror)
        pos2 = "even if unusually enthusiastic or excessively flattering"
               (restored suppression implication)
      Asymmetry test (pre-registered per design-review): difference test
      Delta_neg - Delta_pos with cluster-bootstrap CI on per-target deltas.

Outputs analyze_e5b_e7.json with the factorial decomposition on
pseudowords + the asymmetry contrast on both stimulus sets.
"""
from __future__ import annotations
import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

RESULTS_DIR = Path("experiments/results/post-review-extensions")
OUT_PATH = RESULTS_DIR / "analyze_e5b_e7.json"

FICTIONAL_4 = ["adirenia", "che_pact", "khantelan", "vellinkov"]
ORIG_7 = FICTIONAL_4 + ["clean_drinking_water", "tooth_decay", "standard_brick"]


def load_cells_by_prefix(prefix: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RESULTS_DIR / f"cell_{prefix}*.json"))):
        d = json.load(open(f))
        for t in d["trials"]:
            rows.append({
                "cell": t["cell"], "target": t["target_id"],
                "system": t["system"], "template_idx": t["template_idx"],
                "rating": t["rating"], "digit_raw_mass": t["digit_raw_mass"],
            })
    return pd.DataFrame(rows)


# ====================================================================
# Part 1: E5b factorial decomposition on 15 pseudoword targets (W3 fix)
# ====================================================================

def analyze_e5b():
    print("=" * 70)
    print("Part 1: E5b factorial on 15 pseudoword targets (W3 fix)")
    print("=" * 70)
    df = load_cells_by_prefix("E5b_")
    if df.empty:
        print("WARNING: no E5b data found; skipping Part 1.")
        return {}
    df["H"] = df["cell"].str[4].astype(int)
    df["A"] = df["cell"].str[5].astype(int)
    df["L"] = df["cell"].str[6].astype(int)

    pseudowords = sorted(df["target"].unique().tolist())
    print(f"Pseudoword targets: {len(pseudowords)}")
    print(f"Total E5b trial rows: {len(df)}")

    # Aggregate templates -> (cell, target) means
    agg = (df.groupby(["cell", "target", "H", "A", "L"])["rating"]
             .mean().reset_index())
    print(f"Aggregated to {len(agg)} (cell, target) means.\n")

    print("8 cell means on pseudowords:")
    cell_table = {}
    for cell in sorted(df["cell"].unique()):
        H, A, L = int(cell[4]), int(cell[5]), int(cell[6])
        m = float(agg[agg["cell"] == cell]["rating"].mean())
        print(f"  {cell} (H={H} A={A} L={L}):  mean = {m:.3f}")
        cell_table[cell] = {"H": H, "A": A, "L": L, "mean_pseudowords": m}

    # Primary factorial fit with target fixed effects
    formula = "rating ~ H*A*L + C(target)"
    m_psw = smf.ols(formula, data=agg).fit()
    print(f"\n=== Pseudoword-15 factorial (target FE, n={len(agg)}, R^2={m_psw.rsquared:.4f}) ===")
    terms = ["H", "A", "L", "H:A", "H:L", "A:L", "H:A:L"]
    coefs_psw = {}
    for t in terms:
        if t not in m_psw.params.index:
            continue
        coef = m_psw.params[t]
        se = m_psw.bse[t]
        ci = m_psw.conf_int().loc[t]
        p = m_psw.pvalues[t]
        tag = " [PRE-REG]" if t == "L" else ""
        print(f"  {t:<10s} {coef:>+9.4f}  [{ci[0]:+.3f},{ci[1]:+.3f}]  p={p:.4g}{tag}")
        coefs_psw[t] = {"coef": float(coef), "se": float(se),
                        "ci": [float(ci[0]), float(ci[1])], "p": float(p)}

    # Compare to the E5 fictional-4 fit (from prior analysis)
    e5_known = {
        "H": -0.17, "A": -0.68, "L": -1.23,
        "H:A": +0.35, "H:L": +0.52, "A:L": +0.55, "H:A:L": -0.32,
    }
    print(f"\n=== Comparison: E5b (15 pseudowords) vs E5 (4 fictional) coefficients ===")
    print(f"  {'term':<10s} {'E5b psw15':>11s} {'E5 fic4':>10s} {'difference':>11s}")
    for t in terms:
        if t not in coefs_psw:
            continue
        diff = coefs_psw[t]["coef"] - e5_known[t]
        print(f"  {t:<10s} {coefs_psw[t]['coef']:>+11.3f} {e5_known[t]:>+10.3f}  {diff:>+11.3f}")

    return {
        "n_pseudoword_targets": len(pseudowords),
        "cell_means_pseudowords": cell_table,
        "factorial_coefficients_psw15": coefs_psw,
        "factorial_coefficients_e5_fic4_reference": e5_known,
        "L_dominant_in_psw15": (
            abs(coefs_psw.get("L", {}).get("coef", 0)) >
            max(abs(coefs_psw.get("H", {}).get("coef", 0)),
                abs(coefs_psw.get("A", {}).get("coef", 0)))
            if "L" in coefs_psw else None
        ),
    }


# ====================================================================
# Part 2: E7 positive-license asymmetry test (W2 fix)
# ====================================================================

def _cell_mean_per_target(df: pd.DataFrame, cell: str, targets: list[str]) -> dict:
    """Mean rating per (cell, target) for the given targets."""
    sub = df[(df["cell"] == cell) & (df["target"].isin(targets))]
    return sub.groupby("target")["rating"].mean().to_dict()


def _per_target_delta(target_means_a: dict, target_means_b: dict) -> np.ndarray:
    """Compute per-target a - b for shared targets."""
    shared = sorted(set(target_means_a) & set(target_means_b))
    return np.array([target_means_a[t] - target_means_b[t] for t in shared])


def analyze_e7_asymmetry():
    print("\n" + "=" * 70)
    print("Part 2: E7 asymmetry test (W2 fix)")
    print("=" * 70)

    # Load all relevant cells: E5_*, E5b_*, E7_*
    df_all = pd.concat([
        load_cells_by_prefix("E5_"),
        load_cells_by_prefix("E5b_"),
        load_cells_by_prefix("E7_"),
    ], ignore_index=True)
    if df_all.empty:
        print("WARNING: no data found; skipping Part 2.")
        return {}

    # Aggregate to (cell, target) means
    agg = df_all.groupby(["cell", "target"])["rating"].mean().reset_index()
    agg_dict = {(r["cell"], r["target"]): r["rating"] for _, r in agg.iterrows()}

    def cm(cell: str, targets: list[str]) -> dict:
        return {t: agg_dict[(cell, t)] for t in targets if (cell, t) in agg_dict}

    contrasts = {}

    def _report(label, cell_neg, cell_pos, baseline_cell, targets):
        m_neg = cm(cell_neg, targets)
        m_pos = cm(cell_pos, targets)
        m_base = cm(baseline_cell, targets)
        shared = sorted(set(m_neg) & set(m_pos) & set(m_base))
        if not shared:
            print(f"  [{label}] no shared targets (missing data)")
            return None
        d_neg = np.array([m_neg[t] - m_base[t] for t in shared])
        d_pos = np.array([m_pos[t] - m_base[t] for t in shared])
        diff = d_neg - d_pos  # per-target paired difference
        # Bootstrap on the per-target paired differences
        rng = np.random.default_rng(2026)
        boot = np.array([
            rng.choice(diff, size=len(diff), replace=True).mean()
            for _ in range(5000)
        ])
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        t_stat, p_two = stats.ttest_1samp(diff, 0.0)
        print(f"  [{label}]  n_shared={len(shared)}")
        print(f"     Delta_neg ({cell_neg} - {baseline_cell}): mean = {d_neg.mean():+.3f}")
        print(f"     Delta_pos ({cell_pos} - {baseline_cell}): mean = {d_pos.mean():+.3f}")
        print(f"     (Delta_neg) - (Delta_pos): mean = {diff.mean():+.3f}  "
              f"95% bootstrap CI = [{ci_lo:+.3f}, {ci_hi:+.3f}]  "
              f"t({len(diff)-1}) = {t_stat:.2f}, p = {p_two:.4f}")
        ratio = (abs(d_pos).mean() / abs(d_neg).mean()) if abs(d_neg).mean() > 1e-6 else float("nan")
        print(f"     |Delta_pos| / |Delta_neg| (effect-size ratio): {ratio:.3f}")
        return {
            "n_shared": len(shared),
            "delta_neg_mean": float(d_neg.mean()),
            "delta_pos_mean": float(d_pos.mean()),
            "diff_mean": float(diff.mean()),
            "diff_ci": [float(ci_lo), float(ci_hi)],
            "t_stat": float(t_stat),
            "p_two": float(p_two),
            "effect_ratio_pos_over_neg": float(ratio) if ratio == ratio else None,
        }

    # ORIGINAL TARGETS (7 = fictional-4 + 3 calibration). License-only and
    # full-mirror, each with the simple positive ("pos") and the
    # suppression-implied positive ("pos2").

    print("\n--- ORIGINAL 7 targets ---")
    print("\n[A] License-only contrast (no H, no A):")
    contrasts["orig_license_only_simple_pos"] = _report(
        "orig E5_001 vs E7_pos_001",  "E5_001", "E7_pos_001_orig", "E5_000", ORIG_7)
    contrasts["orig_license_only_pos2"] = _report(
        "orig E5_001 vs E7_pos2_001", "E5_001", "E7_pos2_001_orig", "E5_000", ORIG_7)
    print("\n[B] Full-mirror contrast (H+A+license):")
    contrasts["orig_full_mirror_simple_pos"] = _report(
        "orig E5_111 vs E7_pos_111",  "E5_111", "E7_pos_111_orig", "E5_000", ORIG_7)
    contrasts["orig_full_mirror_pos2"] = _report(
        "orig E5_111 vs E7_pos2_111", "E5_111", "E7_pos2_111_orig", "E5_000", ORIG_7)

    # PSEUDOWORD 15 targets. Baseline is E5b_000.
    df_e5b = load_cells_by_prefix("E5b_")
    psw_15 = sorted(df_e5b["target"].unique().tolist()) if not df_e5b.empty else []
    if psw_15:
        print(f"\n--- PSEUDOWORD {len(psw_15)} targets ---")
        print("\n[A] License-only contrast (no H, no A):")
        contrasts["psw_license_only_simple_pos"] = _report(
            "psw E5b_001 vs E7_pos_001",  "E5b_001", "E7_pos_001_psw", "E5b_000", psw_15)
        contrasts["psw_license_only_pos2"] = _report(
            "psw E5b_001 vs E7_pos2_001", "E5b_001", "E7_pos2_001_psw", "E5b_000", psw_15)
        print("\n[B] Full-mirror contrast (H+A+license):")
        contrasts["psw_full_mirror_simple_pos"] = _report(
            "psw E5b_111 vs E7_pos_111",  "E5b_111", "E7_pos_111_psw", "E5b_000", psw_15)
        contrasts["psw_full_mirror_pos2"] = _report(
            "psw E5b_111 vs E7_pos2_111", "E5b_111", "E7_pos2_111_psw", "E5b_000", psw_15)

    # Interpretation logic
    print("\n" + "=" * 70)
    print("Asymmetry test interpretation")
    print("=" * 70)
    summary_lines = []
    print("Comparing |Δ_neg| to |Δ_pos|. Asymmetry is supported when |Δ_neg| > |Δ_pos|.")
    print("(diff = Δ_neg - Δ_pos; the SIGN of diff is not the same as the MAGNITUDE comparison")
    print(" because Δ_neg is large-negative and Δ_pos can be small-negative, near-zero, or small-positive.)")
    print()
    for key, c in contrasts.items():
        if c is None:
            continue
        d_pos = c["delta_pos_mean"]; d_neg = c["delta_neg_mean"]
        ratio = c["effect_ratio_pos_over_neg"]  # |Δ_pos| / |Δ_neg|
        diff = c["diff_mean"]; ci = c["diff_ci"]
        # Verdict based on magnitude ratio, not sign of diff
        if ratio is None:
            verdict = "indeterminate (Δ_neg too small)"
        elif ratio < 0.5:
            verdict = f"STRONG ASYMMETRY (|Δ_pos|/|Δ_neg| = {ratio:.2f}, < 0.5)"
        elif ratio < 0.8:
            verdict = f"GRADED ASYMMETRY (|Δ_pos|/|Δ_neg| = {ratio:.2f}, in [0.5, 0.8])"
        elif ratio < 1.2:
            verdict = f"NEAR-SYMMETRY (|Δ_pos|/|Δ_neg| = {ratio:.2f}, in [0.8, 1.2])"
        else:
            verdict = f"REVERSE ASYMMETRY (|Δ_pos|/|Δ_neg| = {ratio:.2f}, > 1.2)"
        line = f"  [{key}] Δ_neg={d_neg:+.3f}  Δ_pos={d_pos:+.3f}  ratio={ratio:.3f}  -> {verdict}"
        print(line)
        summary_lines.append(line)

    return {
        "contrasts": contrasts,
        "summary_lines": summary_lines,
    }


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    p1 = analyze_e5b()
    p2 = analyze_e7_asymmetry()
    out = {"E5b_factorial_pseudowords": p1, "E7_asymmetry_test": p2}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
