"""Analysis of E5 (2x2x2 factorial decomposition) and E6 (pseudoword
external-validity replication).

Design follows the LLM review of the analysis plan (verdict: major-revision,
all fixes applied here):

E5:
  - Aggregate to (cell x target) means: 8 cells x 7 targets = 56 rows.
    The 5 prompt templates per (cell,target) are deterministic single
    samples (greedy decoding) -- different question framings, NOT
    independent replicates -- so they are averaged, not treated as n=280.
  - PRIMARY model: 4 fictional targets only (the original ambiguous-valence
    stimuli). rating ~ H + A + L + interactions + C(target), OLS with
    target fixed effects. 4 targets x 8 cells = 32 rows.
  - SECONDARY model: all 7 targets (adds 3 valence-spanned calibration
    targets). 56 rows.
  - The factorial intercept = E5_000 ("research participant" baseline).
    NOTE: E5_000 is NOT M0. E5 cannot test below-M0 overshoot; it tests
    whether each factor shifts the rating relative to the bare prompt.
  - L (negativity-license) is the pre-registered confirmatory factor;
    H, A, interactions are exploratory.
  - Cluster bootstrap (resample targets) is reported as a SENSITIVITY
    analysis only -- 7 (or 4) clusters is too few for reliable CIs.
  - Calibration check: on the 3 valence-spanned targets, forced-honesty
    must preserve valence ordering (water high / decay low / brick mid).

E6:
  - 15 new pseudoword targets, 3 conditions (default/forced/control).
    Average 5 templates -> 45 (target,condition) means.
  - Per-target Delta_forced = forced - default; one-sample t-test on the
    15 deltas + binomial sign test vs 0.5.
  - Tempered claim: within-M3 replication across stimuli, NOT a
    below-base overshoot claim (no M0 baseline for these targets).

Saved to analyze_extensions.json.
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
OUT_PATH = RESULTS_DIR / "analyze_extensions.json"

FICTIONAL = ["adirenia", "che_pact", "khantelan", "vellinkov"]
CALIBRATION = {"clean_drinking_water": "positive",
               "tooth_decay": "negative",
               "standard_brick": "neutral"}


def load_cells(prefix: str) -> pd.DataFrame:
    """Load all trial rows for cells matching prefix into a DataFrame."""
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
# E5: 2x2x2 factorial decomposition
# ====================================================================

def analyze_e5():
    print("=" * 70)
    print("E5: 2x2x2 factorial decomposition (honesty x anti-politeness x license)")
    print("=" * 70)
    df = load_cells("E5_")

    # Decode H, A, L from cell name E5_HAL
    df["H"] = df["cell"].str[3].astype(int)
    df["A"] = df["cell"].str[4].astype(int)
    df["L"] = df["cell"].str[5].astype(int)

    # --- Template-variance diagnostic (review point 5) ---------------
    # Check the 5 template ratings within each (cell,target) are tightly
    # clustered relative to the cell effects.
    tv = df.groupby(["cell", "target"])["rating"].std()
    cell_means_all = df.groupby("cell")["rating"].mean()
    print(f"\nTemplate-variance diagnostic:")
    print(f"  mean within-(cell,target) SD across templates : {tv.mean():.4f}")
    print(f"  SD of the 8 cell means                        : {cell_means_all.std():.4f}")
    print(f"  ratio (template noise / cell signal)          : "
          f"{tv.mean() / cell_means_all.std():.3f}")

    # --- Aggregate to (cell, target) means: 56 rows -----------------
    agg = (df.groupby(["cell", "target", "H", "A", "L"])["rating"]
             .mean().reset_index())
    print(f"\nAggregated to {len(agg)} (cell x target) means.")

    # --- 8 cell means table (sufficient statistics) -----------------
    print("\n8 cell means (averaged over targets):")
    cell_table = {}
    for cell in sorted(df["cell"].unique()):
        H, A, L = int(cell[3]), int(cell[4]), int(cell[5])
        m_all = float(agg[agg["cell"] == cell]["rating"].mean())
        m_fic = float(agg[(agg["cell"] == cell) &
                          (agg["target"].isin(FICTIONAL))]["rating"].mean())
        label = (f"H={H} A={A} L={L}")
        print(f"  {cell} ({label}):  all-7 mean = {m_all:.3f}   "
              f"fictional-4 mean = {m_fic:.3f}")
        cell_table[cell] = {"H": H, "A": A, "L": L,
                            "mean_all7": m_all, "mean_fictional4": m_fic}

    # --- PRIMARY model: 4 fictional targets, target fixed effects ----
    print("\n" + "-" * 70)
    print("PRIMARY factorial model: 4 fictional targets, OLS + target FE")
    print("-" * 70)
    fic = agg[agg["target"].isin(FICTIONAL)].copy()
    formula = "rating ~ H*A*L + C(target)"
    m_primary = smf.ols(formula, data=fic).fit()
    primary_coefs = _report_factorial(m_primary, "PRIMARY (fictional-4)")

    # --- SECONDARY model: all 7 targets ------------------------------
    print("\n" + "-" * 70)
    print("SECONDARY factorial model: all 7 targets, OLS + target FE")
    print("-" * 70)
    m_secondary = smf.ols(formula, data=agg).fit()
    secondary_coefs = _report_factorial(m_secondary, "SECONDARY (all-7)")

    # --- Cluster bootstrap (SENSITIVITY ONLY, 4 clusters) -----------
    print("\n" + "-" * 70)
    print("SENSITIVITY: target-cluster bootstrap on fictional-4")
    print("(UNRELIABLE -- 4 clusters far below the ~15-30 threshold;")
    print(" reported for transparency only)")
    print("-" * 70)
    boot_ci = _cluster_bootstrap(fic, formula, FICTIONAL, n_boot=2000)
    for term, (lo, hi) in boot_ci.items():
        print(f"  {term:<20s} 95% boot CI [{lo:+.3f}, {hi:+.3f}]")

    # --- Calibration valence-ordering check -------------------------
    print("\n" + "-" * 70)
    print("Calibration check: forced-honesty (E5_111) valence ordering")
    print("-" * 70)
    calib = {}
    for tid, expected in CALIBRATION.items():
        r000 = float(agg[(agg.cell == "E5_000") & (agg.target == tid)]["rating"].iloc[0])
        r111 = float(agg[(agg.cell == "E5_111") & (agg.target == tid)]["rating"].iloc[0])
        print(f"  {tid:<22s} ({expected:<8s}): E5_000={r000:.2f}  E5_111={r111:.2f}")
        calib[tid] = {"expected_valence": expected,
                      "E5_000": r000, "E5_111": r111}
    water = calib["clean_drinking_water"]["E5_111"]
    decay = calib["tooth_decay"]["E5_111"]
    brick = calib["standard_brick"]["E5_111"]
    ordering_ok = water > brick > decay
    print(f"  Valence ordering preserved under forced-honesty "
          f"(water > brick > decay): {ordering_ok}")

    return {
        "template_variance": {
            "mean_within_celltarget_sd": float(tv.mean()),
            "cell_mean_sd": float(cell_means_all.std()),
            "ratio": float(tv.mean() / cell_means_all.std()),
        },
        "cell_means": cell_table,
        "primary_fictional4": primary_coefs,
        "secondary_all7": secondary_coefs,
        "sensitivity_cluster_bootstrap_ci": {k: list(v) for k, v in boot_ci.items()},
        "calibration_check": calib,
        "calibration_ordering_preserved": bool(ordering_ok),
    }


def _report_factorial(model, label):
    """Print + collect factorial coefficients (H/A/L main + interactions)."""
    terms = ["H", "A", "L", "H:A", "H:L", "A:L", "H:A:L"]
    out = {}
    print(f"  {'term':<10s} {'coef':>9s} {'std err':>9s} {'95% CI':>20s} "
          f"{'p':>9s}")
    for t in terms:
        if t not in model.params.index:
            continue
        coef = model.params[t]
        se = model.bse[t]
        ci = model.conf_int().loc[t]
        p = model.pvalues[t]
        tag = " [PRE-REG]" if t == "L" else ""
        print(f"  {t:<10s} {coef:>+9.4f} {se:>9.4f} "
              f"[{ci[0]:+.3f},{ci[1]:+.3f}] {p:>9.4f}{tag}")
        out[t] = {"coef": float(coef), "se": float(se),
                  "ci": [float(ci[0]), float(ci[1])], "p": float(p)}
    print(f"  R^2 = {model.rsquared:.4f}, n = {int(model.nobs)}")
    out["_rsquared"] = float(model.rsquared)
    out["_n"] = int(model.nobs)
    return out


def _cluster_bootstrap(data, formula, clusters, n_boot=2000, seed=7):
    """Resample whole targets with replacement; refit; collect coef CIs."""
    rng = np.random.default_rng(seed)
    terms = ["H", "A", "L", "H:A", "H:L", "A:L", "H:A:L"]
    samples = {t: [] for t in terms}
    for _ in range(n_boot):
        drawn = rng.choice(clusters, size=len(clusters), replace=True)
        parts = []
        for i, c in enumerate(drawn):
            d = data[data["target"] == c].copy()
            d["target"] = f"{c}_boot{i}"  # unique so FE stays identified
            parts.append(d)
        boot_df = pd.concat(parts, ignore_index=True)
        try:
            m = smf.ols(formula, data=boot_df).fit()
            for t in terms:
                if t in m.params.index:
                    samples[t].append(m.params[t])
        except Exception:
            continue
    ci = {}
    for t in terms:
        if samples[t]:
            ci[t] = (float(np.percentile(samples[t], 2.5)),
                     float(np.percentile(samples[t], 97.5)))
    return ci


# ====================================================================
# E6: pseudoword external-validity replication
# ====================================================================

def analyze_e6():
    print("\n" + "=" * 70)
    print("E6: pseudoword replication (15 new targets, within-M3)")
    print("=" * 70)
    df = load_cells("E6_")
    df["condition"] = df["cell"].str.replace("E6_", "", regex=False)

    # Aggregate templates -> (target, condition) means
    agg = df.groupby(["target", "condition"])["rating"].mean().unstack()
    targets = sorted(agg.index.tolist())
    print(f"\n{len(targets)} pseudoword targets, conditions: "
          f"{sorted(agg.columns.tolist())}")

    delta_forced = (agg["forced"] - agg["default"]).to_numpy()
    delta_control = (agg["control"] - agg["default"]).to_numpy()
    forced_minus_control = (agg["forced"] - agg["control"]).to_numpy()

    print("\nPer-target ratings and deltas:")
    print(f"  {'target':<14s} {'default':>8s} {'forced':>8s} {'control':>8s} "
          f"{'d_forced':>9s}")
    for t in targets:
        print(f"  {t:<14s} {agg.loc[t,'default']:>8.3f} "
              f"{agg.loc[t,'forced']:>8.3f} {agg.loc[t,'control']:>8.3f} "
              f"{agg.loc[t,'forced']-agg.loc[t,'default']:>+9.3f}")

    # One-sample t-test on the 15 per-target deltas
    t_stat, p_t = stats.ttest_1samp(delta_forced, 0.0)
    # Binomial sign test
    n_neg = int((delta_forced < 0).sum())
    n = len(delta_forced)
    p_binom = stats.binomtest(n_neg, n, 0.5).pvalue

    mean_df = float(delta_forced.mean())
    sd_df = float(delta_forced.std(ddof=1))
    se_df = sd_df / np.sqrt(n)
    ci_lo, ci_hi = mean_df - 1.96 * se_df, mean_df + 1.96 * se_df

    print(f"\nDelta_forced = forced - default across {n} pseudoword targets:")
    print(f"  mean = {mean_df:+.4f}  (SD {sd_df:.4f}, SE {se_df:.4f})")
    print(f"  95% CI (t-based): [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  one-sample t-test vs 0: t({n-1}) = {t_stat:.3f}, p = {p_t:.5f}")
    print(f"  targets with Delta_forced < 0: {n_neg}/{n}")
    print(f"  binomial sign test vs 0.5: p = {p_binom:.5f}")

    mean_dc = float(delta_control.mean())
    mean_fmc = float(forced_minus_control.mean())
    t_fmc, p_fmc = stats.ttest_1samp(forced_minus_control, 0.0)
    print(f"\n  Delta_control = control - default: mean = {mean_dc:+.4f}")
    print(f"  forced - control: mean = {mean_fmc:+.4f}  "
          f"(t = {t_fmc:.3f}, p = {p_fmc:.5f})")
    print("  NOTE: 'forced - control' is forced-honesty minus neutral-accuracy;")
    print("  it is NOT a clean anti-politeness-specific contrast because the")
    print("  control prompt differs from forced on multiple components.")

    if ci_hi < 0:
        interp = (f"Forced-honesty rating depression REPLICATES across "
                  f"{n} novel pseudoword targets within M3 (mean Delta_forced "
                  f"= {mean_df:+.3f}, 95% CI excludes 0, {n_neg}/{n} targets "
                  f"negative). This is a within-M3 replication-across-stimuli "
                  f"result; it does NOT test below-M0 overshoot (no M0 "
                  f"baseline for these targets).")
    else:
        interp = (f"Forced-honesty effect does NOT cleanly replicate on the "
                  f"new pseudoword targets (95% CI on Delta_forced includes 0).")
    print(f"\nInterpretation: {interp}")

    return {
        "n_targets": n,
        "mean_delta_forced": mean_df,
        "sd_delta_forced": sd_df,
        "ci_delta_forced": [ci_lo, ci_hi],
        "t_test": {"t": float(t_stat), "df": n - 1, "p": float(p_t)},
        "sign_test": {"n_negative": n_neg, "n": n, "p": float(p_binom)},
        "mean_delta_control": mean_dc,
        "mean_forced_minus_control": mean_fmc,
        "per_target": {t: {"default": float(agg.loc[t, "default"]),
                           "forced": float(agg.loc[t, "forced"]),
                           "control": float(agg.loc[t, "control"])}
                       for t in targets},
        "interpretation": interp,
    }


def main():
    e5 = analyze_e5()
    e6 = analyze_e6()
    out = {"E5_factorial": e5, "E6_pseudoword_replication": e6}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
