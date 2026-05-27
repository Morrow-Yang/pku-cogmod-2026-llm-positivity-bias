"""Trial-level bootstrap CIs for M3-asymmetric's α+ and α- per stage.

Resamples trials with replacement, refits M3-asymmetric, repeats N_BOOT
times. Reports percentile CI on each α and on the diagnostic α-(M3) − α+(M3).

Computationally: ~5s per fit × 500 bootstraps ≈ 40 minutes single-thread.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from data import load_trials
from models import Model3AsymmetricBelief

DIR_NEW = Path("experiments/results/per-stage-localization-forced-honesty-overshoot")
DIR_PRIOR = Path("experiments/results/stage-base-vs-instruct-positivity-prior")
OUT_DIR = Path("experiments/results/cognitive-model-comparison-polite-speech-post")

N_BOOT = 500
RNG_SEED = 42


def fit_m3(df, init=None):
    m = Model3AsymmetricBelief()
    # Need to re-initialize internal target/condition lists from this df
    if init is None:
        init = m.init_params(df)
    res = minimize(
        m.neg_log_likelihood, init, args=(df,),
        method="L-BFGS-B", bounds=m.bounds(df),
        options={"maxiter": 2000, "ftol": 1e-9},
    )
    return m, res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_trials(DIR_NEW, DIR_PRIOR)
    n = len(df)
    print(f"Loaded {n} trials.\n")

    # Point estimate on the full dataset for warm-start
    print("Fitting M3-asymmetric on full data for point estimate + warm start...")
    m_point, r_point = fit_m3(df)
    pn = m_point.param_names(df)
    point_estimate = {p: float(v) for p, v in zip(pn, r_point.x)}
    print(f"  Point estimate: -LL = {r_point.fun:.2f}")
    print(f"  α+ : M1={point_estimate['alpha_M1_pos']:.4f}  "
          f"M2={point_estimate['alpha_M2_pos']:.4f}  "
          f"M3={point_estimate['alpha_M3_pos']:.4f}")
    print(f"  α- : M1={point_estimate['alpha_M1_neg']:.4f}  "
          f"M2={point_estimate['alpha_M2_neg']:.4f}  "
          f"M3={point_estimate['alpha_M3_neg']:.4f}")

    warm_init = r_point.x.copy()
    rng = np.random.default_rng(RNG_SEED)

    n_failed = 0
    boot_records = []  # list of dicts of param -> value
    print(f"\nRunning {N_BOOT} bootstrap resamples...")
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        df_boot = df.iloc[idx].reset_index(drop=True)
        try:
            # Use a fresh model so internal target/condition lists reflect the
            # bootstrap sample (some targets may be missing in any one resample)
            m_b = Model3AsymmetricBelief()
            init_b = m_b.init_params(df_boot)
            # Try warm-start if dimensions match
            if len(init_b) == len(warm_init):
                init_b = warm_init.copy()
            res = minimize(
                m_b.neg_log_likelihood, init_b, args=(df_boot,),
                method="L-BFGS-B", bounds=m_b.bounds(df_boot),
                options={"maxiter": 1000, "ftol": 1e-8},
            )
            if not res.success and res.fun > 1e10:
                n_failed += 1
                continue
            pn_b = m_b.param_names(df_boot)
            boot_records.append({p: float(v) for p, v in zip(pn_b, res.x)})
        except Exception as e:
            n_failed += 1
            if i < 3:
                print(f"  [warn] bootstrap {i}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  done {i+1}/{N_BOOT}")
    print(f"Done. {len(boot_records)} successful, {n_failed} failed.")

    # Per-parameter summary
    summary = {}
    keys_of_interest = [
        "alpha_M1_pos", "alpha_M1_neg",
        "alpha_M2_pos", "alpha_M2_neg",
        "alpha_M3_pos", "alpha_M3_neg",
    ]
    for k in keys_of_interest:
        vals = np.array([r.get(k, np.nan) for r in boot_records])
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        summary[k] = {
            "point": point_estimate.get(k, None),
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)),
            "ci_95_lo": float(np.percentile(vals, 2.5)),
            "ci_95_hi": float(np.percentile(vals, 97.5)),
            "n_boot": len(vals),
        }

    # Diagnostic differences α-(s) − α+(s) per stage
    print("\nBootstrap CIs:")
    print(f"  {'param':<18} {'point':>9} {'mean':>9} {'SD':>9} {'CI_95_lo':>10} {'CI_95_hi':>10}")
    for k in keys_of_interest:
        s = summary.get(k)
        if s:
            print(f"  {k:<18} {s['point']:>9.4f} {s['mean']:>9.4f} "
                  f"{s['sd']:>9.4f} {s['ci_95_lo']:>10.4f} {s['ci_95_hi']:>10.4f}")

    diff_summary = {}
    for s in ["M1", "M2", "M3"]:
        diffs = np.array([
            r.get(f"alpha_{s}_neg", np.nan) - r.get(f"alpha_{s}_pos", np.nan)
            for r in boot_records
        ])
        diffs = diffs[~np.isnan(diffs)]
        diff_summary[f"alpha_{s}_neg_minus_pos"] = {
            "point": (point_estimate[f"alpha_{s}_neg"] - point_estimate[f"alpha_{s}_pos"]),
            "mean": float(diffs.mean()),
            "sd": float(diffs.std(ddof=1)),
            "ci_95_lo": float(np.percentile(diffs, 2.5)),
            "ci_95_hi": float(np.percentile(diffs, 97.5)),
            "frac_above_zero": float((diffs > 0).mean()),
            "n_boot": len(diffs),
        }

    print("\nDiagnostic differences α-(s) − α+(s):")
    print(f"  {'stage':<6} {'point':>9} {'mean':>9} {'CI_95_lo':>10} {'CI_95_hi':>10} {'P(diff>0)':>11}")
    for s in ["M1", "M2", "M3"]:
        d = diff_summary[f"alpha_{s}_neg_minus_pos"]
        print(f"  {s:<6} {d['point']:>+9.4f} {d['mean']:>+9.4f} "
              f"{d['ci_95_lo']:>+10.4f} {d['ci_95_hi']:>+10.4f} {d['frac_above_zero']:>11.3f}")

    out = {
        "n_boot_requested": N_BOOT,
        "n_boot_successful": len(boot_records),
        "n_failed": n_failed,
        "point_estimate": point_estimate,
        "param_ci": summary,
        "diff_ci": diff_summary,
    }
    out_path = OUT_DIR / "bootstrap_M3.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
