"""Fit M0/M1/M2/M3/M3-symmetric/M4/M5 on the full trial dataset.

Uses multi-start L-BFGS-B for the models with > 25 params to avoid local
minima (M4, M5). Saves results to JSON.

Per review feedback (2026-05-21): convergence fix + linear baseline + nested
symmetric null for M3.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from data import load_trials
from models import (Model0LinearAdditive, Model1Null, Model2KundaGated,
                    Model3AsymmetricBelief, Model3SymmetricBelief,
                    Model4RSAPoliteSpeaker, Model5RSALicenseConditional)

DIR_NEW = Path("experiments/results/per-stage-localization-forced-honesty-overshoot")
DIR_PRIOR = Path("experiments/results/stage-base-vs-instruct-positivity-prior")
OUT_DIR = Path("experiments/results/cognitive-model-comparison-polite-speech-post")


def _single_fit(model, df, init, maxiter=10000):
    return minimize(
        model.neg_log_likelihood, init, args=(df,),
        method="L-BFGS-B", bounds=model.bounds(df),
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7},
    )


def fit_multi_start(model, df, n_starts=5, seed=0, maxiter=10000):
    """Run several random starts; return the best result."""
    rng = np.random.default_rng(seed)
    init = model.init_params(df)
    bounds = model.bounds(df)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    best = _single_fit(model, df, init, maxiter=maxiter)
    for k in range(n_starts - 1):
        # Perturb the deterministic init within bounds, geometric step
        perturb = rng.normal(0.0, 0.3, size=len(init))
        # For small bounds keep init in range; expand for non-bounded params
        new_init = np.clip(init + perturb * (hi - lo) * 0.1, lo + 1e-6, hi - 1e-6)
        res = _single_fit(model, df, new_init, maxiter=maxiter)
        if res.fun < best.fun:
            best = res
    return best


def aic_bic(neg_ll, k, n):
    ll = -neg_ll
    return 2 * k - 2 * ll, k * np.log(n) - 2 * ll


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_trials(DIR_NEW, DIR_PRIOR)
    n = len(df)
    print(f"Loaded {n} trials.\n")

    models = [
        ("M0_linear_additive", Model0LinearAdditive(), 3),
        ("M1_null", Model1Null(), 1),
        ("M2_kunda_gated", Model2KundaGated(), 3),
        ("M3_symmetric_belief", Model3SymmetricBelief(), 3),
        ("M3_asymmetric_belief", Model3AsymmetricBelief(), 5),
        ("M4_rsa_polite_speaker", Model4RSAPoliteSpeaker(), 7),
        ("M5_rsa_license_conditional", Model5RSALicenseConditional(), 5),
    ]

    summary = []
    print("=" * 90)
    print(f"{'Model':<32} {'starts':>7} {'conv':>5} {'k':>4} {'-LL':>11} {'AIC':>11} {'BIC':>11}")
    print("=" * 90)
    for name, m, n_starts in models:
        r = fit_multi_start(m, df, n_starts=n_starts)
        k = m.n_params(df)
        aic, bic = aic_bic(r.fun, k, n)
        pn = m.param_names(df)
        params_dict = {p: float(v) for p, v in zip(pn, r.x)}
        summary.append({
            "model": name,
            "converged": bool(r.success),
            "n_starts": n_starts,
            "n_params": k,
            "neg_log_lik": float(r.fun),
            "log_lik": float(-r.fun),
            "aic": float(aic),
            "bic": float(bic),
            "n_trials": n,
        })
        with open(OUT_DIR / f"fitted_params_{name}.json", "w") as f:
            json.dump({
                "model": name, "n_trials": n, "params": params_dict,
                "converged": bool(r.success), "neg_log_lik": float(r.fun),
                "n_params": k, "aic": float(aic), "bic": float(bic),
            }, f, indent=2)
        print(f"{name:<32} {n_starts:>7} {str(r.success)[0]:>5} {k:>4} "
              f"{r.fun:>11.2f} {aic:>11.2f} {bic:>11.2f}")

    base = next(r for r in summary if r["model"] == "M1_null")
    for row in summary:
        row["delta_aic_vs_M1"] = row["aic"] - base["aic"]
        row["delta_bic_vs_M1"] = row["bic"] - base["bic"]

    with open(OUT_DIR / "fit_summary.json", "w") as f:
        json.dump({
            "comparisons": summary,
            "best_by_bic": min(summary, key=lambda r: r["bic"])["model"],
            "best_by_aic": min(summary, key=lambda r: r["aic"])["model"],
        }, f, indent=2)

    print(f"\nBest by AIC: {min(summary, key=lambda r: r['aic'])['model']}")
    print(f"Best by BIC: {min(summary, key=lambda r: r['bic'])['model']}")

    # Nested LR test: M3 vs M3-symmetric
    m3 = next(r for r in summary if r["model"] == "M3_asymmetric_belief")
    m3s = next(r for r in summary if r["model"] == "M3_symmetric_belief")
    lr_stat = 2 * (m3s["neg_log_lik"] - m3["neg_log_lik"])
    df_diff = m3["n_params"] - m3s["n_params"]
    print(f"\nLikelihood-ratio test (M3 vs M3-symmetric):")
    print(f"  LR statistic: {lr_stat:.3f}  df: {df_diff}  "
          f"(χ²(3) 95% critical = 7.81; χ²(3) 99% critical = 11.34)")
    print(f"  Asymmetry supported at 95%? {lr_stat > 7.81}")


if __name__ == "__main__":
    main()
