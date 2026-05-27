"""Stage-specific LR test: is the M3-specific asymmetry significant?

Compares:
  - Full M3-asymmetric (α+ and α- free at every stage)
  - M3-only-symmetric variant: α+(M3) = α-(M3); M1/M2 stay free

LR = 2(−LL_constrained − −LL_full), df = 1. χ²(1) at 95% = 3.84.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from data import load_trials
from models import Model3AsymmetricBelief, Model3AsymmetricM3Sym

DIR_NEW = Path("experiments/results/per-stage-localization-forced-honesty-overshoot")
DIR_PRIOR = Path("experiments/results/stage-base-vs-instruct-positivity-prior")


def fit(model_class, df):
    m = model_class()
    init = m.init_params(df)
    res = minimize(
        m.neg_log_likelihood, init, args=(df,),
        method="L-BFGS-B", bounds=m.bounds(df),
        options={"maxiter": 5000, "ftol": 1e-10},
    )
    return m, res


def main():
    df = load_trials(DIR_NEW, DIR_PRIOR)
    n = len(df)
    print(f"Loaded {n} trials.\n")

    print("Fitting full M3-asymmetric (α+, α- free at every stage)...")
    m_full, r_full = fit(Model3AsymmetricBelief, df)
    print(f"  k = {m_full.n_params(df)}  -LL = {r_full.fun:.4f}")

    print("\nFitting constrained M3-m3sym (α+(M3) = α-(M3), M1/M2 free)...")
    m_con, r_con = fit(Model3AsymmetricM3Sym, df)
    print(f"  k = {m_con.n_params(df)}  -LL = {r_con.fun:.4f}")

    print(f"\n  -LL diff (constrained - full): {r_con.fun - r_full.fun:+.4f}")

    lr_stat = 2 * (r_con.fun - r_full.fun)
    df_diff = m_full.n_params(df) - m_con.n_params(df)
    chi2_95 = 3.84  # χ²(1) at 0.05
    chi2_99 = 6.63  # χ²(1) at 0.01
    chi2_999 = 10.83  # χ²(1) at 0.001
    print(f"\nLikelihood-ratio test (M3-asymmetric vs M3-m3sym):")
    print(f"  LR statistic: {lr_stat:.4f}")
    print(f"  df: {df_diff}")
    print(f"  χ²(1) critical: 3.84 (p=0.05), 6.63 (p=0.01), 10.83 (p=0.001)")
    if lr_stat > chi2_999:
        verdict = "p < 0.001 — M3-specific asymmetry strongly supported"
    elif lr_stat > chi2_99:
        verdict = "p < 0.01 — M3-specific asymmetry supported"
    elif lr_stat > chi2_95:
        verdict = "p < 0.05 — M3-specific asymmetry significant at conventional threshold"
    else:
        verdict = "p > 0.05 — M3-specific asymmetry NOT significant"
    print(f"  Verdict: {verdict}")

    # Print fitted M3-only param under constraint
    pn_con = m_con.param_names(df)
    n_t = len(m_con._targets)
    print(f"\n  Constrained model α(M3) shared value: {r_con.x[n_t + 2]:.4f}")
    print(f"  Full model α+(M3): {r_full.x[n_t + 2]:.4f}  α-(M3): {r_full.x[n_t + 5]:.4f}")


if __name__ == "__main__":
    main()
