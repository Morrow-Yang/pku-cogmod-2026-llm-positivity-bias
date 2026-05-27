"""End-to-end sanity check for Model 1 on real data.

Verifies:
  1. Data loads, schema is as expected
  2. Model 1 likelihood evaluates to a finite number at init
  3. scipy.optimize.minimize converges
  4. Recovered per-target intercepts match the empirical per-target means
     (Model 1 is the per-target intercept model — analytic optimum IS the mean)
  5. AIC/BIC are sensible (positive, finite)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from data import load_trials
from models import Model1Null

DIR_NEW = Path("experiments/results/per-stage-localization-forced-honesty-overshoot")
DIR_PRIOR = Path("experiments/results/stage-base-vs-instruct-positivity-prior")


def main():
    print("=" * 70)
    print("Model 1 (null, per-target intercepts) end-to-end sanity check")
    print("=" * 70)

    df = load_trials(DIR_NEW, DIR_PRIOR)
    n = len(df)
    print(f"\n[1/5] Data: {n} trials, "
          f"{df['target_id'].nunique()} targets, "
          f"{df['cell'].nunique()} cells")

    m1 = Model1Null()
    pn = m1.param_names(df)
    print(f"\n[2/5] Param count: {len(pn)} = {len(pn)-1} targets + 1 sigma")

    init = m1.init_params(df)
    ll_init = m1.log_likelihood(df, init)
    print(f"      log-lik at init: {ll_init:.2f} (finite? {np.isfinite(ll_init)})")
    assert np.isfinite(ll_init), "Initial log-likelihood is not finite"

    print(f"\n[3/5] Optimizing with L-BFGS-B...")
    res = minimize(
        m1.neg_log_likelihood, init,
        args=(df,),
        method="L-BFGS-B",
        bounds=m1.bounds(df),
    )
    print(f"      converged: {res.success}")
    print(f"      neg-log-lik: {res.fun:.2f}")
    print(f"      n iter: {res.nit}")
    assert res.success, f"Optimizer did not converge: {res.message}"

    print(f"\n[4/5] Analytic vs MLE comparison (Model 1's MLE is per-target mean):")
    targets = sorted(df["target_id"].unique())
    empirical = df.groupby("target_id")["rating"].mean()
    print(f"      {'target':<25s}  empirical    MLE      Δ")
    max_diff = 0.0
    for i, t in enumerate(targets):
        emp = empirical[t]
        mle = res.x[i]
        diff = abs(emp - mle)
        max_diff = max(max_diff, diff)
        print(f"      {t:<25s}  {emp:8.4f}  {mle:8.4f}  {diff:8.4f}")
    print(f"      max |Δ|: {max_diff:.5f}")
    assert max_diff < 0.01, (
        f"MLE intercepts deviate from empirical means by {max_diff:.4f} "
        f"— Model 1's analytic optimum should match exactly."
    )

    sigma_mle = res.x[-1]
    n_params = len(pn)
    aic = 2 * n_params - 2 * (-res.fun)
    bic = n_params * np.log(n) - 2 * (-res.fun)
    print(f"\n[5/5] Fit quality:")
    print(f"      sigma_MLE: {sigma_mle:.4f}")
    print(f"      empirical pooled within-target σ: "
          f"{np.sqrt(df.groupby('target_id')['rating'].var(ddof=0).mean()):.4f}")
    print(f"      AIC: {aic:.2f}")
    print(f"      BIC: {bic:.2f}")
    assert np.isfinite(aic) and aic > 0
    assert np.isfinite(bic) and bic > 0

    print("\n" + "=" * 70)
    print("All checks passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
