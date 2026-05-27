"""Sanity check for Model 2 (Kunda multiplicative): does it beat Model 1?

Predictions:
  - α_M1, α_M2, α_M3 should be roughly increasing (politeness motive grows
    with post-training amount). M3's α should be substantially > 0.
  - β for forced / anti-politeness conditions should be NEGATIVE (the motive
    pulls down, not up).
  - ΔAIC(M2 - M1) should be < 0 (M2 explains the manipulation effects M1
    can't see).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from data import load_trials
from models import Model1Null, Model2KundaGated

DIR_NEW = Path("experiments/results/per-stage-localization-forced-honesty-overshoot")
DIR_PRIOR = Path("experiments/results/stage-base-vs-instruct-positivity-prior")


def fit(model, df):
    init = model.init_params(df)
    res = minimize(
        model.neg_log_likelihood, init, args=(df,),
        method="L-BFGS-B", bounds=model.bounds(df),
    )
    return res


def aic_bic(neg_ll, n_params, n):
    ll = -neg_ll
    return 2 * n_params - 2 * ll, n_params * np.log(n) - 2 * ll


def main():
    df = load_trials(DIR_NEW, DIR_PRIOR)
    n = len(df)
    print(f"Loaded {n} trials.\n")

    print("=" * 70)
    print("Model 1 (null)")
    print("=" * 70)
    m1 = Model1Null()
    r1 = fit(m1, df)
    aic1, bic1 = aic_bic(r1.fun, m1.n_params(df), n)
    print(f"  converged: {r1.success}  k={m1.n_params(df)}  "
          f"-LL={r1.fun:.2f}  AIC={aic1:.2f}  BIC={bic1:.2f}")

    print("\n" + "=" * 70)
    print("Model 2 (Kunda gated)")
    print("=" * 70)
    m2 = Model2KundaGated()
    r2 = fit(m2, df)
    aic2, bic2 = aic_bic(r2.fun, m2.n_params(df), n)
    print(f"  converged: {r2.success}  k={m2.n_params(df)}  "
          f"-LL={r2.fun:.2f}  AIC={aic2:.2f}  BIC={bic2:.2f}")

    print("\nFitted parameters:")
    pn = m2.param_names(df)
    for name, val in zip(pn, r2.x):
        print(f"  {name:<35s}: {val:+.4f}")

    print(f"\nM0 amplitude: 0 (fixed)")
    print(f"β_default:    +1 (fixed anchor)")

    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    print(f"  ΔAIC  (M2 - M1): {aic2 - aic1:+.2f}  "
          f"(<0 favors M2; >10 is strong)")
    print(f"  ΔBIC  (M2 - M1): {bic2 - bic1:+.2f}  "
          f"(<0 favors M2; >10 is strong)")
    print(f"  Δ(-LL) (M1 - M2): {r1.fun - r2.fun:+.2f}  "
          f"(positive = M2 fits better)")

    # Diagnostic: predicted vs empirical for the key M3 cells
    print("\nDiagnostic — predicted vs empirical cell means:")
    print(f"  {'stage':<5} {'condition':<25} {'n':>4}  {'empirical':>10}  {'M2 pred':>10}  {'resid':>8}")
    grouped = df.groupby(["stage", "condition"])
    pred = m2.predict_means(df, r2.x[:-1])
    df_pred = df.copy()
    df_pred["pred"] = pred
    for (stage, cond), grp in grouped:
        emp = grp["rating"].mean()
        pr = df_pred.loc[grp.index, "pred"].mean()
        print(f"  {stage:<5} {cond:<25} {len(grp):>4}  "
              f"{emp:>10.3f}  {pr:>10.3f}  {emp-pr:>+8.3f}")

    print("\nSanity flags:")
    n_t = len(m2._targets)
    n_c = len(m2._cells)
    deltas = r2.x[n_t:n_t + n_c]
    cell_keys = m2._cells

    # Build a quick dict for lookup
    d_map = dict(zip(cell_keys, deltas))

    # M3-forced should be the strongest negative (overshoot)
    key = ("M3", "forced")
    if key in d_map:
        print(f"  δ_M3_forced negative?  {d_map[key] < 0}  ({d_map[key]:+.3f})")
    # M3-anti-politeness-strong should also be strongly negative
    key = ("M3", "anti-politeness-strong")
    if key in d_map:
        print(f"  δ_M3_anti-politeness-strong negative?  {d_map[key] < 0}  ({d_map[key]:+.3f})")
    # M1-default should be positive (politeness shift up)
    key = ("M1", "default")
    if key in d_map:
        print(f"  δ_M1_default positive?  {d_map[key] > 0}  ({d_map[key]:+.3f})")
    # M2-default should also be positive and likely larger
    key = ("M2", "default")
    if key in d_map:
        print(f"  δ_M2_default positive?  {d_map[key] > 0}  ({d_map[key]:+.3f})")


if __name__ == "__main__":
    main()
