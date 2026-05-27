"""Sanity check for Model 4 (RSA polite-speaker) — fits all of M1, M2, M4
and reports head-to-head comparison.

Expected:
  - M4 should fit at least as well as M2 in -LL (more flexible).
  - M4's BIC may be worse than M2's because M4 has more params (3-utility
    mixture per cell, while M2 has one shift per cell).
  - For M3-forced and M3-anti-politeness cells, ω_neg should dominate.
  - For M1-default / M2-default / M3-default, ω_pos should dominate.
  - For M0 cells, the anchor (M0, default) keeps ω_inf ≈ 1; other M0 cells
    should also have low ω_pos and ω_neg.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from data import load_trials
from models import Model1Null, Model2KundaGated, Model4RSAPoliteSpeaker

DIR_NEW = Path("experiments/results/per-stage-localization-forced-honesty-overshoot")
DIR_PRIOR = Path("experiments/results/stage-base-vs-instruct-positivity-prior")


def fit(model, df):
    init = model.init_params(df)
    res = minimize(
        model.neg_log_likelihood, init, args=(df,),
        method="L-BFGS-B", bounds=model.bounds(df),
        options={"maxiter": 1000},
    )
    return res


def aic_bic(neg_ll, n_params, n):
    ll = -neg_ll
    return 2 * n_params - 2 * ll, n_params * np.log(n) - 2 * ll


def summarize(name, res, n_params, n):
    aic, bic = aic_bic(res.fun, n_params, n)
    print(f"  {name:<35s}  converged: {res.success}  "
          f"k={n_params:>3}  -LL={res.fun:>10.2f}  "
          f"AIC={aic:>10.2f}  BIC={bic:>10.2f}")
    return aic, bic, res.fun


def cell_diagnostic(df, model, params, name):
    """Per-cell predicted vs empirical with M4 weight decomposition."""
    pred = model.predict_means(df, params[:-1])
    d = df.copy()
    d["pred"] = pred

    print(f"\n  {name} — per-cell predicted vs empirical:")
    if isinstance(model, Model4RSAPoliteSpeaker):
        # Recover (w_inf, w_pos, w_neg) per cell
        n_t = len(model._targets)
        n_c = len(model._cells)
        l_pos = params[n_t:n_t + n_c]
        l_neg = params[n_t + n_c:n_t + 2 * n_c]
        l_pos_map = dict(zip(model._cells, l_pos))
        l_neg_map = dict(zip(model._cells, l_neg))
        l_pos_map[model.ANCHOR_CELL] = model.ANCHOR_LOGIT
        l_neg_map[model.ANCHOR_CELL] = model.ANCHOR_LOGIT

        print(f"  {'stage':<5} {'condition':<28} {'n':>4} {'emp':>7} {'pred':>7} "
              f"{'ω_inf':>7} {'ω_pos':>7} {'ω_neg':>7}")
        for (stage, cond), grp in d.groupby(["stage", "condition"]):
            emp = grp["rating"].mean()
            pr = grp["pred"].mean()
            lp = l_pos_map[(stage, cond)]
            ln = l_neg_map[(stage, cond)]
            logits = np.array([0.0, lp, ln])
            w = np.exp(logits - logits.max())
            w = w / w.sum()
            print(f"  {stage:<5} {cond:<28} {len(grp):>4} {emp:>7.3f} {pr:>7.3f} "
                  f"{w[0]:>7.3f} {w[1]:>7.3f} {w[2]:>7.3f}")
    else:
        print(f"  {'stage':<5} {'condition':<28} {'n':>4} {'emp':>7} {'pred':>7} {'resid':>8}")
        for (stage, cond), grp in d.groupby(["stage", "condition"]):
            emp = grp["rating"].mean()
            pr = grp["pred"].mean()
            print(f"  {stage:<5} {cond:<28} {len(grp):>4} {emp:>7.3f} {pr:>7.3f} {emp - pr:>+8.3f}")


def main():
    df = load_trials(DIR_NEW, DIR_PRIOR)
    n = len(df)
    print(f"Loaded {n} trials.\n")

    print("=" * 70)
    print("Head-to-head: M1 vs M2 vs M4")
    print("=" * 70)
    m1 = Model1Null()
    r1 = fit(m1, df)
    aic1, bic1, nll1 = summarize("M1 (null)", r1, m1.n_params(df), n)

    m2 = Model2KundaGated()
    r2 = fit(m2, df)
    aic2, bic2, nll2 = summarize("M2 (Kunda gated)", r2, m2.n_params(df), n)

    m4 = Model4RSAPoliteSpeaker()
    r4 = fit(m4, df)
    aic4, bic4, nll4 = summarize("M4 (RSA polite-speaker)", r4, m4.n_params(df), n)

    print("\n" + "=" * 70)
    print("ΔAIC / ΔBIC (vs M1, lower is better)")
    print("=" * 70)
    for name, aic, bic in [("M1", aic1, bic1), ("M2", aic2, bic2), ("M4", aic4, bic4)]:
        print(f"  {name:<5}  ΔAIC={aic - aic1:+10.2f}  ΔBIC={bic - bic1:+10.2f}")

    print("\n" + "=" * 70)
    print("M4 vs M2 (the cognitive-content comparison)")
    print("=" * 70)
    print(f"  ΔAIC (M4 - M2): {aic4 - aic2:+.2f}")
    print(f"  ΔBIC (M4 - M2): {bic4 - bic2:+.2f}")
    print(f"  Δ(-LL) (M2 - M4): {nll2 - nll4:+.2f}  (positive = M4 fits better)")

    cell_diagnostic(df, m4, r4.x, "M4")


if __name__ == "__main__":
    main()
