"""Parametric-bootstrap LR test for the stage-specific M3 asymmetry inversion.

Background
----------
Standard chi-squared LR test (M3-asymmetric vs M3-m3sym, df=1) gives LR=2.24,
p_chi2 = 0.13. But alpha-(M1) and alpha-(M2) sit at the lower bound zero in
both the full and the constrained fits. Self & Liang (1987) show that with
nuisance parameters at the boundary, the asymptotic chi-squared distribution
is invalid — the LR_b distribution under H0 is a (potentially complicated)
mixture of chi-squared distributions on different degrees of freedom.

The parametric bootstrap LR test gives a boundary-aware p-value: simulate
data from the constrained model, refit both, and use the empirical LR_b
distribution as the null reference.

Per E1 design review:
- Do NOT clip simulated y to [1, 7]; the fitted model is Gaussian, so the
  bootstrap must simulate from a Gaussian (clipping makes the simulation
  inconsistent with the fitted model).
- Use B=1000 (vs B=500) for tighter upper-tail p-value precision.
- Standard "+1" correction in the p-value: (#{LR_b >= LR_obs} + 1) / (B+1).
- Report: empirical p, percentiles of LR_b, fraction of LR_b < 0 (numerical
  diagnostics).

Run time: ~3-5 seconds per bootstrap with multi-start L-BFGS-B ~= 1-2 hours
single-threaded for B=1000. Saved to bootstrap_lr_m3.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_trials
from models import Model3AsymmetricBelief, Model3AsymmetricM3Sym


DIR_NEW = Path("experiments/results/per-stage-localization-forced-honesty-overshoot")
DIR_PRIOR = Path("experiments/results/stage-base-vs-instruct-positivity-prior")
OUT_DIR = Path("experiments/results/cognitive-model-comparison-polite-speech-post")
OUT_PATH = OUT_DIR / "bootstrap_lr_m3.json"

B = 1000          # bootstrap replicates (per LLM-review recommendation)
SEED = 42         # master RNG seed
N_STARTS = 3      # multi-start L-BFGS-B per fit
MAXITER = 2000
FTOL = 1e-9


def fit_model(model, df, n_starts=N_STARTS, seed=0):
    """Multi-start L-BFGS-B fit. Returns (best_neg_log_lik, best_params)."""
    init = model.init_params(df)
    bounds = model.bounds(df)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    rng = np.random.default_rng(seed)

    res = minimize(model.neg_log_likelihood, init, args=(df,),
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": MAXITER, "ftol": FTOL})
    best = (res.fun, res.x.copy())

    for k in range(n_starts - 1):
        perturb = rng.normal(0.0, 0.15, size=len(init))
        new_init = np.clip(init + perturb * (hi - lo) * 0.1,
                           lo + 1e-6, hi - 1e-6)
        r = minimize(model.neg_log_likelihood, new_init, args=(df,),
                     method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": MAXITER, "ftol": FTOL})
        if r.fun < best[0]:
            best = (r.fun, r.x.copy())
    return best


def simulate_dataset(model, df_template, params_with_sigma, rng):
    """Simulate y from `model` at `params_with_sigma` on the X-structure of
    df_template. Returns a NEW DataFrame with replaced 'rating' column.

    Per E1 review: NO clipping. The fitted model is Gaussian; simulating
    with clipping is inconsistent with the fitted distribution.
    """
    sigma = params_with_sigma[-1]
    mu = model.predict_means(df_template, params_with_sigma[:-1])
    noise = rng.normal(0.0, sigma, size=len(mu))
    y_synth = mu + noise
    df_new = df_template.copy()
    df_new["rating"] = y_synth
    return df_new


def main():
    print("=" * 70)
    print("E1: Parametric-bootstrap LR test for M3 stage-specific asymmetry")
    print("=" * 70)
    print(f"B = {B}, n_starts = {N_STARTS}, seed = {SEED}")
    print()

    # Step 1: load data
    df = load_trials(DIR_NEW, DIR_PRIOR)
    n = len(df)
    print(f"Loaded N = {n} unique observations.\n")

    # Step 2: fit M3-asym and M3-m3sym on real data
    print("Fitting M3-asym (full) on real data...")
    m_full = Model3AsymmetricBelief()
    _ = m_full.init_params(df)  # warm internal state
    nll_full_real, _ = fit_model(m_full, df)
    print(f"  -LL_full(real) = {nll_full_real:.4f}")

    print("Fitting M3-m3sym (constrained) on real data...")
    m_red = Model3AsymmetricM3Sym()
    _ = m_red.init_params(df)
    nll_red_real, theta_red_real = fit_model(m_red, df)
    print(f"  -LL_red(real)  = {nll_red_real:.4f}")

    lr_obs = 2 * (nll_red_real - nll_full_real)
    print(f"  Observed LR    = {lr_obs:.4f}  (df=1, chi^2 p = "
          f"{1.0 - _chi2_cdf_1df(lr_obs):.4f})")

    # Step 3: bootstrap loop
    print(f"\nRunning {B} parametric bootstrap replicates...")
    master_rng = np.random.default_rng(SEED)
    boot_seeds = master_rng.integers(0, 2**31 - 1, size=B)

    lr_boot = np.full(B, np.nan)
    n_failures = 0
    out_of_support = 0
    for b, seed_b in enumerate(boot_seeds):
        rng_b = np.random.default_rng(int(seed_b))
        df_synth = simulate_dataset(m_red, df, theta_red_real, rng_b)

        # Track how many synthetic y values fell outside Likert support [1, 7]
        # (informational; we are NOT clipping)
        out_of_support += int(((df_synth["rating"] < 1.0) |
                               (df_synth["rating"] > 7.0)).sum())

        try:
            m_f = Model3AsymmetricBelief(); _ = m_f.init_params(df_synth)
            nll_f, _ = fit_model(m_f, df_synth, n_starts=N_STARTS,
                                  seed=int(seed_b))
            m_r = Model3AsymmetricM3Sym(); _ = m_r.init_params(df_synth)
            nll_r, _ = fit_model(m_r, df_synth, n_starts=N_STARTS,
                                  seed=int(seed_b) + 1)
            lr_boot[b] = 2 * (nll_r - nll_f)
        except Exception as e:
            n_failures += 1
            if n_failures <= 3:
                print(f"  [warn] b={b}: {type(e).__name__}: {e}")

        if (b + 1) % 50 == 0:
            done = b + 1
            ok = (~np.isnan(lr_boot[:done])).sum()
            elapsed = ok  # placeholder; could time
            tail = (lr_boot[:done] >= lr_obs).sum()
            print(f"  done {done}/{B}  (ok={ok}, tail#={tail}, "
                  f"running p~{(tail+1)/(ok+1):.4f})")

    # Step 4: report
    valid = lr_boot[~np.isnan(lr_boot)]
    n_neg = (valid < 0).sum()
    n_ge = (valid >= lr_obs).sum()
    p_empirical = (n_ge + 1) / (len(valid) + 1)

    total_obs_in_sim = len(df) * B
    pct_out = 100.0 * out_of_support / total_obs_in_sim if total_obs_in_sim else 0.0

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Observed LR                 : {lr_obs:.4f}")
    print(f"Chi^2(1) p-value            : {1.0 - _chi2_cdf_1df(lr_obs):.4f}")
    print(f"B (bootstrap replicates)    : {B}")
    print(f"Successful fits             : {len(valid)}/{B}")
    print(f"Convergence failures        : {n_failures}")
    print(f"LR_b < 0 (numerical issues) : {n_neg} ({100*n_neg/max(1,len(valid)):.1f}%)")
    print(f"#{{LR_b >= LR_obs}}            : {n_ge}/{len(valid)}")
    print(f"Empirical p-value           : {p_empirical:.4f}")
    print()
    print("LR_b distribution percentiles (under H0 simulation):")
    for q in [5, 25, 50, 75, 90, 95, 99]:
        v = float(np.percentile(valid, q))
        print(f"  P{q:2d} = {v:.4f}")
    print()
    print(f"Diagnostic: synthetic y values outside [1, 7]: "
          f"{out_of_support}/{total_obs_in_sim} ({pct_out:.2f}%)")
    if pct_out > 10.0:
        print("  WARNING: > 10% of simulated y outside Likert support.")
        print("  Gaussian-model approximation may be questionable.")
    print()

    # Pre-registered interpretation
    if p_empirical < 0.05:
        interpretation = ("M3-specific inversion SUPPORTED by boundary-aware "
                          "bootstrap LR (p < 0.05).")
    elif p_empirical < 0.20:
        interpretation = ("Marginal: bootstrap p ~~ chi^2 p; the chi^2 "
                          "approximation was roughly correct.")
    else:
        interpretation = ("M3-specific inversion NOT supported under proper "
                          "boundary-aware inference; chi^2 was anti-conservative.")
    print(f"Pre-registered interpretation: {interpretation}")

    # Step 5: save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "lr_observed": float(lr_obs),
        "chi2_p_value": float(1.0 - _chi2_cdf_1df(lr_obs)),
        "B": B,
        "n_valid": int(len(valid)),
        "n_failures": int(n_failures),
        "n_negative_lr": int(n_neg),
        "n_ge_lr_obs": int(n_ge),
        "empirical_p_value": float(p_empirical),
        "percentiles": {f"P{q}": float(np.percentile(valid, q))
                        for q in [5, 25, 50, 75, 90, 95, 99]},
        "lr_boot": valid.tolist(),
        "synth_out_of_support_pct": float(pct_out),
        "interpretation": interpretation,
        "seed": SEED,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


def _chi2_cdf_1df(x):
    """CDF of chi-squared(1) for sanity-check display only."""
    from math import erf, sqrt
    return erf(sqrt(max(x, 0.0)) / sqrt(2.0))


if __name__ == "__main__":
    main()
