"""E3: Hierarchical Bayesian M3-asymmetric via PyMC.

Goals:
  1. Posterior credible intervals on alpha+/alpha- per stage (proper
     uncertainty quantification, no boundary-LR issues).
  2. Per-target shrinkage on the b_target intercepts.
  3. A principled answer to "is the M3 inversion supported?" via the
     posterior of alpha-(M3) - alpha+(M3).

Model:
    b_target  ~ Normal(mu_b, sigma_b)           # random intercept per target
    alpha_+(s) ~ HalfNormal(0, 1)               # positive learning rate per stage (M1..M3)
    alpha_-(s) ~ HalfNormal(0, 1)               # negative learning rate per stage
    I(c) free per condition (anchor: I_default = 1)
    sigma ~ HalfNormal(0, 1)
    mu[i] = b_target[i] + sign(dir(cond[i])) * alpha(stage[i], dir(cond[i])) *
            I(cond[i]) * g(b_target[i])
    y[i] ~ Normal(mu[i], sigma)

Where g(b) = 4(b-1)(7-b)/36 clipped to [0, 1] (ambiguity gate, same as ML fit).

Sampling: NUTS, 4 chains x 2000 draws (1000 tune + 1000 sample), target_accept=0.95.
Saved: posterior credible intervals + key contrasts to hierarchical_bayesian_m3.json.

Run time: ~10-20 min on laptop CPU (no GPU needed for this size).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_trials


DIR_NEW = Path("experiments/results/per-stage-localization-forced-honesty-overshoot")
DIR_PRIOR = Path("experiments/results/stage-base-vs-instruct-positivity-prior")
OUT_DIR = Path("experiments/results/cognitive-model-comparison-polite-speech-post")
OUT_PATH = OUT_DIR / "hierarchical_bayesian_m3.json"

ANCHOR_CONDITION = "default"


def main():
    print("=" * 70)
    print("E3: Hierarchical Bayesian M3-asym (PyMC NUTS)")
    print("=" * 70)

    df = load_trials(DIR_NEW, DIR_PRIOR)
    n = len(df)
    print(f"Loaded N = {n} unique observations.\n")

    targets = sorted(df["target_id"].unique().tolist())
    stages = ["M0", "M1", "M2", "M3"]  # M0 anchored to alpha = 0
    free_stages = ["M1", "M2", "M3"]
    conditions = sorted(df["condition"].unique().tolist())
    free_conditions = [c for c in conditions if c != ANCHOR_CONDITION]

    # Cell direction (fixed, same as M3-asym ML model)
    anchor_mean = float(df[df["condition"] == ANCHOR_CONDITION]["rating"].mean())
    cond_means = df.groupby("condition")["rating"].mean()
    cond_direction = {
        c: (+1 if float(cond_means[c]) >= anchor_mean else -1)
        for c in cond_means.index
    }
    cond_direction[ANCHOR_CONDITION] = +1

    # Encode indices
    target_idx = df["target_id"].map({t: i for i, t in enumerate(targets)}).to_numpy()
    stage_idx = df["stage"].map({s: i for i, s in enumerate(stages)}).to_numpy()
    cond_idx = df["condition"].map({c: i for i, c in enumerate(conditions)}).to_numpy()
    direction = df["condition"].map(cond_direction).to_numpy().astype(np.float64)
    y_obs = df["rating"].to_numpy().astype(np.float64)

    print(f"  targets: {len(targets)}, stages: {len(stages)}, conditions: {len(conditions)}")
    print(f"  free stages: {free_stages}")
    print(f"  free conditions: {free_conditions}")
    print(f"  anchor condition: {ANCHOR_CONDITION} (intensity = 1)")
    print()

    # --- Build PyMC model ----------------------------------------------
    with pm.Model() as model:
        # Per-target intercepts (random effects with partial pooling)
        mu_b = pm.Normal("mu_b", mu=4.5, sigma=2.0)
        sigma_b = pm.HalfNormal("sigma_b", sigma=2.0)
        b_target = pm.Normal("b_target", mu=mu_b, sigma=sigma_b,
                             shape=len(targets))

        # Per-stage learning rates. M0 fixed at 0 by anchor.
        # We model alpha_+ and alpha_- for M1, M2, M3 (3 stages each).
        alpha_pos_free = pm.HalfNormal("alpha_pos_free", sigma=1.0,
                                       shape=len(free_stages))
        alpha_neg_free = pm.HalfNormal("alpha_neg_free", sigma=1.0,
                                       shape=len(free_stages))

        # Build full per-stage vectors: [0, alpha_+(M1), alpha_+(M2), alpha_+(M3)]
        zeros = pt.zeros(1)
        alpha_pos_full = pt.concatenate([zeros, alpha_pos_free])
        alpha_neg_full = pt.concatenate([zeros, alpha_neg_free])

        # Per-condition intensity (anchor at I_default = 1)
        I_free = pm.HalfNormal("I_free", sigma=2.0, shape=len(free_conditions))
        # Build I-vector with anchor inserted
        anchor_pos = conditions.index(ANCHOR_CONDITION)
        I_parts = []
        free_iter = iter(range(len(free_conditions)))
        for i, c in enumerate(conditions):
            if c == ANCHOR_CONDITION:
                I_parts.append(pt.ones(1))
            else:
                I_parts.append(I_free[next(free_iter)].reshape((1,)))
        I_full = pt.concatenate(I_parts)

        # Per-observation b_t and ambiguity gate
        b_t_obs = b_target[target_idx]
        g_obs = pt.clip(4.0 * (b_t_obs - 1.0) * (7.0 - b_t_obs) / 36.0, 0.0, 1.0)

        # Alpha lookup per observation (vectorized switch via direction)
        ap_obs = alpha_pos_full[stage_idx]
        an_obs = alpha_neg_full[stage_idx]
        alpha_obs = pt.where(direction > 0, ap_obs, an_obs)

        I_obs = I_full[cond_idx]

        mu = b_t_obs + direction * alpha_obs * I_obs * g_obs
        sigma = pm.HalfNormal("sigma", sigma=1.5)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)

    print("Model built. Sampling 4 chains x 2000 draws (1000 tune)...")
    print()

    with model:
        idata = pm.sample(
            draws=1000, tune=1000, chains=4, cores=4,
            target_accept=0.95, return_inferencedata=True,
            random_seed=2027,
            progressbar=False,
        )

    print("\nSampling complete.\n")

    # --- Diagnostics ---------------------------------------------------
    # arviz 1.x renamed hdi_prob -> ci_prob in summary(). Be defensive about
    # both the kwarg and the returned column names across arviz versions.
    try:
        summary = az.summary(
            idata,
            var_names=["mu_b", "sigma_b", "alpha_pos_free", "alpha_neg_free", "sigma"],
            ci_prob=0.95,
        )
    except TypeError:
        summary = az.summary(
            idata,
            var_names=["mu_b", "sigma_b", "alpha_pos_free", "alpha_neg_free", "sigma"],
        )
    print(summary)
    print()

    # Convergence: r_hat and ess. Column names vary across arviz versions.
    def _col(df, *candidates):
        for c in candidates:
            if c in df.columns:
                return df[c].to_numpy()
        return np.array([np.nan])

    rhats = _col(summary, "r_hat", "rhat", "r_hat_")
    max_rhat = float(np.nanmax(rhats))
    print(f"Max r_hat: {max_rhat:.4f}  (target: < 1.01)")
    ess = _col(summary, "ess_bulk", "ess", "ess_mean")
    min_ess = float(np.nanmin(ess))
    print(f"Min ESS bulk: {min_ess:.0f}  (target: > 400 for 4 chains)")

    # --- Posterior contrasts ------------------------------------------
    # All HDIs computed directly as numpy percentiles on the posterior
    # samples — robust to arviz API drift.
    posterior = idata.posterior

    def _samples(var, idx):
        """Flatten posterior samples for var[..., idx] across chains+draws."""
        dim = f"{var}_dim_0"
        return posterior[var].isel({dim: idx}).values.flatten()

    diff_flat = _samples("alpha_neg_free", 2) - _samples("alpha_pos_free", 2)  # M3 idx 2

    print("\nPer-stage learning rates (95% credible interval = 2.5/97.5 percentiles):")
    print(f"  {'stage':<6} {'alpha+ mean':>12} {'alpha+ 95% CI':>22} "
          f"{'alpha- mean':>12} {'alpha- 95% CI':>22}")
    contrasts = {}
    for i, s in enumerate(free_stages):
        ap = _samples("alpha_pos_free", i)
        an = _samples("alpha_neg_free", i)
        ap_mean, an_mean = float(ap.mean()), float(an.mean())
        ap_lo, ap_hi = float(np.percentile(ap, 2.5)), float(np.percentile(ap, 97.5))
        an_lo, an_hi = float(np.percentile(an, 2.5)), float(np.percentile(an, 97.5))
        print(f"  {s:<6} {ap_mean:>12.4f}  [{ap_lo:>+.3f},{ap_hi:>+.3f}] "
              f"{an_mean:>12.4f}  [{an_lo:>+.3f},{an_hi:>+.3f}]")
        contrasts[s] = {
            "alpha_pos_mean": ap_mean, "alpha_pos_hdi": [ap_lo, ap_hi],
            "alpha_neg_mean": an_mean, "alpha_neg_hdi": [an_lo, an_hi],
        }

    print()
    print(f"Diagnostic contrast: alpha-(M3) - alpha+(M3)")
    diff_mean = float(np.mean(diff_flat))
    diff_hdi_lo = float(np.percentile(diff_flat, 2.5))
    diff_hdi_hi = float(np.percentile(diff_flat, 97.5))
    diff_pos = float(np.mean(diff_flat > 0))
    print(f"  Posterior mean: {diff_mean:+.4f}")
    print(f"  95% HDI:        [{diff_hdi_lo:+.4f}, {diff_hdi_hi:+.4f}]")
    print(f"  P(diff > 0):    {diff_pos:.4f}")

    if diff_hdi_lo > 0:
        interp = "M3 inversion SUPPORTED (95% HDI strictly above zero)"
    elif diff_pos > 0.95:
        interp = "M3 inversion strongly suggested (P>0 > 95%, HDI straddles 0 narrowly)"
    elif diff_pos > 0.80:
        interp = "M3 inversion suggested but not strongly supported (P>0 ~ 80-95%)"
    else:
        interp = "M3 inversion NOT supported by Bayesian inference"
    print(f"  Interpretation: {interp}")

    # --- Save ----------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "model": "hierarchical_bayesian_M3_asym",
        "n_observations": int(n),
        "n_targets": len(targets),
        "free_stages": free_stages,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess,
        "stage_contrasts": contrasts,
        "M3_diff": {
            "posterior_mean": diff_mean,
            "hdi_2.5_pct": diff_hdi_lo,
            "hdi_97.5_pct": diff_hdi_hi,
            "P_diff_gt_0": diff_pos,
        },
        "interpretation": interp,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
