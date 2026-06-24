"""Posterior-predictive checks for the winning cognitive models.

A cognitive model earns trust not by BIC alone but by REPRODUCING the
qualitative phenomena it was built to explain. We check two signatures:

  (1) Below-base overshoot: the M3-forced cell rates fictional/novel targets
      BELOW the M0-default base-model baseline (the dissociation that motivates
      the whole project). We test whether the fitted model regenerates a
      negative (M3-forced − M0-default) contrast.

  (2) Per-cell fit: simulated cell means (with fitted σ noise) bracket the
      observed cell means — i.e. the model is not systematically biased.

For each model we simulate N synthetic datasets from the fitted parameters,
recompute the summary statistics, and compare to the observed values
(posterior-predictive p-value = fraction of sims at least as extreme).

Run from repo root.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from data import load_trials
from models import Model2KundaGated, Model3AsymmetricBelief

DIR_NEW = Path("experiments/results/per-stage-localization-forced-honesty-overshoot")
DIR_PRIOR = Path("experiments/results/stage-base-vs-instruct-positivity-prior")
OUT_DIR = Path("experiments/results/cognitive-model-comparison-polite-speech-post")
N_SIM = 2000
SEED = 0


def fit(model, df, n_starts=5, maxiter=10000):
    init = model.init_params(df); bounds = model.bounds(df)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    rng = np.random.default_rng(SEED)
    run = lambda x0: minimize(model.neg_log_likelihood, x0, args=(df,),
                              method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7})
    best = run(init)
    for _ in range(n_starts - 1):
        x0 = np.clip(init + rng.normal(0, 0.3, len(init)) * (hi - lo) * 0.1,
                     lo + 1e-6, hi - 1e-6)
        r = run(x0)
        if r.fun < best.fun:
            best = r
    return best.x


def overshoot_contrast(df, ratings):
    """(M3, forced) mean − (M0, default) mean over the SAME novel targets."""
    novel = {"adirenia", "che_pact", "vellinkov", "khantelan"}
    m = (df["stage"] == "M3") & (df["condition"] == "forced") & df["target_id"].isin(novel)
    b = (df["stage"] == "M0") & (df["condition"] == "default") & df["target_id"].isin(novel)
    if m.sum() == 0 or b.sum() == 0:
        return np.nan
    return float(ratings[m.to_numpy()].mean() - ratings[b.to_numpy()].mean())


def run_model(name, ModelClass):
    df = load_trials(DIR_NEW, DIR_PRIOR).reset_index(drop=True)
    model = ModelClass()
    params = fit(model, df)
    sigma = params[-1]
    mu = model.predict_means(df, params[:-1])
    obs = df["rating"].to_numpy()

    # (1) overshoot signature
    obs_contrast = overshoot_contrast(df, obs)
    rng = np.random.default_rng(SEED)
    sim_contrasts = np.empty(N_SIM)
    for i in range(N_SIM):
        sim = mu + rng.normal(0, sigma, len(mu))
        sim_contrasts[i] = overshoot_contrast(df, sim)
    # PP p-value: fraction of sims with contrast <= 0 (model regenerates overshoot)
    frac_neg = float(np.mean(sim_contrasts < 0))
    ci = [float(np.percentile(sim_contrasts, 2.5)),
          float(np.percentile(sim_contrasts, 97.5))]

    # (2) per-cell predicted vs observed
    cell_obs = df.groupby(["stage", "condition"])["rating"].mean()
    df_pred = df.assign(pred=mu)
    cell_pred = df_pred.groupby(["stage", "condition"])["pred"].mean()
    sse = float(((cell_obs - cell_pred) ** 2).sum())
    ss_tot = float(((cell_obs - cell_obs.mean()) ** 2).sum())
    r2_cell = 1 - sse / ss_tot
    # overall trial-level R²
    r2_trial = 1 - float(np.sum((obs - mu) ** 2)) / float(np.sum((obs - obs.mean()) ** 2))

    res = {
        "model": name,
        "observed_overshoot_contrast": obs_contrast,
        "sim_overshoot_mean": float(sim_contrasts.mean()),
        "sim_overshoot_95ci": ci,
        "frac_sims_overshoot_negative": frac_neg,
        "regenerates_overshoot": bool(obs_contrast < 0 and ci[0] < 0),
        "cell_level_R2": r2_cell,
        "trial_level_R2": r2_trial,
        "n_sim": N_SIM,
    }
    print(f"\n[{name}]")
    print(f"  observed (M3-forced − M0-default) on novel targets: {obs_contrast:+.3f}")
    print(f"  model-simulated contrast: {sim_contrasts.mean():+.3f}  95% CI {ci}")
    print(f"  fraction of sims with negative (below-base) contrast: {frac_neg:.3f}")
    print(f"  cell-level R² = {r2_cell:.3f}   trial-level R² = {r2_trial:.3f}")
    return res


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"checks": [run_model("M2_kunda_gated", Model2KundaGated),
                      run_model("M3_asymmetric_belief", Model3AsymmetricBelief)]}
    with open(OUT_DIR / "ppc.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR/'ppc.json'}")


if __name__ == "__main__":
    main()
