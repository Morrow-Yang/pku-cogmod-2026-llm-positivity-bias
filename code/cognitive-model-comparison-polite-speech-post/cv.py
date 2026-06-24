"""Out-of-sample cross-validation for the cognitive-model comparison.

WHY k-fold over trials (not leave-one-target-out / leave-one-cell-out):
every model here carries a per-target intercept b_t and (for the structured
models) a per-cell shift δ_{s,c}. Holding out a whole target or a whole cell
leaves that group's parameter unfit, so its held-out prediction is undefined —
LOTO/LOCO are ill-posed for this design. Instead we use K-fold over trials,
STRATIFIED by (stage, condition) cell so every training fold still covers every
target and every cell. This measures genuine out-of-sample predictive accuracy
and is what distinguishes a parsimonious structured model from an overfit one
(e.g. M4/M5 with 34–48 per-cell parameters): in-sample BIC can be gamed by
parameters, held-out predictive log-likelihood cannot.

Primary metric: total held-out Gaussian predictive log-likelihood (higher =
better). Secondary: held-out RMSE.

Run from repo root:  .venv/bin/python experiments/code/.../cv.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from data import load_trials
from models import (Model0LinearAdditive, Model1Null, Model2KundaGated,
                    Model2NoGate, Model2FreeGate,
                    Model3AsymmetricBelief, Model3SymmetricBelief,
                    Model4RSAPoliteSpeaker, Model5RSALicenseConditional)

DIR_NEW = Path("experiments/results/per-stage-localization-forced-honesty-overshoot")
DIR_PRIOR = Path("experiments/results/stage-base-vs-instruct-positivity-prior")
OUT_DIR = Path("experiments/results/cognitive-model-comparison-polite-speech-post")

K_FOLDS = 10
SEED = 0


def _fit_one(ModelClass, train_df, n_starts=3, maxiter=10000):
    """Fit a fresh model instance on train_df; return (model, params)."""
    m = ModelClass()
    init = m.init_params(train_df)
    bounds = m.bounds(train_df)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    rng = np.random.default_rng(SEED)

    def run(x0):
        return minimize(m.neg_log_likelihood, x0, args=(train_df,),
                        method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7})

    best = run(init)
    for _ in range(n_starts - 1):
        x0 = np.clip(init + rng.normal(0, 0.3, len(init)) * (hi - lo) * 0.1,
                     lo + 1e-6, hi - 1e-6)
        r = run(x0)
        if r.fun < best.fun:
            best = r
    return m, best.x


def _heldout_loglik(model, params, test_df) -> tuple[float, float]:
    """Gaussian predictive log-lik and squared-error sum on held-out rows."""
    sigma = params[-1]
    mu = model.predict_means(test_df, params[:-1])
    resid = test_df["rating"].to_numpy() - mu
    n = len(resid)
    ll = float(-0.5 * np.sum(resid ** 2) / sigma ** 2
               - n * np.log(sigma) - 0.5 * n * np.log(2 * np.pi))
    return ll, float(np.sum(resid ** 2))


def stratified_folds(df, k, seed):
    """Assign each row a fold 0..k-1, stratified within (stage, condition)."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(df), dtype=int)
    for _, idx in df.groupby(["stage", "condition"]).groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


def main():
    df = load_trials(DIR_NEW, DIR_PRIOR).reset_index(drop=True)
    n = len(df)
    print(f"Loaded {n} trials. {K_FOLDS}-fold stratified CV.\n")
    fold = stratified_folds(df, K_FOLDS, SEED)

    models = [
        ("M0_linear_additive", Model0LinearAdditive),
        ("M1_null", Model1Null),
        ("M2_kunda_gated", Model2KundaGated),
        ("M2_nogate", Model2NoGate),
        ("M2_freegate", Model2FreeGate),
        ("M3_symmetric_belief", Model3SymmetricBelief),
        ("M3_asymmetric_belief", Model3AsymmetricBelief),
        ("M4_rsa_polite_speaker", Model4RSAPoliteSpeaker),
        ("M5_rsa_license_conditional", Model5RSALicenseConditional),
    ]

    results = []
    print("=" * 78)
    print(f"{'Model':<32} {'cv_loglik':>12} {'cv_rmse':>10} {'cv_ll/trial':>12}")
    print("=" * 78)
    for name, Cls in models:
        total_ll, total_sse = 0.0, 0.0
        for f in range(K_FOLDS):
            train_df = df[fold != f].reset_index(drop=True)
            test_df = df[fold == f].reset_index(drop=True)
            model, params = _fit_one(Cls, train_df)
            ll, sse = _heldout_loglik(model, params, test_df)
            total_ll += ll; total_sse += sse
        rmse = float(np.sqrt(total_sse / n))
        results.append({"model": name, "cv_loglik": total_ll,
                        "cv_rmse": rmse, "cv_loglik_per_trial": total_ll / n})
        print(f"{name:<32} {total_ll:>12.2f} {rmse:>10.4f} {total_ll/n:>12.4f}")

    best = max(results, key=lambda r: r["cv_loglik"])
    print(f"\nBest by held-out log-likelihood: {best['model']}")
    with open(OUT_DIR / "cv.json", "w") as fh:
        json.dump({"k_folds": K_FOLDS, "n_trials": n,
                   "results": results, "best_by_cv_loglik": best["model"]},
                  fh, indent=2)
    print(f"Wrote {OUT_DIR/'cv.json'}")


if __name__ == "__main__":
    main()
