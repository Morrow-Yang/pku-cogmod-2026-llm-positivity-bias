"""E2: Model-recovery confusion matrix.

For each of the 7 candidate models, simulate K datasets at its MLE,
then fit ALL 7 models to each synthetic dataset. Record which model wins
by BIC. Outputs a 7x7 confusion matrix:

    M[i, j] = fraction of times, when the data was generated from model i,
              model j had the lowest BIC.

Interpretation:
    - Diagonal entries should be high (correct identification).
    - Off-diagonal entries identify confusable model pairs.
    - If M2_ag wins when M0_lin is the generator, the cognitive model
      "wins" on data with no cognitive structure (over-fitting concern).
    - If a generator's diagonal is low (<70%), that model is not uniquely
      identified by the data design.

Per E2 design review: K=20 sims per generator × 7 generators × 7 fitters
= 980 fits. At ~3 sec per multi-start fit, ~50 min single-threaded.

Saved to model_recovery_matrix.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_trials
from models import (
    Model0LinearAdditive, Model1Null, Model2KundaGated,
    Model3SymmetricBelief, Model3AsymmetricBelief,
    Model4RSAPoliteSpeaker, Model5RSALicenseConditional,
)


DIR_NEW = Path("experiments/results/per-stage-localization-forced-honesty-overshoot")
DIR_PRIOR = Path("experiments/results/stage-base-vs-instruct-positivity-prior")
OUT_DIR = Path("experiments/results/cognitive-model-comparison-polite-speech-post")
OUT_PATH = OUT_DIR / "model_recovery_matrix.json"

K = 20            # simulations per generator (per E2 review)
SEED = 1729       # master RNG seed
N_STARTS = 3
MAXITER = 1500
FTOL = 1e-8

MODEL_CLASSES = [
    ("M0_lin",   Model0LinearAdditive),
    ("M1_null",  Model1Null),
    ("M2_ag",    Model2KundaGated),       # ambiguity-gated cell-shift
    ("M3_sym",   Model3SymmetricBelief),
    ("M3_asym",  Model3AsymmetricBelief),
    ("M4_RSA",   Model4RSAPoliteSpeaker),
    ("M5_RSA+",  Model5RSALicenseConditional),
]


def fit_one(model_class, df, n_starts=N_STARTS, seed=0):
    """Multi-start L-BFGS-B fit of a fresh model instance to df.
    Returns (neg_log_lik, n_params, bic_value, params)."""
    m = model_class()
    init = m.init_params(df)
    bounds = m.bounds(df)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    rng = np.random.default_rng(seed)

    res = minimize(m.neg_log_likelihood, init, args=(df,),
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": MAXITER, "ftol": FTOL})
    best = (res.fun, res.x.copy())

    for k in range(n_starts - 1):
        perturb = rng.normal(0.0, 0.15, size=len(init))
        new_init = np.clip(init + perturb * (hi - lo) * 0.1,
                           lo + 1e-6, hi - 1e-6)
        r = minimize(m.neg_log_likelihood, new_init, args=(df,),
                     method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": MAXITER, "ftol": FTOL})
        if r.fun < best[0]:
            best = (r.fun, r.x.copy())

    n_params = m.n_params(df)
    n = len(df)
    bic = n_params * np.log(n) - 2 * (-best[0])
    return best[0], n_params, bic, best[1], m


def simulate(model_instance, df_template, params_with_sigma, rng):
    """Simulate y from `model_instance` at given params. NO CLIPPING."""
    sigma = params_with_sigma[-1]
    mu = model_instance.predict_means(df_template, params_with_sigma[:-1])
    noise = rng.normal(0.0, sigma, size=len(mu))
    df_new = df_template.copy()
    df_new["rating"] = mu + noise
    return df_new


def main():
    print("=" * 70)
    print("E2: Model-recovery confusion matrix")
    print("=" * 70)
    print(f"K = {K} sims per generator, {len(MODEL_CLASSES)} models, seed = {SEED}")
    print()

    df = load_trials(DIR_NEW, DIR_PRIOR)
    n = len(df)
    print(f"Loaded N = {n} unique observations.\n")

    # Step 1: fit each model on the real data to get MLE generator parameters
    print("Fitting each model on real data (generator MLEs)...")
    generator_mles = {}
    for name, cls in MODEL_CLASSES:
        nll, k, bic, params, m_inst = fit_one(cls, df)
        generator_mles[name] = (m_inst, params)
        print(f"  {name:<10s} k={k:>3d}  -LL={nll:>9.2f}  BIC={bic:>9.2f}")

    # Step 2: build the K × len(MODEL_CLASSES) × len(MODEL_CLASSES) BIC tensor.
    # bic_tensor[k, i, j] = BIC when fitting model j on the k-th sim from generator i.
    n_models = len(MODEL_CLASSES)
    bic_tensor = np.full((K, n_models, n_models), np.nan)
    convergence_fail = 0

    master_rng = np.random.default_rng(SEED)

    for i, (gen_name, gen_cls) in enumerate(MODEL_CLASSES):
        gen_inst, gen_params = generator_mles[gen_name]
        print(f"\n--- Generator: {gen_name} ({i+1}/{n_models}) ---")
        for k_sim in range(K):
            seed_k = int(master_rng.integers(0, 2**31 - 1))
            rng_sim = np.random.default_rng(seed_k)
            df_synth = simulate(gen_inst, df, gen_params, rng_sim)

            for j, (fit_name, fit_cls) in enumerate(MODEL_CLASSES):
                try:
                    nll, k_p, bic, _, _ = fit_one(fit_cls, df_synth,
                                                   n_starts=N_STARTS,
                                                   seed=seed_k + j)
                    bic_tensor[k_sim, i, j] = bic
                except Exception as e:
                    convergence_fail += 1
                    if convergence_fail <= 3:
                        print(f"  [warn] sim={k_sim} gen={gen_name} fit={fit_name}: "
                              f"{type(e).__name__}: {e}")
            best_j = int(np.nanargmin(bic_tensor[k_sim, i]))
            best_name = MODEL_CLASSES[best_j][0]
            print(f"  sim {k_sim+1:>2d}/{K}  winner = {best_name:<10s}  "
                  f"(BIC={bic_tensor[k_sim, i, best_j]:.2f})")

    # Step 3: compute confusion matrix
    confusion = np.zeros((n_models, n_models))
    for i in range(n_models):
        winners = np.nanargmin(bic_tensor[:, i, :], axis=1)
        for j in range(n_models):
            confusion[i, j] = float((winners == j).sum()) / K

    print("\n" + "=" * 70)
    print("Confusion matrix (rows = generator, cols = winner)")
    print("=" * 70)
    print(f"{'gen \\ win':<10s}" + " ".join(f"{n:>9s}" for n, _ in MODEL_CLASSES))
    for i, (gen_name, _) in enumerate(MODEL_CLASSES):
        row = " ".join(f"{confusion[i, j]:>9.2f}" for j in range(n_models))
        print(f"{gen_name:<10s} {row}")
    print()

    diag = np.diag(confusion)
    print(f"Mean diagonal (correct ID rate): {diag.mean():.3f}")
    print(f"Min diagonal (worst-identified):  {diag.min():.3f}  "
          f"(generator: {MODEL_CLASSES[int(np.argmin(diag))][0]})")
    print(f"Convergence failures: {convergence_fail}/"
          f"{K*n_models*n_models}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "K": K,
        "model_names": [n for n, _ in MODEL_CLASSES],
        "confusion_matrix": confusion.tolist(),
        "diagonal": diag.tolist(),
        "mean_diagonal": float(diag.mean()),
        "min_diagonal": float(diag.min()),
        "convergence_failures": int(convergence_fail),
        "seed": SEED,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
