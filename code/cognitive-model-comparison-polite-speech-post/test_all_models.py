"""Head-to-head fit of M1-M5 on the trial-level dataset.

Reports:
  - Convergence, -LL, k, AIC, BIC for each model
  - ΔAIC / ΔBIC vs M1 (null) and vs M4 (the best mechanism-rich baseline)
  - Key diagnostic parameters for M3 (α±) and M5 (β_chat, β_license, γ_k)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from data import load_trials
from models import (Model1Null, Model2KundaGated, Model3AsymmetricBelief,
                    Model4RSAPoliteSpeaker, Model5RSALicenseConditional)

DIR_NEW = Path("experiments/results/per-stage-localization-forced-honesty-overshoot")
DIR_PRIOR = Path("experiments/results/stage-base-vs-instruct-positivity-prior")


def fit(model, df, maxiter=2000):
    init = model.init_params(df)
    res = minimize(
        model.neg_log_likelihood, init, args=(df,),
        method="L-BFGS-B", bounds=model.bounds(df),
        options={"maxiter": maxiter, "ftol": 1e-9},
    )
    return res


def aic_bic(neg_ll, k, n):
    ll = -neg_ll
    return 2 * k - 2 * ll, k * np.log(n) - 2 * ll


def main():
    df = load_trials(DIR_NEW, DIR_PRIOR)
    n = len(df)
    print(f"Loaded {n} trials.\n")

    models = [
        ("M1_null", Model1Null()),
        ("M2_kunda_gated", Model2KundaGated()),
        ("M3_asymmetric_belief", Model3AsymmetricBelief()),
        ("M4_rsa_polite_speaker", Model4RSAPoliteSpeaker()),
        ("M5_rsa_license_conditional", Model5RSALicenseConditional()),
    ]

    results = {}
    print("=" * 85)
    print(f"{'Model':<32} {'conv':>6} {'k':>4} {'-LL':>11} {'AIC':>11} {'BIC':>11}")
    print("=" * 85)
    for name, m in models:
        r = fit(m, df)
        k = m.n_params(df)
        aic, bic = aic_bic(r.fun, k, n)
        results[name] = {"model": m, "res": r, "k": k, "aic": aic, "bic": bic, "nll": r.fun}
        print(f"{name:<32} {str(r.success):>6} {k:>4} {r.fun:>11.2f} {aic:>11.2f} {bic:>11.2f}")

    base_aic = results["M1_null"]["aic"]
    base_bic = results["M1_null"]["bic"]
    print("\n" + "=" * 85)
    print(f"ΔAIC / ΔBIC vs M1 (null) — lower is better")
    print("=" * 85)
    for name in results:
        r = results[name]
        print(f"  {name:<32}  ΔAIC = {r['aic'] - base_aic:+10.2f}   "
              f"ΔBIC = {r['bic'] - base_bic:+10.2f}")

    print("\n" + "=" * 85)
    print(f"M5 vs M4 (license-conditional vs full RSA)")
    print("=" * 85)
    daic = results["M5_rsa_license_conditional"]["aic"] - results["M4_rsa_polite_speaker"]["aic"]
    dbic = results["M5_rsa_license_conditional"]["bic"] - results["M4_rsa_polite_speaker"]["bic"]
    print(f"  ΔAIC (M5 - M4): {daic:+10.2f}  (M5 wins if < 0; -10 is strong)")
    print(f"  ΔBIC (M5 - M4): {dbic:+10.2f}  (M5 wins if < 0; -10 is strong)")
    print("  Interpretation: M5 has fewer free params; if BIC favors M5, the")
    print("    license-conditional (chat × license) interaction is a more")
    print("    parsimonious account of the negative pull than per-cell ω_neg.")

    print("\n" + "=" * 85)
    print(f"M3 diagnostic — asymmetric learning rates (α+ vs α- per stage)")
    print("=" * 85)
    m3 = results["M3_asymmetric_belief"]["model"]
    r3 = results["M3_asymmetric_belief"]["res"]
    n_t = len(m3._targets)
    n_s = len(m3.FREE_STAGES)
    a_pos = r3.x[n_t:n_t + n_s]
    a_neg = r3.x[n_t + n_s:n_t + 2 * n_s]
    print(f"  {'stage':<6} {'α_pos':>8} {'α_neg':>8} {'α_neg - α_pos':>15}")
    for i, s in enumerate(m3.FREE_STAGES):
        print(f"  {s:<6} {a_pos[i]:>8.4f} {a_neg[i]:>8.4f} {a_neg[i] - a_pos[i]:>+15.4f}")
    print("  Interpretation: α_neg(M3) > α_pos(M3) would indicate anti-optimistic")
    print("    updating at the post-DPO stage — the M3 model 'learns' to dial")
    print("    down ratings under negativity license more than it dials up under")
    print("    politeness license.")

    print("\n" + "=" * 85)
    print(f"M5 diagnostic — license-conditional γ_k")
    print("=" * 85)
    m5 = results["M5_rsa_license_conditional"]["model"]
    r5 = results["M5_rsa_license_conditional"]["res"]
    n_t = len(m5._targets)
    n_c = len(m5._cells)
    n_s = len(m5.FREE_STAGES)
    beta_chat = r5.x[n_t + n_c]
    beta_license = r5.x[n_t + n_c + 1]
    gamma_k_per_stage = r5.x[n_t + n_c + 2:n_t + n_c + 2 + n_s]
    print(f"  β_chat                : {beta_chat:+.4f}  (chat-template alone, no license)")
    print(f"  β_license             : {beta_license:+.4f}  (license alone, no chat)")
    for s, g in zip(m5.FREE_STAGES, gamma_k_per_stage):
        print(f"  γ_k({s})             : {g:+.4f}  (interaction at {s})")
    print()
    print("  Interpretation:")
    print("    - γ_k > 0 AND large relative to β_chat, β_license → the")
    print("      license-conditional account holds (the novelty claim).")
    print("    - γ_k near 0 → no interaction; either factor alone suffices.")
    print("    - β_chat or β_license large alone → simpler one-factor account.")
    print()
    print("  Note: standard errors / CIs require recovery analysis or Hessian")
    print("  inversion — not in this script. Run `recovery.py` for that.")


if __name__ == "__main__":
    main()
