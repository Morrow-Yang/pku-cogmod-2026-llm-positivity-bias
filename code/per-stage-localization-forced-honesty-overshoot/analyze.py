"""Analysis for per-stage-localization-forced-honesty-overshoot.

Combines NEW results (C9-C14, this experiment) with PRIOR results (C0-C8,
stage-base-vs-instruct-positivity-prior). Computes per-stage overshoot deltas
with bootstrap CIs + Holm-Bonferroni + baseline-artifact veto + calibration veto,
per the pre-registered success criterion (post-Review-LLM 2026-05-18).

Cell mapping:
  M0 default = C0   (prior)              M0 control = C9  (NEW)   M0 forced = C10 (NEW)
  M1 default = C1   (prior)              M1 control = C11 (NEW)   M1 forced = C12 (NEW)
  M2 default = C2   (prior)              M2 control = C13 (NEW)   M2 forced = C14 (NEW)
  M3 default = C4   (prior)              M3 control = C8  (prior) M3 forced = C5  (prior)

Output: structured summary to stdout + JSON file.

Usage:
  python analyze.py \
    --new-results /root/autodl-tmp/free-choice-project/results/per-stage-localization-forced-honesty-overshoot \
    --prior-results /root/autodl-tmp/free-choice-project/results/stage-base-vs-instruct-positivity-prior \
    --out /root/autodl-tmp/free-choice-project/results/per-stage-localization-forced-honesty-overshoot/summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CELL_TO_STAGE_COND = {
    "C0":  ("M0", "default"),
    "C1":  ("M1", "default"),
    "C2":  ("M2", "default"),
    "C3":  ("M3", "default-completion"),
    "C4":  ("M3", "default"),
    "C5":  ("M3", "forced"),
    "C8":  ("M3", "control"),
    "C9":  ("M0", "control"),
    "C10": ("M0", "forced"),
    "C11": ("M1", "control"),
    "C12": ("M1", "forced"),
    "C13": ("M2", "control"),
    "C14": ("M2", "forced"),
    # post paperreview.ai 2026-05-19 (Round 1)
    "C20": ("M3", "alt-control-truthful"),
    "C21": ("M3", "alt-control-objective"),
    "C22": ("M3", "anti-politeness-strong"),
    "C23": ("M3", "anti-politeness-mild"),
    # post paperreview.ai 2026-05-19 second review (Round 2)
    "C24": ("M3", "anti-politeness-no-neg-words"),
    "C25": ("M3", "forced-completion"),
    "C26": ("M3", "control-completion"),
}

FICTIONAL = {"adirenia", "che_pact", "vellinkov", "khantelan"}
CALIBRATION_POS = {"clean_drinking_water"}
CALIBRATION_NEG = {"tooth_decay"}
CALIBRATION_NEUTRAL = {"standard_brick"}


def load_results(*dirs: Path, min_mass: float = 0.0) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for p in sorted(Path(d).glob("cell_*_target_*.json")):
            data = json.load(open(p))
            rows.extend(data["trials"])
    df = pd.DataFrame(rows)
    df["stage"] = df["cell"].map(lambda c: CELL_TO_STAGE_COND.get(c, (None, None))[0])
    df["condition"] = df["cell"].map(lambda c: CELL_TO_STAGE_COND.get(c, (None, None))[1])
    if min_mass > 0:
        before = len(df)
        df = df[df["digit_raw_mass"] >= min_mass].reset_index(drop=True)
        print(f"[low-mass filter] kept {len(df)}/{before} trials with digit_raw_mass >= {min_mass}")
    # Template index = prompt_idx // 10 (10 trials per template, 5 templates per cell)
    df["template"] = df["prompt_idx"] // 10
    return df


def bootstrap_delta_ci(a, b, n_resamples=5000, alpha=0.05, seed=42):
    """Trial-level (independent) bootstrap. Kept as the fast fallback / sanity check."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    b = np.asarray(b)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        ra = rng.choice(a, size=len(a), replace=True)
        rb = rng.choice(b, size=len(b), replace=True)
        deltas[i] = ra.mean() - rb.mean()
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return {
        "mean_delta": float(a.mean() - b.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "below_zero": hi < 0,
        "method": "trial-level",
    }


def cluster_bootstrap_delta_ci(df_a, df_b, cluster_keys=("target_id", "template"),
                                n_resamples=5000, alpha=0.05, seed=42):
    """Cluster bootstrap (per paperreview.ai 2026-05-19 Q3): resamples (target, template)
    clusters with replacement, pools trials within sampled clusters, computes delta.

    Respects the dependence structure: trials within a target share content; trials within
    a template share phrasing. With 4 targets × 5 templates = 20 clusters per cell,
    cluster bootstrap CIs are typically 1.2-1.5× wider than trial-level — properly
    accounting for the within-cluster correlation.
    """
    rng = np.random.default_rng(seed)
    # Group by cluster key tuple
    a_groups = {k: g["rating"].values for k, g in df_a.groupby(list(cluster_keys))}
    b_groups = {k: g["rating"].values for k, g in df_b.groupby(list(cluster_keys))}
    a_keys = list(a_groups.keys())
    b_keys = list(b_groups.keys())
    if not a_keys or not b_keys:
        return {"status": "MISSING_DATA", "n_clusters_a": len(a_keys), "n_clusters_b": len(b_keys)}

    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        sa = rng.choice(len(a_keys), size=len(a_keys), replace=True)
        sb = rng.choice(len(b_keys), size=len(b_keys), replace=True)
        a_trials = np.concatenate([a_groups[a_keys[idx]] for idx in sa])
        b_trials = np.concatenate([b_groups[b_keys[idx]] for idx in sb])
        deltas[i] = a_trials.mean() - b_trials.mean()
    all_a = np.concatenate([a_groups[k] for k in a_keys])
    all_b = np.concatenate([b_groups[k] for k in b_keys])
    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return {
        "mean_delta": float(all_a.mean() - all_b.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "n_a": int(len(all_a)),
        "n_b": int(len(all_b)),
        "n_clusters_a": len(a_keys),
        "n_clusters_b": len(b_keys),
        "below_zero": hi < 0,
        "method": f"cluster-bootstrap[{','.join(cluster_keys)}]",
    }


def cell_ratings(df, cell, targets):
    """Trial-level ratings (legacy interface)."""
    sel = (df["cell"] == cell) & (df["target_id"].isin(targets))
    return df.loc[sel, "rating"].values


def cell_df(df, cell, targets):
    """Trial-level dataframe slice — needed by cluster bootstrap."""
    return df[(df["cell"] == cell) & (df["target_id"].isin(targets))].copy()


def print_digit_mass_distribution(df):
    """Per-cell digit_raw_mass distribution (mean, P5, P95). Tier-1 manipulation check
    per paperreview.ai 2026-05-19 Q3."""
    print("\n=== Per-cell digit-anchor probability mass (mean / P5 / P95) ===")
    print(f"  {'cell':>5s} {'n':>5s} {'mean':>6s} {'P5':>6s} {'P95':>6s} {'min':>6s}  notes")
    for cell in sorted(df["cell"].unique()):
        m = df[df["cell"] == cell]["digit_raw_mass"].values
        if len(m) == 0:
            continue
        note = "LOW" if m.mean() < 0.7 else ""
        print(f"  {cell:>5s} {len(m):>5d} {m.mean():>6.3f} {np.quantile(m, 0.05):>6.3f} "
              f"{np.quantile(m, 0.95):>6.3f} {m.min():>6.3f}  {note}")


def per_stage_overshoot(df):
    """Primary test: C5(M_k) - C0(M0_base). Pooled on 4 fictional targets."""
    base = cell_ratings(df, "C0", FICTIONAL)
    rows = []
    for stage, forced_cell in [("M0", "C10"), ("M1", "C12"), ("M2", "C14"), ("M3", "C5")]:
        forced = cell_ratings(df, forced_cell, FICTIONAL)
        if len(forced) == 0:
            rows.append({"stage": stage, "cell": forced_cell, "status": "MISSING_DATA"})
            continue
        ci = bootstrap_delta_ci(forced, base)
        ci.update({"stage": stage, "cell": forced_cell,
                   "interpretation": f"{forced_cell}({stage}) - C0(M0_base)"})
        rows.append(ci)
    return rows


def per_stage_anti_politeness(df):
    """Confirmatory: C5(M_k) - C8(M_k). Pooled on 4 fictional targets."""
    pairs = [("M0", "C10", "C9"), ("M1", "C12", "C11"),
             ("M2", "C14", "C13"), ("M3", "C5", "C8")]
    rows = []
    for stage, forced_cell, control_cell in pairs:
        forced = cell_ratings(df, forced_cell, FICTIONAL)
        control = cell_ratings(df, control_cell, FICTIONAL)
        if len(forced) == 0 or len(control) == 0:
            rows.append({"stage": stage, "forced": forced_cell, "control": control_cell,
                         "status": "MISSING_DATA"})
            continue
        ci = bootstrap_delta_ci(forced, control)
        ci.update({"stage": stage, "forced_cell": forced_cell, "control_cell": control_cell,
                   "interpretation": f"{forced_cell}({stage}) - {control_cell}({stage})"})
        rows.append(ci)
    return rows


def holm_closed_test(overshoot_results, n_tests=3):
    """Closed-testing Holm-Bonferroni across the 3 within-Tülu stages (M0/M1/M2/M3).
    Restrict to within-Tülu sub-chain for the localization claim (M0 is the base baseline
    so M0+forced does not represent a stage in the chain — it's a control). Apply Holm
    across the 3 stages M1, M2, M3 sorted by most negative effect first."""
    candidates = [r for r in overshoot_results if r.get("stage") in {"M1", "M2", "M3"} and "below_zero" in r]
    sorted_tests = sorted(candidates, key=lambda t: t["mean_delta"])
    cumulative_pass = True
    for i, t in enumerate(sorted_tests):
        holm_alpha = 0.05 / max(1, n_tests - i)
        t["holm_alpha"] = holm_alpha
        t["holm_rank"] = i + 1
        if not t["below_zero"]:
            cumulative_pass = False
        t["holm_significant"] = cumulative_pass and t["below_zero"]
    return sorted_tests


def baseline_artifact_veto(df):
    """Test whether the prompt itself has a baseline negativity effect on M0.
    If C10 - C0 has CI below 0, some fraction of stage overshoots is prompt artifact."""
    C0 = cell_ratings(df, "C0", FICTIONAL)
    C10 = cell_ratings(df, "C10", FICTIONAL)
    C9 = cell_ratings(df, "C9", FICTIONAL)
    if len(C10) == 0 or len(C9) == 0:
        return {"status": "MISSING_DATA"}
    forced_vs_base = bootstrap_delta_ci(C10, C0)
    control_vs_base = bootstrap_delta_ci(C9, C0)
    return {
        "forced_vs_base_M0": forced_vs_base,
        "control_vs_base_M0": control_vs_base,
        "veto_triggered": forced_vs_base["below_zero"],
        "note": ("VETO TRIGGERED: M0+forced drops below M0 default. Subtract this "
                 "delta from each stage overshoot before declaring the mask installed."
                 if forced_vs_base["below_zero"]
                 else "VETO CLEAN: M0+forced does not significantly drop below M0 default. "
                      "Stage overshoots are not explainable by prompt artifact alone."),
    }


def calibration_veto(df):
    """Test whether forced-honesty on M0 shifts calibration entities toward ground truth
    (semantically appropriate) or uniformly negative (a general negativity bias)."""
    per_target = []
    for kind, targets in [("positive", CALIBRATION_POS), ("negative", CALIBRATION_NEG),
                          ("neutral", CALIBRATION_NEUTRAL)]:
        for tgt in targets:
            forced = df.query("cell == 'C10' and target_id == @tgt")["rating"].values
            control = df.query("cell == 'C9' and target_id == @tgt")["rating"].values
            if len(forced) == 0:
                continue
            per_target.append({
                "kind": kind,
                "target": tgt,
                "mean_forced_C10": float(forced.mean()),
                "mean_control_C9": float(control.mean()) if len(control) > 0 else None,
                "n_forced": int(len(forced)),
                "ground_truth_direction": kind,
            })

    veto_triggered = False
    msg = "VETO CLEAN: forced-honesty shifts calibration entities appropriately."
    pos_means = [r["mean_forced_C10"] for r in per_target if r["kind"] == "positive"]
    neg_means = [r["mean_forced_C10"] for r in per_target if r["kind"] == "negative"]
    if pos_means and neg_means:
        # Trigger if BOTH positive and negative entities land below the neutral midpoint (3.5)
        # under forced-honesty (= general negativity bias rather than semantic appropriateness)
        if pos_means[0] < 3.5 and neg_means[0] < 3.5:
            veto_triggered = True
            msg = (f"VETO TRIGGERED: positive entity 'clean drinking water' at {pos_means[0]:.2f} "
                   f"and negative entity 'tooth decay' at {neg_means[0]:.2f} both below midpoint. "
                   f"Forced-honesty has a general negativity bias rather than truth-discovery.")
        elif pos_means[0] >= 4.5 and neg_means[0] <= 3.5:
            msg = (f"VETO CLEAN: positive entity {pos_means[0]:.2f}, negative entity {neg_means[0]:.2f}. "
                   f"Forced-honesty maintains valence-appropriate ratings.")

    return {"per_target": per_target, "veto_triggered": veto_triggered, "note": msg}


def m3_calibration_veto(df):
    """Per paperreview.ai 2026-05-19 Q1: M3-level calibration veto.
    Tests whether forced-honesty on M3 (where the overshoot occurs) preserves
    valence-appropriate ratings on the 3 calibration entities, or uniformly
    pushes them negative."""
    rows = []
    for kind, targets in [("positive", CALIBRATION_POS),
                          ("negative", CALIBRATION_NEG),
                          ("neutral", CALIBRATION_NEUTRAL)]:
        for tgt in targets:
            default = df.query("cell == 'C4' and target_id == @tgt")["rating"].values
            forced  = df.query("cell == 'C5' and target_id == @tgt")["rating"].values
            control = df.query("cell == 'C8' and target_id == @tgt")["rating"].values
            if len(default) == 0 or len(forced) == 0:
                continue
            rows.append({
                "kind": kind,
                "target": tgt,
                "C4_default": float(default.mean()),
                "C5_forced":  float(forced.mean()),
                "C8_control": float(control.mean()) if len(control) > 0 else None,
                "delta_C5_C4": float(forced.mean() - default.mean()),
            })

    pos = [r["C5_forced"] for r in rows if r["kind"] == "positive"]
    neg = [r["C5_forced"] for r in rows if r["kind"] == "negative"]
    veto_triggered = False
    msg = "M3 CALIBRATION VETO CLEAN: forced-honesty on M3 preserves valence-appropriate ratings."
    if pos and neg:
        if pos[0] < 3.5 and neg[0] < 3.5:
            veto_triggered = True
            msg = (f"M3 CALIBRATION VETO TRIGGERED: positive entity {pos[0]:.2f} and "
                   f"negative entity {neg[0]:.2f} both below midpoint under forced-honesty on M3. "
                   f"Generalized negativity at M3, not anti-politeness-specific.")
    return {"per_target": rows, "veto_triggered": veto_triggered, "note": msg}


def alt_control_decomposition(df):
    """Per paperreview.ai 2026-05-19 Q2: anti-politeness-specific decomposition under
    alternative neutral control phrasings (C20, C21). Confirms the decomposition is
    robust to control-prompt wording."""
    rows = []
    forced = cell_df(df, "C5", FICTIONAL)
    if len(forced) == 0:
        return [{"status": "MISSING_DATA"}]
    for ctl_cell, label in [("C8", "C8 (orig: please rate accurately)"),
                            ("C20", "C20 (alt: be truthful)"),
                            ("C21", "C21 (alt: be precise/objective)")]:
        ctl = cell_df(df, ctl_cell, FICTIONAL)
        if len(ctl) == 0:
            rows.append({"control_cell": ctl_cell, "label": label, "status": "MISSING_DATA"})
            continue
        ci = cluster_bootstrap_delta_ci(forced, ctl)
        ci["control_cell"] = ctl_cell
        ci["label"] = label
        rows.append(ci)
    return rows


def dose_response(df):
    """Per paperreview.ai 2026-05-19 Q5: dose-response on anti-politeness intensity.
    Does stronger anti-politeness produce a larger overshoot? Compare C5, C22, C23 vs C0."""
    base = cell_df(df, "C0", FICTIONAL)
    if len(base) == 0:
        return [{"status": "MISSING_DATA"}]
    rows = []
    for fcell, label in [("C23", "C23 (mild: answer directly)"),
                         ("C5",  "C5  (medium: be honest, not polite)"),
                         ("C22", "C22 (strong: blunt + avoid euphemisms + no hedging)")]:
        forced = cell_df(df, fcell, FICTIONAL)
        if len(forced) == 0:
            rows.append({"cell": fcell, "label": label, "status": "MISSING_DATA"})
            continue
        ci = cluster_bootstrap_delta_ci(forced, base)
        ci["cell"] = fcell
        ci["label"] = label
        rows.append(ci)
    return rows


def negativity_deconfound(df):
    """Per paperreview.ai Round-2 Q5: does the overshoot reproduce when the C5 prompt
    is stripped of explicit 'negative or controversial' wording (C24)? If C24 - C0
    has CI strictly below 0, the overshoot survives the deconfound."""
    base = cell_df(df, "C0", FICTIONAL)
    if len(base) == 0:
        return {"status": "MISSING_DATA"}
    rows = []
    for fcell, label in [("C5",  "C5 (orig: includes 'negative or controversial')"),
                         ("C24", "C24 (de-neg-words: politeness/euphemism/hedging only)")]:
        forced = cell_df(df, fcell, FICTIONAL)
        if len(forced) == 0:
            rows.append({"cell": fcell, "label": label, "status": "MISSING_DATA"})
            continue
        ci = cluster_bootstrap_delta_ci(forced, base)
        ci["cell"] = fcell
        ci["label"] = label
        rows.append(ci)
    return rows


def chat_template_vs_completion(df):
    """Per paperreview.ai Round-2 Q6: at M3, is the overshoot chat-template-driven or RLHF-driven?
    Compare:
      C5  (chat-template forced) vs C25 (completion-style forced): if both overshoot, RLHF-driven;
                                                                   if only C5 overshoots, chat-template-driven.
      C5 - C8  (chat-template anti-politeness share) vs C25 - C26 (completion-style share)."""
    base = cell_df(df, "C0", FICTIONAL)
    rows = []
    # Overshoot at M3 in each format
    for fcell, label in [("C5",  "C5  (chat-template forced)   vs C0 base"),
                         ("C25", "C25 (completion-style forced) vs C0 base")]:
        forced = cell_df(df, fcell, FICTIONAL)
        if len(forced) == 0 or len(base) == 0:
            rows.append({"cell": fcell, "label": label, "status": "MISSING_DATA"})
            continue
        ci = cluster_bootstrap_delta_ci(forced, base)
        ci["cell"] = fcell
        ci["label"] = label
        rows.append(ci)
    # Anti-politeness shares in each format
    chat_forced = cell_df(df, "C5", FICTIONAL)
    chat_control = cell_df(df, "C8", FICTIONAL)
    if len(chat_forced) and len(chat_control):
        ci = cluster_bootstrap_delta_ci(chat_forced, chat_control)
        ci["label"] = "C5 - C8   (chat-template anti-politeness share)"
        rows.append(ci)
    cpl_forced = cell_df(df, "C25", FICTIONAL)
    cpl_control = cell_df(df, "C26", FICTIONAL)
    if len(cpl_forced) and len(cpl_control):
        ci = cluster_bootstrap_delta_ci(cpl_forced, cpl_control)
        ci["label"] = "C25 - C26 (completion-style anti-politeness share)"
        rows.append(ci)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-results", required=True, type=Path)
    ap.add_argument("--prior-results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-mass", type=float, default=0.0,
                    help="Filter trials with digit_raw_mass < min-mass (low-mass sensitivity check)")
    args = ap.parse_args()

    df = load_results(args.new_results, args.prior_results, min_mass=args.min_mass)
    print(f"=== Loaded {len(df)} trials, {df['cell'].nunique()} cells (min_mass={args.min_mass}) ===")
    print(df.groupby(["stage", "condition"]).size().to_string())
    print()
    print_digit_mass_distribution(df)
    print()

    overshoot = per_stage_overshoot(df)
    print("=== Per-stage overshoot (C5(M_k) - C0 on 4 fictional pooled) ===")
    for r in overshoot:
        if "mean_delta" in r:
            sig = "**" if r["below_zero"] else "  "
            print(f"  {sig} {r['interpretation']:<32s} delta={r['mean_delta']:+.3f}  "
                  f"95% CI=[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  n={r['n_a']}")
        else:
            print(f"     {r['stage']} ({r['cell']}): {r['status']}")
    print()

    holm = holm_closed_test(overshoot, n_tests=3)
    print("=== Holm-Bonferroni (closed testing, 3 stages M1/M2/M3) ===")
    for t in holm:
        sig = "**" if t["holm_significant"] else "  "
        print(f"  {sig} rank={t['holm_rank']}  {t['stage']}  delta={t['mean_delta']:+.3f}  "
              f"holm_alpha={t['holm_alpha']:.4f}  holm_significant={t['holm_significant']}")
    print()

    anti = per_stage_anti_politeness(df)
    print("=== Per-stage anti-politeness-specific share (C5(M_k) - C8(M_k)) ===")
    for r in anti:
        if "mean_delta" in r:
            sig = "**" if r["below_zero"] else "  "
            print(f"  {sig} {r['interpretation']:<32s} delta={r['mean_delta']:+.3f}  "
                  f"95% CI=[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
        else:
            print(f"     {r['stage']}: {r['status']}")
    print()

    base_veto = baseline_artifact_veto(df)
    print("=== Baseline-artifact veto ===")
    print(f"  {base_veto.get('note', base_veto.get('status'))}")
    if "forced_vs_base_M0" in base_veto:
        fb = base_veto["forced_vs_base_M0"]
        print(f"  C10 - C0 (forced on M0 - default on M0): delta={fb['mean_delta']:+.3f}  "
              f"CI=[{fb['ci_lo']:+.3f}, {fb['ci_hi']:+.3f}]")
    print()

    calib_veto = calibration_veto(df)
    print("=== Calibration veto (M0, prefix-injected forced-honesty C10) ===")
    print(f"  {calib_veto['note']}")
    for r in calib_veto["per_target"]:
        ctl = f", C9={r['mean_control_C9']:.2f}" if r['mean_control_C9'] is not None else ""
        print(f"    [{r['kind']:>8s}] {r['target']:>22s}: C10={r['mean_forced_C10']:.2f}{ctl}")
    print()

    # NEW (paperreview.ai 2026-05-19): M3 calibration veto
    m3_calib = m3_calibration_veto(df)
    print("=== M3 calibration veto (chat-template forced-honesty C5 on M3) ===")
    print(f"  {m3_calib['note']}")
    for r in m3_calib["per_target"]:
        print(f"    [{r['kind']:>8s}] {r['target']:>22s}: "
              f"C4(default)={r['C4_default']:.2f} → C5(forced)={r['C5_forced']:.2f} "
              f"(Δ={r['delta_C5_C4']:+.2f}), C8(ctrl)={r['C8_control'] if r['C8_control'] is None else f'{r['C8_control']:.2f}'}")
    print()

    # NEW: alt-control-prompt robustness of the decomposition
    alt_ctrl = alt_control_decomposition(df)
    print("=== Alt-control decomposition (anti-politeness-specific share, cluster bootstrap) ===")
    for r in alt_ctrl:
        if "mean_delta" in r:
            sig = "**" if r["below_zero"] else "  "
            print(f"  {sig} C5 - {r['control_cell']:<5s} ({r['label']:50s}) "
                  f"Δ={r['mean_delta']:+.3f}  CI=[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  "
                  f"n_clusters={r.get('n_clusters_a','?')}")
        else:
            print(f"     {r.get('control_cell','?')}: {r.get('status','?')}")
    print()

    # NEW: dose-response on anti-politeness intensity
    dose = dose_response(df)
    print("=== Dose-response: anti-politeness intensity vs M0 base overshoot (cluster bootstrap) ===")
    for r in dose:
        if "mean_delta" in r:
            sig = "**" if r["below_zero"] else "  "
            print(f"  {sig} {r['label']:55s}  Δ vs C0 = {r['mean_delta']:+.3f}  "
                  f"CI=[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
        else:
            print(f"     {r.get('cell','?')}: {r.get('status','?')}")
    print()

    # NEW (Round-2 Q5): negativity-deconfound
    deconf = negativity_deconfound(df)
    print("=== Negativity-wording deconfound (overshoot under C24 with NO 'negative/controversial' wording) ===")
    for r in deconf:
        if "mean_delta" in r:
            sig = "**" if r["below_zero"] else "  "
            print(f"  {sig} {r['label']:60s}  Δ vs C0 = {r['mean_delta']:+.3f}  "
                  f"CI=[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
        else:
            print(f"     {r.get('cell','?')}: {r.get('status','?')}")
    print()

    # NEW (Round-2 Q6): chat-template vs completion-style disentanglement
    fmt_dis = chat_template_vs_completion(df)
    print("=== Chat-template vs completion-style at M3 (overshoot + anti-politeness share) ===")
    for r in fmt_dis:
        if "mean_delta" in r:
            sig = "**" if r["below_zero"] else "  "
            print(f"  {sig} {r['label']:50s}  Δ = {r['mean_delta']:+.3f}  "
                  f"CI=[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
        else:
            print(f"     {r.get('cell','?')}: {r.get('status','?')}")
    print()

    # NEW: cluster-bootstrap re-validation of the headline overshoot
    print("=== Cluster-bootstrap CI on the headline M3 overshoot (C5 - C0) ===")
    forced_m3 = cell_df(df, "C5", FICTIONAL)
    base_m0 = cell_df(df, "C0", FICTIONAL)
    if len(forced_m3) and len(base_m0):
        ci_cluster = cluster_bootstrap_delta_ci(forced_m3, base_m0)
        print(f"  Cluster bootstrap (target × template): Δ={ci_cluster['mean_delta']:+.3f}  "
              f"CI=[{ci_cluster['ci_lo']:+.3f}, {ci_cluster['ci_hi']:+.3f}]  "
              f"({ci_cluster['n_clusters_a']} × {ci_cluster['n_clusters_b']} clusters)")
        ci_trial = bootstrap_delta_ci(forced_m3["rating"].values, base_m0["rating"].values)
        print(f"  Trial-level bootstrap (for comparison): Δ={ci_trial['mean_delta']:+.3f}  "
              f"CI=[{ci_trial['ci_lo']:+.3f}, {ci_trial['ci_hi']:+.3f}]")
        ratio = (ci_cluster['ci_hi'] - ci_cluster['ci_lo']) / max(1e-6, ci_trial['ci_hi'] - ci_trial['ci_lo'])
        print(f"  CI width ratio (cluster / trial): {ratio:.2f}x")
    print()

    # Verdict
    mask_at = {}
    for r in holm:
        if r.get("stage") in {"M1", "M2"} and r.get("holm_significant"):
            mask_at[r["stage"]] = r

    if "M1" in mask_at:
        case = "A: Tulu SFT (M1) installs the polite-speech mask"
    elif "M2" in mask_at:
        case = "B: Tulu DPO (M2) is the critical step (M1 absent)"
    else:
        case = ("C: Tulu recipe does not install the mask within M1/M2; "
                "M3 phenomenon is recipe-dependent (cannot attribute to RLHF specifically without "
                "deconfounding work; M2-vs-M3 conflates 5+ factors)")

    print(f"=== Verdict: Case {case} ===")

    summary = {
        "n_trials_total": int(len(df)),
        "n_cells": int(df["cell"].nunique()),
        "overshoot_per_stage": overshoot,
        "overshoot_holm_corrected": holm,
        "anti_politeness_per_stage": anti,
        "baseline_artifact_veto": base_veto,
        "calibration_veto": calib_veto,
        "verdict_case": case,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"=== Summary written to {args.out} ===")


if __name__ == "__main__":
    main()
