"""Item 3 probe + RSA analysis on M3 hidden states.

Reads hidden-state .npz files from extract_hidden_states.py and the
existing rating data from cell_*.json (E5, E5b, E7, E8). Performs three
analyses per (layer, position):

A. Probe ceiling: within-condition (E5_000) cross-validated R^2 -- the
   upper bound on how well a linear probe can recover ratings.

B. Probe transfer: train probe on E5_000 hidden states; test on each
   other condition's hidden states. The prediction is the probe's output
   on the held-out condition; we compare it to (1) the observed rating
   on that condition and (2) the probe's prediction on E5_000 for the
   same target.
   - Stable representations: probe predictions across conditions are
     similar to E5_000 predictions for the same target, even though
     observed ratings differ.
   - Shifted representations: probe predictions shift toward the
     observed ratings.

C. RSA: cosine similarity between same-target hidden states across
   condition pairs.
   - Reference baseline = E5_000 vs E8_verbose (valence-irrelevant
     prompt change). Other condition pairs are interpretable only
     relative to this baseline.

Per design-review framing: this analysis can show "evidence against
representational shift under honesty prompts" (consistent with either
expression-policy OR motivated-cognition-v1) but cannot prove
expression-policy specifically. Motivated-cognition-v2 (representations
themselves shift) is what's being tested here.

Output: probe_analysis.json + simple per-layer summary printed.
"""
from __future__ import annotations
import glob
import json
import re
from pathlib import Path
import numpy as np

PROBES_DIR = Path("experiments/results/post-review-extensions-probes")
CELLS_DIR = Path("experiments/results/post-review-extensions")
OUT_PATH = PROBES_DIR / "probe_analysis.json"

# Conditions matched to where rating data lives
# cell_id_for_rating[condition_key in HS file] = (cell_id_prefix, target_set)
# E5_000 / E5_001 / E5_111 rating data: cell_E5_xxx_target_<tid>.json (orig targets)
# But also cell_E5b_xxx_target_<tid>.json (pseudoword targets)
# E7_pos_xxx rating data: cell_E7_pos_xxx_orig or _psw_target_<tid>.json
# E8_verbose: no rating data collected (only hidden states)
CONDITION_TO_CELL_PREFIX = {
    "E5_000": ["E5_000", "E5b_000"],
    "E5_001": ["E5_001", "E5b_001"],
    "E5_111": ["E5_111", "E5b_111"],
    "E7_pos_001": ["E7_pos_001_orig", "E7_pos_001_psw"],
    "E7_pos_111": ["E7_pos_111_orig", "E7_pos_111_psw"],
    "E8_verbose": [],  # no rating data
}


def load_hidden_states():
    """Load all .npz files into a structured dict.
    Returns: {cond: {target: {tpl: {"name": (n_layers, hidden), "last": (n_layers, hidden)}}}}
    """
    data = {}
    files = sorted(glob.glob(str(PROBES_DIR / "hs_cond_*.npz")))
    if not files:
        return data, 0, 0
    n_layers = None
    hidden_dim = None
    for f in files:
        name = Path(f).name  # hs_cond_<cond>_target_<tid>_tpl_<idx>.npz
        m = re.match(r"hs_cond_(.+?)_target_(.+?)_tpl_(\d+)\.npz", name)
        if not m:
            continue
        cond, tid, tpl = m.group(1), m.group(2), int(m.group(3))
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        if n_layers is None:
            n_layers = meta["n_layers"]
            hidden_dim = meta["hidden_dim"]
        data.setdefault(cond, {}).setdefault(tid, {})[tpl] = {
            "name": z["hidden_targetname"],
            "last": z["hidden_lasttoken"],
        }
    return data, n_layers, hidden_dim


def load_ratings():
    """Load observed ratings from cell_*.json files.
    Returns: {cond: {target: {tpl: rating}}}
    """
    ratings = {}
    for cond, cell_prefixes in CONDITION_TO_CELL_PREFIX.items():
        for prefix in cell_prefixes:
            for f in sorted(glob.glob(str(CELLS_DIR / f"cell_{prefix}_target_*.json"))):
                d = json.load(open(f))
                for trial in d["trials"]:
                    tid = trial["target_id"]
                    tpl = trial["template_idx"]
                    ratings.setdefault(cond, {}).setdefault(tid, {})[tpl] = trial["rating"]
    return ratings


def build_xy(data, ratings, cond, position):
    """Build the (X, y, ids) arrays for a given condition, position.
    Returns: X (n, hidden_dim), y (n,), ids list of (target, tpl).
    Only includes samples where rating data exists.
    """
    X, y, ids = [], [], []
    if cond not in data:
        return None, None, None
    for tid, by_tpl in data[cond].items():
        for tpl, hs in by_tpl.items():
            r = ratings.get(cond, {}).get(tid, {}).get(tpl)
            if r is None:
                continue
            X.append(hs[position])  # shape (n_layers, hidden_dim)
            y.append(float(r))
            ids.append((tid, tpl))
    if not X:
        return None, None, None
    return np.stack(X), np.array(y), ids


def linear_regression(X_train, y_train, X_test):
    """Closed-form linear regression with regularization.
    Returns y_pred on X_test.
    """
    # Ridge regression with small lambda to avoid singularity
    lam = 1.0
    Xt_aug = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    # (X'X + lambda I)^{-1} X' y
    XtX = Xt_aug.T @ Xt_aug + lam * np.eye(Xt_aug.shape[1])
    XtX[-1, -1] = 0  # don't regularize intercept
    w = np.linalg.solve(XtX, Xt_aug.T @ y_train)
    Xte_aug = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    return Xte_aug @ w


def r2(y_true, y_pred):
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot < 1e-9:
        return float("nan")
    return 1 - ss_res / ss_tot


def main():
    print("Loading hidden states ...", flush=True)
    data, n_layers, hidden_dim = load_hidden_states()
    if not data:
        print("ERROR: no hidden-state files found. Run extract_hidden_states.py first.")
        return
    print(f"  Loaded conditions: {sorted(data.keys())}")
    print(f"  n_layers = {n_layers}, hidden_dim = {hidden_dim}")
    print()

    print("Loading ratings ...", flush=True)
    ratings = load_ratings()
    for c in sorted(ratings.keys()):
        n_ratings = sum(len(v) for v in ratings[c].values())
        print(f"  {c}: {n_ratings} ratings")
    print()

    # Build X/y per condition (averaged later across positions/layers as needed)
    positions = ["name", "last"]
    results = {pos: {} for pos in positions}

    for position_idx, position in enumerate(positions):
        print(f"\n{'='*70}\nPosition: {position}\n{'='*70}")

        # Build (n_samples, n_layers, hidden_dim) for each condition
        Xy = {}
        for cond in sorted(data.keys()):
            X, y, ids = build_xy(data, ratings, cond, position)
            Xy[cond] = (X, y, ids)

        # ===== A. Probe ceiling on E5_000 =====
        # 5-fold leave-one-template-out cross-validation
        X0, y0, ids0 = Xy.get("E5_000", (None, None, None))
        if X0 is None:
            print("WARNING: no E5_000 data; skipping ceiling.")
            continue

        n0 = X0.shape[0]
        tpls0 = np.array([t for (_, t) in ids0])
        ceiling_r2_per_layer = []
        for L in range(n_layers):
            preds = np.zeros(n0)
            for held in sorted(np.unique(tpls0)):
                mask_te = tpls0 == held
                mask_tr = ~mask_te
                Xtr = X0[mask_tr, L, :]
                Xte = X0[mask_te, L, :]
                ytr = y0[mask_tr]
                preds[mask_te] = linear_regression(Xtr, ytr, Xte)
            r2_cv = r2(y0, preds)
            ceiling_r2_per_layer.append(r2_cv)
        print(f"\nProbe ceiling (5-fold CV on E5_000), R^2 by layer:")
        best_layer = int(np.nanargmax(ceiling_r2_per_layer))
        print(f"  Best layer: {best_layer}, R^2 = {ceiling_r2_per_layer[best_layer]:.3f}")
        print(f"  Layer 0..31 R^2 (every 4): " +
              "  ".join(f"L{L}={ceiling_r2_per_layer[L]:+.2f}"
                       for L in range(0, n_layers, 4)))

        # ===== B. Probe transfer (train on E5_000, test on other conds) =====
        # At the best ceiling layer, train a single probe on all E5_000.
        # Predict on each other condition.
        transfer = {}
        for L in [best_layer, n_layers - 1]:
            transfer[f"L{L}"] = {}
            X0L = X0[:, L, :]
            for cond in sorted(data.keys()):
                if cond == "E5_000":
                    continue
                Xc, yc, idsc = Xy[cond]
                if Xc is None:
                    continue
                Xc_L = Xc[:, L, :]
                # Train probe on E5_000, predict on cond
                preds_cond = linear_regression(X0L, y0, Xc_L)
                # If we have ratings on this cond, compute R^2 of probe vs observed
                r2_vs_observed = r2(yc, preds_cond) if yc is not None and len(yc) > 1 else None
                # Per-target: compare probe prediction on cond vs probe prediction on E5_000
                # for the same target. If similar, representations are stable.
                # Build per-target probe predictions on E5_000 (avg across templates).
                preds_0 = linear_regression(X0L, y0, X0L)
                per_target_0 = {}
                for (tid, _), p in zip(ids0, preds_0):
                    per_target_0.setdefault(tid, []).append(p)
                per_target_0 = {t: float(np.mean(v)) for t, v in per_target_0.items()}
                per_target_cond = {}
                for (tid, _), p in zip(idsc, preds_cond):
                    per_target_cond.setdefault(tid, []).append(p)
                per_target_cond = {t: float(np.mean(v)) for t, v in per_target_cond.items()}
                # Shared targets, compute correlation and mean shift
                shared = sorted(set(per_target_0) & set(per_target_cond))
                preds_0_arr = np.array([per_target_0[t] for t in shared])
                preds_c_arr = np.array([per_target_cond[t] for t in shared])
                corr = float(np.corrcoef(preds_0_arr, preds_c_arr)[0, 1]) if len(shared) > 1 else float("nan")
                shift = float((preds_c_arr - preds_0_arr).mean())
                transfer[f"L{L}"][cond] = {
                    "r2_vs_observed_ratings": r2_vs_observed,
                    "per_target_probe_corr_with_E5_000": corr,
                    "mean_probe_shift_vs_E5_000": shift,
                }
        print(f"\nProbe transfer (train on E5_000, test on cond), at best layer {best_layer}:")
        for cond, m in transfer[f"L{best_layer}"].items():
            r2s = m["r2_vs_observed_ratings"]
            r2_str = f"R^2 vs obs = {r2s:.2f}" if r2s is not None else "no obs ratings"
            print(f"  {cond:<12s}  per-target probe corr w/ E5_000 = {m['per_target_probe_corr_with_E5_000']:+.3f}, "
                  f"mean shift = {m['mean_probe_shift_vs_E5_000']:+.3f}  ({r2_str})")

        # ===== C. RSA (cross-condition cosine similarity for same target) =====
        # Use averaged-across-templates hidden states for each (cond, target).
        # IMPORTANT: load directly from `data` (not via Xy), so the E8_verbose
        # baseline (which has no rating data) is included.
        rsa = {}
        for L in [best_layer, n_layers - 1]:
            rsa[f"L{L}"] = {}
            per_cond_target = {}
            for cond in sorted(data.keys()):
                acc = {}
                for tid, by_tpl in data[cond].items():
                    layer_vecs = [hs[position][L] for hs in by_tpl.values()]
                    acc[tid] = np.mean(np.stack(layer_vecs).astype(np.float32), axis=0)
                per_cond_target[cond] = acc

            # Compute per-target cosine similarity between condition pairs
            def cos(a, b):
                na = np.linalg.norm(a); nb = np.linalg.norm(b)
                if na < 1e-9 or nb < 1e-9: return 0.0
                return float(np.dot(a, b) / (na * nb))

            anchor = "E5_000"
            if anchor not in per_cond_target:
                continue
            for cond in sorted(per_cond_target.keys()):
                if cond == anchor:
                    continue
                shared = sorted(set(per_cond_target[anchor]) & set(per_cond_target[cond]))
                sims = np.array([
                    cos(per_cond_target[anchor][t], per_cond_target[cond][t])
                    for t in shared
                ])
                rsa[f"L{L}"][cond] = {
                    "mean_cosine_with_E5_000": float(sims.mean()),
                    "median": float(np.median(sims)),
                    "n_targets": int(len(shared)),
                }
        print(f"\nRSA (mean cosine similarity with E5_000, per target avgd across templates), best layer {best_layer}:")
        baseline = rsa[f"L{best_layer}"].get("E8_verbose", {}).get("mean_cosine_with_E5_000")
        print(f"  E8_verbose (valence-irrelevant baseline): {baseline:+.4f}" if baseline is not None else "  (no baseline)")
        for cond, m in rsa[f"L{best_layer}"].items():
            if cond == "E8_verbose":
                continue
            note = ""
            if baseline is not None:
                delta = m["mean_cosine_with_E5_000"] - baseline
                if delta < -0.05:
                    note = " *** LOWER THAN BASELINE -> potential rep shift"
                elif delta < -0.02:
                    note = " (slightly lower than baseline)"
            print(f"  {cond:<12s}: cosine = {m['mean_cosine_with_E5_000']:+.4f}{note}")

        results[position] = {
            "n_layers": n_layers,
            "best_ceiling_layer": int(best_layer),
            "ceiling_r2_per_layer": [float(r) for r in ceiling_r2_per_layer],
            "transfer": transfer,
            "rsa": rsa,
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
