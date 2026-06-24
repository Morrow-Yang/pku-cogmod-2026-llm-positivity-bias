"""Cognitive models for the polite-speech post-training comparison.

Each model is a class with:
  - `param_names(df)` → ordered list of free parameter names (sigma is last)
  - `init_params(df)` → reasonable starting values
  - `bounds(df)` → list of (lo, hi) tuples for scipy.optimize
  - `predict_means(df, params)` → numpy array of predicted ratings, one per row
  - `log_likelihood(df, params)` → trial-level Gaussian sum log-likelihood

This module currently implements Model 1 (null) only. Models 2-6 will land in
follow-up commits after Model 1 is end-to-end verified on real data.

Conventions
-----------
- Rating support: continuous expected value in [1, 7] (Likert digit_distribution mean).
- Trial-level fitting: every row contributes one Gaussian log-density. We do not
  pre-average over (target, cell) — see analyze.py for cell-level summaries.
- σ is the last parameter in every flat parameter vector, by convention. This
  lets the optimizer treat noise as a free parameter without special-casing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

LIKERT_MIN, LIKERT_MAX = 1.0, 7.0


class CognitiveModel:
    """Base class. Subclasses implement param_names/init/bounds/predict_means."""
    name: str = "base"

    def param_names(self, df: pd.DataFrame) -> list[str]:
        raise NotImplementedError

    def init_params(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def bounds(self, df: pd.DataFrame) -> list[tuple[float, float]]:
        raise NotImplementedError

    def predict_means(self, df: pd.DataFrame, params: np.ndarray) -> np.ndarray:
        """Return one predicted rating per row of df. Excludes sigma."""
        raise NotImplementedError

    def n_params(self, df: pd.DataFrame) -> int:
        return len(self.param_names(df))

    def log_likelihood(self, df: pd.DataFrame, params: np.ndarray) -> float:
        """Trial-level Gaussian log-likelihood. Last param is sigma."""
        sigma = params[-1]
        if sigma <= 0:
            return -np.inf
        mu = self.predict_means(df, params[:-1])
        resid = df["rating"].to_numpy() - mu
        n = len(resid)
        return float(-0.5 * np.sum(resid ** 2) / sigma ** 2
                     - n * np.log(sigma)
                     - 0.5 * n * np.log(2 * np.pi))

    def neg_log_likelihood(self, params: np.ndarray, df: pd.DataFrame) -> float:
        return -self.log_likelihood(df, params)


# ---------------------------------------------------------------------------
# Model 1: target-only null
#
# Each target has its own intercept μ_t. No effect of stage, condition, or any
# manipulation. This is the "null" against which mechanism-bearing models must
# show improvement. If a model is not preferred over Model 1, it has failed to
# explain the manipulation effects we are trying to understand.
#
# Parameters: K target intercepts + σ.
# ---------------------------------------------------------------------------
class Model1Null(CognitiveModel):
    name = "M1_target_null"

    def __init__(self):
        self._targets: list[str] | None = None

    def _ensure_targets(self, df: pd.DataFrame) -> list[str]:
        if self._targets is None:
            self._targets = sorted(df["target_id"].unique().tolist())
        return self._targets

    def param_names(self, df: pd.DataFrame) -> list[str]:
        targets = self._ensure_targets(df)
        return [f"mu_{t}" for t in targets] + ["sigma"]

    def init_params(self, df: pd.DataFrame) -> np.ndarray:
        targets = self._ensure_targets(df)
        means = df.groupby("target_id")["rating"].mean()
        mu0 = np.array([means.get(t, 4.0) for t in targets], dtype=float)
        sigma0 = float(df["rating"].std())
        return np.concatenate([mu0, [sigma0]])

    def bounds(self, df: pd.DataFrame) -> list[tuple[float, float]]:
        targets = self._ensure_targets(df)
        return [(LIKERT_MIN, LIKERT_MAX)] * len(targets) + [(1e-3, 5.0)]

    def predict_means(self, df: pd.DataFrame, params: np.ndarray) -> np.ndarray:
        targets = self._ensure_targets(df)
        mu_map = dict(zip(targets, params))
        return df["target_id"].map(mu_map).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Model 2: Kunda gated motivated-cognition model
#
# Cognitive content (Kunda 1990): the polite-speech shift on a given trial is
# the product of a (stage, condition)-specific motivational shift δ_{s,c} and
# an ambiguity gate g(b_t) that suppresses shifts when the evidence about the
# target is unambiguous.
#
#   μ(t, s, c) = b_t + δ_{s,c} · g(b_t)
#   g(b) = 4·(b - 1)·(7 - b) / 36           (∈ [0,1], peak 1 at b=4)
#
# The anchor cell is (M0, default): δ_{M0, default} ≡ 0. With that constraint,
# b_t IS the model's "natural" rating for target t at the M0-default cell
# (free parameter, fit jointly with all other params), and every other
# δ_{s,c} is the polite-speech shift the model applies on top.
#
# Compared to a saturated 2-way ANOVA on (cell × target), this model imposes
# the rank-1 constraint that the SAME δ_{s,c} applies across all targets,
# modulated only by the per-target ambiguity gate. That's the testable
# Kunda-claim — motivation × ambiguity is the lever.
#
# Parameters:
#   - K target intercepts b_t
#   - (cells - 1) cell shifts δ_{s,c}; the (M0, default) cell is anchored
#   - σ
# ---------------------------------------------------------------------------
class Model2KundaGated(CognitiveModel):
    name = "M2_kunda_gated"
    ANCHOR_CELL = ("M0", "default")

    def __init__(self):
        self._targets: list[str] | None = None
        self._cells: list[tuple[str, str]] | None = None  # free cells, anchor excluded

    def _ensure_setup(self, df: pd.DataFrame):
        if self._targets is None:
            self._targets = sorted(df["target_id"].unique().tolist())
        if self._cells is None:
            all_cells = sorted(set(zip(df["stage"], df["condition"])))
            self._cells = [c for c in all_cells if c != self.ANCHOR_CELL]

    def param_names(self, df: pd.DataFrame) -> list[str]:
        self._ensure_setup(df)
        return ([f"b_{t}" for t in self._targets]
                + [f"delta_{s}_{c}" for s, c in self._cells]
                + ["sigma"])

    def init_params(self, df: pd.DataFrame) -> np.ndarray:
        self._ensure_setup(df)
        # b_t init: empirical M0-default mean per target (the anchor), fall
        # back to grand mean for targets that don't appear in C0.
        base = df[df["cell"] == "C0"].groupby("target_id")["rating"].mean()
        grand = float(df["rating"].mean())
        b0 = np.array([float(base.get(t, grand)) for t in self._targets])

        # δ_{s,c} init: empirical cell mean minus anchor-cell mean, divided by
        # an average gate (0.7). Crude but in the right ballpark.
        anchor_mean = float(df[(df["stage"] == self.ANCHOR_CELL[0])
                               & (df["condition"] == self.ANCHOR_CELL[1])]["rating"].mean())
        cell_means = df.groupby(["stage", "condition"])["rating"].mean()
        d0 = np.array([(float(cell_means[(s, c)]) - anchor_mean) / 0.7
                       for s, c in self._cells])

        sigma0 = float(df["rating"].std()) * 0.7
        return np.concatenate([b0, d0, [sigma0]])

    def bounds(self, df: pd.DataFrame) -> list[tuple[float, float]]:
        self._ensure_setup(df)
        return ([(LIKERT_MIN, LIKERT_MAX)] * len(self._targets)
                + [(-5.0, 5.0)] * len(self._cells)
                + [(1e-3, 5.0)])

    @staticmethod
    def _gate(b: np.ndarray) -> np.ndarray:
        g = 4.0 * (b - 1.0) * (7.0 - b) / 36.0
        return np.clip(g, 0.0, 1.0)

    def predict_means(self, df: pd.DataFrame, params: np.ndarray) -> np.ndarray:
        self._ensure_setup(df)
        n_t = len(self._targets)
        n_c = len(self._cells)
        b = params[:n_t]
        d = params[n_t:n_t + n_c]
        b_map = dict(zip(self._targets, b))
        d_map = dict(zip(self._cells, d))
        d_map[self.ANCHOR_CELL] = 0.0

        b_t = df["target_id"].map(b_map).to_numpy(dtype=float)
        cell_key = list(zip(df["stage"], df["condition"]))
        d_sc = np.array([d_map[k] for k in cell_key], dtype=float)
        return b_t + d_sc * self._gate(b_t)


# ---------------------------------------------------------------------------
# Model 2-nogate: ablation of the ambiguity gate (g ≡ 1).
#
# Identical to Model2KundaGated but with the gate disabled, so the SAME cell
# shift δ_{s,c} applies to every target regardless of how ambiguous (mid-scale)
# its base rating is. This is the critical test of whether the Kunda
# "ambiguity is the lever" claim earns its keep: if M2-nogate fits as well or
# better (by BIC / CV), the ambiguity gate adds no explanatory value and the
# motivated-reasoning interpretation should be downgraded to a plain cell-shift.
# Same parameter count as M2 (the gate has no free parameters), so AIC/BIC
# differences are pure goodness-of-fit.
# ---------------------------------------------------------------------------
class Model2NoGate(Model2KundaGated):
    name = "M2_nogate"

    @staticmethod
    def _gate(b: np.ndarray) -> np.ndarray:
        return np.ones_like(b)


# ---------------------------------------------------------------------------
# Model 2-freegate: free-form ambiguity gate.
#
# Replaces the hard-coded quadratic gate (peak at b=4, fixed width) with a
# Gaussian bump whose peak location p and width w are FREE parameters:
#
#   g(b) = exp(−(b − p)² / (2 w²))      (peak 1 at b = p)
#
# This tests whether the ASSUMED gate shape (peak exactly at the scale midpoint,
# parabolic falloff) is what the data want, or whether the apparent
# "ambiguity gating" is an artifact of the fixed functional form. Adds 2 free
# parameters (p, w) over M2.
# ---------------------------------------------------------------------------
class Model2FreeGate(Model2KundaGated):
    name = "M2_freegate"

    def param_names(self, df: pd.DataFrame) -> list[str]:
        self._ensure_setup(df)
        return ([f"b_{t}" for t in self._targets]
                + [f"delta_{s}_{c}" for s, c in self._cells]
                + ["gate_peak", "gate_width", "sigma"])

    def init_params(self, df: pd.DataFrame) -> np.ndarray:
        base = super().init_params(df)            # [b..., d..., sigma]
        sigma0 = base[-1]
        return np.concatenate([base[:-1], [4.0, 1.5, sigma0]])

    def bounds(self, df: pd.DataFrame) -> list[tuple[float, float]]:
        self._ensure_setup(df)
        return ([(LIKERT_MIN, LIKERT_MAX)] * len(self._targets)
                + [(-5.0, 5.0)] * len(self._cells)
                + [(1.0, 7.0), (0.3, 6.0)]        # gate_peak, gate_width
                + [(1e-3, 5.0)])

    def predict_means(self, df: pd.DataFrame, params: np.ndarray) -> np.ndarray:
        self._ensure_setup(df)
        n_t = len(self._targets)
        n_c = len(self._cells)
        b = params[:n_t]
        d = params[n_t:n_t + n_c]
        peak = params[n_t + n_c]
        width = params[n_t + n_c + 1]
        b_map = dict(zip(self._targets, b))
        d_map = dict(zip(self._cells, d))
        d_map[self.ANCHOR_CELL] = 0.0
        b_t = df["target_id"].map(b_map).to_numpy(dtype=float)
        cell_key = list(zip(df["stage"], df["condition"]))
        d_sc = np.array([d_map[k] for k in cell_key], dtype=float)
        gate = np.exp(-((b_t - peak) ** 2) / (2.0 * width ** 2))
        return b_t + d_sc * gate


# ---------------------------------------------------------------------------
# Model 4: RSA polite-speaker, three-utility decomposition (Yoon 2020-style)
#
# A simplification of Yoon, Tessler, Goodman & Frank (2020) [[polite-speech-
# emerges-competing-social-goals]] for our continuous-Likert setting. The
# pragmatic speaker chooses utterance w from a softmax over a three-utility
# composition:
#
#   U_total(w; s, ω) = ω_inf · U_inf(w; s) + ω_pos · U_pos(w) + ω_neg · U_neg(w)
#
# where (ω_inf, ω_pos, ω_neg) is a simplex (sums to 1). The full continuous
# expected-rating reduces (under the right utility shapes) to a mixture:
#
#   μ(t, s, c) = ω_inf(s, c) · b_t  +  ω_pos(s, c) · H_pos  +  ω_neg(s, c) · H_neg
#
# Interpretation: each cell (stage, condition) installs a mixture of three
# pulls — toward the truth (b_t), toward the positive-social anchor (H_pos=7,
# "kind"), and toward the negative-social anchor (H_neg=1, "anti-polite /
# blunt"). The anchor cell (M0, default) is fixed at (1, 0, 0) — the base
# model is informationally truthful, with no politeness mask of either sign.
#
# The simplex is parameterized via softmax over free logits l_pos(s,c) and
# l_neg(s,c) with l_inf ≡ 0, so optimization stays unconstrained. The anchor
# cell gets fixed dummy logits of -10 (≈ 0 weight).
#
# Parameters:
#   - 9 target intercepts b_t (informational truth)
#   - 2 · (n_cells - 1) logits (l_pos, l_neg) for non-anchor cells
#   - σ
# ---------------------------------------------------------------------------
class Model4RSAPoliteSpeaker(CognitiveModel):
    name = "M4_rsa_polite_speaker"
    ANCHOR_CELL = ("M0", "default")
    H_POS = 7.0
    H_NEG = 1.0
    ANCHOR_LOGIT = -10.0  # ≈ ω_pos = ω_neg = 0 in softmax

    def __init__(self):
        self._targets: list[str] | None = None
        self._cells: list[tuple[str, str]] | None = None

    def _ensure_setup(self, df: pd.DataFrame):
        if self._targets is None:
            self._targets = sorted(df["target_id"].unique().tolist())
        if self._cells is None:
            all_cells = sorted(set(zip(df["stage"], df["condition"])))
            self._cells = [c for c in all_cells if c != self.ANCHOR_CELL]

    def param_names(self, df: pd.DataFrame) -> list[str]:
        self._ensure_setup(df)
        return ([f"b_{t}" for t in self._targets]
                + [f"l_pos_{s}_{c}" for s, c in self._cells]
                + [f"l_neg_{s}_{c}" for s, c in self._cells]
                + ["sigma"])

    def init_params(self, df: pd.DataFrame) -> np.ndarray:
        self._ensure_setup(df)
        base = df[df["cell"] == "C0"].groupby("target_id")["rating"].mean()
        grand = float(df["rating"].mean())
        b0 = np.array([float(base.get(t, grand)) for t in self._targets])

        # Initialize logits by sign of cell-vs-anchor mean shift.
        anchor_mean = float(df[(df["stage"] == self.ANCHOR_CELL[0])
                               & (df["condition"] == self.ANCHOR_CELL[1])]["rating"].mean())
        cell_means = df.groupby(["stage", "condition"])["rating"].mean()
        l_pos0 = np.full(len(self._cells), -3.0)
        l_neg0 = np.full(len(self._cells), -3.0)
        for i, key in enumerate(self._cells):
            shift = float(cell_means[key]) - anchor_mean
            if shift > 0:
                l_pos0[i] = -1.5  # moderate positive pull
            elif shift < 0:
                l_neg0[i] = -1.5

        sigma0 = float(df["rating"].std()) * 0.7
        return np.concatenate([b0, l_pos0, l_neg0, [sigma0]])

    def bounds(self, df: pd.DataFrame) -> list[tuple[float, float]]:
        self._ensure_setup(df)
        return ([(LIKERT_MIN, LIKERT_MAX)] * len(self._targets)
                + [(-8.0, 4.0)] * len(self._cells)
                + [(-8.0, 4.0)] * len(self._cells)
                + [(1e-3, 5.0)])

    def predict_means(self, df: pd.DataFrame, params: np.ndarray) -> np.ndarray:
        self._ensure_setup(df)
        n_t = len(self._targets)
        n_c = len(self._cells)
        b = params[:n_t]
        l_pos = params[n_t:n_t + n_c]
        l_neg = params[n_t + n_c:n_t + 2 * n_c]

        b_map = dict(zip(self._targets, b))
        l_pos_map = dict(zip(self._cells, l_pos))
        l_neg_map = dict(zip(self._cells, l_neg))
        l_pos_map[self.ANCHOR_CELL] = self.ANCHOR_LOGIT
        l_neg_map[self.ANCHOR_CELL] = self.ANCHOR_LOGIT

        cell_key = list(zip(df["stage"], df["condition"]))
        lp = np.array([l_pos_map[k] for k in cell_key], dtype=float)
        ln = np.array([l_neg_map[k] for k in cell_key], dtype=float)
        b_t = df["target_id"].map(b_map).to_numpy(dtype=float)

        # softmax(0, lp, ln) → (w_inf, w_pos, w_neg)
        logits = np.stack([np.zeros_like(lp), lp, ln], axis=1)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        z = e.sum(axis=1, keepdims=True)
        w = e / z
        w_inf, w_pos, w_neg = w[:, 0], w[:, 1], w[:, 2]

        return w_inf * b_t + w_pos * self.H_POS + w_neg * self.H_NEG


# ---------------------------------------------------------------------------
# Model 3: Sign-asymmetric valence sensitivity
#   (inspired by — NOT an implementation of — optimism-bias asymmetry,
#    Sharot 2011; Lefebvre 2017)
#
# IMPORTANT framing (2026-06-21): Lefebvre/Sharot's α+/α− are LEARNING RATES on
# reward prediction errors in a sequential Rescorla-Wagner update over repeated
# outcomes. Our paradigm has NO trial sequence and NO outcomes to learn from
# (each rating is an independent greedy forward pass), so a true RW updating
# model is structurally inapplicable. We therefore borrow only the *asymmetry
# motif*: a static, sign-dependent sensitivity to positive- vs negative-valence
# prompt pulls. α± here are SENSITIVITIES, not learning rates. We cite Sharot/
# Lefebvre as conceptual motivation, not as the implemented model.
#
# Cognitive content: the per-trial shift is the product of (a) a per-stage
# sign-dependent sensitivity α(s, ±), (b) a condition-specific intensity, and
# (c) the ambiguity gate g(b_t).
#
# Direction dir(c) ∈ {+, -} is a FIXED, A-PRIORI INPUT from prompt design (NOT
# the realized data sign — see _ensure_setup). Negativity-licensing cells
# (forced, anti-politeness-*, *-completion neg variants) get α−(s); every other
# cell (default, control, alt-control-*, control-/default-completion) gets α+(s).
#
#   μ(t, s, c) = b_t + sign(dir(c)) · α(s, dir(c)) · I(c) · g(b_t)
#
# Identifiability: α(M0, +) ≡ α(M0, -) ≡ 0 (M0 base model has no learning
# from instruction). intensity(default) ≡ 1 (anchor to set the α scale).
#
# Diagnostic: if α(M3, -) > α(M3, +), the M3 instruct model is "anti-
# optimistic" — it discounts polite-up pulls relative to anti-polite-down
# pulls, consistent with the below-base overshoot finding.
#
# Parameters:
#   - 9 target intercepts b_t
#   - 3 stages × 2 directions = 6 rates α(s, ±); M0 fixed at 0
#   - n_conditions - 1 intensities (anchor at default = 1)
#   - σ
# ---------------------------------------------------------------------------
class Model3AsymmetricBelief(CognitiveModel):
    name = "M3_asymmetric_belief"
    FREE_STAGES = ["M1", "M2", "M3"]
    ANCHOR_CONDITION = "default"

    def __init__(self):
        self._targets: list[str] | None = None
        self._conditions: list[str] | None = None  # free conditions, anchor excluded
        self._cond_direction: dict[str, int] | None = None  # +1 or -1, fixed per cond

    def _ensure_setup(self, df: pd.DataFrame):
        if self._targets is None:
            self._targets = sorted(df["target_id"].unique().tolist())
        if self._conditions is None:
            all_conds = sorted(df["condition"].unique().tolist())
            self._conditions = [c for c in all_conds if c != self.ANCHOR_CONDITION]
        if self._cond_direction is None:
            # A-PRIORI valence map (de-circularized, 2026-06-21). Direction is
            # fixed by PROMPT DESIGN, not by the realized data sign: negativity-
            # licensing prompts (forced / anti-politeness / their completion
            # variants) are negative-direction (−); every other condition
            # (default, control, alt-control-*, control-completion, default-
            # completion) is positive-direction (+). This removes the earlier
            # circularity where dir(c) was read from cond_means >= anchor_mean
            # and then α± were fit to the same data — partly baking in α− > α+.
            # NEG_LICENSE_CONDITIONS is the same a-priori set used by M5.
            all_conds = sorted(df["condition"].unique().tolist())
            self._cond_direction = {
                c: (-1 if c in NEG_LICENSE_CONDITIONS else +1)
                for c in all_conds
            }
            # The anchor condition (default) is positive-direction: its intensity
            # is fixed at 1, so M1-default predicts b_t + α+(M1)·g(b_t).
            self._cond_direction[self.ANCHOR_CONDITION] = +1

    def param_names(self, df: pd.DataFrame) -> list[str]:
        self._ensure_setup(df)
        return ([f"b_{t}" for t in self._targets]
                + [f"alpha_{s}_pos" for s in self.FREE_STAGES]
                + [f"alpha_{s}_neg" for s in self.FREE_STAGES]
                + [f"I_{c}" for c in self._conditions]
                + ["sigma"])

    def init_params(self, df: pd.DataFrame) -> np.ndarray:
        self._ensure_setup(df)
        base = df[df["cell"] == "C0"].groupby("target_id")["rating"].mean()
        grand = float(df["rating"].mean())
        b0 = np.array([float(base.get(t, grand)) for t in self._targets])

        alpha_pos0 = np.full(len(self.FREE_STAGES), 0.5)
        alpha_neg0 = np.full(len(self.FREE_STAGES), 0.5)

        # Initialize intensities from |empirical shift| / |anchor shift|
        anchor_mean = float(df[df["condition"] == self.ANCHOR_CONDITION]["rating"].mean())
        cond_means = df.groupby("condition")["rating"].mean()
        I0 = np.array([abs(float(cond_means[c]) - anchor_mean) + 0.3
                       for c in self._conditions])

        sigma0 = float(df["rating"].std()) * 0.7
        return np.concatenate([b0, alpha_pos0, alpha_neg0, I0, [sigma0]])

    def bounds(self, df: pd.DataFrame) -> list[tuple[float, float]]:
        self._ensure_setup(df)
        return ([(LIKERT_MIN, LIKERT_MAX)] * len(self._targets)
                + [(0.0, 5.0)] * len(self.FREE_STAGES)   # α_pos
                + [(0.0, 5.0)] * len(self.FREE_STAGES)   # α_neg
                + [(0.0, 5.0)] * len(self._conditions)   # intensities
                + [(1e-3, 5.0)])

    @staticmethod
    def _gate(b: np.ndarray) -> np.ndarray:
        g = 4.0 * (b - 1.0) * (7.0 - b) / 36.0
        return np.clip(g, 0.0, 1.0)

    def predict_means(self, df: pd.DataFrame, params: np.ndarray) -> np.ndarray:
        self._ensure_setup(df)
        n_t = len(self._targets)
        n_s = len(self.FREE_STAGES)
        n_c = len(self._conditions)
        b = params[:n_t]
        a_pos = params[n_t:n_t + n_s]
        a_neg = params[n_t + n_s:n_t + 2 * n_s]
        I = params[n_t + 2 * n_s:n_t + 2 * n_s + n_c]

        # Per-stage rate maps. M0 has zero learning.
        alpha_pos_map = {s: a for s, a in zip(self.FREE_STAGES, a_pos)}
        alpha_neg_map = {s: a for s, a in zip(self.FREE_STAGES, a_neg)}
        alpha_pos_map["M0"] = 0.0
        alpha_neg_map["M0"] = 0.0

        # Per-condition intensity map. Anchor has intensity 1.
        I_map = {c: i for c, i in zip(self._conditions, I)}
        I_map[self.ANCHOR_CONDITION] = 1.0

        b_map = dict(zip(self._targets, b))

        stages = df["stage"].to_numpy()
        conds = df["condition"].to_numpy()
        b_t = df["target_id"].map(b_map).to_numpy(dtype=float)
        g_b = self._gate(b_t)

        # Vectorize: for each row, pick α(stage, dir(cond)) · intensity(cond) · sign
        n = len(df)
        out = np.empty(n)
        for i in range(n):
            s = stages[i]; c = conds[i]
            d = self._cond_direction[c]
            alpha = alpha_pos_map[s] if d > 0 else alpha_neg_map[s]
            out[i] = b_t[i] + d * alpha * I_map[c] * g_b[i]
        return out


NEG_LICENSE_CONDITIONS = {
    "forced", "anti-politeness-mild", "anti-politeness-no-neg-words",
    "anti-politeness-strong", "forced-completion",
}


def _has_chat(stage: str, cond: str, cell_format_map: dict) -> int:
    """Return 1 if this (stage, cond) cell uses chat-template format, else 0."""
    # We look up via the CELL_TO_FORMAT registry (passed in as cell_format_map),
    # but our DF only knows (stage, condition). We need to know which cell maps
    # to which (stage, condition) — that's done at data-loading time. Here we
    # re-derive by passing the cell_format flag through df["format"].
    raise NotImplementedError("Use df['format'] column directly")


# ---------------------------------------------------------------------------
# Model 5: RSA polite-speaker + license-conditional γ_k
#
# Restricts Model 4's per-cell negative-pull logit l_neg to a 2-way ANOVA on
# (chat-template × negativity-license) with main effects and interaction:
#
#   l_neg(s, c) = β_chat · has_chat(s, c)
#               + β_license · has_neg_license(c)
#               + γ_k · has_chat(s, c) · has_neg_license(c)
#
# The l_pos structure is kept per-cell as in M4 (the polite-up shifts are
# the "uninteresting" politeness component we already characterized).
#
# Diagnostic γ_k > 0 (with CI strictly above 0) is the NOVEL claim of this
# project: there is an interaction between chat-template format and explicit
# negativity-license instruction that drives below-base overshoot beyond what
# either factor alone explains. Each main effect tested separately:
#   - β_chat > 0 alone would mean chat-template causes the overshoot (would
#     contradict our finding that completion-mode forced cells don't overshoot)
#   - β_license > 0 alone would mean any neg-license causes it (would
#     contradict the M0-forced cell that doesn't overshoot)
#   - γ_k > 0 with β_chat, β_license small is our predicted signature
#
# has_chat is read from df["format"] == "chat".
# has_neg_license is read from condition ∈ NEG_LICENSE_CONDITIONS.
#
# Parameters:
#   - 9 target intercepts b_t
#   - 27 cell-level l_pos (non-anchor cells, as in M4)
#   - 3 negative-pull params: β_chat, β_license, γ_k
#   - σ
# ---------------------------------------------------------------------------
class Model5RSALicenseConditional(CognitiveModel):
    name = "M5_rsa_license_conditional"
    ANCHOR_CELL = ("M0", "default")
    H_POS = 7.0
    H_NEG = 1.0
    L_NEG_BASELINE = -8.0   # baseline log-odds when both indicators are 0

    def __init__(self):
        self._targets: list[str] | None = None
        self._cells: list[tuple[str, str]] | None = None

    def _ensure_setup(self, df: pd.DataFrame):
        if self._targets is None:
            self._targets = sorted(df["target_id"].unique().tolist())
        if self._cells is None:
            all_cells = sorted(set(zip(df["stage"], df["condition"])))
            self._cells = [c for c in all_cells if c != self.ANCHOR_CELL]

    FREE_STAGES = ["M1", "M2", "M3"]

    def param_names(self, df: pd.DataFrame) -> list[str]:
        self._ensure_setup(df)
        return ([f"b_{t}" for t in self._targets]
                + [f"l_pos_{s}_{c}" for s, c in self._cells]
                + ["beta_chat", "beta_license"]
                + [f"gamma_k_{s}" for s in self.FREE_STAGES]
                + ["sigma"])

    def init_params(self, df: pd.DataFrame) -> np.ndarray:
        self._ensure_setup(df)
        base = df[df["cell"] == "C0"].groupby("target_id")["rating"].mean()
        grand = float(df["rating"].mean())
        b0 = np.array([float(base.get(t, grand)) for t in self._targets])

        anchor_mean = float(df[(df["stage"] == self.ANCHOR_CELL[0])
                               & (df["condition"] == self.ANCHOR_CELL[1])]["rating"].mean())
        cell_means = df.groupby(["stage", "condition"])["rating"].mean()
        l_pos0 = np.full(len(self._cells), -3.0)
        for i, key in enumerate(self._cells):
            if float(cell_means[key]) > anchor_mean:
                l_pos0[i] = -1.5

        beta_chat0 = 0.3
        beta_license0 = 0.3
        gamma_k0 = np.array([0.3, 0.3, 2.0])  # M3 anticipated bigger
        sigma0 = float(df["rating"].std()) * 0.7
        return np.concatenate([
            b0, l_pos0,
            [beta_chat0, beta_license0],
            gamma_k0,
            [sigma0]
        ])

    def bounds(self, df: pd.DataFrame) -> list[tuple[float, float]]:
        self._ensure_setup(df)
        return ([(LIKERT_MIN, LIKERT_MAX)] * len(self._targets)
                + [(-8.0, 4.0)] * len(self._cells)
                + [(0.0, 10.0)]    # beta_chat ≥ 0
                + [(0.0, 10.0)]    # beta_license ≥ 0
                + [(0.0, 10.0)] * len(self.FREE_STAGES)  # γ_k per stage ≥ 0
                + [(1e-3, 5.0)])

    def predict_means(self, df: pd.DataFrame, params: np.ndarray) -> np.ndarray:
        self._ensure_setup(df)
        n_t = len(self._targets)
        n_c = len(self._cells)
        n_s = len(self.FREE_STAGES)
        b = params[:n_t]
        l_pos = params[n_t:n_t + n_c]
        beta_chat = params[n_t + n_c]
        beta_license = params[n_t + n_c + 1]
        gamma_k_per_stage = params[n_t + n_c + 2:n_t + n_c + 2 + n_s]
        gamma_k_map = {s: g for s, g in zip(self.FREE_STAGES, gamma_k_per_stage)}
        gamma_k_map["M0"] = 0.0

        b_map = dict(zip(self._targets, b))
        l_pos_map = dict(zip(self._cells, l_pos))
        l_pos_map[self.ANCHOR_CELL] = -8.0

        has_chat = (df["format"] == "chat").to_numpy(dtype=float)
        has_license = df["condition"].isin(NEG_LICENSE_CONDITIONS).to_numpy(dtype=float)
        gamma_k = df["stage"].map(gamma_k_map).to_numpy(dtype=float)

        cell_key = list(zip(df["stage"], df["condition"]))
        lp = np.array([l_pos_map[k] for k in cell_key], dtype=float)

        ln = (self.L_NEG_BASELINE
              + beta_chat * has_chat
              + beta_license * has_license
              + gamma_k * has_chat * has_license)

        b_t = df["target_id"].map(b_map).to_numpy(dtype=float)

        logits = np.stack([np.zeros_like(lp), lp, ln], axis=1)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        z = e.sum(axis=1, keepdims=True)
        w = e / z
        w_inf, w_pos, w_neg = w[:, 0], w[:, 1], w[:, 2]

        return w_inf * b_t + w_pos * self.H_POS + w_neg * self.H_NEG


# ---------------------------------------------------------------------------
# Model 0: Linear additive fixed-effects baseline (pure ANOVA, no cognitive
# content). Added after Review LLM flagged the missing "no cognition" baseline.
#
#   μ(t, s, c) = b_t + β_s + γ_c
#
# This tests whether any of the cognitive-content models actually beat a
# vanilla additive ANOVA. If a cognitive model loses to M0_linear, the
# "cognitive structure" it adds is decorative.
#
# Identifiability: β_M0 ≡ 0 (anchor on stage), γ_default ≡ 0 (anchor on
# condition). The remaining b_t intercepts absorb the global mean.
# ---------------------------------------------------------------------------
class Model0LinearAdditive(CognitiveModel):
    name = "M0_linear_additive"
    ANCHOR_STAGE = "M0"
    ANCHOR_CONDITION = "default"

    def __init__(self):
        self._targets: list[str] | None = None
        self._stages: list[str] | None = None
        self._conditions: list[str] | None = None

    def _ensure_setup(self, df: pd.DataFrame):
        if self._targets is None:
            self._targets = sorted(df["target_id"].unique().tolist())
        if self._stages is None:
            all_s = sorted(df["stage"].unique().tolist())
            self._stages = [s for s in all_s if s != self.ANCHOR_STAGE]
        if self._conditions is None:
            all_c = sorted(df["condition"].unique().tolist())
            self._conditions = [c for c in all_c if c != self.ANCHOR_CONDITION]

    def param_names(self, df: pd.DataFrame) -> list[str]:
        self._ensure_setup(df)
        return ([f"b_{t}" for t in self._targets]
                + [f"beta_{s}" for s in self._stages]
                + [f"gamma_{c}" for c in self._conditions]
                + ["sigma"])

    def init_params(self, df: pd.DataFrame) -> np.ndarray:
        self._ensure_setup(df)
        base = df[df["cell"] == "C0"].groupby("target_id")["rating"].mean()
        grand = float(df["rating"].mean())
        b0 = np.array([float(base.get(t, grand)) for t in self._targets])
        stage_means = df.groupby("stage")["rating"].mean()
        cond_means = df.groupby("condition")["rating"].mean()
        anchor_s = float(stage_means[self.ANCHOR_STAGE])
        anchor_c = float(cond_means[self.ANCHOR_CONDITION])
        beta0 = np.array([float(stage_means[s]) - anchor_s for s in self._stages])
        gamma0 = np.array([float(cond_means[c]) - anchor_c for c in self._conditions])
        sigma0 = float(df["rating"].std()) * 0.7
        return np.concatenate([b0, beta0, gamma0, [sigma0]])

    def bounds(self, df: pd.DataFrame) -> list[tuple[float, float]]:
        self._ensure_setup(df)
        return ([(LIKERT_MIN, LIKERT_MAX)] * len(self._targets)
                + [(-3.0, 3.0)] * len(self._stages)
                + [(-3.0, 3.0)] * len(self._conditions)
                + [(1e-3, 5.0)])

    def predict_means(self, df: pd.DataFrame, params: np.ndarray) -> np.ndarray:
        self._ensure_setup(df)
        n_t = len(self._targets)
        n_s = len(self._stages)
        n_c = len(self._conditions)
        b = params[:n_t]
        beta = params[n_t:n_t + n_s]
        gamma = params[n_t + n_s:n_t + n_s + n_c]
        b_map = dict(zip(self._targets, b))
        beta_map = dict(zip(self._stages, beta))
        beta_map[self.ANCHOR_STAGE] = 0.0
        gamma_map = dict(zip(self._conditions, gamma))
        gamma_map[self.ANCHOR_CONDITION] = 0.0
        b_t = df["target_id"].map(b_map).to_numpy(dtype=float)
        b_s = df["stage"].map(beta_map).to_numpy(dtype=float)
        g_c = df["condition"].map(gamma_map).to_numpy(dtype=float)
        return b_t + b_s + g_c


# ---------------------------------------------------------------------------
# Model 3-symmetric: the nested-null control for M3 asymmetric.
#
# Identical to Model 3, but with α+(s) ≡ α-(s) ≡ α(s) — i.e., the same
# learning rate for positive and negative direction cells. Per stage.
#
# Likelihood-ratio test M3 vs M3-symmetric: if M3 doesn't beat its symmetric
# nested null by more than the χ²(3) critical value (8.81 at 0.05), the
# asymmetric finding is not supported by the data.
# ---------------------------------------------------------------------------
class Model3SymmetricBelief(Model3AsymmetricBelief):
    name = "M3_symmetric_belief"

    def param_names(self, df: pd.DataFrame) -> list[str]:
        self._ensure_setup(df)
        return ([f"b_{t}" for t in self._targets]
                + [f"alpha_{s}" for s in self.FREE_STAGES]
                + [f"I_{c}" for c in self._conditions]
                + ["sigma"])

    def init_params(self, df: pd.DataFrame) -> np.ndarray:
        self._ensure_setup(df)
        base = df[df["cell"] == "C0"].groupby("target_id")["rating"].mean()
        grand = float(df["rating"].mean())
        b0 = np.array([float(base.get(t, grand)) for t in self._targets])
        alpha0 = np.full(len(self.FREE_STAGES), 0.4)
        anchor_mean = float(df[df["condition"] == self.ANCHOR_CONDITION]["rating"].mean())
        cond_means = df.groupby("condition")["rating"].mean()
        I0 = np.array([abs(float(cond_means[c]) - anchor_mean) + 0.3
                       for c in self._conditions])
        sigma0 = float(df["rating"].std()) * 0.7
        return np.concatenate([b0, alpha0, I0, [sigma0]])

    def bounds(self, df: pd.DataFrame) -> list[tuple[float, float]]:
        self._ensure_setup(df)
        return ([(LIKERT_MIN, LIKERT_MAX)] * len(self._targets)
                + [(0.0, 5.0)] * len(self.FREE_STAGES)   # single α per stage
                + [(0.0, 5.0)] * len(self._conditions)
                + [(1e-3, 5.0)])

    def predict_means(self, df: pd.DataFrame, params: np.ndarray) -> np.ndarray:
        self._ensure_setup(df)
        n_t = len(self._targets)
        n_s = len(self.FREE_STAGES)
        n_c = len(self._conditions)
        b = params[:n_t]
        alpha = params[n_t:n_t + n_s]
        I = params[n_t + n_s:n_t + n_s + n_c]
        alpha_map = {s: a for s, a in zip(self.FREE_STAGES, alpha)}
        alpha_map["M0"] = 0.0
        I_map = {c: i for c, i in zip(self._conditions, I)}
        I_map[self.ANCHOR_CONDITION] = 1.0
        b_map = dict(zip(self._targets, b))

        stages = df["stage"].to_numpy()
        conds = df["condition"].to_numpy()
        b_t = df["target_id"].map(b_map).to_numpy(dtype=float)
        g_b = self._gate(b_t)
        n = len(df)
        out = np.empty(n)
        for i in range(n):
            s = stages[i]; c = conds[i]
            d = self._cond_direction[c]
            out[i] = b_t[i] + d * alpha_map[s] * I_map[c] * g_b[i]
        return out


# ---------------------------------------------------------------------------
# Model 3-M3sym: the stage-specific nested null for M3-asymmetric.
#
# Identical to Model 3-asymmetric, but α+(M3) ≡ α-(M3) ≡ α(M3) — i.e., the
# stage-M3 cell forces symmetric learning rate, while M1 and M2 remain free
# to be asymmetric. This isolates the question "is the M3-specific reversal
# of asymmetry statistically necessary?" from the broader LR test that lumps
# the strong M1/M2 asymmetry with M3's marginal one.
#
# LR test M3-asymmetric vs M3-M3sym: df = 1.
#   χ²(1) 95% critical = 3.84; 99% = 6.63.
# ---------------------------------------------------------------------------
class Model3AsymmetricM3Sym(Model3AsymmetricBelief):
    name = "M3_asymmetric_m3sym"

    def param_names(self, df: pd.DataFrame) -> list[str]:
        self._ensure_setup(df)
        return ([f"b_{t}" for t in self._targets]
                + ["alpha_M1_pos", "alpha_M2_pos", "alpha_M3"]
                + ["alpha_M1_neg", "alpha_M2_neg"]
                + [f"I_{c}" for c in self._conditions]
                + ["sigma"])

    def init_params(self, df: pd.DataFrame) -> np.ndarray:
        self._ensure_setup(df)
        base = df[df["cell"] == "C0"].groupby("target_id")["rating"].mean()
        grand = float(df["rating"].mean())
        b0 = np.array([float(base.get(t, grand)) for t in self._targets])
        # 3 pos rates (M1, M2, M3-shared) + 2 neg rates (M1, M2)
        alpha0 = np.array([0.5, 0.5, 0.2, 0.0, 0.0])
        anchor_mean = float(df[df["condition"] == self.ANCHOR_CONDITION]["rating"].mean())
        cond_means = df.groupby("condition")["rating"].mean()
        I0 = np.array([abs(float(cond_means[c]) - anchor_mean) + 0.3
                       for c in self._conditions])
        sigma0 = float(df["rating"].std()) * 0.7
        return np.concatenate([b0, alpha0, I0, [sigma0]])

    def bounds(self, df: pd.DataFrame) -> list[tuple[float, float]]:
        self._ensure_setup(df)
        return ([(LIKERT_MIN, LIKERT_MAX)] * len(self._targets)
                + [(0.0, 5.0)] * 5
                + [(0.0, 5.0)] * len(self._conditions)
                + [(1e-3, 5.0)])

    def predict_means(self, df: pd.DataFrame, params: np.ndarray) -> np.ndarray:
        self._ensure_setup(df)
        n_t = len(self._targets)
        n_c = len(self._conditions)
        b = params[:n_t]
        alpha_M1_pos = params[n_t]
        alpha_M2_pos = params[n_t + 1]
        alpha_M3 = params[n_t + 2]  # shared
        alpha_M1_neg = params[n_t + 3]
        alpha_M2_neg = params[n_t + 4]
        I = params[n_t + 5:n_t + 5 + n_c]

        alpha_pos_map = {"M1": alpha_M1_pos, "M2": alpha_M2_pos, "M3": alpha_M3, "M0": 0.0}
        alpha_neg_map = {"M1": alpha_M1_neg, "M2": alpha_M2_neg, "M3": alpha_M3, "M0": 0.0}
        I_map = {c: i for c, i in zip(self._conditions, I)}
        I_map[self.ANCHOR_CONDITION] = 1.0
        b_map = dict(zip(self._targets, b))

        stages = df["stage"].to_numpy()
        conds = df["condition"].to_numpy()
        b_t = df["target_id"].map(b_map).to_numpy(dtype=float)
        g_b = self._gate(b_t)
        n = len(df)
        out = np.empty(n)
        for i in range(n):
            s = stages[i]; c = conds[i]
            d = self._cond_direction[c]
            alpha = alpha_pos_map[s] if d > 0 else alpha_neg_map[s]
            out[i] = b_t[i] + d * alpha * I_map[c] * g_b[i]
        return out


REGISTRY = {
    "M0_linear": Model0LinearAdditive,
    "M1": Model1Null,
    "M2": Model2KundaGated,
    "M2_nogate": Model2NoGate,
    "M2_freegate": Model2FreeGate,
    "M3": Model3AsymmetricBelief,
    "M3_symmetric": Model3SymmetricBelief,
    "M3_m3sym": Model3AsymmetricM3Sym,
    "M4": Model4RSAPoliteSpeaker,
    "M5": Model5RSALicenseConditional,
}
