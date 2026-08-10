"""Single-variable A/B: is the RDD HMM+CPI sign-flip a label-identity artifact?

Context
-------
``scripts/rdd_lead_2004.py`` reimplements the documented RDD leading-indicator
study on the full SPY 2004+ sample (5667 daily bars). Its expanding-fit run
reported, for the CPI-only ``{acc, real}`` feature set:

  - corr(regime0, r_{t+5..t+15}) = **+0.081** (the 2014+ run was −0.03)
  - discrimination gap @21 = **+0.98%** (same sign in EARLY and LATE halves)
  - but ``relabel_conflicts = 63/103 refits (61%)``.

The confound under test: the original harness relabels the two HMM states by
per-refit **mean forward return** (regime0 = higher forward return at h=10),
tracked across refits by overlap matching. In 61% of refits the new fit's
forward-return ordering contradicted the tracked labels — if the regime0/1
*semantic* flips inside the run, the discrimination gap and series-level corr
are computed on partially mislabeled points, which could inflate or invert the
+0.081 / +0.98% numbers.

This script replaces **only** the state-relabel anchor and re-runs the
identical expanding-fit harness:

  - **A (primary) ``mean_acc``**: regime0 = the state with the *lower* fitted
    emission mean of ``z_acc`` (inflation acceleration) — cost-push easing =
    risk-on. The anchor is a fitted model parameter (``means_``), deterministic
    per fit, recomputed fresh after every refit (no cross-refit carry).
  - **B (control) ``var_acc``**: the stock ``rank_states_by_vol`` convention —
    regime0 = lowest fitted emission *variance* of the ``z_acc`` feature. The
    RDD docs said this anchor is *not* meaningful for macro features; included
    to bound anchor-sensitivity.
  - **baseline ``fwd_return``**: the original forward-return anchor, re-run
    through this same loop and the same conflict definition, so the stability
    comparison is apples-to-apples.

Conflict definition (the stability measure, lower = better)
-----------------------------------------------------------
At every refit k≥1 the new anchor map (raw state → regime0/1, from the new
fit's own fitted parameters) is aligned to the previous refit's anchor map via
overlap matching over the shared training window. A ``relabel_conflict`` is
counted if any overlap-matched state identity changed semantic regime — i.e.
the anchor flipped which persistent state is risk-on. The original script's
63/103 was a *different* (forward-return-ordering) definition; the baseline
``fwd_return`` run here reproduces the original anchor under the *same*
definition used for A and B, making the rates directly comparable.

Falsifiable outcomes
--------------------
- If ``mean_acc`` keeps band corr ≈ +0.08 and gap@21 ≈ +0.98% while
  relabel_conflicts drops to ≤5% (vs the baseline's high rate): the sign-flip
  is real and label-stable → the {acc,real} line has a stable footing.
- If under ``mean_acc`` the band corr reverts toward ≤0 / gap@21 collapses, or
  conflicts stay >20%: the +0.081/+0.98% was a label-identity artifact → the
  RDD CPI-only line stays REFINE/DEAD.

Deterministic: fixed ``random_state=42`` for every HMM fit.

Usage::

    uv run python scripts/rdd_lead_2004_anchorab.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from hmmlearn import hmm

_proj = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_proj))

from scripts.rdd_lead_2004 import (  # noqa: E402
    MIN_TRAIN_SIZE,
    RANDOM_STATE,
    RETRAIN_INTERVAL,
    _band_corr,
    _fwd_at_cursor,
    _match_new_to_old,
    _remap_by_fwd_return,
    _report_feature_set,
    build_aligned_features,
    discrimination_gap,
    load_spy_daily,
    persistent_flips,
)
from src.bt.strategies.rdd import load_yields  # noqa: E402
from src.indicators.macro._shared import load_cpi_price_index  # noqa: E402

MIN_RUN = 5
PIVOT_R = 5
MIN_SWING = 0.03
H_ANCHOR = 10
N_REGIMES = 2
ACC_COL = 0  # feature column index of z_acc in the feature frame (verified: col 0)


class _FittedHMM(Protocol):
    """A fitted GaussianHMM exposing emission means/covariances."""

    means_: np.ndarray
    covars_: np.ndarray


Anchor = Literal["mean_acc", "var_acc", "fwd_return"]


# --------------------------------------------------------------------------- #
# stable anchor remaps (pure: deterministic function of the fitted parameters)
# --------------------------------------------------------------------------- #


def _remap_by_emission_mean(
    model: _FittedHMM, n_regimes: int, acc_col: int = ACC_COL
) -> dict[int, int]:
    """Map raw HMM state → regime by the fitted emission mean of ``z_acc``.

    Regime 0 = the state with the **lower** ``z_acc`` emission mean (inflation
    acceleration easing → risk-on); regime 1 = the higher-acc state. The mean
    is a fitted model parameter, so the map is deterministic per fit and does
    not depend on assignment counts or forward returns. Ties broken by state
    index (deterministic).
    """
    means = np.asarray(model.means_, dtype=float)
    acc_mean = {s: float(means[s, acc_col]) for s in range(n_regimes)}
    order = sorted(range(n_regimes), key=lambda s: (acc_mean[s], s))
    return {raw: reg for reg, raw in enumerate(order)}


def _remap_by_emission_var(
    model: _FittedHMM, n_regimes: int, var_col: int = ACC_COL
) -> dict[int, int]:
    """Map raw HMM state → regime by fitted emission variance of ``z_acc``.

    Regime 0 = lowest variance, regime 1 = highest — the stock
    ``rank_states_by_vol`` convention, which the RDD docs flagged as *not*
    meaningful for macro features. Mirrors ``rank_states_by_vol`` including its
    handling of both hmmlearn covars layouts (ndim 3 = (n, n_dim, n_dim) diag
    stack, ndim 2 = (n, n_dim) row profile). Ties by state index.
    """
    covars = np.asarray(model.covars_, dtype=float)
    var_by_state: dict[int, float] = {}
    for s in range(n_regimes):
        if covars.ndim == 3:
            var_by_state[s] = float(covars[s, var_col, var_col])
        elif covars.ndim == 2:
            var_by_state[s] = float(covars[s, var_col])
        else:
            var_by_state[s] = float(covars[s])
    order = sorted(var_by_state, key=lambda s: (var_by_state[s], s))
    return {raw: reg for reg, raw in enumerate(order)}


def _anchor_fn(anchor: Anchor) -> Callable[[_FittedHMM, int], dict[int, int]]:
    """Dispatch to the anchor remap for ``anchor``."""
    if anchor == "mean_acc":
        return _remap_by_emission_mean
    if anchor == "var_acc":
        return _remap_by_emission_var
    raise ValueError(f"unreachable anchor: {anchor!r}")  # fwd_return handled in loop


# --------------------------------------------------------------------------- #
# anchored regime inference (expanding refit, cursor-safe)
# --------------------------------------------------------------------------- #


def infer_regime_series_anchored(
    features: pd.DataFrame,
    *,
    anchor: Anchor = "mean_acc",
    close: pd.Series | None = None,
    h_anchor: int = H_ANCHOR,
    n_regimes: int = N_REGIMES,
    min_train_size: int = MIN_TRAIN_SIZE,
    retrain_interval: int = RETRAIN_INTERVAL,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, dict]:
    """Infer a 0/1 regime series over ``features`` with a stable state anchor.

    Expanding, cursor-safe refit (data ≤ cursor only), identical cadence to
    ``rdd_lead_2004.infer_regime_series``. The difference is **only the
    anchor**: after every refit the raw-state → regime map is recomputed fresh
    from the new fit's *fitted parameters* (``means_`` for ``mean_acc``,
    ``covars_`` for ``var_acc``) or, for the ``fwd_return`` baseline, from the
    forward returns observable at that cursor. No anchor state is carried
    across refits.

    ``relabel_conflicts`` counts refits where the new anchor map disagrees with
    the previous refit's anchor map on an overlap-matched state identity (the
    persistent state's semantic flipped).

    Returns ``(regime_series, meta)`` aligned to ``features.index`` (NaN
    warmup); ``meta`` carries ``n_refits`` and ``relabel_conflicts``.
    """
    assert close is not None or anchor != "fwd_return", (
        "fwd_return anchor requires the close series"
    )
    idx = features.index
    n = len(features)
    regimes = pd.Series(np.nan, index=idx, dtype=float)
    meta: dict = {"anchor": anchor, "n_refits": 0, "relabel_conflicts": 0}

    model: hmm.GaussianHMM | None = None
    raw_to_regime: dict[int, int] = {}
    raw_hist: pd.Series = pd.Series(np.nan, index=idx, dtype=float)
    last_fit = -1

    for i in range(n):
        if i < min_train_size:
            continue
        cursor = idx[i]
        X = features.iloc[: i + 1].dropna()
        if len(X) < min_train_size:
            continue
        if model is None or i - last_fit >= retrain_interval:
            new_model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type="diag",
                random_state=random_state,
                n_iter=200,
                tol=1e-3,
            )
            new_model.fit(X.to_numpy())
            new_states = new_model.predict(X.to_numpy())
            if anchor == "fwd_return":
                assert close is not None
                fwd_train = _fwd_at_cursor(close, cursor, h_anchor).reindex(X.index)
                new_map = _remap_by_fwd_return(
                    new_states, fwd_train.to_numpy(dtype=float), n_regimes
                )
            else:
                new_map = _anchor_fn(anchor)(new_model, n_regimes)
            if model is not None:
                shared = raw_hist.dropna().index.intersection(X.index)
                if len(shared):
                    old_raw = raw_hist.reindex(shared).to_numpy(dtype=int)
                    new_raw = (
                        pd.Series(new_states, index=X.index)
                        .reindex(shared)
                        .to_numpy(dtype=int)
                    )
                    match = _match_new_to_old(new_raw, old_raw, n_regimes)
                    flipped = any(
                        nw in new_map
                        and ow in raw_to_regime
                        and new_map[nw] != raw_to_regime[ow]
                        for nw, ow in match.items()
                    )
                    if flipped:
                        meta["relabel_conflicts"] += 1
            raw_to_regime = new_map
            model = new_model
            last_fit = i
            meta["n_refits"] += 1
        assert model is not None
        st = int(model.predict(X.to_numpy())[-1])
        raw_hist.iloc[i] = st
        if st in raw_to_regime:
            regimes.iloc[i] = float(raw_to_regime[st])
    return regimes, meta


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def _print_headline(
    label: str,
    close: pd.Series,
    regimes: pd.Series,
    meta: dict,
) -> None:
    """One-line summary: band corr, gap@21, flips, and the stability measure."""
    flips = persistent_flips(regimes, min_run=MIN_RUN)
    band = _band_corr(regimes, close)
    g = discrimination_gap(regimes, close, (21,))
    gap21 = float(g.loc[g["h"] == 21, "gap"].iloc[0])
    n_conf = int(meta["relabel_conflicts"])
    n_ref = int(meta["n_refits"])
    pct = 100.0 * n_conf / max(n_ref, 1)
    print(
        f"[{label}]  band_corr(r0, r_+5..+15)={band:+.3f}  "
        f"gap@21={100 * gap21:+.2f}%  flips={len(flips)}  "
        f"conflicts={n_conf}/{n_ref} ({pct:.0f}%)"
    )


def main() -> None:
    close = load_spy_daily()
    cpi = load_cpi_price_index()
    yld = load_yields()
    print(
        f"SPY daily: {len(close)} bars  {close.index[0].date()} → {close.index[-1].date()}"
    )

    runs: list[tuple[str, bool, Anchor]] = [
        ("{acc, real}    anchor=mean_acc (emission-mean z_acc)", False, "mean_acc"),
        (
            "{acc, real}    anchor=var_acc (rank_states_by_vol control)",
            False,
            "var_acc",
        ),
        ("{acc, real}    anchor=fwd_return (original, baseline)", False, "fwd_return"),
        ("{acc, real, rdisc}  anchor=mean_acc (emission-mean z_acc)", True, "mean_acc"),
    ]

    for name, with_rdisc, anchor in runs:
        features = build_aligned_features(close, cpi, yld, with_rdisc=with_rdisc)
        regimes, meta = infer_regime_series_anchored(
            features, anchor=anchor, close=close, h_anchor=H_ANCHOR
        )
        _print_headline(name, close, regimes, meta)
        _report_feature_set(
            name,
            close,
            regimes,
            meta,
            "anchored-expanding",
            MIN_RUN,
            PIVOT_R,
            MIN_SWING,
        )


if __name__ == "__main__":
    main()
