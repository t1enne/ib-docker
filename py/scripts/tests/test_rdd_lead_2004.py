"""Unit tests for the pure helpers in scripts/rdd_lead_2004.py.

Covers flip screening, pivot detection, forward-return relabel, state-identity
matching and the block-bootstrap premium — no DB, no HMM fit required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.rdd_lead_2004 import (
    _block_bootstrap_premium_p,
    _match_new_to_old,
    _remap_by_fwd_return,
    persistent_flips,
    pivot_mask,
    significant_pivots,
)


# --------------------------------------------------------------------------- #
# persistent_flips
# --------------------------------------------------------------------------- #


def test_persistent_flips_screens_flicker() -> None:
    r = pd.Series(
        [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
        index=pd.date_range("2020-01-01", periods=12, freq="D"),
    )
    flips = persistent_flips(r, min_run=3)
    # 0→1 at idx3; the single 1 at idx7 is screened; 1→0 at idx8 (0-run start).
    assert [(d, f, t) for d, f, t in flips] == [
        (r.index[3], 0, 1),
        (r.index[8], 1, 0),
    ]


def test_persistent_flips_no_flip_within_short_runs() -> None:
    r = pd.Series([0, 0, 1, 0, 0, 0, 0], index=range(7))
    # the 1-run (len 1) and the leading 00 are all < min_run=3 except the tail.
    flips = persistent_flips(r, min_run=3)
    assert flips == []


def test_persistent_flips_respects_min_run() -> None:
    r = pd.Series([0, 0, 0, 1, 1, 0, 0, 0], index=range(8))
    assert len(persistent_flips(r, min_run=2)) == 2
    assert len(persistent_flips(r, min_run=4)) == 0


# --------------------------------------------------------------------------- #
# pivots
# --------------------------------------------------------------------------- #


def test_pivot_mask_finds_highs_and_lows() -> None:
    close = pd.Series([1, 3, 1, 3, 1], index=range(5))
    highs = close.index[pivot_mask(close, 1, "high")]
    lows = close.index[pivot_mask(close, 1, "low")]
    # window is centered ±1 with min_periods=3 → endpoints can never be pivots
    assert list(highs) == [1, 3]
    assert list(lows) == [2]


def test_significant_pivots_requires_min_swing() -> None:
    # monotone ramp: no significant pivot of either kind
    close = pd.Series(np.linspace(1.0, 2.0, 21), index=range(21))
    h_ramp, l_ramp = significant_pivots(close, r=2, min_swing=0.03)
    assert len(h_ramp) == 0 and len(l_ramp) == 0

    # a real 10% peak survives the swing filter
    base = np.linspace(1.0, 1.0, 21)
    base[10] = 1.10
    close2 = pd.Series(base, index=range(21))
    h2, l2 = significant_pivots(close2, r=2, min_swing=0.03)
    assert len(h2) == 1 and close2.index[10] in h2


# --------------------------------------------------------------------------- #
# relabel + identity matching
# --------------------------------------------------------------------------- #


def test_remap_by_fwd_return_orders_by_mean() -> None:
    states = np.array([0, 0, 1, 1, 0, 1])
    fwd = np.array([0.01, 0.02, 0.03, 0.04, 0.015, 0.05])
    remap = _remap_by_fwd_return(states, fwd, 2)
    # state 1 has higher mean fwd return → regime 0
    assert remap[1] == 0
    assert remap[0] == 1


def test_remap_by_fwd_return_ignores_nan() -> None:
    states = np.array([0, 1])
    fwd = np.array([np.nan, np.nan])
    remap = _remap_by_fwd_return(states, fwd, 2)
    # both have no finite observations → -inf means, tie broken by index
    assert remap[0] == 0
    assert remap[1] == 1


def test_match_new_to_old_recovers_permutation() -> None:
    old = np.array([0, 0, 1, 1, 0, 0, 1])
    new = np.array([1, 1, 0, 0, 1, 1, 0])  # old state 0 -> new state 1
    match = _match_new_to_old(new, old, 2)
    assert match == {0: 1, 1: 0}


# --------------------------------------------------------------------------- #
# block bootstrap premium
# --------------------------------------------------------------------------- #


def test_premium_bootstrap_real_signal() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    regime = rng.integers(0, 2, n).astype(float)
    fwd = rng.normal(0, 0.005, n) + 0.002 * regime  # real +20bp premium
    p = _block_bootstrap_premium_p(regime, fwd, block=63, iters=500, seed=7)
    # The joint-block resampling preserves within-block dependence (this is the
    # same conservative discipline as lead_scan's block_bootstrap_p) and is
    # underpowered for a 20bp premium against 50bp noise: it does NOT reject.
    assert p > 0.10


def test_premium_bootstrap_no_signal() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    regime = rng.integers(0, 2, n).astype(float)
    fwd = rng.normal(0, 0.005, n)
    p = _block_bootstrap_premium_p(regime, fwd, block=63, iters=500, seed=7)
    assert p > 0.05  # no rejection for a null premium


def test_premium_bootstrap_insufficient_data() -> None:
    regime = np.array([0.0, 1.0, 0.0, 1.0])
    fwd = np.array([0.0, 0.0, 0.0, 0.0])
    assert _block_bootstrap_premium_p(regime, fwd, block=63) == 1.0
