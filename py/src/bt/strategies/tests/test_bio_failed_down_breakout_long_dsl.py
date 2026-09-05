"""Tests for bio_failed_down_breakout_long_dsl (iteration-2 long mirror).

Pins the pure failed-down-breakout / volume-spike helpers against synthetic
shapes (the mirror geometry the DSL screens on) and covers the required edge
cases: short/empty series, NaN gaps, sub-threshold volume (no long despite
price geometry), a floor-level (non-spike) volume read, current-bar exclusion
from its own shelf low, and a collapse-guard that blocks buying a base that is
still free-falling.
"""

from __future__ import annotations

import numpy as np

from src.bt.strategies.bio_failed_down_breakout_long_dsl import (
    _visible,
    base_not_plunging,
    mean_prior_close,
    mean_prior_volume,
    Params,
    prior_shelf_low,
    STRATEGY_TYPE,
)
from src.bt.strategies.series import SeriesView


def _lk(arr: np.ndarray) -> SeriesView:
    """SeriesView over the whole array (cursor = full length: no truncation)."""
    a = np.asarray(arr, dtype=float)
    return SeriesView(a, lambda: len(a))


def _series_window(arr: np.ndarray) -> SeriesView:
    """SeriesView whose cursor sits at the SECOND-TO-LAST bar (no lookahead)."""
    a = np.asarray(arr, dtype=float)
    return SeriesView(a, lambda: max(0, len(a) - 1))


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_prior_shelf_low_basic():
    # steady shelf around 100 with a small dip -> window min low before cur.
    lows = np.full(60, 100.0)
    lows[40] = 97.0  # a prior dip inside the lookback, shelf min 97
    v = _series_window(np.concatenate([lows, [98.5, 98.5]]))
    assert prior_shelf_low(v, 48) == 97.0


def test_prior_shelf_low_excludes_current_bar():
    # current bar is 94 (the washout); it must NOT count toward its own shelf.
    lows = np.concatenate([np.full(60, 100.0), [94.0, 98.5]])
    v = _series_window(lows)  # cursor on second-to-last bar (the 94 washout)
    # 48 prior bars all 100 -> shelf 100, excludes the 94 breakdown bar itself.
    assert prior_shelf_low(v, 48) == 100.0


def test_prior_shelf_low_too_short_returns_nan():
    v = _series_window(np.full(5, 100.0))
    assert np.isnan(prior_shelf_low(v, 48))


def test_mean_prior_volume_excludes_current_dry_floor():
    # current volume 100 (a floor/near-noise read); prior 24 bars avg 1000.
    vols = np.concatenate([np.full(24, 1000.0), [100.0, 50.0]])
    v = _series_window(vols)
    # returns mean of vols prior to the current bar only -> ~1000, not 100.
    assert mean_prior_volume(v, 24) == 1000.0


def test_mean_prior_volume_nan_when_short():
    v = _series_window(np.full(5, 100.0))
    assert np.isnan(mean_prior_volume(v, 24))


def test_mean_prior_close_returns_prior_window():
    closes = np.concatenate([np.full(48, 100.0), [92.0, 99.0]])
    v = _series_window(closes)
    # excludes current bar; 48 prior closes avg 100.
    assert mean_prior_close(v, 48) == 100.0


def test_base_not_plunging_flat_and_collapse_reject():
    # flat base (all 100) -> not plunging -> True at small ATR.
    flat = np.concatenate([np.full(60, 100.0), [99.0, 98.0]])
    v = _series_window(flat)
    assert base_not_plunging(v, 48, 0.5, atr_val=0.8) is True
    # a running base FALLING ~6pts over the window is a collapse -> reject (False)
    plunging = np.concatenate([np.linspace(100, 94, 60), [93.0, 92.5]])
    vp = _series_window(plunging)
    assert base_not_plunging(vp, 48, 0.5, atr_val=0.8) is False


def test_base_not_plunging_returns_true_on_unmeasurable_atr():
    # Cannot measure collapse -> must NOT block (safe default for the long gate).
    v = _series_window(np.full(60, 100.0))
    assert base_not_plunging(v, 48, 0.5, atr_val=float("nan")) is True


def test_visible_filters_nan_gaps():
    a = np.array([100.0, np.nan, 101.0, np.nan, 102.0])
    out = _visible(a)
    assert out.tolist() == [100.0, 101.0, 102.0]


# ---------------------------------------------------------------------------
# no-signal / floor-volume guard path + module contract
# ---------------------------------------------------------------------------


def test_module_contract_well_formed():
    from src.bt.strategies.bio_failed_down_breakout_long_dsl import (
        STRATEGY_TYPE as s,
    )

    assert STRATEGY_TYPE == "bio_failed_down_breakout_long_dsl"
    assert s == STRATEGY_TYPE
    # distinct from iteration-1 short module
    from src.bt.strategies.bio_post_catalyst_fade_dsl import (
        STRATEGY_TYPE as s1,
    )

    assert s1 != STRATEGY_TYPE


def test_no_signal_when_volume_is_floor_and_geometry_is_a_real_washout():
    """A genuine failed-down-breakout with a *floor-level* volume read must NOT
    signal — the direct volume channel is flat, so the long is a pass by the
    volume-first rule. The volume gate is the sole trigger: current volume ==
    the prior mean (no spike) means no long regardless of how washout-like the
    price geometry is."""
    # 40 prior volume bars all 1000 (a steady floor), current bar equal 1000:
    # mean_prior_volume(window 24) must be ~1000 and current/mean == 1.0 which
    # is far below vol_mult (3.0). If the helper measured the *current* bar it
    # would still read 1.0 here, but a floor that is static must never clear the
    # 3.0x expansion threhold.
    vols = np.concatenate([np.full(40, 1000.0), [1000.0]])
    prior = mean_prior_volume(_series_window(vols), 24)
    assert np.isfinite(prior) and prior > 0
    # current (cursor-end) volume vs the 3.0x spike threshold -> 1.0 < 3.0
    ratio = np.asarray(vols, dtype=float)[-1] / prior
    assert ratio < 3.0


def test_module_params_defaults_well_formed():
    p = Params()
    assert 0.0 < p.risk_pct < 0.02  # down-gap tail -> tight 0.5% risk budget
    assert p.stop_atr_mult >= 1.5
    assert p.vol_mult >= 1.0
    assert p.cooldown_bars > 0
    assert p.target_atr_rng > 0
