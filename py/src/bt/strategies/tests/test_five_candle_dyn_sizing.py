"""Tests for dynamic confidence sizing in five_candle_bias_displacement."""

from __future__ import annotations

from dataclasses import replace

from src.bt.strategies.five_candle_bias_displacement_dsl import Params, dyn_risk_mult


def _p(**over: object) -> Params:
    base = Params(dyn_sizing=True, lookback=5, bias_threshold=4)
    for k, v in over.items():
        base = replace(base, **{k: v})
    return base


def test_disabled_returns_flat_1():
    assert dyn_risk_mult(Params(), aligned=5, cf_body=4.0, pb_body=1.0) == 1.0


def test_neutral_pass_keeps_base():
    # aligned 4 (< lookback 5) and confirm below decisive floor 3.0 -> flat
    p = _p()
    assert dyn_risk_mult(p, aligned=4, cf_body=2.5, pb_body=1.0) == 1.0


def test_full_alignment_boosts():
    p = _p(full_alignment_boost=1.5)
    assert dyn_risk_mult(p, aligned=5, cf_body=2.5, pb_body=1.0) == 1.5


def test_decisive_confirm_boosts():
    p = _p(decisive_confirm_boost=2.0, decisive_confirm_floor=3.0)
    assert dyn_risk_mult(p, aligned=4, cf_body=4.0, pb_body=1.0) == 2.0


def test_signals_compose_multiplicatively():
    p = _p(full_alignment_boost=1.5, decisive_confirm_boost=2.0)
    assert dyn_risk_mult(p, aligned=5, cf_body=4.0, pb_body=1.0) == 3.0


def test_boost_clamped_at_max():
    p = _p(full_alignment_boost=2.0, decisive_confirm_boost=2.0, max_dyn_risk_mult=2.5)
    assert dyn_risk_mult(p, aligned=5, cf_body=4.0, pb_body=1.0) == 2.5


def test_boost_never_below_floor():
    p = _p(min_dyn_risk_mult=1.0)
    assert dyn_risk_mult(p, aligned=3, cf_body=1.0, pb_body=5.0) == 1.0


def test_decisive_floor_boundary_uses_ge():
    p = _p(decisive_confirm_boost=1.5, decisive_confirm_floor=3.0)
    assert dyn_risk_mult(p, aligned=4, cf_body=3.0, pb_body=1.0) == 1.5
