"""Tests for the momentum_spy_gate screen (SPY AE-gate momentum variant)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bt.screen.screens import init_screen, resolve_screen_params
from src.bt.screen.screens import momentum as momentum_module
from src.bt.screen.screens import momentum_spy_gate
from src.bt.screen.screens.momentum import Params as MomentumParams
from src.bt.screen.screens.momentum_spy_gate import Params as SpyParams
from src.bt.screen.types import ScreenResult, ScreenState


def _ts(v: str) -> pd.Timestamp:
    ts = pd.Timestamp(v)
    assert isinstance(ts, pd.Timestamp)
    return ts


def _uprun_df(n: int = 200) -> pd.DataFrame:
    """Steady 60%-gain uptrend daily frame — long enough to pass warmup.

    Won't necessarily form a valid compression coil, but exercises the full
    scoring path (coil detection → box replay → setup/flat scoring) over many
    bars, which is what the delegation test needs.
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    closes = 100.0 * (1.0 + np.linspace(0, 0.6, n))
    opens = np.concatenate([[100.0], closes[:-1]])
    high = np.maximum(opens, closes) * 1.004
    low = np.minimum(opens, closes) * 0.996
    return pd.DataFrame(
        {
            "open": opens,
            "high": high,
            "low": low,
            "close": closes,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


def _state(symbols: list[str] = ("ABC",)) -> ScreenState:
    """ScreenState over synthetic frames, no benchmark embedded (un-gated)."""
    frames = tuple((s, _uprun_df()) for s in symbols)
    return ScreenState(
        ts=frames[0][1]["close"].index[-1],
        frames=frames,
        trend={s: "BULL" for s in symbols},
        vol={s: "MED_VOL" for s in symbols},
    )


def test_discovery_registers_spy_gate_screen():
    mod = init_screen("momentum_spy_gate")
    assert mod.SCREEN_TYPE == "momentum_spy_gate"
    assert callable(mod.on_state)
    assert getattr(mod, "Params", None) is not None


def test_resolve_params_uses_spy_gate_defaults():
    p = resolve_screen_params("momentum_spy_gate", {})
    assert isinstance(p, SpyParams)
    assert p.benchmark == "SPY"
    assert p.min_gain == 0.35
    assert p.body_atr_ratio == 0.30
    assert p.min_hover_bars == 4
    assert p.regime_min_strength == 0.12
    assert p.ae_lookback == 40
    # Regime observer alias reflects the SPY benchmark.
    assert p.regime_symbol == "SPY"


def test_resolve_params_accepts_call_overrides_and_ignores_extras():
    p = resolve_screen_params("momentum_spy_gate", {"min_gain": 0.5, "bogus": 1})
    assert p.min_gain == 0.5
    assert p.benchmark == "SPY"  # untouched default retained
    # Trading-only keys from the backtest pass are ignored (not in Params).


def test_spy_gate_is_parameter_delta_over_base_momentum():
    # Only the documentable knob/gate deltas differ; base momentum defaults
    # stay on the touched fields' base screen defaults.
    base = MomentumParams()
    spy = SpyParams()
    assert (spy.benchmark, spy.min_gain, spy.body_atr_ratio, spy.min_hover_bars)
    assert spy.regime_min_strength == 0.12
    assert spy.ae_lookback == 40
    # Fields we don't restate must match the base (no accidental shadowing).
    for field in ("big_lookback", "comp_window", "hover_tol", "decay_bars", "trend_ma"):
        assert getattr(spy, field) == getattr(base, field)


def test_on_state_delegates_identical_to_base_momentum():
    """Feeding the same state + equivalent Params yields identical results.

    The variant must be a pure re-parameterization of the base momentum screen —
    never a silent fork of the scoring math.
    """
    state = _state()
    # Same effective config both ways: SPY gate + tightened knobs.
    directly = SpyParams()
    via_base = MomentumParams(
        benchmark="SPY",
        min_gain=0.35,
        body_atr_ratio=0.30,
        min_hover_bars=4,
        regime_min_strength=0.12,
        ae_lookback=40,
    )
    spy_out = momentum_spy_gate.on_state(state, directly)
    base_out = momentum_module.on_state(state, via_base)
    assert len(spy_out) == len(base_out) == 1
    assert spy_out == base_out
    r = spy_out[0]
    assert isinstance(r, ScreenResult)
    assert r.symbol == "ABC"
    assert r.timestamp == state.ts
    assert 0.0 <= r.score <= 1.0
    assert r.action in ("long", "flat")
    assert r.signals

    # And the defaults route through init_screen + resolve_screen_params
    # identically to calling on_state directly.
    resolved = resolve_screen_params("momentum_spy_gate", {})
    assert momentum_spy_gate.on_state(state, resolved) == spy_out
