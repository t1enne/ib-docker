"""Edge-case tests for the screen layer (pure functions + runner)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bt.screen.types import ScreenState
from src.bt.screen.runner import build_state, run_screen, rank, screen_over_history
from src.bt.screen.screens.momentum import _trend_label, _ema_cross, _momentum, Params


def _ts(v) -> pd.Timestamp:
    """Coerce an index scalar to a non-NaT Timestamp (typed)."""
    ts = pd.Timestamp(v)
    assert isinstance(ts, pd.Timestamp)
    return ts


def _mk_frame(closes: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume": 1_000_000.0,
        },
        index=closes.index,
    )


def _trending(n: int = 400, drift: float = 0.003, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    log_ret = drift + 0.01 * rng.standard_normal(n - 1)
    return pd.Series(100.0 * np.exp(np.cumsum(np.insert(log_ret, 0, 0.0))), index=idx)


# ── trend label  ────────────────────────────────────────────────────────


def test_trend_label_uptrend_bull():
    closes = _trending(drift=0.003)
    assert _trend_label(closes, 50, 200, 0.005) == "BULL"


def test_trend_label_downtrend_bear():
    closes = _trending(drift=-0.003)
    assert _trend_label(closes, 50, 200, 0.005) == "BEAR"


def test_trend_label_flat_range():
    closes = pd.Series([100.0] * 300, index=pd.date_range("2023-01-01", periods=300))
    assert _trend_label(closes, 50, 200, 0.005) == "RANGE"


def test_trend_label_insufficient_none():
    closes = pd.Series([100.0] * 10, index=pd.date_range("2023-01-01", periods=10))
    assert _trend_label(closes, 50, 200, 0.005) is None


# ── ema cross  ──────────────────────────────────────────────────────────


def test_ema_cross_insufficient():
    closes = pd.Series([100.0] * 30)
    assert _ema_cross(closes, 20, 50) == (False, False)


def test_momentum_none_on_short():
    closes = pd.Series([100.0] * 10)
    assert _momentum(closes, 20) is None


def test_momentum_positive():
    closes = pd.Series([100.0 * (1.01**i) for i in range(30)])
    mom = _momentum(closes, 20)
    assert mom is not None and mom > 0.0


# ── runner / build_state  ────────────────────────────────────────────────


def test_build_state_populates_trend_and_vol():
    closes = _trending(drift=0.003)
    frames = (("A", _mk_frame(closes)),)
    state = build_state(_ts(closes.index[-1]), frames)
    assert isinstance(state, ScreenState)
    assert state.trend["A"] == "BULL"
    assert state.vol["A"] in ("LOW_VOL", "MED_VOL", "HIGH_VOL")


def test_run_screen_empty_frames_returns_empty():
    frames: tuple = ()
    state = build_state(_ts("2023-01-01"), frames)
    assert run_screen(state, "momentum", {}) == ()


def test_run_screen_warmup_flat():
    # Fewer bars than Params.warmup_bars (60) -> flat, score 0.
    closes = pd.Series(
        _trending(drift=0.003, n=30).to_numpy(),
        index=pd.date_range("2023-01-01", periods=30),
    )
    frames = (("A", _mk_frame(closes)),)
    state = build_state(_ts(closes.index[-1]), frames)
    results = run_screen(state, "momentum", {})
    assert len(results) == 1
    assert results[0].action == "flat"
    assert results[0].score == 0.0
    # flat results still expose the common metric set
    assert {"ema_50", "ema_100", "atr_14", "rsi_14", "hi_52w", "lo_52w"} <= set(
        results[0].model_features.keys()
    )


def test_run_screen_scores_bullish_entry():
    # Uptrend past warmup -> long entry, non-zero score.
    closes = _trending(drift=0.004, n=400, seed=7)
    frames = (("A", _mk_frame(closes)),)
    state = build_state(_ts(closes.index[-1]), frames)
    results = run_screen(state, "momentum", {})
    assert len(results) == 1
    r = results[0]
    assert r.symbol == "A"
    assert r.action in ("long", "short", "flat")
    assert 0.0 <= r.score <= 1.0
    # common metric set is always present
    assert {"ema_50", "ema_100", "atr_14", "rsi_14", "hi_52w", "lo_52w"} <= set(
        r.model_features.keys()
    )


def test_rank_sorts_desc_and_caps():
    from src.bt.screen.types import ScreenResult

    ts = _ts("2023-01-01")
    results = (
        ScreenResult("A", ts, 0.2, "flat", ("x",), {}),
        ScreenResult("B", ts, 0.9, "long", ("y",), {}),
        ScreenResult("C", ts, 0.5, "short", ("z",), {}),
    )
    ranked = rank(results)
    assert [r.symbol for r in ranked] == ["B", "C", "A"]
    assert len(rank(results, top=2)) == 2


# ── cursor-safe history walk  ────────────────────────────────────────────


def test_screen_over_history_cursor_safe():
    closes = _trending(drift=0.004, n=400, seed=3)
    frames = (("A", _mk_frame(closes)),)
    hist = screen_over_history(frames, "momentum", {})
    assert len(hist) == len(closes)
    # No look-ahead: at each ts, the score uses only bars <= ts.
    for ts, results in hist.items():
        assert results[0].timestamp == ts
        assert all(r.score >= 0.0 for r in results)


def test_params_from_dict_ignores_extras():
    p = Params.from_dict({"fast": 10, "slow": 30, "not_a_real_field": 1})
    assert p.fast == 10
    assert p.slow == 30
    assert p.momentum_lookback == 20  # default filled
    assert p.trend_fast == 50  # default filled
