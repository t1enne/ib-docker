"""Tests for kalman_pairs model updater."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.kalman.model_updater import create_kalman_pairs_updater
from src.bt.state.factories import create_initial_backtest_state
from src.bt.engine.candle_store import CandleStore, CandleRows
from src.bt.engine.utils import merge_bt_state
from src.bt.state import Candle


def _make_candle_rows(
    s1: str, s2: str, n: int = 300, seed: int = 42
) -> tuple[CandleRows, pd.DatetimeIndex]:
    """Build CandleRows dict with synthetic cointegrated price data."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")

    p2 = 100 + rng.standard_normal(n).cumsum() * 0.5
    p1 = 0.5 + 0.8 * p2 + rng.normal(0, 0.3, n)

    rows: CandleRows = {}
    for sym, prices in [(s1, p1), (s2, p2)]:
        ts_ns = np.array([t.to_datetime64() for t in idx], dtype="datetime64[ns]")
        rows[(sym, "1d")] = {
            "timestamp": ts_ns,
            "open": np.array(prices * 0.999, dtype=float),
            "high": np.array(prices * 1.01, dtype=float),
            "low": np.array(prices * 0.99, dtype=float),
            "close": np.array(prices, dtype=float),
            "volume": np.ones(n, dtype=float) * 1000,
            "_len": np.array([n], dtype=int),
        }
    return rows, idx


# ── Model updater ───────────────────────────────────────────────


def test_updater_populates_kalman_fields():
    """After warmup, model_state.kalman_* fields are set."""
    rows, idx = _make_candle_rows("A", "B", n=300)
    state = create_initial_backtest_state(
        ["A", "B"], 100000.0, pd.Timestamp("2024-01-01")
    )
    store = CandleStore(rows)
    state = merge_bt_state(state, dict(candles=store))

    updater = create_kalman_pairs_updater(
        pair=("A", "B"), warmup_bars=150, ols_warmup=50, z_window=20
    )

    for i in range(len(idx)):
        ts = idx[i]
        store.advance(ts)
        for sym in ["A", "B"]:
            close = float(rows[(sym, "1d")]["close"][i])
            candle = Candle(
                timestamp=ts,
                symbol=sym,
                open=close * 0.999,
                high=close * 1.01,
                low=close * 0.99,
                close=close,
                volume=1000.0,
                interval="1d",
            )
            state = updater(state, candle)

    ms = state.model_state
    assert ms.kalman_spread is not None
    assert ms.kalman_z_score is not None
    assert ms.kalman_beta is not None
    assert ms.kalman_alpha is not None
    assert ms.kalman_n_steps > 0


def test_updater_respects_warmup():
    """Before warmup_bars, kalman_z_score stays None."""
    rows, idx = _make_candle_rows("A", "B", n=10)
    state = create_initial_backtest_state(
        ["A", "B"], 100000.0, pd.Timestamp("2024-01-01")
    )
    store = CandleStore(rows)
    state = merge_bt_state(state, dict(candles=store))

    updater = create_kalman_pairs_updater(
        pair=("A", "B"), warmup_bars=150, ols_warmup=50
    )

    for i in range(len(idx)):
        ts = idx[i]
        store.advance(ts)
        for sym in ["A", "B"]:
            close = float(rows[(sym, "1d")]["close"][i])
            candle = Candle(
                timestamp=ts,
                symbol=sym,
                open=close * 0.999,
                high=close * 1.01,
                low=close * 0.99,
                close=close,
                volume=1000.0,
                interval="1d",
            )
            state = updater(state, candle)

    assert state.model_state.kalman_z_score is None


def test_updater_deterministic():
    """Same data twice → same kalman_z_score at the end."""
    rows, idx = _make_candle_rows("A", "B", n=300)

    def run_once() -> float | None:
        state = create_initial_backtest_state(
            ["A", "B"], 100000.0, pd.Timestamp("2024-01-01")
        )
        store = CandleStore(rows)
        state = merge_bt_state(state, dict(candles=store))
        updater = create_kalman_pairs_updater(
            pair=("A", "B"), warmup_bars=150, ols_warmup=50, z_window=20
        )
        for i in range(len(idx)):
            ts = idx[i]
            store.advance(ts)
            for sym in ["A", "B"]:
                close = float(rows[(sym, "1d")]["close"][i])
                state = updater(
                    state,
                    Candle(
                        timestamp=ts,
                        symbol=sym,
                        open=close * 0.999,
                        high=close * 1.01,
                        low=close * 0.99,
                        close=close,
                        volume=1000.0,
                        interval="1d",
                    ),
                )
        return state.model_state.kalman_z_score

    a = run_once()
    b = run_once()
    assert a is not None
    assert b is not None
    assert np.isclose(a, b)


def test_updater_beta_reasonable():
    """Kalman beta should be close to the true 0.8 hedge ratio."""
    rows, idx = _make_candle_rows("A", "B", n=300)
    state = create_initial_backtest_state(
        ["A", "B"], 100000.0, pd.Timestamp("2024-01-01")
    )
    store = CandleStore(rows)
    state = merge_bt_state(state, dict(candles=store))

    updater = create_kalman_pairs_updater(
        pair=("A", "B"), warmup_bars=150, ols_warmup=50, z_window=20
    )

    for i in range(len(idx)):
        ts = idx[i]
        store.advance(ts)
        for sym in ["A", "B"]:
            close = float(rows[(sym, "1d")]["close"][i])
            state = updater(
                state,
                Candle(
                    timestamp=ts,
                    symbol=sym,
                    open=close * 0.999,
                    high=close * 1.01,
                    low=close * 0.99,
                    close=close,
                    volume=1000.0,
                    interval="1d",
                ),
            )

    beta = state.model_state.kalman_beta
    assert beta is not None
    assert 0.5 < beta < 1.2  # true beta = 0.8


def test_updater_zscore_is_tradable():
    """Rolling z-score of spread should be in tradable ±range (not tiny t-stat)."""
    rows, idx = _make_candle_rows("A", "B", n=300, seed=99)
    state = create_initial_backtest_state(
        ["A", "B"], 100000.0, pd.Timestamp("2024-01-01")
    )
    store = CandleStore(rows)
    state = merge_bt_state(state, dict(candles=store))

    updater = create_kalman_pairs_updater(
        pair=("A", "B"), warmup_bars=150, ols_warmup=50, z_window=20
    )

    z_vals: list[float] = []
    for i in range(len(idx)):
        ts = idx[i]
        store.advance(ts)
        for sym in ["A", "B"]:
            close = float(rows[(sym, "1d")]["close"][i])
            state = updater(
                state,
                Candle(
                    timestamp=ts,
                    symbol=sym,
                    open=close * 0.999,
                    high=close * 1.01,
                    low=close * 0.99,
                    close=close,
                    volume=1000.0,
                    interval="1d",
                ),
            )
        z = state.model_state.kalman_z_score
        if z is not None and abs(z) < 1e-10:
            continue
        if z is not None:
            z_vals.append(z)

    assert len(z_vals) > 0
    # Rolling z-score should be tradable magnitude, not raw t-stat (~0.005).
    # Synthetic cointegrated data is tight; real pairs produce > ±1 regularly.
    max_abs = max(abs(v) for v in z_vals)
    assert max_abs > 0.5, f"max |z| = {max_abs:.4f}, expected > 0.5 (tradable, not tiny t-stat)"
