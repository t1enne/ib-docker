"""Tests for regime model updater integration with backtest pipeline."""

from __future__ import annotations

import pandas as pd

from src.bt.regime.detectors import create_sma_detector
from src.bt.regime.model_updater import (
    create_regime_model_updater,
    create_regime_model_updater_for_symbols,
)
from src.bt.regime.types import REGIME_INT_TO_LABEL
from typing import cast

from src.bt.state import (
    Candle,
    create_initial_backtest_state,
)
from src.bt.engine.utils import merge_bt_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candles_df(prices: pd.Series) -> pd.DataFrame:
    """Make a candles-style DataFrame from a price series."""
    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices,
            "volume": 1_000_000,
        },
        index=prices.index,
    )


def _make_tick(timestamp: pd.Timestamp, symbol: str, price: float) -> Candle:
    return Candle(
        timestamp=timestamp,
        symbol=symbol,
        open=price * 0.999,
        high=price * 1.002,
        low=price * 0.998,
        close=price,
        volume=1_000_000,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegimeModelUpdater:
    def test_single_symbol_updates_regime(self) -> None:
        """A bull market should produce BULL regime in model_state."""
        idx = pd.date_range("2024-01-01", periods=300, freq="D")
        rng = pd.Series(100.0 * (1.0005 ** pd.Series(range(300), index=idx)))
        candles_df = _make_candles_df(rng)

        state = create_initial_backtest_state(
            symbols=["AAPL"],
            initial_capital=10_000,
            start_timestamp=idx[0],
        )
        state = merge_bt_state(state, dict(candles={"AAPL": candles_df}))

        detector = create_sma_detector(fast_window=20, slow_window=50)
        update = create_regime_model_updater(detector)

        tick = _make_tick(idx[-1], "AAPL", float(rng.iloc[-1]))
        result = update(state, tick)

        assert result.model_state.current_regime is not None
        label = REGIME_INT_TO_LABEL.get(result.model_state.current_regime)
        assert label == "BULL", f"Expected BULL, got {label}"

    def test_bear_market_sets_bear_regime(self) -> None:
        """A bear market should produce BEAR."""
        idx = pd.date_range("2024-01-01", periods=300, freq="D")
        rng = pd.Series(100.0 * (0.999 ** pd.Series(range(300), index=idx)))
        candles_df = _make_candles_df(rng)

        state = create_initial_backtest_state(
            symbols=["AAPL"],
            initial_capital=10_000,
            start_timestamp=idx[0],
        )
        state = merge_bt_state(state, dict(candles={"AAPL": candles_df}))

        detector = create_sma_detector(fast_window=20, slow_window=50)
        update = create_regime_model_updater(detector)

        tick = _make_tick(idx[-1], "AAPL", float(rng.iloc[-1]))
        result = update(state, tick)

        assert result.model_state.current_regime is not None
        label = REGIME_INT_TO_LABEL.get(result.model_state.current_regime)
        assert label == "BEAR", f"Expected BEAR, got {label}"

    def test_empty_candles_no_update(self) -> None:
        """No candles → current_regime stays None."""
        ts = cast(pd.Timestamp, pd.Timestamp("2024-01-01"))
        state = create_initial_backtest_state(
            symbols=["AAPL"],
            initial_capital=10_000,
            start_timestamp=ts,
        )
        detector = create_sma_detector()
        update = create_regime_model_updater(detector)

        tick = _make_tick(ts, "AAPL", 100.0)
        result = update(state, tick)

        assert result.model_state.current_regime is None

    def test_short_candles_no_update(self) -> None:
        """Less than 20 bars → no regime set."""
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        rng = pd.Series([100.0] * 10, index=idx)
        candles_df = _make_candles_df(rng)

        state = create_initial_backtest_state(
            symbols=["AAPL"],
            initial_capital=10_000,
            start_timestamp=idx[0],
        )
        state = merge_bt_state(state, dict(candles={"AAPL": candles_df}))

        detector = create_sma_detector(slow_window=50)
        update = create_regime_model_updater(detector)

        tick = _make_tick(idx[-1], "AAPL", 100.0)
        result = update(state, tick)

        assert result.model_state.current_regime is None

    def test_multi_symbol_updates_on_last_symbol(self) -> None:
        """Multi-symbol: regime updated only when last symbol ticks."""
        idx = pd.date_range("2024-01-01", periods=300, freq="D")
        bull_prices = pd.Series(100.0 * (1.0005 ** pd.Series(range(300), index=idx)))
        bear_prices = pd.Series(100.0 * (0.999 ** pd.Series(range(300), index=idx)))

        state = create_initial_backtest_state(
            symbols=["AAPL", "GOOGL"],
            initial_capital=10_000,
            start_timestamp=idx[0],
        )
        state = merge_bt_state(
            state,
            dict(
                candles={
                    "AAPL": _make_candles_df(bull_prices),
                    "GOOGL": _make_candles_df(bear_prices),
                }
            ),
        )

        detector = create_sma_detector(fast_window=20, slow_window=50)
        update = create_regime_model_updater_for_symbols(detector, ["AAPL", "GOOGL"])

        # Tick on AAPL (not last symbol) — should NOT update
        tick_a = _make_tick(idx[-1], "AAPL", float(bull_prices.iloc[-1]))
        result = update(state, tick_a)
        assert result.model_state.current_regime is None

        # Tick on GOOGL (last symbol) — should update
        tick_g = _make_tick(idx[-1], "GOOGL", float(bear_prices.iloc[-1]))
        result = update(state, tick_g)
        assert result.model_state.current_regime is not None
        # Last symbol is GOOGL = bear → BEAR
        label = REGIME_INT_TO_LABEL.get(result.model_state.current_regime)
        assert label == "BEAR", f"Expected BEAR from GOOGL, got {label}"

    def test_regime_is_immutable_preserved(self) -> None:
        """Verify original state is not mutated."""
        idx = pd.date_range("2024-01-01", periods=300, freq="D")
        rng = pd.Series(100.0 * (1.0005 ** pd.Series(range(300), index=idx)))
        state = create_initial_backtest_state(
            symbols=["AAPL"],
            initial_capital=10_000,
            start_timestamp=idx[0],
        )
        state = merge_bt_state(state, dict(candles={"AAPL": _make_candles_df(rng)}))

        original_regime = state.model_state.current_regime
        detector = create_sma_detector(fast_window=20, slow_window=50)
        update = create_regime_model_updater(detector)

        tick = _make_tick(idx[-1], "AAPL", float(rng.iloc[-1]))
        result = update(state, tick)

        # Original unchanged
        assert state.model_state.current_regime is original_regime
        # Result has new value
        assert result.model_state.current_regime is not None
        assert result.model_state.current_regime != original_regime
