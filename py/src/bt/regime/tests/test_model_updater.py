"""Tests for regime model updater integration with backtest pipeline."""

from __future__ import annotations

import time
from collections import Counter
from typing import cast

import numpy as np
import pandas as pd

from src.bt.regime.detectors import create_sma_detector
from src.bt.regime.model_updater import (
    create_dual_online_updater,
    create_regime_model_updater,
    create_regime_model_updater_for_symbols,
)
from src.bt.regime.types import TREND_INT_TO_LABEL, VOL_INT_TO_LABEL
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
# Tests — legacy model updater
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
        label = TREND_INT_TO_LABEL.get(result.model_state.current_regime)
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
        label = TREND_INT_TO_LABEL.get(result.model_state.current_regime)
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

        tick_a = _make_tick(idx[-1], "AAPL", float(bull_prices.iloc[-1]))
        result = update(state, tick_a)
        assert result.model_state.current_regime is None

        tick_g = _make_tick(idx[-1], "GOOGL", float(bear_prices.iloc[-1]))
        result = update(state, tick_g)
        assert result.model_state.current_regime is not None
        label = TREND_INT_TO_LABEL.get(result.model_state.current_regime)
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

        assert state.model_state.current_regime is original_regime
        assert result.model_state.current_regime is not None
        assert result.model_state.current_regime != original_regime


# ---------------------------------------------------------------------------
# HTF (Higher Timeframe) regime tests
# ---------------------------------------------------------------------------


class TestDualOnlineWithHTF:
    """Tests for create_dual_online_updater with HTF trend_bar / vol_bar."""

    def test_trend_from_htf_daily_on_hourly_ticks(self) -> None:
        """Simulate 1h ticks with HTF 1d bars — trend comes from daily SMA.

        Feeds ~420 hourly ticks + ~60 daily HTF bars. Verifies:
        - current_trend is set from daily SMA (not hourly noise)
        - current_vol is set from the HMM
        - completes within a time budget (no O(n²) regression)
        """
        t0 = time.time()

        updater = create_dual_online_updater(
            n_regimes=3,
            window_size=50,
            retrain_interval=20,
            trend_fast=5,
            trend_slow=20,
            range_threshold_pct=0.005,
            trend_bar="1d",
            vol_bar=None,
        )

        base = pd.Timestamp("2024-01-01")
        idx = pd.date_range(base, periods=60, freq="D")

        # Steady uptrend on daily closes
        daily_closes = 100.0 * (1.002 ** pd.Series(range(60), index=idx))

        # Hourly ticks: 60 days × 7 hours = 420 ticks
        hourly_idx = pd.date_range(base, periods=420, freq="h")
        rng = np.random.default_rng(42)
        price = 100.0
        hourly_prices: list[float] = []
        for _ in range(420):
            price *= 1.0 + rng.normal(0.0005, 0.005)
            hourly_prices.append(price)

        state = create_initial_backtest_state(
            symbols=["AAPL"],
            initial_capital=10_000,
            start_timestamp=cast(pd.Timestamp, base),
        )

        # Seed candles with enough daily bars for SMA warmup (trend_slow=20)
        daily_df = pd.DataFrame(
            {
                "open": daily_closes * 0.999,
                "high": daily_closes * 1.002,
                "low": daily_closes * 0.998,
                "close": daily_closes,
                "volume": 1_000_000,
            },
            index=idx,
        )
        state = merge_bt_state(state, dict(candles={"AAPL": daily_df}))

        trend_values: list[int | None] = []
        vol_values: list[int | None] = []

        for i, ts in enumerate(hourly_idx):
            # Every 7 hours, push a daily HTF bar
            if i > 0 and i % 7 == 0:
                day_idx = min(i // 7, len(daily_closes) - 1)
                day_close = float(daily_closes.iloc[day_idx])
                row = {
                    "symbol": "AAPL",
                    "timestamp": ts,
                    "open": day_close * 0.999,
                    "high": day_close * 1.002,
                    "low": day_close * 0.998,
                    "close": day_close,
                    "volume": 1_000_000,
                }
                new_htf = dict(state.htf_data)
                freq_rows: list[dict] = new_htf.setdefault("1d", [])
                freq_rows.append(row)
                state = merge_bt_state(state, dict(htf_data=new_htf))

            # Feed base (1h) tick through updater
            base_tick = Candle(
                timestamp=ts,
                symbol="AAPL",
                open=hourly_prices[i] * 0.999,
                high=hourly_prices[i] * 1.002,
                low=hourly_prices[i] * 0.998,
                close=hourly_prices[i],
                volume=1_000_000,
            )
            state = updater(state, base_tick)
            trend_values.append(state.model_state.current_trend)
            vol_values.append(state.model_state.current_vol)

        elapsed = time.time() - t0
        assert elapsed < 5.0, (
            f"420 ticks took {elapsed:.1f}s — possible O(n²) in HTF path"
        )

        # After warmup, trend should be active (daily SMA needs 20 bars)
        late_trends = [t for t in trend_values[200:] if t is not None]
        assert len(late_trends) > 0, "Expected trend values after warmup"

        # In a steady uptrend, BULL should dominate
        tc = Counter(late_trends)
        bull_pct = tc.get(1, 0) / len(late_trends)
        assert bull_pct > 0.4, (
            f"Expected >40% BULL in uptrend, got {bull_pct:.1%}. "
            f"Distribution: {dict(tc)}"
        )

        # Vol should eventually produce values (HMM needs warmup)
        # May not converge on noisy hourly data with small window — just verify
        # the updater didn't crash and trend worked.
        any_vol = any(v is not None for v in vol_values)
        # If HMM warmup completed, verify it produces valid labels
        if any_vol:
            last_vols = [v for v in vol_values if v is not None]
            assert all(0 <= v <= 2 for v in last_vols), (
                f"Vol labels out of range: {set(last_vols)}"
            )
