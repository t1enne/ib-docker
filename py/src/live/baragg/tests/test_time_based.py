"""Tests for TimeBarAggregator wrapper class."""

from __future__ import annotations

from typing import cast

import pandas as pd

from src.bt.state.types import Candle
from src.live.baragg.time_based import TimeBarAggregator


def _ts(val: str) -> pd.Timestamp:
    result = cast(pd.Timestamp, pd.Timestamp(val))
    assert not pd.isna(result)
    return result


def _tick(symbol: str, ts: str, *, close: float = 100.0, volume: float = 0.0) -> Candle:
    return Candle(
        timestamp=_ts(ts),
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
    )


class TestTimeBarAggregator:
    def test_returns_completed_bar_on_bucket_roll(self) -> None:
        agg = TimeBarAggregator(interval="5min", symbol="AAPL")
        agg.update(_tick("AAPL", "2025-01-01 10:01:23", close=150.0))
        agg.update(_tick("AAPL", "2025-01-01 10:02:45", close=155.0))
        bars = agg.update(_tick("AAPL", "2025-01-01 10:05:01", close=160.0))
        assert len(bars) == 1
        assert bars[0].open == 150.0
        assert bars[0].high == 155.0
        assert bars[0].close == 155.0

    def test_bars_not_accumulated_across_calls(self) -> None:
        """A bar returned on update() is not returned again."""
        agg = TimeBarAggregator(interval="5min", symbol="AAPL")
        agg.update(_tick("AAPL", "2025-01-01 10:01:23", close=150.0))
        agg.update(_tick("AAPL", "2025-01-01 10:05:01", close=160.0))
        bars = agg.update(_tick("AAPL", "2025-01-01 10:06:00", close=161.0))
        assert bars == ()

    def test_current_bar_after_tick(self) -> None:
        agg = TimeBarAggregator(interval="5min", symbol="AAPL")
        agg.update(_tick("AAPL", "2025-01-01 10:01:23", close=150.0))
        assert agg.current_bar is not None
        assert agg.current_bar.close == 150.0

    def test_reset_clears_state(self) -> None:
        agg = TimeBarAggregator(interval="5min", symbol="AAPL")
        agg.update(_tick("AAPL", "2025-01-01 10:01:23", close=150.0))
        assert agg.current_bar is not None
        agg.reset()
        assert agg.current_bar is None

    def test_multiple_bars_in_sequence(self) -> None:
        agg = TimeBarAggregator(interval="5min", symbol="AAPL")

        agg.update(_tick("AAPL", "2025-01-01 10:01:00", close=100.0))
        agg.update(_tick("AAPL", "2025-01-01 10:03:00", close=102.0))
        bars = agg.update(_tick("AAPL", "2025-01-01 10:05:30", close=103.0))
        assert len(bars) == 1 and bars[0].tick_count == 2

        agg.update(_tick("AAPL", "2025-01-01 10:07:00", close=104.0))
        bars = agg.update(_tick("AAPL", "2025-01-01 10:10:15", close=105.0))
        assert len(bars) == 1 and bars[0].tick_count == 2

        agg.update(_tick("AAPL", "2025-01-01 10:12:00", close=106.0))
        agg.update(_tick("AAPL", "2025-01-01 10:13:00", close=107.0))
        agg.update(_tick("AAPL", "2025-01-01 10:14:00", close=108.0))
        bars = agg.update(_tick("AAPL", "2025-01-01 10:15:01", close=109.0))
        assert len(bars) == 1 and bars[0].tick_count == 4
