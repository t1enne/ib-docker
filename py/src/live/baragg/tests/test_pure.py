"""Tests for pure bar aggregation functions."""

from __future__ import annotations

from typing import cast

import pandas as pd

from src.bt.state.types import Candle
from src.live.baragg.pure import baragg_tick, baragg_current, create_initial_state
from src.live.baragg.types import AggregatorState


# ── Helpers ──────────────────────────────────────────────────────────


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


def _state(symbol: str = "AAPL", interval: str = "5min") -> AggregatorState:
    return create_initial_state(symbol, interval)


# ── First tick ───────────────────────────────────────────────────────


class TestFirstTick:
    def test_partial_created_with_bucket_floor(self) -> None:
        state = baragg_tick(_state(), _tick("AAPL", "2025-01-01 10:01:23", close=150.0))
        assert state.partial is not None
        assert state.partial.open == 150.0
        assert state.partial.tick_count == 1
        assert state.bucket == _ts("2025-01-01 10:00:00")
        assert state.completed == ()


# ── Same-bucket updates ──────────────────────────────────────────────


class TestSameBucketMultiTick:
    def test_aggregates_correctly(self) -> None:
        state = _state()
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:01:23", close=100.0))
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:02:45", close=105.0))
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:03:10", close=98.0))

        assert state.partial is not None
        assert state.partial.open == 100.0  # preserved from first tick
        assert state.partial.high == 105.0
        assert state.partial.low == 98.0
        assert state.partial.close == 98.0  # latest tick
        assert state.partial.tick_count == 3
        assert state.completed == ()
        bar = baragg_current(state)
        assert bar is not None
        assert bar.tick_count == 3

    def test_volume_accumulates(self) -> None:
        state = _state()
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:01:23", volume=100.0))
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:02:45", volume=250.0))
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:03:10", volume=50.0))
        assert state.partial is not None
        assert state.partial.volume == 400.0


# ── Bucket boundary ──────────────────────────────────────────────────


class TestBucketBoundary:
    def test_emits_bar_and_starts_new(self) -> None:
        state = _state()
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:01:23", close=150.0))
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:02:45", close=155.0))
        # Cross into next bucket
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:05:01", close=160.0))

        # Completed bar
        assert len(state.completed) == 1
        bar = state.completed[0]
        assert bar.symbol == "AAPL"
        assert bar.open == 150.0
        assert bar.high == 155.0
        assert bar.low == 150.0
        assert bar.close == 155.0
        assert bar.tick_count == 2
        assert bar.interval == "5min"
        assert bar.timestamp == _ts("2025-01-01 10:00:00")

        # New partial bar started in next bucket
        assert state.partial is not None
        assert state.partial.open == 160.0
        assert state.partial.tick_count == 1
        assert state.bucket == _ts("2025-01-01 10:05:00")

    def test_volume_in_completed_bar(self) -> None:
        state = _state()
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:01:23", volume=100.0))
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:02:45", volume=200.0))
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:05:01", volume=50.0))
        assert state.completed[0].volume == 300.0


# ── Symbol filter ────────────────────────────────────────────────────


class TestSymbolMismatch:
    def test_ignores_wrong_symbol(self) -> None:
        state = _state("AAPL")
        result = baragg_tick(state, _tick("MSFT", "2025-01-01 10:01:23"))
        assert result is state  # identity check: no change
        assert result.partial is None


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_tick_at_exact_bucket_boundary(self) -> None:
        """Tick at exactly the bucket boundary goes into THAT bucket."""
        state = baragg_tick(
            _state(),
            _tick("AAPL", "2025-01-01 10:05:00.000000", close=150.0),
        )
        assert state.bucket == _ts("2025-01-01 10:05:00")
        assert state.completed == ()

    def test_multiple_ticks_same_boundary(self) -> None:
        """Many ticks in same bucket produce no completed bars."""
        state = _state()
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:00:00", close=100.0))
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:04:59", close=102.0))
        assert state.completed == ()
        assert state.partial is not None
        assert state.partial.tick_count == 2

    def test_gap_across_multiple_buckets(self) -> None:
        """Jumping 2+ intervals ahead emits only the immediate prior bucket."""
        state = _state()
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:01:00", close=100.0))
        # Jump 15 minutes (3 buckets) ahead
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:16:00", close=150.0))
        assert len(state.completed) == 1
        assert state.completed[0].timestamp == _ts("2025-01-01 10:00:00")
        assert state.bucket == _ts("2025-01-01 10:15:00")

    def test_1h_interval(self) -> None:
        """Buckets align to UTC hour boundaries."""
        state = _state(interval="1h")
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:15:00", close=100.0))
        assert state.bucket == _ts("2025-01-01 10:00:00")

        state = baragg_tick(state, _tick("AAPL", "2025-01-01 11:05:00", close=110.0))
        assert state.completed[0].timestamp == _ts("2025-01-01 10:00:00")
        assert state.bucket == _ts("2025-01-01 11:00:00")

    def test_15min_interval(self) -> None:
        """Buckets floor to multiples of 15."""
        state = _state(interval="15min")
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:17:00", close=100.0))
        assert state.bucket == _ts("2025-01-01 10:15:00")

    def test_single_tick_bar(self) -> None:
        """Bar with exactly 1 tick: open=high=low=close."""
        state = _state()
        state = baragg_tick(
            state, _tick("AAPL", "2025-01-01 10:01:23", close=150.0, volume=100.0)
        )
        state = baragg_tick(state, _tick("AAPL", "2025-01-01 10:05:05", close=151.0))
        bar = state.completed[0]
        assert bar.open == bar.high == bar.low == bar.close == 150.0
        assert bar.tick_count == 1
        assert bar.volume == 100.0

    def test_current_bar_none_before_first_tick(self) -> None:
        assert baragg_current(_state()) is None
