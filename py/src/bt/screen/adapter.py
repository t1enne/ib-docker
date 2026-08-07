"""Screen adapter — the I/O boundary of the screen layer.

Convert an external OHLCV source (e.g. the data_feed / IBKR cache) into the
``ScreenState`` shape the pure screens consume. All I/O, caching, and any
external detector invocation lives here and nowhere in the pure screen logic.
"""

from __future__ import annotations

import pandas as pd

from src.bt.screen.runner import build_state
from src.bt.screen.types import ScreenState
from src.bt.data_feed import load_candles
from src.data.resample import resample_ohlcv

FrameMap = dict[str, tuple[tuple[str, pd.DataFrame], ...]]


def frames_from_feed(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    bar: str,
) -> tuple[tuple[str, pd.DataFrame], ...]:
    """Load OHLCV frames for a symbol list/date range.

    Returns a ``(symbol, DataFrame)`` tuple aligned to ``ScreenState.frames``.
    Frames with no rows are skipped so warmup gating stays clean.
    """
    return _per_symbol(load_candles(symbols, start, end, bar), symbols)


def frames_by_interval(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    intervals: list[str],
) -> FrameMap:
    """Load hourly OHLCV and resample it to every requested interval.

    Source data is hourly; each ``interval`` (e.g. "1h", "4h", "1d") is
    produced by resampling the same hourly frame. Returns a map keyed by
    interval string of ``(symbol, DataFrame)`` tuples.
    """
    hourly = _per_symbol(load_candles(symbols, start, end, "1h"), symbols)
    out: FrameMap = {}
    for iv in intervals:
        out[iv] = tuple(
            (s, resample_ohlcv(sym_df, iv)) if iv != "1h" else (s, sym_df)
            for s, sym_df in hourly
        )
    return out


def _per_symbol(
    df: pd.DataFrame, symbols: list[str]
) -> tuple[tuple[str, pd.DataFrame], ...]:
    """Split a MultiIndex-column feed into per-symbol frames."""
    out: list[tuple[str, pd.DataFrame]] = []
    for symbol in symbols:
        try:
            sym_df = df.xs(symbol, axis=1)
        except KeyError:
            continue
        if sym_df.empty or "close" not in sym_df.columns:
            continue
        out.append((symbol, sym_df))
    return tuple(out)


def state_from_feed(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    bar: str,
) -> ScreenState:
    """Load latest OHLCV and build a ready-to-score ScreenState.

    The latest timestamp across every symbol's frame is used as ``ts``, so a
    screen called on the returned state scores the most recent bar.
    """
    frames = frames_from_feed(symbols, start, end, bar)
    latest = max(
        (f["close"].index[-1] for _, f in frames),
        default=pd.Timestamp(end),
    )
    assert isinstance(latest, pd.Timestamp)
    return build_state(latest, frames)


def state_per_interval(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    intervals: list[str],
) -> dict[str, ScreenState]:
    """Build a ready-to-score ScreenState per requested interval."""
    by_iv = frames_by_interval(symbols, start, end, intervals)
    states: dict[str, ScreenState] = {}
    for iv, frames in by_iv.items():
        latest = max(
            (f["close"].index[-1] for _, f in frames),
            default=pd.Timestamp(end),
        )
        assert isinstance(latest, pd.Timestamp)
        states[iv] = build_state(latest, frames)
    return states
