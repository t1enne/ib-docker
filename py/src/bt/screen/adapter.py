"""Screen adapter — the I/O boundary of the screen layer.

Convert an external OHLCV source (e.g. the data_feed / IBKR cache) into the
``ScreenState`` shape the pure screens consume. All I/O, caching, and any
external detector invocation lives here and nowhere in the pure screen logic.

Integrity: ``data_feed.load_candles`` aborts the *whole* load when any symbol
has a gap > 96h. A screen over a wide universe must not fail because one stale/
sparse listing (e.g. a delisted or long-suspended symbol) has a hole — so the
adapter loads with the gap check disabled and **drops the gappy symbols**
instead of aborting.
"""

from __future__ import annotations

import pandas as pd

from src.bt.screen.runner import build_state
from src.bt.screen.types import ScreenState
from src.bt.data_feed import load_candles, detect_gaps
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
    Frames with no rows are skipped and symbols with data gaps are dropped so
    warmup gating stays clean and one bad symbol never aborts the screen.
    """
    return _per_symbol(_load_lenient(symbols, start, end, bar), symbols)


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
    hourly = _per_symbol(_load_lenient(symbols, start, end, "1h"), symbols)
    out: FrameMap = {}
    for iv in intervals:
        out[iv] = tuple(
            (s, resample_ohlcv(sym_df, iv)) if iv != "1h" else (s, sym_df)
            for s, sym_df in hourly
        )
    return out


def _load_lenient(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    bar: str,
) -> pd.DataFrame:
    """Load a MultiIndex feed, disabling the whole-load abort on data gaps.

    ``data_feed.load_candles`` raises :class:`DataIntegrityError` if *any*
    symbol has a gap > ``DEFAULT_MAX_GAP``. For a screen (read-only scoring, no
    indicator correctness across a hole) that is too strict: a stale symbol
    should be skipped, not sink the whole universe. We load with the check off,
    then return only the frame so the caller's ``_per_symbol`` can drop symbols
    whose ``detect_gaps`` report is non-empty.
    """
    return load_candles(symbols, start, end, bar, max_gap=pd.Timedelta.max)


def _per_symbol(
    df: pd.DataFrame,
    symbols: list[str],
    *,
    drop_gappy: bool = True,
) -> tuple[tuple[str, pd.DataFrame], ...]:
    """Split a MultiIndex-column feed into per-symbol frames.

    Symbols with no rows, no close data, or (when ``drop_gappy``) a bar gap
    larger than ``DEFAULT_MAX_GAP`` are skipped.
    """
    # Symbols with an actual bar-gap discontinuity (> DEFAULT_MAX_GAP).
    bad = set(detect_gaps(df, symbols)) if drop_gappy else set()
    out: list[tuple[str, pd.DataFrame]] = []
    for symbol in symbols:
        if symbol in bad:
            continue
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
    """Build a ready-to-score ScreenState per requested interval.

    Symbol frames that resampled to zero rows (e.g. a symbol whose source bars
    collapse after ``dropna``) are dropped so an empty frame's index never
    crashes the ``latest`` lookup.
    """
    by_iv = frames_by_interval(symbols, start, end, intervals)
    states: dict[str, ScreenState] = {}
    for iv, frames in by_iv.items():
        nonempty = [(s, f) for s, f in frames if f is not None and not f.empty]
        latest = max(
            (f["close"].index[-1] for _, f in nonempty),
            default=pd.Timestamp(end),
        )
        assert isinstance(latest, pd.Timestamp)
        states[iv] = build_state(latest, tuple(nonempty))
    return states
