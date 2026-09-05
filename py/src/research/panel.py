"""Daily cross-sectional panel construction over the local candle DB.

Separates pure window selection (`select_full_history`) from the DB I/O edge
(`load_daily_panel`) so the cross-sectional math is unit-testable without a
live database.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import cast

import pandas as pd

from src.data.db import query_candles
from src.research.types import PanelInfo

# Retained members must have started within this many days of the earliest
# founder, so a late-added / partial-sync name cannot truncate the window.
_FOUNDER_TOLERANCE_DAYS = 120


@dataclass(frozen=True)
class MemberSpan:
    """Coverage span of one universe member in the local DB."""

    ticker: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_rows: int


def select_full_history(
    spans: dict[str, MemberSpan], min_rows: int = 60
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Keep the founding cohort of full-history members, drop late stragglers.

    A ``full-history`` member is one whose coverage starts within
    ``_FOUNDER_TOLERANCE_DAYS`` (=120) of the earliest-starting member, after
    the ``min_rows`` floor. A much-later start (recent IPO or partial sync)
    would truncate the rectangular window of every other member, so such
    names are dropped instead.

    Args:
        spans: per-ticker coverage spans (from the DB daily load).
        min_rows: below this many rows a member can never be full-history.

    Returns:
        (kept, dropped) ticker tuples. ``kept`` share one start anchor.
    """
    qualified = [t for t, s in spans.items() if s.n_rows >= min_rows]
    if not qualified:
        return (), tuple(sorted(spans))
    if len(qualified) == 1:
        return (tuple(qualified), tuple(sorted(t for t in spans if t != qualified[0])))

    earliest = min(spans[t].start for t in qualified)
    anchor_end = earliest + pd.Timedelta(days=_FOUNDER_TOLERANCE_DAYS)
    kept = [t for t in qualified if spans[t].start <= anchor_end]
    if len(kept) < 2:
        # Degenerate input: anchor on the single longest member instead.
        longest = max(qualified, key=lambda t: (spans[t].end - spans[t].start).days)
        kept = [longest]
    kept_set = set(kept)
    dropped = [t for t in spans if t not in kept_set]
    return tuple(sorted(kept)), tuple(sorted(dropped))


def _daily(ticker: str) -> pd.DataFrame:
    """One ticker's full daily OHLCV frame from the local candle DB."""
    return query_candles(ticker.upper(), bar="1d")


def load_daily_panel(
    universe_symbols: list[str],
    bench: str,
    *,
    from_date: str | None = None,
    to_date: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, PanelInfo]:
    """Edge loader: daily closes for full-history members + the benchmark.

    The benchmark ticker is never a panel member. Returns
    (member_close_frame, bench_close_series, info).

    Raises:
        ValueError: if no full-history member remains, or the benchmark has no
            overlapping daily rows in the chosen window.
    """
    bench = bench.upper()
    members = [s.upper() for s in universe_symbols if s.upper() != bench]
    if not members:
        raise ValueError("universe has no symbols other than the benchmark")

    spans: dict[str, MemberSpan] = {}
    for ticker in members:
        df = _daily(ticker)
        if not df.empty:
            spans[ticker] = MemberSpan(
                ticker, df.index.min(), df.index.max(), int(len(df))
            )
    kept, dropped = select_full_history(spans)
    if not kept:
        raise ValueError(f"no full-history members with daily data (bench={bench})")

    lo = max(spans[t].start for t in kept)
    hi = min(spans[t].end for t in kept)
    lo, hi = _narrow_window(lo, hi, from_date, to_date)
    if lo > hi:
        raise ValueError("full-history members have disjoint or empty windows")

    bench_raw = _daily(bench)
    bench_sel = bench_raw.loc[lo:hi]
    if bench_sel.empty:
        raise ValueError(f"benchmark {bench} has no daily data in the panel window")

    closes: dict[str, pd.Series] = {}
    for ticker in kept:
        frame = _daily(ticker).loc[lo:hi]
        if not frame.empty:
            closes[ticker] = frame["close"]
    close_frame = pd.DataFrame(closes).sort_index()
    if close_frame.empty:
        raise ValueError("no retained member produced overlapping daily closes")

    info = PanelInfo(
        members=tuple(sorted(close_frame.columns)),
        bench=bench,
        n_members=len(close_frame.columns),
        n_common_rows=int(len(close_frame)),
        common_start=str(close_frame.index.min().date()),
        common_end=str(close_frame.index.max().date()),
        dropped=dropped,
    )
    return close_frame, bench_sel["close"], info


def _narrow_window(
    lo: pd.Timestamp,
    hi: pd.Timestamp,
    from_date: str | None,
    to_date: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Narrow the panel to an optional user date range (inclusive bounds)."""
    if from_date:
        cand = _as_ts(from_date)
        if cand > lo:
            lo = cand
    if to_date:
        cand = _as_ts(to_date)
        if cand < hi:
            hi = cand
    return lo, hi


def _as_ts(date_str: str) -> pd.Timestamp:
    """Parse a validated YYYY-MM-DD string into a Timestamp (no NaT)."""
    year, month, day = (int(part) for part in date_str.split("-"))
    return cast(pd.Timestamp, pd.Timestamp(datetime.date(year, month, day)))
