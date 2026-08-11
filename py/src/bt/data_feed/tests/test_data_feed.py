"""Tests for the data feed gap-integrity guard.

``load_candles`` must abort (raise) when a loaded symbol contains a
discontinuity > 48h between consecutive bars — otherwise rolling/indicator
math treats the two sides of the hole as adjacent and silently corrupts the
backtest. ``detect_gaps`` is the pure detector behind that guard.
"""

from __future__ import annotations

import pandas as pd
import pytest

import src.bt.data_feed as feed
from src.bt.data_feed import (
    DataIntegrityError,
    GapBreak,
    detect_gaps,
    load_candles,
)


def _ts(v: str) -> pd.Timestamp:
    ts = pd.Timestamp(v)
    assert isinstance(ts, pd.Timestamp)
    return ts


def _frame(symbols: list[str], template):
    """Drop the per-symbol rows in ``template`` into a MultiIndex-column frame."""
    dfs = {}
    for s in symbols:
        rows = template(s)
        dfs[s] = pd.DataFrame(
            {
                "open": [1.0] * len(rows),
                "high": [1.0] * len(rows),
                "low": [1.0] * len(rows),
                "close": [1.0] * len(rows),
                "volume": [1.0] * len(rows),
            },
            index=pd.DatetimeIndex(rows),
        )
    return pd.concat(list(dfs.values()), axis=1, keys=dfs.keys(), sort=False)


def _hourly(*stamps: str) -> list[str]:
    return list(pd.Series(stamps).astype("datetime64[ns]"))


def _two_bars(a: str, b: str) -> list[str]:
    return _hourly(a, b)


# ── detect_gaps ────────────────────────────────────────────────────────
def test_detect_gaps_clean_series():
    def template(_s: str) -> list[str]:
        return _hourly("2024-01-01 00:00", "2024-01-01 23:00", "2024-01-02 12:00")

    df = _frame(["A"], template)
    report = detect_gaps(df, ["A"])
    assert report == {}


def test_detect_gaps_reports_gap_over_threshold():
    # 72h between bars 2 and 3.
    def template(_s: str) -> list[str]:
        return _hourly("2024-01-01 00:00", "2024-01-02 00:00", "2024-01-05 00:00")

    df = _frame(["A"], template)
    report = detect_gaps(df, ["A"])
    assert "A" in report
    breaks = report["A"]
    assert len(breaks) == 1
    b = breaks[0]
    assert isinstance(b, GapBreak)
    assert b.prev_ts == pd.Timestamp("2024-01-02 00:00")
    assert b.next_ts == pd.Timestamp("2024-01-05 00:00")
    assert b.duration == pd.Timedelta(hours=72)


def test_detect_gaps_ignores_gap_within_threshold():
    # 24h gap is fine (e.g. missing a trading day).
    def template(_s: str) -> list[str]:
        return _two_bars("2024-01-01 00:00", "2024-01-02 00:00")

    df = _frame(["A"], template)
    assert detect_gaps(df, ["A"]) == {}


def test_detect_gaps_skips_missing_symbol():
    def template(_s: str) -> list[str]:
        return _hourly("2024-01-01 00:00", "2024-01-05 00:00")

    df = _frame(["A"], template)
    assert detect_gaps(df, ["B"]) == {}


def test_detect_gaps_single_bar_and_empty():
    assert (
        detect_gaps(_frame(["A"], lambda s: _hourly("2024-01-01 00:00")), ["A"]) == {}
    )
    assert detect_gaps(_frame(["A"], lambda s: _hourly()), ["A"]) == {}


def test_detect_gaps_per_symbol_isolation():
    def template(s: str) -> list[str]:
        if s == "A":
            return _hourly("2024-01-01 00:00", "2024-01-05 00:00")
        return _two_bars("2024-01-01 00:00", "2024-01-02 00:00")

    df = _frame(["A", "B"], template)
    report = detect_gaps(df, ["A", "B"]).get("A", ())
    assert len(report) == 1
    assert detect_gaps(df, ["B"]) == {}


# ── load_candles guard ─────────────────────────────────────────────────
def _load_df(*, hourly_spans: dict):
    def factory(symbol):
        span = hourly_spans[symbol]
        return _hourly(*span)

    # get_local_candles yields a plain OHLCV frame (single-symbol columns).
    # xs(symbol, axis=1) drops the (symbol, field) MultiIndex level 0.
    def fake_get_local_candles(symbol, start=None, end=None, bar="1h"):
        return _frame([symbol], factory).xs(symbol, axis=1)

    return fake_get_local_candles


def test_load_candles_raises_on_gap(monkeypatch):
    monkeypatch.setattr(
        feed,
        "get_local_candles",
        _load_df(
            hourly_spans={
                "A": ("2024-01-01 00:00", "2024-01-02 00:00"),
                "B": ("2024-01-01 00:00", "2024-01-04 00:00"),  # 72h gap
            }
        ),
    )
    with pytest.raises(DataIntegrityError) as exc:
        load_candles(["A", "B"], _ts("2024-01-01"), _ts("2024-01-05"), "1h")
    assert "B" in str(exc.value)
    assert "gap" in str(exc.value)


def test_load_candles_passes_when_clean(monkeypatch):
    monkeypatch.setattr(
        feed,
        "get_local_candles",
        _load_df(
            hourly_spans={
                "A": ("2024-01-01 00:00", "2024-01-02 00:00"),
                "B": ("2024-01-01 00:00", "2024-01-01 23:00"),
            }
        ),
    )
    df = load_candles(["A", "B"], _ts("2024-01-01"), _ts("2024-01-02"), "1h")
    assert set(df.columns.get_level_values(0)) == {"A", "B"}


def test_load_candles_max_gap_disables_guard(monkeypatch):
    monkeypatch.setattr(
        feed,
        "get_local_candles",
        _load_df(
            hourly_spans={
                "A": ("2024-01-01 00:00", "2024-01-05 00:00"),  # 96h gap
            }
        ),
    )
    # Large sentinel skips the check entirely.
    df = load_candles(
        ["A"],
        _ts("2024-01-01"),
        _ts("2024-01-06"),
        "1h",
        max_gap=pd.Timedelta.max,
    )
    assert len(df["A"].index) == 2
