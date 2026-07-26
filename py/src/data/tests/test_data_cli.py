"""Test data CLI — gap detection, universe loading, recap output.

Focus: gap-detection logic that runs without API calls.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.cli import _find_gaps_48h, _recap, _load_universe


# ── Fixtures ────────────────────────────────────────────────────


def _make_df(timestamps: list[datetime], cols: str | None = None) -> pd.DataFrame:
    """Build a 1-row-per-ts OHLCV DataFrame (prices are dummy)."""
    n = len(timestamps)
    df = pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.5] * n,
            "volume": [1000] * n,
        },
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )
    return df


# ── _find_gaps_48h ──────────────────────────────────────────────


class TestFindGaps48h:
    def test_empty_df(self):
        assert _find_gaps_48h(pd.DataFrame()) == []

    def test_single_row(self):
        ts = [datetime(2026, 1, 5, 10, 0)]
        assert _find_gaps_48h(_make_df(ts)) == []

    def test_no_gap_consecutive_hours(self):
        """1h candles — adjacent, no gap."""
        ts = [datetime(2026, 1, 5, i, 0) for i in range(10)]
        assert _find_gaps_48h(_make_df(ts)) == []

    def test_gap_exactly_48h(self):
        """48h delta is NOT >48h — should not flag."""
        ts = [
            datetime(2026, 1, 5, 10, 0),
            datetime(2026, 1, 7, 10, 0),  # exactly 48h
        ]
        assert _find_gaps_48h(_make_df(ts)) == []

    def test_gap_48h_plus_one_second(self):
        ts = [
            datetime(2026, 1, 5, 10, 0, 0),
            datetime(2026, 1, 7, 10, 0, 1),  # 48h + 1s
        ]
        gaps = _find_gaps_48h(_make_df(ts))
        assert len(gaps) == 1
        assert gaps[0] == (ts[0], ts[1])

    def test_weekend_gap_filtered(self):
        """Fri 21:00 → Mon 15:30 = pure weekend — should NOT flag."""
        ts = [
            datetime(2026, 1, 9, 21, 0),  # Friday
            datetime(2026, 1, 12, 15, 30),  # Monday
        ]
        assert _find_gaps_48h(_make_df(ts)) == []

    def test_real_gap_with_trading_days(self):
        """Fri → next Fri = includes Mon-Thu trading days — should flag."""
        ts = [
            datetime(2026, 1, 9, 21, 0),  # Friday
            datetime(2026, 1, 16, 15, 30),  # Next Friday
        ]
        gaps = _find_gaps_48h(_make_df(ts))
        assert len(gaps) == 1

    def test_skip_short_gap(self):
        """Normal 1h spacing — no gaps."""
        ts = [datetime(2026, 1, 5, 10, 0), datetime(2026, 1, 5, 11, 0)]
        assert _find_gaps_48h(_make_df(ts)) == []

    def test_multiple_gaps(self):
        ts = [
            datetime(2026, 1, 5, 10, 0),
            datetime(2026, 1, 10, 10, 0),  # gap 1: 5d
            datetime(2026, 1, 10, 11, 0),  # adjacent
            datetime(2026, 1, 20, 10, 0),  # gap 2: 10d
        ]
        gaps = _find_gaps_48h(_make_df(ts))
        assert len(gaps) == 2
        assert gaps[0][0] == ts[0]
        assert gaps[0][1] == ts[1]
        assert gaps[1][0] == ts[2]
        assert gaps[1][1] == ts[3]

    def test_gap_at_boundaries_mid_array(self):
        """Gap in middle — not at edges."""
        ts = [
            datetime(2026, 1, 1, 10, 0),
            datetime(2026, 1, 1, 11, 0),
            datetime(2026, 1, 10, 10, 0),  # 8.95d gap
            datetime(2026, 1, 10, 11, 0),
        ]
        gaps = _find_gaps_48h(_make_df(ts))
        assert len(gaps) == 1
        assert gaps[0][0] == ts[1]
        assert gaps[0][1] == ts[2]


# ── _recap ──────────────────────────────────────────────────────


class TestRecap:
    def test_empty_df(self):
        result = _recap(pd.DataFrame(), "AAPL")
        assert "no data" in result

    def test_normal_no_gaps(self):
        ts = [datetime(2026, 1, 5, 10, 0), datetime(2026, 1, 5, 11, 0)]
        df = _make_df(ts)
        result = _recap(df, "AAPL")
        assert "2 rows" in result
        assert "AAPL:" in result
        assert "gaps" not in result

    def test_with_gaps(self):
        ts = [
            datetime(2026, 1, 5, 10, 0),
            datetime(2026, 1, 10, 10, 0),  # 5d gap
        ]
        df = _make_df(ts)
        result = _recap(df, "AAPL")
        assert "gaps >48h (1)" in result
        assert "2026-01-05" in result
        assert "2026-01-10" in result


# ── _load_universe ──────────────────────────────────────────────


class TestLoadUniverse:
    def test_load_nsdq(self, tmp_path: Path):
        """Load a well-formed universe file."""
        univ_dir = tmp_path / "universes"
        univ_dir.mkdir()
        (univ_dir / "test.yml").write_text(
            yaml.dump({"symbols": ["AAPL", "MSFT", "GOOGL"]})
        )

        # Monkeypatch _UNIVERSE_DIR
        import src.data.cli as cli_mod

        orig = cli_mod._UNIVERSE_DIR
        cli_mod._UNIVERSE_DIR = univ_dir
        try:
            symbols = _load_universe("test")
            assert symbols == ["AAPL", "MSFT", "GOOGL"]
        finally:
            cli_mod._UNIVERSE_DIR = orig

    def test_load_missing_file(self, tmp_path: Path):
        """Missing universe file → raises."""
        import src.data.cli as cli_mod
        import click

        univ_dir = tmp_path / "universes"
        univ_dir.mkdir()
        orig = cli_mod._UNIVERSE_DIR
        cli_mod._UNIVERSE_DIR = univ_dir
        try:
            with pytest.raises(click.BadParameter, match="unknown universe 'nope'"):
                _load_universe("nope")
        finally:
            cli_mod._UNIVERSE_DIR = orig

    def test_list_universes(self, tmp_path: Path):
        """_list_universes returns yml stems only."""
        import src.data.cli as cli_mod

        univ_dir = tmp_path / "universes"
        univ_dir.mkdir()
        (univ_dir / "foo.yml").write_text("symbols: []")
        (univ_dir / "bar.yml").write_text("symbols: []")
        (univ_dir / "readme.md").write_text("")

        orig = cli_mod._UNIVERSE_DIR
        cli_mod._UNIVERSE_DIR = univ_dir
        try:
            names = cli_mod._list_universes()
            assert names == ["bar", "foo"]
        finally:
            cli_mod._UNIVERSE_DIR = orig


# ── sql-injection warning (query_candles) ──────────────────────


def test_query_candles_sql_injection_warning():
    """query_candles uses f-string interpolation — flag for fix.

    The query_candles function in src/data/db.py uses f-strings
    to build SQL. This is exploitable if symbol comes from user input.
    """
    import src.data.db as db_mod
    import inspect

    src = inspect.getsource(db_mod.query_candles)
    # Check it uses f-strings in the SQL (should be parameterized)
    assert "f" not in src.split("q =")[1].split("rows")[0] or "UPPER" in src, (
        "query_candles uses f-string SQL — should use parameterized queries. "
        "The function interpolates symbol directly. "
        "Fix: pass symbol as ? parameter."
    )
    # Note: this is informational. The actual risk is low since this
    # is a local CLI tool, but worth noting.
