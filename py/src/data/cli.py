"""Data CLI — download, query, and preview market data.

Usage:
    py data dl AAPL MSFT --from 2026-01-01
    py data dl --universe universes/nsdq.json --from 2026-01-01
    py data query AAPL --from 2026-01-01 --to 2026-06-01
    py data preview AAPL MSFT --from 2026-01-01
    py data preview --universe universes/sector.json --from 2024-01-01
"""

from __future__ import annotations
from src.data import load_universe_config

import asyncio
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from src.utils import to_optional_ts
from src.data.db import query_candles
from src.data.types import PreviewResult
from src.data.xcal import is_non_trading_day

# ── Gap detection helpers ──────────────────────────────────────


def _all_non_trading(gap_start: datetime, gap_end: datetime) -> bool:
    """Check if every date in (gap_start, gap_end) is a non-trading day."""
    d = gap_start.date() + timedelta(days=1)
    end = gap_end.date()
    while d < end:
        dt = datetime.combine(d, datetime.min.time())
        if not is_non_trading_day(dt):
            return False
        d += timedelta(days=1)
    return True


def _find_gaps_48h(df: pd.DataFrame) -> list[tuple[datetime, datetime]]:
    """Find gaps >48h between consecutive candles in a DataFrame.

    Filters out gaps that consist entirely of non-trading days
    (weekends + NYSE holidays).

    Args:
        df: DataFrame with DatetimeIndex, sorted chronologically.

    Returns:
        List of (gap_start, gap_end) tuples for real data gaps.
    """
    if df.empty or len(df) < 2:
        return []

    gaps: list[tuple[datetime, datetime]] = []
    idx = df.index
    for i in range(len(idx) - 1):
        delta = idx[i + 1] - idx[i]
        if delta > timedelta(hours=48):
            gap_start = idx[i].to_pydatetime()
            gap_end = idx[i + 1].to_pydatetime()
            if not _all_non_trading(gap_start, gap_end):
                gaps.append((gap_start, gap_end))
    return gaps


# ── Recap output ────────────────────────────────────────────────


def _recap(df: pd.DataFrame, label: str) -> str:
    """Build a recap string for a symbol's data.

    Shows: start, end, total rows, and any gaps >48h.
    """
    if df.empty:
        return f"{label}: no data"

    gaps = _find_gaps_48h(df)
    gap_lines = ""
    if gaps:
        pieces = "\n  ".join(
            f"{g[0].strftime('%Y-%m-%d %H:%M')} → {g[1].strftime('%Y-%m-%d %H:%M')}"
            for g in gaps
        )
        gap_lines = f"\n  gaps >48h ({len(gaps)}):\n  {pieces}"

    return (
        f"{label}: {len(df)} rows"
        f"  {df.index[0].strftime('%Y-%m-%d %H:%M')}"
        f" → {df.index[-1].strftime('%Y-%m-%d %H:%M')}"
        f"{gap_lines}"
    )


def _print_recap(
    symbols: tuple[str, ...],
    from_date: str,
    to_date: Optional[str],
    bar: str = "1h",
) -> None:
    """Query each symbol from DB and print its recap to stderr."""
    for sym in symbols:
        start_ts = to_optional_ts(from_date)
        end_ts = to_optional_ts(to_date)
        df = query_candles(sym.upper(), start_ts, end_ts, bar)
        click.echo(_recap(df, sym.upper()), err=True)


# ── Stdin reader ────────────────────────────────────────────────


def read_ohlcv_stdin() -> pd.DataFrame:
    """Read OHLCV JSON lines from stdin and return a DataFrame.

    Expects: {"t":"2026-01-05T...","o":100.5,"h":...,"l":...,"c":...,"v":...}
    """
    if sys.stdin.isatty():
        return pd.DataFrame()

    records: list[dict] = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            click.echo(f"Warning: skipping invalid JSON line: {line[:80]}", err=True)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["t"])
    df = df.set_index("timestamp").drop(columns=["t"])

    renames = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns=renames)
    return df


# ── CLI group ──────────────────────────────────────────────────


@click.group(name="data")
def data_group():
    """Market data download and query."""


@data_group.command(name="query")
@click.argument("symbols", nargs=-1, required=False)
@click.option(
    "--universe",
    "-U",
    help="Universe file name (e.g. 'nsdq', 'sector'). Overrides positional SYMBOLS.",
)
@click.option("--from", "-f", "from_date", help="Start date (YYYY-MM-DD)")
@click.option("--to", "-t", "to_date", help="End date (YYYY-MM-DD)")
@click.option("--bar", default="1h", help="Bar size (1h, 1d, etc.)")
def query_cmd(
    symbols: tuple[str, ...],
    universe: Optional[str],
    from_date: Optional[str],
    to_date: Optional[str],
    bar: str,
):
    """Query OHLCV candles for SYMBOL from the local database.

    Shows a recap with date range, row count, and gaps >48h.
    """
    start_ts = to_optional_ts(from_date)
    end_ts = to_optional_ts(to_date)
    symbols_list = _resolve_symbol_list(symbols, universe)
    for symbol in symbols_list:
        df = query_candles(symbol.upper(), start_ts, end_ts, bar)
        click.echo(_recap(df, symbol.upper()), err=True)


# ── Universe file helpers ────────────────────────────────────

_UNIVERSE_DIR = Path(__file__).resolve().parent.parent.parent / "universes"


def _list_universes() -> list[str]:
    """Return list of available universe names (stem of each .json file)."""
    if not _UNIVERSE_DIR.is_dir():
        return []
    return sorted(p.stem for p in _UNIVERSE_DIR.glob("*.json") if p.is_file())


def _resolve_symbol_list(
    symbols: tuple[str, ...], universe: Optional[str]
) -> list[str]:
    """Resolve CLI symbol args — universe name or inline list."""
    if universe:
        path = _UNIVERSE_DIR / f"{universe}.json"
        return load_universe_config(str(path)).symbols
    if symbols:
        return list(symbols)
    raise click.UsageError("provide SYMBOLS or --universe/-U")


def _display_preview(
    result: PreviewResult,
    from_date: str,
    to_date: Optional[str],
) -> None:
    """Display a PreviewResult to stderr — shared by preview and dl commands."""
    click.echo(
        f"resolved {result.resolved}, {result.total_gaps} fetch gaps across all symbols\n",
        err=True,
    )

    for plan in result.plans:
        ticker = plan.ticker
        click.echo(f"{ticker} (conid {plan.conid}):", err=True)

        if plan.gaps:
            for g in plan.gaps:
                click.echo(
                    f"  fetch: {g[0].strftime('%Y-%m-%d %H:%M')}"
                    f" → {g[1].strftime('%Y-%m-%d %H:%M')}",
                    err=True,
                )
        else:
            click.echo("  fetch: up to date", err=True)

        start_ts = to_optional_ts(from_date)
        end_ts = to_optional_ts(to_date)
        df = query_candles(ticker.upper(), start_ts, end_ts)
        if not df.empty:
            gaps = _find_gaps_48h(df)
            if gaps:
                for g in gaps:
                    click.echo(
                        f"  gap >48h: {g[0].strftime('%Y-%m-%d %H:%M')}"
                        f" → {g[1].strftime('%Y-%m-%d %H:%M')}",
                        err=True,
                    )
            else:
                click.echo("  gap >48h: none", err=True)
        else:
            click.echo("  gap >48h: no existing data to check", err=True)

        click.echo("", err=True)


@data_group.command(name="dl")
@click.argument("symbols", nargs=-1, required=False)
@click.option(
    "--universe",
    "-U",
    help="Universe file name (e.g. 'nsdq', 'sector'). Overrides positional SYMBOLS.",
)
@click.option(
    "--from", "-f", "from_date", required=True, help="Start date (YYYY-MM-DD)"
)
@click.option("--to", "-t", "to_date", help="End date (YYYY-MM-DD)")
@click.option("--bar", default="1h", help="Bar size (1h, 1d, etc.)")
def dl_cmd(
    symbols: tuple[str, ...],
    universe: Optional[str],
    from_date: str,
    to_date: Optional[str],
    bar: str,
):
    """Download historical data from IBKR for SYMBOLS or --universe.

    Shows remaining gaps after download using same format as preview.
    """
    from src.data.sync import sync_data, preview_sync

    symbols_list = _resolve_symbol_list(symbols, universe)
    f_date = date.fromisoformat(from_date)
    t_date = date.fromisoformat(to_date) if to_date else None

    async def _sync_and_check():
        await sync_data(symbols_list, from_date=f_date, to_date=t_date, bar=bar)
        return await preview_sync(symbols_list, from_date=f_date, to_date=t_date)

    remaining = asyncio.run(_sync_and_check())
    _display_preview(remaining, from_date, to_date)


@data_group.command(name="preview")
@click.argument("symbols", nargs=-1, required=False)
@click.option(
    "--universe",
    "-U",
    help="Universe file name (e.g. 'nsdq', 'sector'). Overrides positional SYMBOLS.",
)
@click.option(
    "--from", "-f", "from_date", required=True, help="Start date (YYYY-MM-DD)"
)
@click.option("--to", "-t", "to_date", help="End date (YYYY-MM-DD)")
def preview_cmd(
    symbols: tuple[str, ...],
    universe: Optional[str],
    from_date: str,
    to_date: Optional[str],
):
    """Preview what gaps exist without downloading.

    Shows both fetch gaps (missing data that would be downloaded)
    and gaps >48h in existing data.
    """
    from src.data.sync import preview_sync

    symbols_list = _resolve_symbol_list(symbols, universe)
    f_date = date.fromisoformat(from_date)
    t_date = date.fromisoformat(to_date) if to_date else None

    async def _run():
        return await preview_sync(symbols_list, from_date=f_date, to_date=t_date)

    result = asyncio.run(_run())
    _display_preview(result, from_date, to_date)
