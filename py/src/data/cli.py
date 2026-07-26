"""Data CLI — download, query, and preview market data.

Usage:
    py data dl AAPL MSFT --from 2026-01-01
    py data dl --universe nsdq --from 2026-01-01
    py data query AAPL --from 2026-01-01 --to 2026-06-01
    py data preview AAPL MSFT --from 2026-01-01
    py data preview --universe sector --from 2024-01-01
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import yaml

from src.utils import to_optional_ts
from src.data.db import query_candles


# ── Gap detection helpers ──────────────────────────────────────


def _find_gaps_48h(df: pd.DataFrame) -> list[tuple[datetime, datetime]]:
    """Find gaps >48h between consecutive candles in a DataFrame.

    Args:
        df: DataFrame with DatetimeIndex, sorted chronologically.

    Returns:
        List of (gap_start, gap_end) tuples for each gap found.
    """
    if df.empty or len(df) < 2:
        return []

    gaps: list[tuple[datetime, datetime]] = []
    idx = df.index
    for i in range(len(idx) - 1):
        delta = idx[i + 1] - idx[i]
        if delta > timedelta(hours=48):
            gaps.append((idx[i].to_pydatetime(), idx[i + 1].to_pydatetime()))
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
@click.argument("symbol")
@click.option("--from", "-f", "from_date", help="Start date (YYYY-MM-DD)")
@click.option("--to", "-t", "to_date", help="End date (YYYY-MM-DD)")
@click.option("--bar", default="1h", help="Bar size (1h, 1d, etc.)")
def query_cmd(symbol: str, from_date: Optional[str], to_date: Optional[str], bar: str):
    """Query OHLCV candles for SYMBOL from the local database.

    Shows a recap with date range, row count, and gaps >48h.
    """
    start_ts = to_optional_ts(from_date)
    end_ts = to_optional_ts(to_date)

    df = query_candles(symbol.upper(), start_ts, end_ts, bar)
    click.echo(_recap(df, symbol.upper()), err=True)


# ── Universe file helpers ────────────────────────────────────

_UNIVERSE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "universes"


def _list_universes() -> list[str]:
    """Return list of available universe names (stem of each .yml file)."""
    if not _UNIVERSE_DIR.is_dir():
        return []
    return sorted(p.stem for p in _UNIVERSE_DIR.glob("*.yml") if p.is_file())


def _load_universe(name: str) -> list[str]:
    """Load symbols from a universe file by name (e.g. 'nsdq')."""
    path = _UNIVERSE_DIR / f"{name}.yml"
    if not path.is_file():
        avail = ", ".join(_list_universes())
        raise click.BadParameter(
            f"unknown universe '{name}'. available: {avail}",
            param_hint="--universe",
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    symbols: list[str] = data.get("symbols", []) if isinstance(data, dict) else []
    return symbols


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

    Provide tickers inline or use --universe to load symbols from
    a named universe file in universes/.

    Shows a recap per symbol after download with date range,
    row count, and gaps >48h.
    """
    from src.data.sync import sync_data

    if universe:
        symbols_list = _load_universe(universe)
    elif symbols:
        symbols_list = list(symbols)
    else:
        raise click.UsageError("provide SYMBOLS or --universe/-U")

    f_date = date.fromisoformat(from_date)
    t_date = date.fromisoformat(to_date) if to_date else None

    async def _run():
        return await sync_data(symbols_list, from_date=f_date, to_date=t_date, bar=bar)

    result = asyncio.run(_run())
    click.echo(
        f"resolved {result.resolved}, fetched {len(result.fetched)} symbols, "
        f"{result.gaps_found} gaps filled",
        err=True,
    )

    _print_recap(tuple(symbols_list), from_date, to_date, bar)


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

    Provide tickers inline or use --universe to load symbols from
    a named universe file in universes/.

    Shows both fetch gaps (missing data that would be downloaded)
    and gaps >48h in existing data.
    """
    from src.data.sync import preview_sync

    if universe:
        symbols_list = _load_universe(universe)
    elif symbols:
        symbols_list = list(symbols)
    else:
        raise click.UsageError("provide SYMBOLS or --universe/-U")

    f_date = date.fromisoformat(from_date)
    t_date = date.fromisoformat(to_date) if to_date else None

    async def _run():
        return await preview_sync(symbols_list, from_date=f_date, to_date=t_date)

    result = asyncio.run(_run())
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

        # Also show >48h gaps in existing data within the range
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
