"""Data module — download and query market data.

CLI: py data dl AAPL MSFT --from 2026-01-01
     py data query AAPL --from 2026-01-01 --to 2026-06-01
     py data preview AAPL MSFT --from 2026-01-01
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date
from typing import Optional

import click
import pandas as pd

from src.utils import to_optional_ts

from src.shared.db import query_candles


# ── JSON lines output helper ──────────────────────────────────────


def _jsonl_out(df: pd.DataFrame) -> str:
    """Convert OHLCV DataFrame to JSON lines string."""
    if df.empty:
        return ""
    lines: list[str] = []
    for idx, row in df.iterrows():
        rec = {
            "t": idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx),
            "o": row.get("open"),
            "h": row.get("high"),
            "l": row.get("low"),
            "c": row.get("close"),
            "v": row.get("volume"),
        }
        lines.append(json.dumps(rec, default=str))
    return "\n".join(lines)


def _csv_out(df: pd.DataFrame) -> str:
    """Convert OHLCV DataFrame to CSV string."""
    if df.empty:
        return ""
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
        out = out.rename(columns={"Date": "timestamp", "index": "timestamp"})
    return out.to_csv(index=False)


def _write_output(df: pd.DataFrame, fmt: str) -> None:
    """Write DataFrame to stdout in the requested format."""
    if fmt == "csv":
        click.echo(_csv_out(df))
    else:
        click.echo(_jsonl_out(df))


# ── Stdin reader ──────────────────────────────────────────────────


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


# ── CLI group ─────────────────────────────────────────────────────


@click.group(name="data")
def data_group():
    """Market data download and query."""


@data_group.command(name="query")
@click.argument("symbol")
@click.option("--from", "-f", "from_date", help="Start date (YYYY-MM-DD)")
@click.option("--to", "-t", "to_date", help="End date (YYYY-MM-DD)")
@click.option("--bar", default="1h", help="Bar size (1h, 1d, etc.)")
@click.option(
    "--format", "-F", "fmt", type=click.Choice(["jsonl", "csv"]), default="jsonl"
)
def query_cmd(
    symbol: str, from_date: Optional[str], to_date: Optional[str], bar: str, fmt: str
):
    """Query OHLCV candles for SYMBOL from the local database."""
    start_ts = to_optional_ts(from_date)
    end_ts = to_optional_ts(to_date)

    df = query_candles(symbol.upper(), start_ts, end_ts, bar)
    _write_output(df, fmt)


@data_group.command(name="dl")
@click.argument("symbols", nargs=-1, required=True)
@click.option(
    "--from", "-f", "from_date", required=True, help="Start date (YYYY-MM-DD)"
)
@click.option("--to", "-t", "to_date", help="End date (YYYY-MM-DD)")
@click.option("--bar", default="1h", help="Bar size (1h, 1d, etc.)")
@click.option(
    "--format", "-F", "fmt", type=click.Choice(["jsonl", "csv"]), default="jsonl"
)
def dl_cmd(
    symbols: tuple[str, ...], from_date: str, to_date: Optional[str], bar: str, fmt: str
):
    """Download historical data from IBKR for SYMBOLS."""
    from src.syncm import sync_data

    f_date = date.fromisoformat(from_date)
    t_date = date.fromisoformat(to_date) if to_date else None

    async def _run():
        return await sync_data(list(symbols), from_date=f_date, to_date=t_date, bar=bar)

    result = asyncio.run(_run())
    click.echo(
        json.dumps(
            {
                "resolved": result.resolved,
                "fetched": len(result.fetched),
                "gaps_found": result.gaps_found,
            }
        ),
        err=True,
    )

    for sym in symbols:
        start_ts = to_optional_ts(from_date)
        end_ts = to_optional_ts(to_date)
        df = query_candles(sym.upper(), start_ts, end_ts, bar)
        _write_output(df, fmt)


@data_group.command(name="preview")
@click.argument("symbols", nargs=-1, required=True)
@click.option(
    "--from", "-f", "from_date", required=True, help="Start date (YYYY-MM-DD)"
)
@click.option("--to", "-t", "to_date", help="End date (YYYY-MM-DD)")
def preview_cmd(symbols: tuple[str, ...], from_date: str, to_date: Optional[str]):
    """Preview what gaps exist without downloading."""
    from src.syncm import preview_sync

    f_date = date.fromisoformat(from_date)
    t_date = date.fromisoformat(to_date) if to_date else None

    async def _run():
        return await preview_sync(list(symbols), from_date=f_date, to_date=t_date)

    result = asyncio.run(_run())
    out = {
        "resolved": result.resolved,
        "total_gaps": result.total_gaps,
        "plans": [
            {
                "ticker": p.ticker,
                "conid": p.conid,
                "gaps": [[str(g[0]), str(g[1])] for g in p.gaps],
            }
            for p in result.plans
        ],
    }
    click.echo(json.dumps(out, indent=2))
