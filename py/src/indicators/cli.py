"""Indicator helpers — standalone functions for technical analysis.

Each indicator is a pure function operating on pandas Series.
No state, no side effects. Agent-friendly: every indicator is also
a CLI command that reads OHLCV from stdin and writes to stdout.
"""

from __future__ import annotations

import json
import sys
from typing import Optional

import click
import pandas as pd

from src.bt.indicators import (
    adx,
    atr,
    bollinger_bands,
    ema,
    macd,
    momentum,
    rsi,
    sma,
    volatility,
)


# ── Stdin reader ──────────────────────────────────────────────────


def _read_ohlcv_stdin() -> pd.DataFrame:
    """Read OHLCV JSON lines from stdin."""
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
            pass
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["t"])
    df = df.set_index("timestamp").drop(columns=["t"])
    renames = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns=renames)
    return df


def _get_close(df: pd.DataFrame, symbol: Optional[str]) -> pd.Series:
    """Extract close prices, optionally filtering by symbol if MultiIndex."""
    if symbol and "symbol" in df.columns:
        mask = df["symbol"].str.upper() == symbol.upper()
        return df.loc[mask, "close"]
    if "close" in df.columns:
        return df["close"]
    return pd.Series(dtype=float)


# ── Output helpers ────────────────────────────────────────────────


def _write_series(result: pd.Series, name: str) -> None:
    """Write a Series as JSON lines with timestamp + value."""
    for ts, val in result.dropna().items():
        click.echo(json.dumps({"t": ts.isoformat() if hasattr(ts, "isoformat") else str(ts), name: round(float(val), 6)}, default=str))


def _write_df(result: pd.DataFrame) -> None:
    """Write a DataFrame as JSON lines."""
    for idx, row in result.dropna().iterrows():
        rec: dict = {"t": idx.isoformat() if hasattr(idx, "isoformat") else str(idx)}
        for col in result.columns:
            rec[col] = round(float(row[col]), 6) if not np.isnan(row[col]) else None
        click.echo(json.dumps(rec, default=str))


# ── CLI group ─────────────────────────────────────────────────────


@click.group(name="ind")
def ind_group():
    """Technical indicators from OHLCV data (stdin or DB)."""


@ind_group.command(name="ema")
@click.option("--span", type=int, default=20, help="EMA span")
@click.option("--symbol", "-s", help="Ticker symbol (optional, for multi-symbol input)")
@click.option("--from", "-f", "from_date", help="Start date for DB query (YYYY-MM-DD)")
@click.option("--to", "-t", "to_date", help="End date for DB query (YYYY-MM-DD)")
@click.option("--bar", default="1h", help="Bar size for DB query")
def ema_cmd(span: int, symbol: Optional[str], from_date: Optional[str], to_date: Optional[str], bar: str):
    """Exponential Moving Average."""
    df = _read_ohlcv_stdin()
    if df.empty and from_date:
        from src.shared.db import query_candles
        assert symbol, "--symbol required for DB query"
        df = query_candles(symbol.upper(), pd.Timestamp(from_date) if from_date else None, pd.Timestamp(to_date) if to_date else None, bar)
    close = _get_close(df, symbol)
    result = ema(close, span)
    _write_series(result, f"ema_{span}")


@ind_group.command(name="rsi")
@click.option("--window", "-w", type=int, default=14, help="RSI period")
@click.option("--symbol", "-s", help="Ticker symbol")
@click.option("--from", "-f", "from_date")
@click.option("--to", "-t", "to_date")
@click.option("--bar", default="1h")
def rsi_cmd(window: int, symbol: Optional[str], from_date: Optional[str], to_date: Optional[str], bar: str):
    """Relative Strength Index."""
    df = _read_ohlcv_stdin()
    if df.empty and from_date:
        from src.shared.db import query_candles
        assert symbol
        df = query_candles(symbol.upper(), pd.Timestamp(from_date) if from_date else None, pd.Timestamp(to_date) if to_date else None, bar)
    close = _get_close(df, symbol)
    result = rsi(close, window)
    _write_series(result, "rsi")


@ind_group.command(name="atr")
@click.option("--window", "-w", type=int, default=14, help="ATR period")
@click.option("--symbol", "-s")
@click.option("--from", "-f", "from_date")
@click.option("--to", "-t", "to_date")
@click.option("--bar", default="1h")
def atr_cmd(window: int, symbol: Optional[str], from_date: Optional[str], to_date: Optional[str], bar: str):
    """Average True Range."""
    df = _read_ohlcv_stdin()
    if df.empty and from_date:
        from src.shared.db import query_candles
        assert symbol
        df = query_candles(symbol.upper(), pd.Timestamp(from_date) if from_date else None, pd.Timestamp(to_date) if to_date else None, bar)
    result = atr(df["high"], df["low"], df["close"], window)
    _write_series(result, "atr")


@ind_group.command(name="macd")
@click.option("--fast", type=int, default=12)
@click.option("--slow", type=int, default=26)
@click.option("--signal", type=int, default=9)
@click.option("--symbol", "-s")
@click.option("--from", "-f", "from_date")
@click.option("--to", "-t", "to_date")
@click.option("--bar", default="1h")
def macd_cmd(fast: int, slow: int, signal: int, symbol: Optional[str], from_date: Optional[str], to_date: Optional[str], bar: str):
    """MACD indicator."""
    df = _read_ohlcv_stdin()
    if df.empty and from_date:
        from src.shared.db import query_candles
        assert symbol
        df = query_candles(symbol.upper(), pd.Timestamp(from_date) if from_date else None, pd.Timestamp(to_date) if to_date else None, bar)
    close = _get_close(df, symbol)
    result = macd(close, fast, slow, signal)
    _write_df(result)


@ind_group.command(name="bbands")
@click.option("--window", "-w", type=int, default=20)
@click.option("--num-std", type=float, default=2.0)
@click.option("--symbol", "-s")
@click.option("--from", "-f", "from_date")
@click.option("--to", "-t", "to_date")
@click.option("--bar", default="1h")
def bbands_cmd(window: int, num_std: float, symbol: Optional[str], from_date: Optional[str], to_date: Optional[str], bar: str):
    """Bollinger Bands."""
    df = _read_ohlcv_stdin()
    if df.empty and from_date:
        from src.shared.db import query_candles
        assert symbol
        df = query_candles(symbol.upper(), pd.Timestamp(from_date) if from_date else None, pd.Timestamp(to_date) if to_date else None, bar)
    close = _get_close(df, symbol)
    upper, middle, lower = bollinger_bands(close, window, num_std)
    _write_df(pd.DataFrame({"upper": upper, "middle": middle, "lower": lower}, index=close.index))


@ind_group.command(name="adx")
@click.option("--window", "-w", type=int, default=14)
@click.option("--symbol", "-s")
@click.option("--from", "-f", "from_date")
@click.option("--to", "-t", "to_date")
@click.option("--bar", default="1h")
def adx_cmd(window: int, symbol: Optional[str], from_date: Optional[str], to_date: Optional[str], bar: str):
    """Average Directional Index."""
    df = _read_ohlcv_stdin()
    if df.empty and from_date:
        from src.shared.db import query_candles
        assert symbol
        df = query_candles(symbol.upper(), pd.Timestamp(from_date) if from_date else None, pd.Timestamp(to_date) if to_date else None, bar)
    result = adx(df["high"], df["low"], df["close"], window)
    _write_series(result, "adx")


@ind_group.command(name="vol")
@click.option("--window", "-w", type=int, default=20)
@click.option("--annualized/--raw", default=True)
@click.option("--symbol", "-s")
@click.option("--from", "-f", "from_date")
@click.option("--to", "-t", "to_date")
@click.option("--bar", default="1h")
def vol_cmd(window: int, annualized: bool, symbol: Optional[str], from_date: Optional[str], to_date: Optional[str], bar: str):
    """Rolling volatility."""
    df = _read_ohlcv_stdin()
    if df.empty and from_date:
        from src.shared.db import query_candles
        assert symbol
        df = query_candles(symbol.upper(), pd.Timestamp(from_date) if from_date else None, pd.Timestamp(to_date) if to_date else None, bar)
    close = _get_close(df, symbol)
    result = volatility(close, window, annualized)
    _write_series(result, "volatility")
