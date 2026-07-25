"""Spread module CLI — pairs trading analysis.

Usage:
    py spread analyze AAPL MSFT --from 2024-01-01
    py kalman pairs AAPL MSFT | py ind ema --span 50  (compose via pipes)
"""

from __future__ import annotations

import json
import sys
from typing import Optional

import click
import pandas as pd

from src.kalman.pure import run_pairs_kalman
from src.kalman.types import PairsKalmanConfig


@click.group(name="spread")
def spread_group():
    """Pairs spread analysis with Kalman filter."""


@spread_group.command(name="analyze")
@click.argument("symbol1", required=False)
@click.argument("symbol2", required=False)
@click.option("--stdin", "use_stdin", is_flag=True, help="Read OHLCV from stdin (two symbols)")
@click.option("--from", "-f", "from_date")
@click.option("--to", "-t", "to_date")
@click.option("--bar", default="1d")
@click.option("--process-noise", "-q", type=float, default=1e-4)
@click.option("--measurement-noise", "-r", type=float, default=1e-3)
@click.option("--mean-halflife", "-m", type=int, default=50)
def spread_analyze(
    symbol1: Optional[str],
    symbol2: Optional[str],
    use_stdin: bool,
    from_date: Optional[str],
    to_date: Optional[str],
    bar: str,
    process_noise: float,
    measurement_noise: float,
    mean_halflife: int,
):
    """Analyze spread between two symbols using Kalman [α, β] model.

    Output: JSON lines with t_stat, beta, alpha, spread, innovation_S.
    """
    if use_stdin or not symbol1:
        df = _read_ohlcv_stdin()
        if df.empty:
            raise click.UsageError("No data: pipe OHLCV or provide SYMBOL1 SYMBOL2")
        if "symbol" in df.columns:
            syms = df["symbol"].unique()
            if len(syms) < 2:
                raise click.UsageError(f"Need 2 symbols, got {list(syms)}")
            p1 = df[df["symbol"] == syms[0]]["close"]
            p2 = df[df["symbol"] == syms[1]]["close"]
            sym1_name, sym2_name = str(syms[0]), str(syms[1])
        else:
            raise click.UsageError("Stdin data must include 'symbol' column")
    else:
        from src.shared.db import query_candles
        start_ts = pd.Timestamp(from_date) if from_date else None
        end_ts = pd.Timestamp(to_date) if to_date else None
        df1 = query_candles(symbol1.upper(), start_ts, end_ts, bar)
        df2 = query_candles(symbol2.upper(), start_ts, end_ts, bar)
        p1 = df1["close"]
        p2 = df2["close"]
        sym1_name, sym2_name = symbol1, symbol2

    config = PairsKalmanConfig(
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        mean_halflife=mean_halflife,
    )

    result = run_pairs_kalman(p1, p2, config)

    n = len(result.t_stat)
    for i in range(n):
        ts = result.t_stat.index[i]
        rec = {
            "t": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "t_stat": round(float(result.t_stat.iloc[i]), 4),
            "beta": round(float(result.beta.iloc[i]), 6),
            "alpha": round(float(result.alpha.iloc[i]), 6),
            "spread": round(float(result.spread.iloc[i]), 6),
            "innovation_S": round(float(result.innovation_S.iloc[i]), 8),
        }
        click.echo(json.dumps(rec, default=str))


def _read_ohlcv_stdin() -> pd.DataFrame:
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
