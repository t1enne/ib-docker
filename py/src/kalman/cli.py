"""Kalman filter CLI — standalone command group.

Usage:
    py kalman run AAPL --q 1e-5 --r 1e-3
    py data query AAPL --from 2026-01-01 | py kalman run --stdin
    py kalman pairs AAPL MSFT --q 1e-4 --r 1e-3
"""

from __future__ import annotations

import json
import sys
from typing import Optional, cast

import click
import pandas as pd

from src.kalman.pure import run_filter, run_pairs_kalman, compute_stats
from src.kalman.types import KalmanConfig, PairsKalmanConfig
from src.utils import to_optional_ts


# ── Helpers ───────────────────────────────────────────────────────


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


def _write_kalman_result(result, include_stats: bool = True) -> None:
    """Write FilterResult as JSON lines."""
    n = len(result.filtered)
    for i in range(n):
        ts = result.filtered.index[i]
        rec = {
            "t": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "filtered": round(float(result.filtered.iloc[i]), 6),
            "predicted": round(float(result.predicted.iloc[i]), 6),
            "upper_ci": round(float(result.upper_ci.iloc[i]), 6),
            "lower_ci": round(float(result.lower_ci.iloc[i]), 6),
            "residual": round(float(result.residuals.iloc[i]), 6),
            "kalman_gain": round(float(result.kalman_gains.iloc[i]), 6),
            "velocity": round(float(result.velocity.iloc[i]), 6),
        }
        click.echo(json.dumps(rec, default=str))


def _write_pairs_result(result) -> None:
    """Write PairsKalmanResult as JSON lines."""
    n = len(result.t_stat)
    for i in range(n):
        ts = result.t_stat.index[i]
        rec = {
            "t": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "alpha": round(float(result.alpha.iloc[i]), 6),
            "beta": round(float(result.beta.iloc[i]), 6),
            "spread": round(float(result.spread.iloc[i]), 6),
            "t_stat": round(float(result.t_stat.iloc[i]), 4),
            "innovation_S": round(float(result.innovation_S.iloc[i]), 8),
        }
        click.echo(json.dumps(rec, default=str))


def _load_prices(
    symbol: Optional[str],
    from_date: Optional[str],
    to_date: Optional[str],
    bar: str,
) -> pd.Series:
    """Load close prices — from stdin, DB, or CLI args."""
    df = _read_ohlcv_stdin()

    if df.empty and symbol:
        from src.shared.db import query_candles

        start_ts = to_optional_ts(from_date)
        end_ts = to_optional_ts(to_date)
        df = query_candles(symbol.upper(), start_ts, end_ts, bar)

    if df.empty:
        raise click.UsageError(
            "No data: provide symbol + dates, or pipe OHLCV via stdin"
        )

    return cast(pd.Series, df["close"])


# ── CLI group ─────────────────────────────────────────────────────


@click.group(name="kalman")
def kalman_group():
    """Kalman filter for price smoothing and pairs trading."""


@kalman_group.command(name="run")
@click.argument("symbol", required=False)
@click.option(
    "--stdin",
    "use_stdin",
    is_flag=True,
    help="Read OHLCV from stdin (no symbol needed)",
)
@click.option("--from", "-f", "from_date", help="Start date (YYYY-MM-DD)")
@click.option("--to", "-t", "to_date", help="End date (YYYY-MM-DD)")
@click.option("--bar", default="1h", help="Bar size")
@click.option("--process-noise", "-q", type=float, default=1e-5)
@click.option("--measurement-noise", "-r", type=float, default=1e-3)
@click.option("--adaptive/--no-adaptive", default=False)
@click.option("--vol-window", type=int, default=20)
@click.option("--stats/--no-stats", default=True, help="Print summary stats to stderr")
def kalman_run(
    symbol: Optional[str],
    use_stdin: bool,
    from_date: Optional[str],
    to_date: Optional[str],
    bar: str,
    process_noise: float,
    measurement_noise: float,
    adaptive: bool,
    vol_window: int,
    stats: bool,
):
    """Run Kalman filter on SYMBOL.

    Pipe OHLCV via stdin or provide SYMBOL + --from/--to to query DB.

    Output: JSON lines with filtered, predicted, CI, residuals, velocity, gains.
    """
    if use_stdin:
        symbol = None

    prices = _load_prices(symbol, from_date, to_date, bar)

    config = KalmanConfig(
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        adaptive=adaptive,
        vol_window=vol_window,
    )

    result = run_filter(prices, config)

    if stats:
        s = compute_stats(prices, result)
        click.echo(
            json.dumps(
                {
                    "rmse": round(s.rmse, 6),
                    "mae": round(s.mae, 6),
                    "coverage_95": round(s.coverage_95, 4),
                    "avg_kalman_gain": round(s.avg_kalman_gain, 6),
                    "n_observations": s.n_observations,
                }
            ),
            err=True,
        )

    _write_kalman_result(result)


@kalman_group.command(name="pairs")
@click.argument("symbol1", required=False)
@click.argument("symbol2", required=False)
@click.option(
    "--stdin", "use_stdin", is_flag=True, help="Read two OHLCV streams from stdin"
)
@click.option("--from", "-f", "from_date")
@click.option("--to", "-t", "to_date")
@click.option("--bar", default="1h")
@click.option("--process-noise", "-q", type=float, default=1e-4)
@click.option("--measurement-noise", "-r", type=float, default=1e-3)
@click.option(
    "--mean-halflife", "-m", type=int, default=50, help="OLS warm-start window"
)
@click.option("--adaptive/--no-adaptive", default=False)
@click.option("--vol-window", type=int, default=20)
def kalman_pairs(
    symbol1: Optional[str],
    symbol2: Optional[str],
    use_stdin: bool,
    from_date: Optional[str],
    to_date: Optional[str],
    bar: str,
    process_noise: float,
    measurement_noise: float,
    mean_halflife: int,
    adaptive: bool,
    vol_window: int,
):
    """Pairs-trading Kalman filter for two symbols.

    Output: JSON lines with alpha, beta, spread, t_stat, innovation_S.
    """
    if use_stdin or not symbol1:
        df = _read_ohlcv_stdin()
        if df.empty:
            raise click.UsageError(
                "No data: pipe OHLCV via stdin or provide SYMBOL1 SYMBOL2"
            )
        # Assume stdin has both symbols
        if "symbol" in df.columns:
            syms = df["symbol"].unique()
            if len(syms) < 2:
                raise click.UsageError(f"Need 2 symbols in stdin, got {list(syms)}")
            p1 = df[df["symbol"] == syms[0]]["close"]
            p2 = df[df["symbol"] == syms[1]]["close"]
        else:
            raise click.UsageError("Stdin data must include 'symbol' column for pairs")
    else:
        assert symbol2 is not None, "symbol2 required when not using stdin"
        from src.shared.db import query_candles

        start_ts = to_optional_ts(from_date)
        end_ts = to_optional_ts(to_date)
        df1 = query_candles(symbol1.upper(), start_ts, end_ts, bar)
        df2 = query_candles(symbol2.upper(), start_ts, end_ts, bar)
        p1 = df1["close"]
        p2 = df2["close"]

    config = PairsKalmanConfig(
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        mean_halflife=mean_halflife,
        adaptive=adaptive,
        vol_window=vol_window,
    )

    result = run_pairs_kalman(p1, p2, config)
    _write_pairs_result(result)


# ── Legacy entry point (backward compat) ──────────────────────────


def kalman(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    process_noise: float = 1e-5,
    measurement_noise: float = 1e-3,
    adaptive: bool = False,
    vol_window: int = 20,
) -> None:
    """Legacy entry point — delegates to CLI run command."""
    from src.shared.db import query_candles
    from src.kalman.pure import run_filter, compute_stats

    start_ts = to_optional_ts(start)
    end_ts = to_optional_ts(end)
    df = query_candles(symbol.upper(), start_ts, end_ts)

    if df.empty:
        click.echo(f"No data found for {symbol}")
        return

    prices = cast(pd.Series, df["close"])
    config = KalmanConfig(
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        adaptive=adaptive,
        vol_window=vol_window,
    )

    result = run_filter(prices, config)
    stats = compute_stats(prices, result)

    # Print stats to stderr
    click.echo(
        json.dumps(
            {
                "symbol": symbol,
                "observations": stats.n_observations,
                "rmse": round(stats.rmse, 6),
                "mae": round(stats.mae, 6),
                "coverage_95": round(stats.coverage_95, 4),
                "avg_kalman_gain": round(stats.avg_kalman_gain, 6),
            }
        ),
        err=True,
    )

    _write_kalman_result(result)
