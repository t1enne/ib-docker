"""HMM (Hidden Markov Model) CLI — regime detection.

Usage:
    py hmm fit AAPL --n-regimes 3 --from 2024-01-01
    py data query AAPL --from 2024-01-01 | py hmm predict --model hmm_models/AAPL.pkl
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd

from src.hmm.hmm import MarketRegimeHMM
from src.utils import to_optional_ts


# ── Helpers ───────────────────────────────────────────────────────


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


def _write_regimes(regimes: pd.Series, probas: Optional[pd.DataFrame] = None) -> None:
    """Write regime labels + optional probabilities as JSON lines."""
    for i, (ts, r) in enumerate(regimes.dropna().items()):
        rec = {
            "t": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),
            "regime": int(r),
        }
        if probas is not None and i < len(probas):
            for col in probas.columns:
                rec[col] = round(float(probas.iloc[i][col]), 6)
        click.echo(json.dumps(rec, default=str))


# ── CLI ───────────────────────────────────────────────────────────


@click.group(name="hmm")
def hmm_group():
    """Hidden Markov Model regime detection."""


@hmm_group.command(name="fit")
@click.argument("symbol", required=False)
@click.option("--stdin", "use_stdin", is_flag=True, help="Read OHLCV from stdin")
@click.option("--from", "-f", "from_date", help="Start date (YYYY-MM-DD)")
@click.option("--to", "-t", "to_date", help="End date (YYYY-MM-DD)")
@click.option("--bar", default="1h")
@click.option(
    "--n-regimes", "-n", type=int, default=3, help="Number of regime states (2 or 3)"
)
@click.option("--vol-window", "-v", type=int, default=20)
@click.option("--momentum-window", "-m", type=int, default=10)
@click.option("--min-train-size", type=int, default=252)
@click.option(
    "--output-dir", "-o", default="./hmm_models", help="Output directory for models"
)
@click.option(
    "--predict/--no-predict", default=True, help="Output regime predictions after fit"
)
def hmm_fit(
    symbol: Optional[str],
    use_stdin: bool,
    from_date: Optional[str],
    to_date: Optional[str],
    bar: str,
    n_regimes: int,
    vol_window: int,
    momentum_window: int,
    min_train_size: int,
    output_dir: str,
    predict: bool,
):
    """Fit HMM on SYMBOL and save the model.

    Output (stderr): model path + regime statistics.
    Output (stdout): regime predictions (JSON lines) if --predict.
    """
    if use_stdin:
        symbol = None

    if symbol:
        from src.shared.db import query_candles

        start_ts = to_optional_ts(from_date)
        end_ts = to_optional_ts(to_date)
        df = query_candles(symbol.upper(), start_ts, end_ts, bar)
    else:
        df = _read_ohlcv_stdin()

    if df.empty:
        raise click.UsageError(
            "No data: provide symbol + dates, or pipe OHLCV via stdin"
        )

    prices = df["close"]
    sym = symbol or "stdin"

    hmm_model = MarketRegimeHMM(
        n_regimes=n_regimes,
        vol_window=vol_window,
        momentum_window=momentum_window,
        min_train_size=min_train_size,
    )

    hmm_model.fit(prices)

    # Save model
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_file = out_path / f"{sym}_hmm_{n_regimes}r.pkl"
    hmm_model.save(str(model_file))

    # Stats to stderr
    stats = hmm_model.get_regime_statistics(prices)
    stats_dict = {
        "model_file": str(model_file),
        "n_regimes": n_regimes,
        "regimes": {},
    }
    for r in range(n_regimes):
        stats_dict["regimes"][str(r)] = {
            "mean_return_annual": round(
                float(stats.mean_return.get(r, float("nan"))), 6
            )
            if not np.isnan(stats.mean_return.get(r, float("nan")))
            else None,
            "volatility": round(float(stats.volatility.get(r, float("nan"))), 6)
            if not np.isnan(stats.volatility.get(r, float("nan")))
            else None,
            "frequency": round(float(stats.frequency.get(r, 0)), 4),
        }

    # Transition matrix
    try:
        transmat = hmm_model.get_transition_matrix()
        stats_dict["transition_matrix"] = {
            str(i): {
                str(j): round(float(transmat.iloc[i, j]), 4) for j in range(n_regimes)
            }
            for i in range(n_regimes)
        }
    except Exception:
        pass

    click.echo(json.dumps(stats_dict, indent=2), err=True)

    if predict:
        regimes = hmm_model.predict(prices)
        probas = hmm_model.predict_proba(prices)
        _write_regimes(regimes, probas)


@hmm_group.command(name="predict")
@click.option(
    "--model", "-m", "model_path", required=True, help="Path to saved HMM model (.pkl)"
)
@click.argument("symbol", required=False)
@click.option("--stdin", "use_stdin", is_flag=True)
@click.option("--from", "-f", "from_date")
@click.option("--to", "-t", "to_date")
@click.option("--bar", default="1h")
@click.option("--probas/--no-probas", default=True, help="Include regime probabilities")
def hmm_predict(
    model_path: str,
    symbol: Optional[str],
    use_stdin: bool,
    from_date: Optional[str],
    to_date: Optional[str],
    bar: str,
    probas: bool,
):
    """Predict regimes using a saved HMM model.

    Output: JSON lines with regime labels + probabilities.
    """
    # Load model
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    hmm_model = MarketRegimeHMM(
        n_regimes=model_data["n_regimes"],
        vol_window=model_data.get("vol_window", 20),
        momentum_window=model_data.get("momentum_window", 10),
        min_train_size=model_data.get("min_train_size", 252),
    )
    hmm_model.model = model_data["model"]
    hmm_model.fitted = True

    if use_stdin:
        symbol = None

    if symbol:
        from src.shared.db import query_candles

        start_ts = to_optional_ts(from_date)
        end_ts = to_optional_ts(to_date)
        df = query_candles(symbol.upper(), start_ts, end_ts, bar)
    else:
        df = _read_ohlcv_stdin()

    if df.empty:
        raise click.UsageError("No data")

    prices = df["close"]
    regimes = hmm_model.predict(prices)
    p_df = hmm_model.predict_proba(prices) if probas else None
    _write_regimes(regimes, p_df)
