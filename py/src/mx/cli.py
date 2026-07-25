"""MX (matrix) module CLI — correlation + cointegration analysis.

Usage:
    py mx matrix AAPL MSFT GOOGL --from 2024-01-01
"""

from __future__ import annotations

import json
from typing import Optional

import click
import pandas as pd

from src.utils import to_optional_ts


@click.group(name="mx")
def mx_group():
    """Correlation and cointegration matrix analysis."""


@mx_group.command(name="matrix")
@click.argument("symbols", nargs=-1, required=True)
@click.option("--from", "-f", "from_date", help="Start date (YYYY-MM-DD)")
@click.option("--end", "-e", "to_date", help="End date (YYYY-MM-DD)")
@click.option("--bar", default="1d")
@click.option("--universe", "-u", default=None, help="Path to universe config file")
def mx_matrix(
    symbols: tuple[str, ...],
    from_date: Optional[str],
    to_date: Optional[str],
    bar: str,
    universe: Optional[str],
):
    """Compute correlation and cointegration matrix for SYMBOLS.

    Output: JSON with correlation matrix, cointegration p-values, and summary.
    """
    from src.shared.db import query_candles

    if universe:
        from src.syncm import load_universe_config

        conf = load_universe_config(universe)
        syms = conf.symbols
    else:
        syms = list(symbols)

    start_ts = to_optional_ts(from_date)
    end_ts = to_optional_ts(to_date)

    # Load closes for all symbols
    closes: dict[str, pd.Series] = {}
    for sym in syms:
        df = query_candles(sym.upper(), start_ts, end_ts, bar)
        if not df.empty:
            closes[sym] = df["close"]

    if len(closes) < 2:
        click.echo(json.dumps({"error": "Need at least 2 symbols with data"}))
        return

    # Align to common index
    aligned = pd.DataFrame(closes).dropna()
    len(aligned.columns)

    # Correlation matrix
    corr = aligned.corr()

    # Cointegration p-values (pairwise)
    from src.utils import symmetric_cointegration_p

    coint_p: dict[str, dict[str, Optional[float]]] = {}
    cols = list(aligned.columns)
    for i, s1 in enumerate(cols):
        coint_p[s1] = {}
        for j, s2 in enumerate(cols):
            if i == j:
                coint_p[s1][s2] = None
            elif j > i:
                p = symmetric_cointegration_p(aligned[s1], aligned[s2])
                coint_p[s1][s2] = round(float(p), 6)
            else:
                coint_p[s1][s2] = coint_p[s2][s1]

    # Output
    result = {
        "symbols": cols,
        "n_observations": len(aligned),
        "correlation": {
            s1: {s2: round(float(corr.loc[s1, s2]), 4) for s2 in cols} for s1 in cols
        },
        "cointegration_pvalues": coint_p,
    }
    click.echo(json.dumps(result, indent=2))
