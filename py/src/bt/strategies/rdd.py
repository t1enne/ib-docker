"""Real-Discount-Dominance (RDD) feature pipeline + macro data loader.

Implements the three leading features from the research spec
(``docs/research/2024-08_RDD-features.md``):

- ``acc``   — inflation acceleration (2nd diff of log CPI, cost-push gauge)
- ``real``  — real (inflation-adjusted) equity level  = ``deflated_log_prices``
- ``rdisc`` — real-discount proxy = nominal 10y yield minus trailing CPI

All three are standardised to rolling z-scores (window ``w_z``) and fused by a
``MarketRegimeHMM`` into a small number of hidden regimes. The *feature* stack
here is pure and cursor-safe: every function reads only values known at (or
before) the caller's cursor ``τ``, and the stepped CPI / yield series are
forward-filled on load so the latest-known print is exactly what a cursor sees.

Engineering additions vs the library's macro CPI indicator
(:func:`src.indicators.macro.load_cpi_price_index` / ``deflated_log_prices``):

- :func:`load_yields` — a structural clone of ``load_cpi_price_index`` for the
  FRED DGS10 10y yield file (``assets/dgs10.csv``). Blank cells (FRED no-print
  days) are coerced to NaN and forward-filled onto a daily grid.
- :func:`zdisc` / :func:`zacc` / :func:`zreal` — the standardised feature
  functions (per-feature, vectorized, one Series out).

The HMM itself lives in the strategy module (``rdd_regime.py``); this module is
deliberately HMM-free so it stays importable in isolation and independently
testable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.indicators.macro import deflated_log_prices

_DEFAULT_YIELD_PATH = Path("assets/dgs10.csv")

_LOG2_DAYS = 252.0


def load_yields(
    source: str | Path | None = None,
) -> pd.Series:
    """Return a daily 10y UST yield series (percent) forward-filled onto days.

    Structural clone of ``load_cpi_price_index`` for ``assets/dgs10.csv``: a
    two-column CSV (``date`` YYYY-MM-DD, ``yld10`` pct float) where FRED
    no-print dates are **blank** cells. Blanks coerce to NaN and are dropped,
    then the parsed series is reindexed to a business-day grid with
    ``method="ffill"`` so the value at any cursor ``τ`` is the latest known
    print (look-ahead-safe, exactly like CPI).
    """
    path = source or _DEFAULT_YIELD_PATH
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Default yield file not found at {_DEFAULT_YIELD_PATH}. "
            "Pass source=... to load_yields."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["yld10"])
    raw = df["yld10"].astype(float)
    raw.index = pd.DatetimeIndex(df["date"])
    s: pd.Series = raw[~raw.index.duplicated()].sort_index()
    if s.empty:
        return pd.Series(dtype=float, name="yld10")
    daily: pd.Series = s.reindex(
        pd.date_range(s.index.min(), s.index.max(), freq="D"), method="ffill"
    )
    return daily.rename("yld10")


def _align_to_grid(series: pd.Series, grid: pd.Index) -> pd.Series:
    """Reindex ``series`` onto ``grid`` with forward-fill (cursor-safe)."""
    return series.reindex(grid, method="ffill")


def _log(series: pd.Series) -> pd.Series:
    """Natural log of a strictly-positive series, NaN-guarded for zeros."""
    guarded = series.replace(0.0, np.nan)
    return pd.Series(np.log(guarded), index=series.index, dtype=float)


def zacc(
    cpi_grid: pd.Series,
    w_a: int = 63,
    w_z: int = 252,
) -> pd.Series:
    """Standardised inflation acceleration (2nd diff of log CPI).

    ``lp = log CPI``; ``acc = lp - 2·lp_{t-1} + lp_{t-2}`` (discrete 2nd
    difference). CPI is stepped to daily from monthly so ``acc`` is a
    spike-train — smooth it with a rolling mean (``w_a``) *before* z-scoring.
    High ``zacc`` = cost-push accelerating → risk-off.
    """
    lp = _log(cpi_grid)
    acc = lp - 2.0 * lp.shift(1) + lp.shift(2)
    sm = acc.rolling(w_a, min_periods=max(1, w_a // 2)).mean()
    return _zscore(sm, w_z)


def zreal(
    nominal: pd.Series,
    cpi_grid: pd.Series,
    w_z: int = 252,
) -> pd.Series:
    """Standardised real equity level ``log(P) - log(CPI)``.

    Reuses ``deflated_log_prices`` (already aligned to ``nominal``), then
    z-scores. High ``zreal`` = over-stretched vs purchasing power (ambiguous on
    its own — fused with ``acc``/``rdisc`` in the joint state).
    """
    real = deflated_log_prices(nominal, cpi_grid)
    return _zscore(real, w_z)


def zrdisc(
    yld: pd.Series,
    cpi_grid: pd.Series,
    k: int = 252,
    w_z: int = 252,
) -> pd.Series:
    """Standardised real-discount proxy ``Y - trailing annualised CPI``.

    ``pi_k = 252·(log CPI_t - log CPI_{t-k})/k`` (annualised trailing inflation);
    ``rdisc = Y_t - pi_k``. High ``zrdisc`` = real discount expensive → risk-off.
    """
    yld_g = _align_to_grid(yld, cpi_grid.index)
    lp = _log(cpi_grid)
    pi = _LOG2_DAYS * (lp - lp.shift(k)) / k
    rdisc = yld_g - pi
    return _zscore(rdisc, w_z)


def _zscore(series: pd.Series, w_z: int) -> pd.Series:
    """Rolling z-score of ``series`` (trailing, cursor-safe)."""
    mean = series.rolling(w_z, min_periods=w_z).mean()
    std = series.rolling(w_z, min_periods=w_z).std()
    return (series - mean) / std


def feature_matrix(
    nominal: pd.Series,
    cpi_grid: pd.Series,
    yld: pd.Series | None,
    w_a: int = 63,
    k: int = 252,
    w_z: int = 252,
    with_rdisc: bool = True,
) -> pd.DataFrame:
    """Build the joint standardised feature frame fed to the HMM.

    Columns: ``[z_acc, z_real, (z_rdisc)]``. Drops any row with a NaN (warmup
    of the z-scores / CPI lead-in) so downstream fits receive clean rows,
    matching ``MarketRegimeHMM``'s ``dropna()`` convention. ``nominal`` must be
    a price Series on the backtest candle grid; ``cpi_grid`` a daily-stepped CPI
    level aligned to the same grid.
    """
    grid = nominal.index
    cpi_g = _align_to_grid(cpi_grid, grid)
    real = zreal(nominal, cpi_g, w_z=w_z)
    acc = zacc(cpi_g, w_a=w_a, w_z=w_z)
    cols: dict[str, pd.Series] = {"z_acc": acc, "z_real": real}
    if with_rdisc and yld is not None and not yld.empty:
        cols["z_rdisc"] = zrdisc(_align_to_grid(yld, grid), cpi_g, k=k, w_z=w_z)
    return pd.DataFrame(cols).dropna()


__all__ = [
    "load_yields",
    "zacc",
    "zreal",
    "zrdisc",
    "feature_matrix",
]
