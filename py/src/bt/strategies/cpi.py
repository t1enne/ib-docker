"""CPI price-index loader and deflator for the inflation mean-reversion strategy.

Source: ``assets/cpi.csv`` — the World Bank "CPI, consumer prices" indicator:
one row per (Country, Country Code, Year) storing the **annual inflation rate
in percent** (e.g. United States, 2021 → 4.6979).

A deflator needs a *price-index level*, so the annual rates are chained into a
cumulative index (base 1.0 at the earliest year) and stepped onto a daily grid
(each year's rate applies across that full year). Because the file ends in
2024 while market data runs into 2026, the last observed rate is forward-filled
over the residual window.

Pure functions here; the file is only read at the I/O boundary (lazily, and
cached) via :func:`load_cpi_price_index`.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

_DEFAULT_PATH = Path("assets/cpi.csv")
_USA = "USA"


def load_cpi_price_index(
    source: str | Path | BinaryIO | None = None,
    country: str = _USA,
) -> pd.Series:
    """Return a daily cumulative CPI price-index level series (base 1.0)."""
    df = _read_frame(source)
    sub = df[df["country_code"].astype(str).str.upper() == country.upper()]
    if sub.empty:
        raise ValueError(f"No CPI rows for country code {country!r} in source")
    sub = sub.sort_values("year")

    years = sub["year"].astype(int).to_numpy()
    rates = sub["cpi"].astype(float).to_numpy() / 100.0
    levels = np.concatenate(([1.0], np.cumprod(1.0 + rates)))
    idx_years = np.concatenate(([years[0]], years + 1))

    annual = pd.Series(
        levels, index=[pd.Timestamp(int(y), 1, 1) for y in idx_years], name="cpi"
    )
    end = annual.index.max()
    daily = annual.reindex(
        pd.date_range(annual.index.min(), end + pd.Timedelta(days=366), freq="D"),
        method="ffill",
    )
    return daily


def deflated_log_prices(
    nominal: pd.Series,
    cpi_index: pd.Series,
) -> pd.Series:
    """real_log = ln(nominal) - ln(cpi), reindexed to ``nominal`` and FF'd.

    Equals the log of real (inflation-adjusted) prices. No re-normalization to
    the nominal level is applied here because the strategy only cares about the
    *z-score* of the series, which is invariant to a constant level shift (a
    constant in log space shifts mean and series equally, leaving z unchanged).
    """
    cpi = cpi_index.reindex(nominal.index, method="ffill")
    log_cpi = pd.Series(
        np.log(cpi.replace(0.0, np.nan)), index=nominal.index, name="lc"
    )
    log_px = pd.Series(
        np.log(nominal.replace(0.0, np.nan)), index=nominal.index, name="lp"
    )
    aligned = pd.concat([log_px, log_cpi], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float, name="real_log")
    return (aligned["lp"] - aligned["lc"]).rename("real_log")


def _read_frame(source: str | Path | BinaryIO | None) -> pd.DataFrame:
    if source is None:
        if not Path(_DEFAULT_PATH).exists():
            raise FileNotFoundError(
                f"Default CPI file not found at {_DEFAULT_PATH}. "
                "Pass source=... to load_cpi_price_index."
            )
        df = pd.read_csv(_DEFAULT_PATH)
    else:
        df = pd.read_csv(source)  # type: ignore[arg-type]
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


__all__ = ["load_cpi_price_index", "deflated_log_prices"]
