"""Macro asset indicators.

The **only** public API is :func:`init_macro_indicator`. It returns a
lookahead-free callable bound to one macro series:

    from src.indicators.macro import init_macro_indicator

    cpi  = init_macro_indicator("cpi")     # loads once at init
    v    = cpi(ts)                         # latest CPI level known at/before ts

    gdp = init_macro_indicator("gdpc1", series=my_preloaded_series)
    g   = gdp(ts)

Usage:

- ``init_macro_indicator("cpi")``   — World Bank chained CPI price-index level.
- ``init_macro_indicator("<name>")`` — FRED series read from ``assets/<name>.csv``.
  Supported names: ``gdpc1``, ``gdp`` (GDP); ``payems``, ``unrate``
  (employment); ``indpro``, ``tcu`` (manufacturing / production);
  ``bopgstb`` (balance of trade).

Each callable is lookahead-free: at a cursor ``ts`` it returns the latest print
with a release date ``<= ts`` and **never** a future observation. The asset
file is read once at init (or a pre-loaded Series is bound directly); repeated
calls are just ``Series.asof`` — no disk I/O per call.

``deflated_log_prices`` is a separate pure, vectorised helper (``ln(nominal) -
ln(cpi)``) used by the RDD strategy; it is not part of the cursor-indicator API.
"""

from __future__ import annotations

from src.indicators.macro._shared import (
    deflated_log_prices,
    init_macro_indicator,
)

__all__ = ["init_macro_indicator", "deflated_log_prices"]
