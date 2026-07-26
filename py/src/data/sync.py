"""Data sync — historical candle data synchronization from IBKR.

Public API:
  sync_data       — Fetch missing candle data for a list of tickers
  preview_sync    — Dry-run: show what gaps exist without fetching
  load_universe_config — Load and validate a universe.yml file

All functions at the I/O boundary are async. Core gap-calculation
logic (calculate_gaps) is pure — no side effects.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from typing import Optional

import yaml

from src.data.types import SymbolSchema, ISymbol
from src.data.ibkr.candles import (
    candles_batch,
    get_existing_range,
    calculate_gaps,
    find_internal_gaps,
    _merge_and_sort_gaps,
)
from src.data.ibkr import get_contract_info, lookup
from src.data.types import (
    FetchPlan,
    PreviewResult,
    ProgressFn,
    SyncResult,
    UniverseConf,
)

logger = logging.getLogger(__name__)


# ── Symbol resolution ─────────────────────────────────────────


async def _get_symbol_for_ticker(ticker: str) -> ISymbol:
    """Resolve a single ticker string to an ISymbol (DB lookup → API fallback)."""
    s = SymbolSchema.get_or_none(SymbolSchema.ticker == ticker)
    if s:
        return s

    logger.info("looking up %s via API", ticker)
    contract = await lookup(ticker)
    conid = int(contract.conid or "-")
    symbol_info = await get_contract_info(conid)
    return symbol_info


async def resolve_symbols(tickers: list[str]) -> list[ISymbol]:
    """Resolve ticker strings to ISymbol objects (DB lookup + API fallback).

    Returns only successfully resolved symbols. Failed resolutions are
    logged and skipped.
    """
    semaphore = asyncio.Semaphore(1)

    async def bounded(ticker: str) -> Optional[ISymbol]:
        async with semaphore:
            try:
                return await _get_symbol_for_ticker(ticker)
            except Exception as e:
                logger.warning("Error resolving %s: %s", ticker, e)
                return None

    results = await asyncio.gather(*[bounded(t) for t in tickers])
    return [s for s in results if s is not None]


# ── Internal helpers ──────────────────────────────────────────


async def _get_candles(
    symbols: list[ISymbol],
    from_date: date,
    to_date: Optional[date] = None,
    bar: str = "1h",
) -> list[int]:
    """Fetch missing candle data for resolved symbols.

    Returns list of conids that were successfully fetched.
    Errors propagate — no silent swallowing.
    """
    from_dt = datetime.combine(from_date, datetime.min.time())
    to_dt = datetime.combine(to_date, datetime.max.time()) if to_date else None

    return await candles_batch(
        [symbol.conid for symbol in symbols],
        from_datetime=from_dt,
        to_datetime=to_dt,
        bar=bar,
    )


# ── Public API ────────────────────────────────────────────────


async def sync_data(
    tickers: list[str],
    from_date: date,
    to_date: Optional[date] = None,
    bar: str = "1h",
    on_progress: Optional[ProgressFn] = None,
) -> SyncResult:
    """Sync historical candle data for the given tickers.

    Args:
        tickers: List of ticker symbols to sync
        from_date: Start date for historical data (required)
        to_date: End date for historical data (defaults to now)
        bar: Bar size (e.g. "1h", "1d")
        on_progress: Optional progress callback (I/O boundary)

    Returns:
        SyncResult with resolved count, fetched conids, and gap count.

    Raises:
        ValueError: If no symbols could be resolved
    """
    if on_progress:
        on_progress("Resolving symbols...", 0, len(tickers))

    symbols = await resolve_symbols(tickers)

    if on_progress:
        on_progress(
            f"Resolved {len(symbols)}/{len(tickers)} symbols",
            len(symbols),
            len(tickers),
        )

    if not symbols:
        raise ValueError(f"No symbols resolved for tickers: {tickers}")

    if on_progress:
        on_progress("Fetching candles...", 0, len(symbols))

    fetched = await _get_candles(symbols, from_date, to_date, bar)

    if on_progress:
        on_progress("Sync complete", len(symbols), len(symbols))

    return SyncResult(
        resolved=len(symbols),
        fetched=fetched,
        gaps_found=len(fetched),
    )


async def preview_sync(
    tickers: list[str],
    from_date: date,
    to_date: Optional[date] = None,
) -> PreviewResult:
    """Dry-run: show what gaps exist without making any API fetch calls.

    Resolves symbols, checks existing DB data, calculates gap plans.
    No candles are fetched — this is a pure preview.

    Args:
        tickers: List of ticker symbols to preview
        from_date: Start date for gap analysis
        to_date: End date for gap analysis (defaults to now)

    Returns:
        PreviewResult with per-symbol FetchPlans and total gap count.
    """
    symbols = await resolve_symbols(tickers)

    to_dt = (
        datetime.combine(to_date, datetime.max.time()) if to_date else datetime.now()
    )
    from_dt = datetime.combine(from_date, datetime.min.time())

    plans: list[FetchPlan] = []
    for symbol in symbols:
        oldest, newest = await get_existing_range(symbol.ticker)
        edge_gaps = calculate_gaps(from_dt, to_dt, oldest, newest)
        internal_gaps = await asyncio.to_thread(
            find_internal_gaps, symbol.ticker, from_dt, to_dt
        )
        all_gaps = _merge_and_sort_gaps(edge_gaps + internal_gaps)
        plans.append(
            FetchPlan(
                ticker=symbol.ticker,
                conid=symbol.conid,
                gaps=all_gaps,
            )
        )

    total_gaps = sum(len(p.gaps) for p in plans)
    return PreviewResult(
        resolved=len(symbols),
        total_gaps=total_gaps,
        plans=plans,
    )


def load_universe_config(file_path: str) -> UniverseConf:
    """Load and validate a universe configuration YAML file.

    Validates:
      - File exists and is readable
      - Content is a YAML dict (not list, scalar, or empty)
      - Required 'symbols' key is present and non-empty

    Args:
        file_path: Path to the universe.yml file

    Returns:
        Frozen UniverseConf with validated fields.

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the content is malformed or missing required fields
    """
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Universe config not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {file_path}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid universe config: expected a mapping (dict), "
            f"got {type(data).__name__}"
        )

    if "symbols" not in data:
        raise ValueError(
            "Universe config missing required 'symbols' field. "
            "Expected format:\n"
            "  symbols:\n"
            "    - AAPL\n"
            "    - MSFT\n"
            "  from_date: '2024-01-01'  # optional\n"
            "  to_date: '2024-12-31'    # optional\n"
            "  bar: '1h'                # optional, default: 1h"
        )

    symbols = data.get("symbols", [])
    if not isinstance(symbols, list) or len(symbols) == 0:
        raise ValueError(
            "Universe config 'symbols' must be a non-empty list of tickers"
        )

    return UniverseConf(
        symbols=symbols,
        from_date=data.get("from_date"),
        to_date=data.get("to_date"),
        bar=data.get("bar", "1h"),
    )


# Re-export for convenience
__all__ = [
    "sync_data",
    "preview_sync",
    "load_universe_config",
    "SyncResult",
    "PreviewResult",
    "UniverseConf",
]
