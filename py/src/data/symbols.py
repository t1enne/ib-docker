"""Symbol resolution and universe configuration loading.

Shared primitives used by the `data dl` and `data preview` commands.
These used to live in the monolithic sync module; they are split out so
each command module imports only what it needs.

This module is I/O-bound (DB lookups + IBKR API fallback) but keeps the
per-ticker logic private. The public surface:

  resolve_symbols       — ticker strings → resolved ISymbol objects
  load_universe_config  — load/validate a universe .json file
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from src.data.ibkr import get_contract_info, lookup
from src.data.types import ISymbol, SymbolSchema, UniverseConf

logger = logging.getLogger(__name__)


# ── Per-ticker resolution ──────────────────────────────────────


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


# ── Universe configuration ─────────────────────────────────────


def load_universe_config(file_path: str) -> UniverseConf:
    """Load and validate a universe configuration JSON file.

    Validates:
      - File exists and is readable
      - Content is a JSON object (not array, scalar, or empty)
      - Required 'symbols' key is present and non-empty

    Args:
        file_path: Path to the universe .json file

    Returns:
        Frozen UniverseConf with validated fields.

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the content is malformed or missing required fields
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Universe config not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid universe config: expected a JSON object, "
            f"got {type(data).__name__}"
        )

    if "symbols" not in data:
        raise ValueError(
            "Universe config missing required 'symbols' field. "
            "Expected format:\n"
            '  { "symbols": ["AAPL", "MSFT"] }\n'
        )

    symbols = data.get("symbols", [])
    if not isinstance(symbols, list) or len(symbols) == 0:
        raise ValueError(
            "Universe config 'symbols' must be a non-empty list of tickers"
        )

    return UniverseConf(
        symbols=[sym.upper() for sym in symbols],
        from_date=data.get("from_date"),
        to_date=data.get("to_date"),
        bar=data.get("bar", "1h"),
    )
