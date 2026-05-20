from src.db.models import SymbolSchema, ISymbol
from dataclasses import dataclass
from typing import cast, Any, Dict, List, Optional
import asyncio
from datetime import date, datetime
import yaml

from src.syncm.ibkr_layer.candles import candles_batch, date_to_timestamp
from src.syncm.ibkr_layer import get_contract_info, lookup


@dataclass
class UniverseConf:
    symbols: List[str]
    from_date: Optional[date] = None


async def _get_symbol_for_ticker(ticker: str) -> ISymbol:
    s = SymbolSchema.get_or_none(SymbolSchema.ticker == ticker)
    if s:
        return s

    print(f"looking up {ticker}")
    contract = await lookup(ticker)
    conid = int(contract.conid or "-")
    symbol_info = await get_contract_info(conid)
    return symbol_info


async def resolve_symbols(tickers: list[str]) -> list[ISymbol]:
    """Resolve ticker strings to ISymbol objects (DB lookup + API fallback).

    Returns only successfully resolved symbols. Failed resolutions are
    logged and skipped.
    """
    semaphore = asyncio.Semaphore(2)

    async def bounded(ticker: str) -> Optional[ISymbol]:
        async with semaphore:
            try:
                return await _get_symbol_for_ticker(ticker)
            except Exception as e:
                print(f"Error resolving {ticker}: {e}")
                return None

    results = await asyncio.gather(*[bounded(t) for t in tickers])
    return [s for s in results if s is not None]


async def _get_candles(symbols: list[ISymbol], from_date: date):
    await candles_batch(
        [symbol.conid for symbol in symbols],
        from_datetime=datetime.combine(from_date, datetime.min.time()),
    )


async def sync_data(tickers: list[str], from_date: date) -> None:
    """Sync historical candle data for the given tickers starting from from_date.

    Args:
        tickers: List of ticker symbols to sync
        from_date: Start date for historical data (required — no default)

    Raises:
        ValueError: If no symbols could be resolved
    """
    symbols = await resolve_symbols(tickers)
    if not symbols:
        raise ValueError(f"No symbols resolved for tickers: {tickers}")
    await _get_candles(symbols, from_date)


def load_universe_config(file_path: str) -> UniverseConf:
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return UniverseConf(**data)


__all__ = ["sync_data", "load_universe_config"]
