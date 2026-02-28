from src.utils import get_days_from_now
from src.db.models import SymbolSchema, ISymbol
from dataclasses import dataclass
from typing import cast, Any, Dict, List, Optional
import asyncio
from datetime import date, datetime
import yaml

from src.syncm.ibkr_layer.candles import candles_batch
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


def get_symbols(tickers: list[str]):
    semaphore = asyncio.Semaphore(2)

    async def bounded_resolve(ticker: str) -> Optional[ISymbol]:
        async with semaphore:
            try:
                return await _get_symbol_for_ticker(ticker)
            except Exception as e:
                print(f"Error resolving {ticker}: {e}")
                return None

    return [bounded_resolve(ticker) for ticker in tickers]


async def _get_candles(symbols: list[ISymbol], from_date: date):
    try:
        await candles_batch(
            [symbol.conid for symbol in symbols],
            lookback=get_days_from_now(from_date),
            from_date=from_date,
        )
    except Exception as e:
        print(f"Error syncing {symbols}: {e}")


async def sync_data(tickers: list[str], from_date: Optional[date] = None):
    from_date = from_date or date.today()
    _symbols: list[ISymbol] = await asyncio.gather(*get_symbols(tickers))
    symbols = [s for s in _symbols if s is not None]
    await _get_candles(symbols, from_date)


def load_universe_config(file_path: str) -> UniverseConf:
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return UniverseConf(**data)


__all__ = ["sync_data", "load_universe_config", "get_symbols"]
