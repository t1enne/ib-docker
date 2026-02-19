from dataclasses import dataclass
from typing import cast, Any, Dict, List, Optional
import asyncio
import datetime

import yaml
from src.syncm.ibkr_layer import candles, get_contract_info, lookup
from src.db.models import get_ohlcv_model


@dataclass
class UniverseConf:
    universe: List[str]
    intervals: List[str]
    start_date: Optional[datetime.date] = None


async def sync_data(config: UniverseConf):
    tasks = []
    for ticker in config.universe:
        for interval in config.intervals:
            tasks.append(sync_symbol(ticker, interval))

    await asyncio.gather(*tasks, return_exceptions=True)


async def sync_symbol(ticker: str, interval: str):
    try:
        conid = await lookup(ticker)
        symbol_info = await get_contract_info(conid)
        print(f"Syncing {ticker} ({symbol_info.id}) for {interval}")
        model = get_ohlcv_model(interval)
        await candles(conid, bar=cast(Any, interval))
        print(f"  Synced {ticker} for {interval}")
    except Exception as e:
        print(f"Error syncing {ticker} for {interval}: {e}")


def load_universe_config(file_path: str) -> UniverseConf:
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return UniverseConf(**data)


__all__ = ["sync_data", "load_universe_config"]
