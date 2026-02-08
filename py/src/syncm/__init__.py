from dataclasses import dataclass
from typing import cast, Any, Dict, List
import asyncio
import datetime

import yaml
from src.syncm.ibkr_layer import candles, get_contract_info, search_contract
from src.db.models import get_ohlcv_model


@dataclass
class UniverseConf:
    universe: List[str]
    intervals: List[str]
    start_date: str


async def sync_data(config: UniverseConf):
    tasks = []
    for ticker in config.universe:
        for interval in config.intervals:
            tasks.append(sync_symbol(ticker, interval, config.start_date))

    await asyncio.gather(*tasks, return_exceptions=True)


async def sync_symbol(ticker: str, interval: str, start_date: str):
    try:
        conid = await search_contract(ticker)
        symbol_info = await get_contract_info(conid)
        print(f"Syncing {ticker} ({symbol_info.id}) for {interval}")

        model = get_ohlcv_model(interval)

        # Get last timestamp
        last_record = (
            model.select()
            .where(model.symbol_id == symbol_info.id)
            .order_by(model.timestamp.desc())
            .first()
        )

        if last_record:
            start_dt = datetime.datetime.fromtimestamp(last_record.timestamp / 1000)
            start_time = start_dt.strftime("%Y%m%d-%H:%M:%S")
            print(f"  Found last data at {start_dt}, syncing from there")
        else:
            start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            start_time = start_dt.strftime("%Y%m%d-%H:%M:%S")
            print(f"  No data found, syncing from {start_date}")

        await candles(conid, bar=cast(Any, interval), startTime=start_time)
        print(f"  Synced {ticker} for {interval}")
    except Exception as e:
        print(f"Error syncing {ticker} for {interval}: {e}")


def load_universe_config(file_path: str) -> UniverseConf:
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return UniverseConf(**data)


__all__ = ["sync_data", "load_universe_config"]
