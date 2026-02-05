import asyncio
import datetime
from typing import List
from dataclasses import dataclass

import yaml

from src.trading.candles import candles
from src.trading.shared import get_contract_info, search_contract
from src.db.models import get_ohlcv_model

from src.consts import BAR_INTERVAL


@dataclass
class UniverseConfig:
    universe: List[str]
    intervals: List[str]
    start_date: str

    def __post_init__(self):
        if not self.universe:
            raise ValueError("universe cannot be empty")
        if not self.intervals:
            raise ValueError("intervals cannot be empty")
        for interval in self.intervals:
            if interval not in BAR_INTERVAL:
                raise ValueError(f"Invalid interval: {interval}")
        # Simple date validation
        try:
            datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("start_date must be in YYYY-MM-DD format")


async def sync_data(config: UniverseConfig):
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

        await candles(conid, bar=interval, startTime=start_time)
        print(f"  Synced {ticker} for {interval}")
    except Exception as e:
        print(f"Error syncing {ticker} for {interval}: {e}")


def load_universe_config(file_path: str) -> UniverseConfig:
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return UniverseConfig(**data)
