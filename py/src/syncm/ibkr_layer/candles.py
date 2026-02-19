from ib_rest_api_client.models import (
    IserverHistoryLastResponse,
    SingleHistoricalBarLast,
    SingleHistoricalBarBidAsk,
    IserverHistoryBidAskResponse,
)
import datetime
import math
import re
from typing import Optional, cast
from src.consts import BAR_INTERVAL
from src.db.models import get_ohlcv_model
from src.db import db

from .shared import client, get_contract_info, auth_client

from ib_rest_api_client.api.trading_market_data.get_iserver_marketdata_history import (
    sync,
    asyncio,
)


# Maximum number of candles per API request
MAX_CANDLES_PER_REQUEST = 1000

# Bar interval to milliseconds mapping
BAR_INTERVAL_MS = {
    "1min": 60 * 1000,
    "5min": 5 * 60 * 1000,
    "15min": 15 * 60 * 1000,
    "30min": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
    "1w": 7 * 24 * 60 * 60 * 1000,
}


async def candles(
    conid: int,
    period: Optional[str] = "1d",
    bar: str = "1h",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
):
    """Fetch candles from IBKR API with chunking support."""

    symbol_info = await get_contract_info(conid)
    print(f"Getting candles for {symbol_info.ticker}")

    model = get_ohlcv_model(bar)

    r = sync(
        client=auth_client,
        conid=conid,
        bar=bar,
        period=cast(str, period),
        start_time=cast(str, start_time),
    )

    if not isinstance(r, IserverHistoryBidAskResponse) or not r.data:
        raise Exception("Unexpected response type")

    data = r.data
    print(f"Inserting {len(data)} candles total")

    insert_data = [
        {
            "symbol_id": symbol_info.id,
            "timestamp": item["t"],
            "open": item["o"],
            "high": item["h"],
            "low": item["l"],
            "close": item["c"],
            "volume": item["v"],
        }
        for item in data
    ]

    try:
        with db.atomic():
            model.insert_many(insert_data).on_conflict(
                conflict_target=(model.timestamp, model.symbol_id),
                update={
                    model.open: model.open,
                    model.high: model.high,
                    model.low: model.low,
                    model.close: model.close,
                    model.volume: model.volume,
                },
            ).execute()
    except Exception as e:
        # Fallback for databases without unique constraint: filter and batch insert
        print(f"Bulk insert failed ({e}), using filtered batch insert")
        # Get existing timestamps for this symbol
        existing_timestamps = {
            row.timestamp
            for row in model.select(model.timestamp).where(
                model.symbol_id == symbol_info.id
            )
        }
        # Filter out existing records
        new_data = [
            item for item in insert_data if item["timestamp"] not in existing_timestamps
        ]
        if new_data:
            # Batch insert in chunks of 1000
            batch_size = 1000
            for i in range(0, len(new_data), batch_size):
                batch = new_data[i : i + batch_size]
                with db.atomic():
                    model.insert_many(batch).execute()
