import asyncio
from pprint import pprint
from src.db.models import CandleSchema
from ib_rest_api_client.models import (
    IserverHistoryLastResponse,
    SingleHistoricalBarLast,
    SingleHistoricalBarBidAsk,
    IserverHistoryBidAskResponse,
)
import math
import re
import pandas as pd
from datetime import date, datetime
from typing import Optional, cast
from src.consts import BAR_INTERVAL
from src.db import db

from .shared import client, get_contract_info, auth_client
from .rate_limiter import RateLimiter, RateLimitConfig, with_retry, batch_items

from ib_rest_api_client.api.trading_market_data import get_iserver_marketdata_history


# Maximum number of candles per API request
MAX_CANDLES_PER_REQUEST = 1000

# Default rate limiter config for candles
DEFAULT_CANDLE_RATE_LIMIT = RateLimitConfig(
    max_concurrent=2,
    min_delay_ms=200,
    max_retries=3,
    base_delay_ms=500,
    max_delay_ms=10000,
)

_candle_limiter = RateLimiter(DEFAULT_CANDLE_RATE_LIMIT)


async def _fetch_candles(
    conid: int,
    bar: str,
    from_date: date,
    to_date: date = date.today(),
    data: list[dict] = [],
) -> list[dict]:
    """Internal function to fetch candles from IBKR API. It fetches {days_left} worth of candles starting from from_date going backwards"""
    days_left = (to_date - from_date).days
    if days_left <= 0:
        return data

    symbol_info = await get_contract_info(conid)
    print(
        f"Getting candles for {symbol_info.ticker} for days {days_left}, from {from_date}, to {to_date}"
    )

    r = get_iserver_marketdata_history.sync(
        client=auth_client,
        conid=conid,
        bar=bar,
        period=f"{days_left}d",
        start_time=to_date.strftime("%Y%m%d-%H:%M:%S"),
    )

    if not isinstance(r, IserverHistoryBidAskResponse) or not r.data:
        print("Unexpected!")
        pprint(r)
        raise Exception("Unexpected response type")

    sorted_data = sorted(r.data, key=lambda x: x.t)
    oldest_ts = sorted_data[0].t
    if not oldest_ts:
        print("Unexpected!")
        raise ValueError("Empty timestamp")

    new_data = data + [
        {
            "conid": symbol_info.conid,
            "ticker": symbol_info.ticker,
            "timestamp": item.t,
            "open": item.o,
            "high": item.h,
            "low": item.l,
            "close": item.c,
            "volume": item.v,
        }
        for item in r.data
    ]

    oldest_date = datetime.fromtimestamp(oldest_ts / 1000).date()
    print(f"Got {len(r.data)} candles, oldest: {oldest_date}")

    return await _fetch_candles(
        conid=conid,
        from_date=from_date,
        to_date=oldest_date,
        bar=bar,
        data=new_data,
    )


@with_retry(max_retries=3, base_delay_ms=1000, max_delay_ms=10000)
async def candles(
    conid: int,
    from_date: date,
    bar: str = "1h",
):
    """Fetch candles from IBKR API with rate limiting and retry support."""
    async with _candle_limiter:
        insert_data = await _fetch_candles(conid, bar, from_date, date.today())
    try:
        with db.atomic():
            CandleSchema.insert_many(insert_data).on_conflict_ignore().execute()
    except Exception as e:
        print(f"Bulk insert failed ({e}), using filtered batch insert")
        raise e


async def candles_batch(
    conids: list[int],
    lookback: int,
    from_date: date,
    max_concurrent: int = 2,
    bar: str = "1h",
) -> list[int]:
    """Fetch candles for multiple conids with rate limiting and batching.

    Args:
        conids: List of conids to fetch candles for
        period: Time period (e.g., "1d", "1w", "1y")
        bar: Bar size (e.g., "1h", "1d")
        start_time: Optional start date
        max_concurrent: Maximum concurrent requests

    Returns:
        List of conids that were successfully fetched
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_limit(c: int) -> int | None:
        try:
            async with semaphore:
                await candles(c, from_date, bar)
                return c
        except Exception as e:
            print(f"Failed to fetch candles for conid {c}: {e}")
            return None

    results = await asyncio.gather(*[fetch_with_limit(c) for c in conids])
    return [r for r in results if r is not None]
