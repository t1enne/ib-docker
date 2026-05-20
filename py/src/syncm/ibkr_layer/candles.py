import time
import asyncio
from pprint import pprint
from src.db.models import CandleSchema
from peewee import fn
from ib_rest_api_client.models import (
    IserverHistoryLastResponse,
    SingleHistoricalBarLast,
    SingleHistoricalBarBidAsk,
    IserverHistoryBidAskResponse,
)
import math
import re
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Optional, cast
from src.consts import BAR_INTERVAL
from src.db import db

from .shared import client, get_contract_info, auth_client
from .rate_limiter import with_retry

from ib_rest_api_client.api.trading_market_data import get_iserver_marketdata_history


# Maximum number of candles per API request
MAX_CANDLES_PER_REQUEST = 1000


def date_to_timestamp(d: date) -> int:
    return int(datetime.combine(d, datetime.min.time()).timestamp() * 1000)


def timestamp_to_datetime(ts: int) -> datetime:
    return datetime.fromtimestamp(ts / 1000)


def calculate_gaps(
    requested_from: datetime,
    requested_to: datetime,
    oldest_existing: Optional[int],
    newest_existing: Optional[int],
) -> list[tuple[datetime, datetime]]:
    """Calculate gaps between requested date range and existing data.

    Pure function that returns a list of (start, end) tuples representing
    missing date ranges that need to be fetched.

    Args:
        requested_from: Start of the requested datetime range
        requested_to: End of the requested datetime range
        oldest_existing: Oldest timestamp in DB (ms), or None if empty
        newest_existing: Newest timestamp in DB (ms), or None if empty

    Returns:
        List of (start_datetime, end_datetime) tuples for gaps to fetch
    """
    gaps: list[tuple[datetime, datetime]] = []
    if oldest_existing is None and newest_existing is None:
        return [(requested_from, requested_to)]

    if oldest_existing is None:
        oldest_existing = newest_existing

    if newest_existing is None:
        newest_existing = oldest_existing

    existing_from = timestamp_to_datetime(cast(int, oldest_existing))
    existing_to = timestamp_to_datetime(cast(int, newest_existing))

    backward_gap_start = requested_from
    backward_gap_end = existing_from.replace(hour=0, minute=0, second=0, microsecond=0)

    if backward_gap_start < backward_gap_end:
        gaps.append((backward_gap_start, backward_gap_end))

    forward_gap_start = (existing_to).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) + timedelta(days=1)
    forward_gap_end = requested_to

    if forward_gap_start < forward_gap_end:
        gaps.append((forward_gap_start, forward_gap_end))

    return gaps


def get_existing_range_sync(ticker: str) -> tuple[Optional[int], Optional[int]]:
    """Returns (oldest_timestamp, newest_timestamp) for ticker, or (None, None) if empty.

    Synchronous — use get_existing_range() from async contexts to avoid blocking the event loop.
    """
    result = (
        CandleSchema.select(
            fn.MIN(CandleSchema.timestamp).alias("oldest"),
            fn.MAX(CandleSchema.timestamp).alias("newest"),
        )
        .where(CandleSchema.ticker == ticker)
        .first()
    )

    if result is None or result.oldest is None:
        return None, None
    return result.oldest, result.newest


async def get_existing_range(ticker: str) -> tuple[Optional[int], Optional[int]]:
    """Async wrapper around get_existing_range_sync to avoid blocking the event loop."""
    return await asyncio.to_thread(get_existing_range_sync, ticker)


async def _fetch_candles_iterative(
    conid: int,
    ticker: str,
    bar: str,
    from_datetime: datetime,
    to_datetime: datetime,
) -> list[dict]:
    """Iteratively fetch candles from IBKR API for a given datetime range.

    Fetches candles going backwards from to_datetime until from_datetime is reached.
    """
    current_to = to_datetime
    accumulated_data: list[dict] = []

    while current_to > from_datetime:
        datetime_to_fetch = current_to - from_datetime
        days_to_fetch = max(1, datetime_to_fetch.days)
        if days_to_fetch <= 0:
            break

        print(
            f"Getting candles for {ticker} for days {days_to_fetch}, "
            f"from {from_datetime}, to {current_to}"
        )

        r = await get_iserver_marketdata_history.asyncio(
            client=auth_client,
            conid=conid,
            bar=bar,
            period=f"{days_to_fetch}d",
            start_time=current_to.strftime("%Y%m%d-%H:%M:%S"),
        )

        if not isinstance(r, IserverHistoryBidAskResponse) or not r.data:
            print("Unexpected response!")
            pprint(r)
            raise Exception("Unexpected response type")

        sorted_data = sorted(r.data, key=lambda x: x.t)
        if not sorted_data or not sorted_data[0].t:
            print("No more data available")
            break

        oldest_ts = sorted_data[0].t

        accumulated_data = accumulated_data + [
            {
                "conid": conid,
                "ticker": ticker,
                "timestamp": item.t,
                "open": item.o,
                "high": item.h,
                "low": item.l,
                "close": item.c,
                "volume": item.v,
            }
            for item in r.data
        ]

        oldest_datetime = timestamp_to_datetime(oldest_ts)
        print(f"Got {len(r.data)} candles, oldest: {oldest_datetime}")

        if oldest_datetime >= current_to or oldest_datetime <= from_datetime:
            break
        current_to = oldest_datetime

    return accumulated_data


@with_retry(max_retries=3, base_delay_ms=1000, max_delay_ms=10000)
async def candles(
    conid: int,
    from_datetime: datetime,
    to_datetime: Optional[datetime] = None,
    bar: str = "1h",
    force_fetch: bool = False,
    ticker: Optional[str] = None,
):
    """Fetch candles from IBKR API with rate limiting and retry support.

    Args:
        conid: Contract ID
        from_datetime: Start datetime for fetching
        to_datetime: End datetime for fetching (defaults to now)
        bar: Bar size (e.g., "1h", "1d")
        force_fetch: If True, ignore existing data and fetch all (default: False)
        ticker: Optional ticker symbol. If not provided, will fetch from API.
    """
    if to_datetime is None:
        to_datetime = datetime.now()

    if ticker is None:
        symbol_info = await get_contract_info(conid)
        ticker = symbol_info.ticker

    oldest_existing, newest_existing = await get_existing_range(ticker)

    gaps = (
        [(from_datetime, to_datetime)]
        if force_fetch
        else calculate_gaps(
            from_datetime, to_datetime, oldest_existing, newest_existing
        )
    )

    if not gaps:
        print(f"No gaps to fetch for {ticker}: data already exists for full range")
        return

    print(f"Fetching {len(gaps)} gap(s) for {ticker}/{conid}: {gaps}")

    for gap_start, gap_end in gaps:
        insert_data = await _fetch_candles_iterative(
            conid, ticker, bar, gap_start, gap_end
        )

        if insert_data:
            try:
                with db.atomic():
                    CandleSchema.insert_many(insert_data).on_conflict_ignore().execute()
            except Exception as e:
                print(f"Bulk insert failed ({e})")
                raise e


async def candles_batch(
    conids: list[int],
    from_datetime: datetime,
    to_datetime: Optional[datetime] = None,
    max_concurrent: int = 2,
    bar: str = "1h",
) -> list[int]:
    """Fetch candles for multiple conids with rate limiting and batching.

    Args:
        conids: List of conids to fetch candles for
        from_datetime: Start datetime for fetching
        to_datetime: End datetime for fetching (defaults to now)
        max_concurrent: Maximum concurrent requests
        bar: Bar size (e.g., "1h", "1d")

    Returns:
        List of conids that were successfully fetched
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_limit(c: int) -> int | None:
        try:
            async with semaphore:
                await candles(c, from_datetime, to_datetime, bar)
                return c
        except Exception as e:
            print(f"Failed to fetch candles for conid {c}: {e}")
            return None

    results = await asyncio.gather(*[fetch_with_limit(c) for c in conids])
    return [r for r in results if r is not None]
