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
from datetime import date, datetime, timedelta
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


def date_to_timestamp(d: date) -> int:
    return int(datetime.combine(d, datetime.min.time()).timestamp() * 1000)


def timestamp_to_date(ts: int) -> date:
    return datetime.fromtimestamp(ts / 1000).date()


def calculate_gaps(
    requested_from: date,
    requested_to: date,
    oldest_existing: Optional[int],
    newest_existing: Optional[int],
) -> list[tuple[date, date]]:
    """Calculate gaps between requested date range and existing data.

    Pure function that returns a list of (start, end) tuples representing
    missing date ranges that need to be fetched.

    Args:
        requested_from: Start of the requested date range
        requested_to: End of the requested date range
        oldest_existing: Oldest timestamp in DB (ms), or None if empty
        newest_existing: Newest timestamp in DB (ms), or None if empty

    Returns:
        List of (start_date, end_date) tuples for gaps to fetch
    """
    gaps: list[tuple[date, date]] = []
    if oldest_existing is None and newest_existing is None:
        return [(requested_from, requested_to)]

    if oldest_existing is None:
        oldest_existing = newest_existing

    if newest_existing is None:
        newest_existing = oldest_existing

    existing_from = timestamp_to_date(cast(int, oldest_existing))
    existing_to = timestamp_to_date(cast(int, newest_existing))

    backward_gap_start = requested_from
    backward_gap_end = existing_from + timedelta(days=1)

    if backward_gap_start <= backward_gap_end:
        gaps.append((backward_gap_start, backward_gap_end))

    forward_gap_start = existing_to - timedelta(days=1)
    forward_gap_end = requested_to

    if forward_gap_start <= forward_gap_end:
        gaps.append((forward_gap_start, forward_gap_end))

    return gaps


def get_existing_range_sync(ticker: str) -> tuple[Optional[int], Optional[int]]:
    """Returns (oldest_timestamp, newest_timestamp) for ticker, or (None, None) if empty."""
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


async def _fetch_candles_iterative(
    conid: int,
    bar: str,
    from_date: date,
    to_date: date,
    data: list[dict] = [],
) -> list[dict]:
    """Iteratively fetch candles from IBKR API for a given date range.

    Fetches candles going backwards from to_date until from_date is reached.
    """
    current_to = to_date
    accumulated_data = data

    while current_to > from_date:
        days_to_fetch = (current_to - from_date).days
        if days_to_fetch <= 0:
            break

        symbol_info = await get_contract_info(conid)
        print(
            f"Getting candles for {symbol_info.ticker} for days {days_to_fetch}, "
            f"from {from_date}, to {current_to}"
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

        oldest_date = timestamp_to_date(oldest_ts)
        print(f"Got {len(r.data)} candles, oldest: {oldest_date}")

        if oldest_date >= current_to or oldest_date <= from_date:
            break
        current_to = oldest_date

    return accumulated_data


@with_retry(max_retries=3, base_delay_ms=1000, max_delay_ms=10000)
async def candles(
    conid: int,
    from_date: date,
    to_date: Optional[date] = None,
    bar: str = "1h",
    force_fetch: bool = False,
    ticker: Optional[str] = None,
):
    """Fetch candles from IBKR API with rate limiting and retry support.

    Args:
        conid: Contract ID
        from_date: Start date for fetching
        to_date: End date for fetching (defaults to today)
        bar: Bar size (e.g., "1h", "1d")
        force_fetch: If True, ignore existing data and fetch all (default: False)
        ticker: Optional ticker symbol. If not provided, will fetch from API.
    """
    if to_date is None:
        to_date = date.today()

    if ticker is None:
        symbol_info = await get_contract_info(conid)
        ticker = symbol_info.ticker

    oldest_existing, newest_existing = get_existing_range_sync(ticker)

    gaps = (
        [(from_date, to_date)]
        if force_fetch
        else calculate_gaps(from_date, to_date, oldest_existing, newest_existing)
    )

    if not gaps:
        print(f"No gaps to fetch for {ticker}: data already exists for full range")
        return

    print(f"Fetching {len(gaps)} gap(s) for {ticker}/{conid}: {gaps}")

    async with _candle_limiter:
        for gap_start, gap_end in gaps:
            insert_data = await _fetch_candles_iterative(conid, bar, gap_start, gap_end)

            if insert_data:
                try:
                    with db.atomic():
                        CandleSchema.insert_many(
                            insert_data
                        ).on_conflict_ignore().execute()
                        # .on_conflict(
                        #     conflict_target=(CandleSchema.timestamp),
                        # ).update(
                        #     open=CandleSchema.open,
                        #     high=CandleSchema.high,
                        #     low=CandleSchema.low,
                        #     close=CandleSchema.close,
                        #     volume=CandleSchema.volume,

                except Exception as e:
                    print(f"Bulk insert failed ({e})")
                    raise e


async def candles_batch(
    conids: list[int],
    lookback: int,
    from_date: date,
    to_date: Optional[date] = None,
    max_concurrent: int = 2,
    bar: str = "1h",
) -> list[int]:
    """Fetch candles for multiple conids with rate limiting and batching.

    Args:
        conids: List of conids to fetch candles for
        lookback: Number of days to look back
        from_date: Start date for fetching
        to_date: End date for fetching (defaults to today)
        max_concurrent: Maximum concurrent requests
        bar: Bar size (e.g., "1h", "1d")

    Returns:
        List of conids that were successfully fetched
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_limit(c: int) -> int | None:
        try:
            async with semaphore:
                await candles(c, from_date, to_date, bar)
                return c
        except Exception as e:
            print(f"Failed to fetch candles for conid {c}: {e}")
            return None

    results = await asyncio.gather(*[fetch_with_limit(c) for c in conids])
    return [r for r in results if r is not None]
