import datetime
import math
import re
from typing import Optional
from src.consts import BAR_INTERVAL
from src.db.models import get_ohlcv_model
from src.db import db

from .shared import client, get_contract_info


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


def validate_candles_args(conid, period, bar, startTime):
    if not isinstance(conid, int):
        raise ValueError("conid must be int")
    if bar not in BAR_INTERVAL:
        raise ValueError(f"bar must be one of {BAR_INTERVAL}")
    if period is not None:
        if not isinstance(period, str) or not re.match(r"\d+d$", period):
            raise ValueError(r"period must be string matching \d+d$")
    if startTime is not None and not isinstance(startTime, str):
        raise ValueError("startTime must be str or None")


def parse_timestamp(timestamp_str: str) -> datetime.datetime:
    """Parse ISO or IBKR timestamp string to timezone-aware datetime."""
    try:
        dt = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except ValueError:
        dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d-%H:%M:%S")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def format_timestamp_for_api(dt: datetime.datetime) -> str:
    """Format datetime for IBKR API."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc).strftime("%Y%m%d-%H:%M:%S")


def get_bar_interval_ms(bar: str) -> int:
    """Get interval in milliseconds for a bar size."""
    return BAR_INTERVAL_MS.get(bar, 60 * 60 * 1000)  # Default to 1h


async def candles(
    conid: int,
    period: Optional[str] = "252d",
    bar: str = "1d",
    startTime: Optional[str] = None,
    endTime: Optional[str] = None,
):
    """Fetch candles from IBKR API with chunking support.

    Args:
        conid: Contract ID
        period: Period string (e.g., "252d") - used when no startTime and no DB data
        bar: Bar interval (e.g., "1h", "1d")
        startTime: Start time in ISO or IBKR format (YYYYMMDD-HH:MM:SS)
        endTime: End time in ISO or IBKR format (optional, defaults to now if not provided)

    The function will:
    1. Check for existing data in DB to avoid overfetching
    2. Split requests into chunks of MAX_CANDLES_PER_REQUEST
    3. Use date utilities for timestamp calculations
    """
    # Validate
    validate_candles_args(conid, period, bar, startTime)

    symbol_info = await get_contract_info(conid)
    print(f"Getting candles for {symbol_info.ticker}")

    model = get_ohlcv_model(bar)

    # Determine the actual start time
    # If startTime is provided, use it
    # Otherwise, check DB for last record
    if startTime:
        current_start_dt = parse_timestamp(startTime)
    else:
        # Check database for existing data
        last_record = (
            model.select()
            .where(model.symbol_id == symbol_info.id)
            .order_by(model.timestamp.desc())
            .first()
        )

        if last_record:
            # Start from after the last record
            last_ts_ms = last_record.timestamp
            current_start_dt = datetime.datetime.fromtimestamp(
                last_ts_ms / 1000, tz=datetime.timezone.utc
            )
            # Add one interval to avoid re-fetching the last candle
            interval_ms = get_bar_interval_ms(bar)
            current_start_dt += datetime.timedelta(milliseconds=interval_ms)
            print(f"Resuming from {current_start_dt} (after last stored record)")
        else:
            # No existing data, use period (default to 252d if not specified)
            days = 252
            if period:
                days = int(period.replace("d", ""))
            current_start_dt = datetime.datetime.now(
                tz=datetime.timezone.utc
            ) - datetime.timedelta(days=days)

    # Determine end time
    if endTime:
        end_dt = parse_timestamp(endTime)
    else:
        end_dt = datetime.datetime.now(tz=datetime.timezone.utc)

    interval_ms = get_bar_interval_ms(bar)
    day_ms = 24 * 60 * 60 * 1000

    # Calculate total candles needed
    time_diff_ms = int((end_dt - current_start_dt).total_seconds() * 1000)
    if time_diff_ms <= 0:
        print("No new candles to fetch")
        return

    total_candles_needed = (time_diff_ms // interval_ms) + 1

    print(
        f"Fetching {total_candles_needed} candles from {current_start_dt} to {end_dt}"
    )

    # Fetch in chunks (request by endTime + period)
    all_data = []
    chunks_fetched = 0
    current_end_dt = end_dt

    while current_end_dt >= current_start_dt:
        remaining_ms = int((current_end_dt - current_start_dt).total_seconds() * 1000)
        if remaining_ms <= 0:
            break

        chunk_ms = min(remaining_ms, interval_ms * (MAX_CANDLES_PER_REQUEST - 1))
        period_days = max(1, int(math.ceil(chunk_ms / day_ms)))
        period_str = f"{period_days}d"

        end_time_str = format_timestamp_for_api(current_end_dt)

        params = {
            "conid": conid,
            "bar": bar,
            "period": period_str,
            "endTime": end_time_str,
        }

        print(f"Fetching chunk {chunks_fetched + 1}: to {end_time_str} ({period_str})")

        try:
            r = await client.get("iserver/marketdata/history", params=params)
            r.raise_for_status()
            data = r.json().get("data", [])
        except Exception as e:
            print(f"Failed getting marketdata: {e}")
            break

        if not data:
            print("No candles in range, moving earlier")
            current_end_dt -= datetime.timedelta(days=period_days)
            continue

        # Filter by requested overall range
        filtered_data = []
        for item in data:
            item_ts_ms = item["t"]
            item_dt = datetime.datetime.fromtimestamp(
                item_ts_ms / 1000, tz=datetime.timezone.utc
            )
            if item_dt > end_dt:
                break
            if item_dt < current_start_dt:
                continue
            filtered_data.append(item)

        all_data.extend(filtered_data)
        chunks_fetched += 1

        print(f"Retrieved {len(filtered_data)} candles in chunk {chunks_fetched}")

        if data:
            # Move to previous chunk using earliest candle time
            first_candle_ts_ms = data[0]["t"]
            current_end_dt = datetime.datetime.fromtimestamp(
                first_candle_ts_ms / 1000, tz=datetime.timezone.utc
            )
            current_end_dt -= datetime.timedelta(milliseconds=interval_ms)

        # Safety check: don't fetch more than needed
        if len(all_data) >= total_candles_needed:
            break

    if not all_data:
        print("No candles found")
        return

    print(f"Inserting {len(all_data)} candles total")

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
        for item in all_data
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

    print(f"Successfully stored {len(insert_data)} candles in {chunks_fetched} chunks")
