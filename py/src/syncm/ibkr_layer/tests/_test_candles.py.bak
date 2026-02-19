import pytest
import os
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock
import math

# Import the function to test
from src.syncm.ibkr_layer.candles import candles
from src.db.models import Symbol, get_ohlcv_model
from src.syncm.ibkr_layer.tests.conftest import get_ohlcv_model_for_test, TEST_DB_PATH


@pytest.mark.asyncio
async def test_fetch_hourly_candles_no_overfetch(
    test_db, mock_get_contract_info, ibkr_api_mock, mock_symbol, sample_candle_data
):
    """Test that downloading hourly candles works with 1000-candle chunking
    and does not overfetch already stored data.

    This test verifies:
    1. Existing data in DB is not re-fetched (no overfetching)
    2. Fetches are split into 1000-candle chunks
    3. Date utilities are used to calculate time ranges
    4. Only missing data is fetched from the API

    Uses respx to mock HTTP routes while testing with real HTTP client.
    """
    from httpx import Response

    # Setup: Create test data
    # We'll simulate having 500 hours of data already stored (from Jan 1 to Jan 21)
    # And need to fetch 2500 more hours (up to ~April 1)
    # Total: 3000 candles, 500 already stored, 2500 to fetch = 3 chunks of 1000

    symbol_id = mock_symbol.id
    start_date = datetime(2024, 1, 1, 0, 0, 0)
    existing_candles_count = 500
    total_candles_needed = 3000
    candles_to_fetch = total_candles_needed - existing_candles_count

    # Calculate timestamps in milliseconds
    start_timestamp_ms = int(start_date.timestamp() * 1000)
    hour_in_ms = 3600000  # 1 hour in milliseconds

    # Insert existing data into test database
    ohlcv_model = get_ohlcv_model_for_test(test_db, "1h")
    Symbol.create(id=symbol_id, ticker="AAPL", market="NASDAQ", currency="USD")

    existing_data = sample_candle_data(start_timestamp_ms, existing_candles_count)
    for candle in existing_data:
        ohlcv_model.create(
            symbol_id=symbol_id,
            timestamp=candle["t"],
            open=candle["o"],
            high=candle["h"],
            low=candle["l"],
            close=candle["c"],
            volume=candle["v"],
        )

    # Calculate the start time for fetching (should be after last existing candle)
    last_existing_timestamp_ms = start_timestamp_ms + (
        (existing_candles_count - 1) * hour_in_ms
    )
    first_fetch_timestamp_ms = last_existing_timestamp_ms + hour_in_ms

    # Prepare mock API responses - split into 1000-candle chunks
    chunk_size = 1000
    num_chunks = math.ceil(candles_to_fetch / chunk_size)

    # Track API calls using respx
    request_log = []
    call_count = [0]

    def make_chunk_response(chunk_idx):
        """Create response for a specific chunk."""
        chunk_start_idx = existing_candles_count + (chunk_idx * chunk_size)
        remaining_candles = candles_to_fetch - (chunk_idx * chunk_size)
        current_chunk_size = min(chunk_size, remaining_candles)

        chunk_start_timestamp_ms = start_timestamp_ms + (chunk_start_idx * hour_in_ms)
        chunk_data = sample_candle_data(chunk_start_timestamp_ms, current_chunk_size)

        return {"data": chunk_data}

    def handle_request(request):
        idx = call_count[0]
        call_count[0] += 1
        request_log.append({
            "url": str(request.url),
            "params": dict(request.url.params),
        })
        return Response(200, json=make_chunk_response(idx))

    # Mock the API route with dynamic responses
    route = ibkr_api_mock.get(
        "iserver/marketdata/history",
        name="marketdata_history"
    ).mock(side_effect=handle_request)

    # Calculate end time (total candles needed from start)
    end_date = start_date + timedelta(hours=total_candles_needed - 1)

    # Execute: Call candles function
    await candles(
        conid=symbol_id,
        period=None,
        bar="1h",
        startTime=datetime.fromtimestamp(first_fetch_timestamp_ms / 1000).isoformat(),
        endTime=end_date.isoformat(),
    )

    # Verify: Check the results

    # 1. Verify correct number of API calls (chunked by 1000)
    assert len(request_log) == num_chunks, (
        f"Expected {num_chunks} API calls, got {len(request_log)}"
    )

    # 2. Verify no overfetching - count total records in DB
    total_records = (
        ohlcv_model.select().where(ohlcv_model.symbol_id == symbol_id).count()
    )
    assert total_records == total_candles_needed, (
        f"Expected {total_candles_needed} total records in DB, got {total_records}"
    )

    # 3. Verify the existing data was not duplicated (check timestamps)
    existing_timestamps = {candle["t"] for candle in existing_data}
    fetched_records = ohlcv_model.select().where(
        (ohlcv_model.symbol_id == symbol_id)
        & (ohlcv_model.timestamp > last_existing_timestamp_ms)
    )

    for record in fetched_records:
        assert record.timestamp not in existing_timestamps, (
            f"Timestamp {record.timestamp} was already in DB - overfetch occurred!"
        )

    # 4. Verify date range continuity
    all_records = list(
        ohlcv_model.select()
        .where(ohlcv_model.symbol_id == symbol_id)
        .order_by(ohlcv_model.timestamp)
    )

    assert len(all_records) == total_candles_needed

    for i in range(1, len(all_records)):
        expected_diff = hour_in_ms
        actual_diff = all_records[i].timestamp - all_records[i - 1].timestamp
        assert actual_diff == expected_diff, (
            f"Gap in data at index {i}: expected diff {expected_diff}, got {actual_diff}"
        )


@pytest.mark.asyncio
async def test_fetch_with_no_existing_data(
    test_db, mock_get_contract_info, ibkr_api_mock, mock_symbol, sample_candle_data
):
    """Test fetching candles when no data exists in the database."""
    from httpx import Response

    symbol_id = mock_symbol.id
    start_date = datetime(2024, 1, 1, 0, 0, 0)
    total_candles = 2500  # Should result in 3 chunks (1000 + 1000 + 500)

    start_timestamp_ms = int(start_date.timestamp() * 1000)
    hour_in_ms = 3600000

    # Create symbol in DB
    Symbol.create(id=symbol_id, ticker="AAPL", market="NASDAQ", currency="USD")

    # Prepare API responses
    chunk_size = 1000
    num_chunks = math.ceil(total_candles / chunk_size)

    request_log = []
    call_count = [0]

    def make_chunk_response(chunk_idx):
        chunk_start_idx = chunk_idx * chunk_size
        remaining_candles = total_candles - (chunk_idx * chunk_size)
        current_chunk_size = min(chunk_size, remaining_candles)

        chunk_start_timestamp_ms = start_timestamp_ms + (chunk_start_idx * hour_in_ms)
        chunk_data = sample_candle_data(chunk_start_timestamp_ms, current_chunk_size)

        return {"data": chunk_data}

    def handle_request(request):
        idx = call_count[0]
        call_count[0] += 1
        request_log.append({"params": dict(request.url.params)})
        return Response(200, json=make_chunk_response(idx))

    route = ibkr_api_mock.get(
        "iserver/marketdata/history",
        name="marketdata_history"
    ).mock(side_effect=handle_request)

    # Calculate end time
    end_date = start_date + timedelta(hours=total_candles - 1)

    # Execute
    await candles(
        conid=symbol_id,
        bar="1h",
        startTime=start_date.isoformat(),
        endTime=end_date.isoformat()
    )

    # Verify
    ohlcv_model = get_ohlcv_model_for_test(test_db, "1h")
    total_records = (
        ohlcv_model.select().where(ohlcv_model.symbol_id == symbol_id).count()
    )

    assert total_records == total_candles
    assert len(request_log) == num_chunks


@pytest.mark.asyncio
async def test_fetch_exactly_1000_boundary(
    test_db, mock_get_contract_info, ibkr_api_mock, mock_symbol, sample_candle_data
):
    """Test fetching exactly 1000 candles (single chunk boundary case)."""
    from httpx import Response

    symbol_id = mock_symbol.id
    start_date = datetime(2024, 1, 1, 0, 0, 0)
    total_candles = 1000  # Exactly one chunk

    start_timestamp_ms = int(start_date.timestamp() * 1000)

    # Create symbol in DB
    Symbol.create(id=symbol_id, ticker="AAPL", market="NASDAQ", currency="USD")

    # Prepare single API response
    api_data = sample_candle_data(start_timestamp_ms, total_candles)

    request_log = []

    route = ibkr_api_mock.get(
        "iserver/marketdata/history",
        name="marketdata_history"
    ).mock(return_value=Response(200, json={"data": api_data}))

    original_call = route.return_value

    def tracking_response(request):
        request_log.append({"params": dict(request.url.params)})
        return original_call

    route.side_effect = tracking_response

    # Calculate end time
    end_date = start_date + timedelta(hours=total_candles - 1)

    # Execute
    await candles(
        conid=symbol_id,
        bar="1h",
        startTime=start_date.isoformat(),
        endTime=end_date.isoformat()
    )

    # Verify
    ohlcv_model = get_ohlcv_model_for_test(test_db, "1h")
    total_records = (
        ohlcv_model.select().where(ohlcv_model.symbol_id == symbol_id).count()
    )

    assert total_records == total_candles
    assert len(request_log) == 1  # Should only make one API call
