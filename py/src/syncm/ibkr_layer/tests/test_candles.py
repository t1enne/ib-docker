import pytest
from datetime import datetime, timedelta
from ib_rest_api_client.models import IserverHistoryBidAskResponse, SingleHistoricalBarBidAsk
from src.syncm.ibkr_layer.candles import (
    calculate_gaps,
    date_to_timestamp,
    get_existing_range_sync,
    get_existing_range,
    candles,
    _fetch_candles_iterative,
    MAX_PERIOD_DAYS,
)


def ts(dt: datetime) -> int:
    """Convert a datetime to milliseconds timestamp (preserves intra-day time)."""
    return int(dt.timestamp() * 1000)


class TestCalculateGaps:
    """Test the calculate_gaps pure function - no mocking needed"""

    def test_empty_db_returns_full_range(self):
        """When DB is empty, return the full requested range as a single gap"""
        result = calculate_gaps(
            requested_from=datetime(2024, 1, 1),
            requested_to=datetime(2024, 12, 31),
            oldest_existing=None,
            newest_existing=None,
        )

        assert result == [(datetime(2024, 1, 1), datetime(2024, 12, 31))]

    def test_no_gaps_when_existing_covers_full_range(self):
        """When existing data covers the full requested range, return empty list"""
        oldest = date_to_timestamp(datetime(2024, 1, 1))
        newest = date_to_timestamp(datetime(2024, 12, 31))

        result = calculate_gaps(
            requested_from=datetime(2024, 1, 1),
            requested_to=datetime(2024, 12, 31),
            oldest_existing=oldest,
            newest_existing=newest,
        )

        assert result == []

    def test_backward_gap_only(self):
        """When existing data is newer than requested, return backward gap"""
        oldest = date_to_timestamp(datetime(2025, 6, 1))
        newest = date_to_timestamp(datetime(2025, 6, 30))

        result = calculate_gaps(
            requested_from=datetime(2024, 1, 1),
            requested_to=datetime(2025, 3, 1),
            oldest_existing=oldest,
            newest_existing=newest,
        )

        # Backward gap ends at the oldest candle timestamp (exact), not midnight
        assert result == [(datetime(2024, 1, 1), datetime(2025, 6, 1))]

    def test_forward_gap_only(self):
        """When existing data is older than requested, return forward gap"""
        oldest = date_to_timestamp(datetime(2024, 1, 1))
        newest = date_to_timestamp(datetime(2024, 6, 30))

        result = calculate_gaps(
            requested_from=datetime(2024, 1, 1),
            requested_to=datetime(2025, 3, 1),
            oldest_existing=oldest,
            newest_existing=newest,
        )

        # Forward gap starts at the newest candle timestamp (exact), not next midnight
        forward_gap = result[0]
        assert forward_gap[0] == datetime(2024, 6, 30)
        assert forward_gap[1] == datetime(2025, 3, 1)

    def test_both_gaps(self):
        """When existing data is in the middle, return both backward and forward gaps"""
        oldest = ts(datetime(2024, 6, 1, 10, 0, 0))  # 10:00 AM
        newest = ts(datetime(2024, 9, 30, 14, 0, 0))  # 2:00 PM

        result = calculate_gaps(
            requested_from=datetime(2024, 1, 1),
            requested_to=datetime(2024, 12, 31),
            oldest_existing=oldest,
            newest_existing=newest,
        )

        assert len(result) == 2
        # Backward: everything before the OLDEST existing candle (exact timestamp)
        assert result[0] == (datetime(2024, 1, 1), datetime(2024, 6, 1, 10, 0, 0))
        # Forward: everything after the NEWEST existing candle (exact timestamp)
        assert result[1] == (datetime(2024, 9, 30, 14, 0, 0), datetime(2024, 12, 31))

    def test_existing_only_oldest(self):
        """When only oldest exists (newest is None), handle correctly"""
        oldest = ts(datetime(2024, 6, 1, 10, 0, 0))

        result = calculate_gaps(
            requested_from=datetime(2024, 1, 1),
            requested_to=datetime(2024, 12, 31),
            oldest_existing=oldest,
            newest_existing=None,
        )

        # Single row: both oldest and newest point to the same timestamp
        assert len(result) == 2
        assert result[0] == (datetime(2024, 1, 1), datetime(2024, 6, 1, 10, 0, 0))
        assert result[1] == (datetime(2024, 6, 1, 10, 0, 0), datetime(2024, 12, 31))

    def test_existing_only_newest(self):
        """When only newest exists (oldest is None), handle correctly"""
        newest = ts(datetime(2024, 6, 30, 14, 0, 0))

        result = calculate_gaps(
            requested_from=datetime(2024, 1, 1),
            requested_to=datetime(2024, 12, 31),
            oldest_existing=None,
            newest_existing=newest,
        )

        # Single row: both oldest and newest point to the same timestamp
        assert len(result) == 2
        assert result[0] == (datetime(2024, 1, 1), datetime(2024, 6, 30, 14, 0, 0))
        assert result[1] == (datetime(2024, 6, 30, 14, 0, 0), datetime(2024, 12, 31))

    def test_existing_touches_requested_boundary(self):
        """When existing data touches the requested boundary, no gap for that side"""
        oldest = date_to_timestamp(datetime(2024, 1, 1))  # exact boundary (date → ms)
        newest = ts(datetime(2024, 12, 30, 14, 0, 0))

        result = calculate_gaps(
            requested_from=datetime(2024, 1, 1),
            requested_to=datetime(2024, 12, 31),
            oldest_existing=oldest,
            newest_existing=newest,
        )

        # oldest == requested_from, so no backward gap
        assert len(result) == 1
        assert result[0] == (datetime(2024, 12, 30, 14, 0, 0), datetime(2024, 12, 31))

    def test_existing_fully_covers_within_day(self):
        """When existing data sits within a day, gaps are at exact timestamps"""
        oldest = ts(datetime(2024, 6, 1, 10, 0, 0))
        newest = ts(datetime(2024, 6, 1, 14, 0, 0))

        result = calculate_gaps(
            requested_from=datetime(2024, 1, 1),
            requested_to=datetime(2024, 12, 31),
            oldest_existing=oldest,
            newest_existing=newest,
        )

        # Both gaps should be present, with exact timestamps
        assert len(result) == 2
        assert result[0] == (datetime(2024, 1, 1), datetime(2024, 6, 1, 10, 0, 0))
        assert result[1] == (datetime(2024, 6, 1, 14, 0, 0), datetime(2024, 12, 31))


class TestGetExistingRangeSync:
    """Test the get_existing_range_sync helper function - pure, no mocking needed"""

    def test_returns_none_for_empty_db(self, mocker):
        """Returns (None, None) when no data exists"""
        mock_select = mocker.patch("src.syncm.ibkr_layer.candles.CandleSchema.select")
        mock_select.return_value.where.return_value.first.return_value = None

        result = get_existing_range_sync("AAPL")

        assert result == (None, None)

    def test_returns_timestamps(self, mocker):
        """Returns (oldest, newest) timestamps when data exists"""
        mock_select = mocker.patch("src.syncm.ibkr_layer.candles.CandleSchema.select")
        mock_result = mocker.MagicMock()
        mock_result.oldest = 1704067200000
        mock_result.newest = 1735689600000
        mock_select.return_value.where.return_value.first.return_value = mock_result

        result = get_existing_range_sync("AAPL")

        assert result == (1704067200000, 1735689600000)


class TestNoOverfetch:
    """Test that we don't overfetch when data already exists"""

    @pytest.fixture(autouse=True)
    def _mock_internal_gaps(self, mocker):
        """Mock find_internal_gaps to avoid DB queries in unit tests."""
        mocker.patch("src.syncm.ibkr_layer.candles.find_internal_gaps", return_value=[])

    @pytest.mark.asyncio
    async def test_skip_fetch_when_data_exists(self, mocker):
        """When existing data covers the requested range, skip API call"""
        oldest_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        newest_ts = int(datetime(2024, 12, 31).timestamp() * 1000)

        mocker.patch(
            "src.syncm.ibkr_layer.candles.get_existing_range",
            return_value=(oldest_ts, newest_ts),
        )
        spy = mocker.patch(
            "src.syncm.ibkr_layer.candles._fetch_candles_iterative", return_value=[]
        )

        await candles(123, datetime(2024, 1, 1), datetime(2024, 12, 31), ticker="AAPL")

        spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_when_data_missing(self, mocker):
        """When requested range is beyond existing data, fetch from API"""
        oldest_ts = int(datetime(2024, 6, 1).timestamp() * 1000)
        newest_ts = int(datetime(2024, 6, 30).timestamp() * 1000)

        mocker.patch(
            "src.syncm.ibkr_layer.candles.get_existing_range",
            return_value=(oldest_ts, newest_ts),
        )
        spy = mocker.patch(
            "src.syncm.ibkr_layer.candles._fetch_candles_iterative", return_value=[]
        )

        await candles(123, datetime(2024, 1, 1), ticker="AAPL")

        spy.assert_called()

    @pytest.mark.asyncio
    async def test_force_fetch_ignores_existing(self, mocker):
        """When force_fetch=True, always fetch regardless of existing data"""
        oldest_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        newest_ts = int(datetime(2024, 12, 31).timestamp() * 1000)

        mocker.patch(
            "src.syncm.ibkr_layer.candles.get_existing_range",
            return_value=(oldest_ts, newest_ts),
        )
        spy = mocker.patch(
            "src.syncm.ibkr_layer.candles._fetch_candles_iterative", return_value=[]
        )

        await candles(123, datetime(2024, 1, 1), ticker="AAPL", force_fetch=True)

        spy.assert_called()

    @pytest.mark.asyncio
    async def test_fetch_empty_db(self, mocker):
        """When DB is empty, fetch full range"""
        mocker.patch(
            "src.syncm.ibkr_layer.candles.get_existing_range",
            return_value=(None, None),
        )
        spy = mocker.patch(
            "src.syncm.ibkr_layer.candles._fetch_candles_iterative", return_value=[]
        )

        await candles(123, datetime(2024, 1, 1), ticker="AAPL")

        spy.assert_called()

    @pytest.mark.asyncio
    async def test_fetches_with_correct_gap_params(self, mocker):
        """Verify _fetch_candles_iterative is called with correct gap parameters"""
        mocker.patch(
            "src.syncm.ibkr_layer.candles.get_existing_range",
            return_value=(None, None),
        )
        spy = mocker.patch(
            "src.syncm.ibkr_layer.candles._fetch_candles_iterative", return_value=[]
        )

        await candles(
            123, datetime(2024, 1, 1), datetime(2024, 12, 31), bar="1h", ticker="AAPL"
        )

        spy.assert_called_once_with(
            123, "AAPL", "1h", datetime(2024, 1, 1), datetime(2024, 12, 31)
        )

    @pytest.mark.asyncio
    async def test_fetches_both_gaps(self, mocker):
        """When there are gaps on both ends, fetch both"""
        oldest_ts = int(datetime(2024, 6, 1, 10, 0, 0).timestamp() * 1000)
        newest_ts = int(datetime(2024, 9, 30, 14, 0, 0).timestamp() * 1000)

        mocker.patch(
            "src.syncm.ibkr_layer.candles.get_existing_range",
            return_value=(oldest_ts, newest_ts),
        )
        spy = mocker.patch(
            "src.syncm.ibkr_layer.candles._fetch_candles_iterative", return_value=[]
        )

        await candles(
            123, datetime(2024, 1, 1), datetime(2024, 12, 31), bar="1h", ticker="AAPL"
        )

        assert spy.call_count == 2
        # Backward gap: from requested_from to oldest existing (exact)
        spy.assert_any_call(
            123, "AAPL", "1h", datetime(2024, 1, 1), datetime(2024, 6, 1, 10, 0, 0)
        )
        # Forward gap: from newest existing (exact) to requested_to
        spy.assert_any_call(
            123, "AAPL", "1h", datetime(2024, 9, 30, 14, 0, 0), datetime(2024, 12, 31)
        )


class TestFetchCandlesIterative:
    """Test _fetch_candles_iterative chunking and period capping."""

    @pytest.fixture(autouse=True)
    def _patch_auth_client(self, mocker):
        """Prevent auth_client dependency from raising."""
        mocker.patch(
            "src.syncm.ibkr_layer.candles.auth_client",
            mocker.MagicMock(),
        )

    def _make_bar(self, timestamp: int) -> SingleHistoricalBarBidAsk:
        """Create a single mock bar with sensible OHLC values."""
        return SingleHistoricalBarBidAsk(
            o=100.0,
            c=101.0,
            h=102.0,
            l=99.0,
            v=1000,
            t=timestamp,
        )

    def _make_response(self, bars: list[SingleHistoricalBarBidAsk]) -> IserverHistoryBidAskResponse:
        """Wrap bars in an IserverHistoryBidAskResponse."""
        return IserverHistoryBidAskResponse(data=bars)

    @pytest.mark.asyncio
    async def test_short_range_single_call(self, mocker):
        """Range within MAX_PERIOD_DAYS -> single API call with correct period."""
        to_dt = datetime(2024, 6, 30, 16, 0, 0)
        from_dt = datetime(2024, 1, 1, 9, 30, 0)
        # ~180 days, well under 365

        bar_timestamps = [1704067200000]  # single bar at oldest edge
        mock_response = self._make_response([self._make_bar(t) for t in bar_timestamps])

        spy = mocker.patch(
            "src.syncm.ibkr_layer.candles.get_iserver_marketdata_history.asyncio",
            return_value=mock_response,
        )

        result = await _fetch_candles_iterative(
            conid=123,
            ticker="AAPL",
            bar="1h",
            from_datetime=from_dt,
            to_datetime=to_dt,
        )

        spy.assert_called_once()
        _, kwargs = spy.call_args
        # ~180 days between Jan 1 09:30 and Jun 30 16:00 = 181 days + partial time
        assert kwargs["period"] == "182d"
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_long_range_triggers_chunking(self, mocker):
        """Range > MAX_PERIOD_DAYS -> two API calls (capped + remainder)."""
        to_dt = datetime(2025, 6, 15, 16, 0, 0)
        from_dt = datetime(2024, 1, 1, 9, 30, 0)
        # ~531 days > 365

        # First batch: oldest is 365 days back from to_dt
        first_oldest_ts = int((to_dt - timedelta(days=365)).timestamp() * 1000)
        # Second batch: oldest is at from_dt
        second_oldest_ts = int(from_dt.timestamp() * 1000)

        first_response = self._make_response([self._make_bar(first_oldest_ts)])
        second_response = self._make_response([self._make_bar(second_oldest_ts)])

        spy = mocker.patch(
            "src.syncm.ibkr_layer.candles.get_iserver_marketdata_history.asyncio",
            side_effect=[first_response, second_response],
        )

        result = await _fetch_candles_iterative(
            conid=123,
            ticker="AAPL",
            bar="1h",
            from_datetime=from_dt,
            to_datetime=to_dt,
        )

        assert spy.call_count == 2

        # First call: capped at MAX_PERIOD_DAYS
        call1_kwargs = spy.call_args_list[0].kwargs
        assert call1_kwargs["period"] == f"{MAX_PERIOD_DAYS}d"
        assert call1_kwargs["start_time"] == to_dt.strftime("%Y%m%d-%H:%M:%S")

        # Second call: uses remaining days
        call2_kwargs = spy.call_args_list[1].kwargs
        expected_remaining = (to_dt - from_dt).days - MAX_PERIOD_DAYS  # ~166
        assert call2_kwargs["period"] == f"{expected_remaining + 1}d"
        # start_time should be first_oldest_ts (the oldest from response 1)
        expected_start = datetime.fromtimestamp(first_oldest_ts / 1000)
        assert call2_kwargs["start_time"] == expected_start.strftime("%Y%m%d-%H:%M:%S")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_empty_response_breaks(self, mocker):
        """API returns no data -> loop breaks gracefully, partial data returned.

        The function raises on empty data (not isinstance check with empty list),
        so we mock the response to have data but with None oldest timestamp.
        """
        to_dt = datetime(2024, 6, 30, 16, 0, 0)
        from_dt = datetime(2024, 1, 1, 9, 30, 0)

        # First call returns data
        first_ts = int((to_dt - timedelta(days=30)).timestamp() * 1000)
        first_response = self._make_response([self._make_bar(first_ts)])

        # Second call returns a bar with None timestamp (breaks the loop)
        bar_with_none_ts = self._make_bar(0)
        bar_with_none_ts.t = None
        second_response = self._make_response([bar_with_none_ts])

        spy = mocker.patch(
            "src.syncm.ibkr_layer.candles.get_iserver_marketdata_history.asyncio",
            side_effect=[first_response, second_response],
        )

        result = await _fetch_candles_iterative(
            conid=123,
            ticker="AAPL",
            bar="1h",
            from_datetime=from_dt,
            to_datetime=to_dt,
        )

        # The second call has data but None timestamp, so loop breaks
        assert spy.call_count == 2
        # The first call's data is returned
        assert len(result) == 1
