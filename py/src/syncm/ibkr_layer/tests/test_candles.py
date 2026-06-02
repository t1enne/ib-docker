import pytest
from datetime import datetime, timedelta
from src.syncm.ibkr_layer.candles import (
    calculate_gaps,
    date_to_timestamp,
    get_existing_range_sync,
    get_existing_range,
    candles,
)


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

        assert result == [(datetime(2024, 7, 1), datetime(2025, 3, 1))]

    def test_both_gaps(self):
        """When existing data is in the middle, return both backward and forward gaps"""
        oldest = date_to_timestamp(datetime(2024, 6, 1))
        newest = date_to_timestamp(datetime(2024, 9, 30))

        result = calculate_gaps(
            requested_from=datetime(2024, 1, 1),
            requested_to=datetime(2024, 12, 31),
            oldest_existing=oldest,
            newest_existing=newest,
        )

        assert len(result) == 2
        assert (datetime(2024, 1, 1), datetime(2024, 6, 1)) in result
        assert (datetime(2024, 10, 1), datetime(2024, 12, 31)) in result

    def test_existing_only_oldest(self):
        """When only oldest exists (newest is None), handle correctly"""
        oldest = date_to_timestamp(datetime(2024, 6, 1))

        result = calculate_gaps(
            requested_from=datetime(2024, 1, 1),
            requested_to=datetime(2024, 12, 31),
            oldest_existing=oldest,
            newest_existing=None,
        )

        assert len(result) == 2
        assert (datetime(2024, 1, 1), datetime(2024, 6, 1)) in result
        assert (datetime(2024, 6, 2), datetime(2024, 12, 31)) in result

    def test_existing_only_newest(self):
        """When only newest exists (oldest is None), handle correctly"""
        newest = date_to_timestamp(datetime(2024, 6, 30))

        result = calculate_gaps(
            requested_from=datetime(2024, 1, 1),
            requested_to=datetime(2024, 12, 31),
            oldest_existing=None,
            newest_existing=newest,
        )

        assert len(result) == 2
        assert (datetime(2024, 1, 1), datetime(2024, 6, 30)) in result
        assert (datetime(2024, 7, 1), datetime(2024, 12, 31)) in result

    def test_existing_touches_requested_boundary(self):
        """When existing data touches the boundary, no gap for that side"""
        oldest = date_to_timestamp(datetime(2024, 1, 2))
        newest = date_to_timestamp(datetime(2024, 12, 30))

        result = calculate_gaps(
            requested_from=datetime(2024, 1, 1),
            requested_to=datetime(2024, 12, 31),
            oldest_existing=oldest,
            newest_existing=newest,
        )

        assert len(result) == 1
        assert (datetime(2024, 1, 1), datetime(2024, 1, 2)) in result


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

    @pytest.mark.asyncio
    async def test_skip_fetch_when_data_exists(self, mocker):
        """When existing data covers the requested range, skip API call"""
        today = datetime.now()
        yesterday = today - timedelta(days=1)

        oldest_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        newest_ts = int(
            datetime.combine(yesterday, datetime.min.time()).timestamp() * 1000
        )

        mocker.patch(
            "src.syncm.ibkr_layer.candles.get_existing_range",
            return_value=(oldest_ts, newest_ts),
        )
        spy = mocker.patch(
            "src.syncm.ibkr_layer.candles._fetch_candles_iterative", return_value=[]
        )

        await candles(123, datetime(2024, 1, 1), to_datetime=yesterday, ticker="AAPL")

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
        today = datetime.now()
        yesterday = today - timedelta(days=1)

        oldest_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        newest_ts = int(
            datetime.combine(yesterday, datetime.min.time()).timestamp() * 1000
        )

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
        oldest_ts = int(datetime(2024, 6, 1).timestamp() * 1000)
        newest_ts = int(datetime(2024, 9, 30).timestamp() * 1000)

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
