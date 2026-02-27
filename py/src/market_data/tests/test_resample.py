"""Tests for market_data resample and cache modules."""

from src.utils import list_to_axes

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.market_data.resample import resample_ohlcv, resample_multiindex
from src.market_data.cache import ResampleCache, update_resample_cache, get_from_cache


def create_hourly_data(start: str, periods: int, symbol: str = "AAPL") -> pd.DataFrame:
    """Create mock hourly OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(start, periods=periods, freq="h")

    base_price = 100.0
    prices = base_price + np.random.randn(periods).cumsum()

    return pd.DataFrame(
        {
            "open": prices + np.random.randn(periods) * 0.5,
            "high": prices + np.abs(np.random.randn(periods)) * 2,
            "low": prices - np.abs(np.random.randn(periods)) * 2,
            "close": prices,
            "volume": np.random.randint(1000, 10000, periods),
        },
        index=dates,
    )


def create_multiindex_candles(
    symbols: list[str], start: str, periods: int
) -> pd.DataFrame:
    """Create mock MultiIndex (symbol, timestamp) OHLCV data."""
    frames = []
    for symbol in symbols:
        df = create_hourly_data(start, periods, symbol)
        df["symbol"] = symbol
        frames.append(df)

    combined = pd.concat(frames)
    combined = combined.reset_index()
    combined = combined.rename(columns={"index": "timestamp"})
    combined = combined.set_index(["symbol", "timestamp"])
    return combined


class TestResampleOHLCV:
    """Tests for resample_ohlcv function."""

    def test_resample_4h_basic(self):
        """Test basic 4h resample."""
        df = create_hourly_data("2023-01-01 00:00", periods=10)

        result = resample_ohlcv(df, "4h", completed_only=False)

        assert len(result) > 0
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_resample_1d_basic(self):
        """Test basic 1D resample."""
        df = create_hourly_data("2023-01-01 00:00", periods=48)

        result = resample_ohlcv(df, "1d", completed_only=False)

        assert len(result) >= 2

    def test_completed_only_filters_incomplete(self):
        """Test that completed_only=True excludes current bucket."""
        df = create_hourly_data("2023-01-01 00:00", periods=10)
        current_ts = pd.Timestamp("2023-01-01 09:30")

        result_all = resample_ohlcv(df, "4h", completed_only=False)
        result_completed = resample_ohlcv(
            df, "4h", completed_only=True, current_ts=current_ts
        )

        assert len(result_completed) <= len(result_all)

    def test_resample_empty_dataframe(self):
        """Test resample on empty DataFrame."""
        df = pd.DataFrame(
            columns=list_to_axes(["open", "high", "low", "close", "volume"])
        )

        result = resample_ohlcv(df, "1h")

        assert result.empty
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_resample_preserves_ohlcv_logic(self):
        """Test that OHLCV aggregation is correct."""
        df = create_hourly_data("2023-01-01 00:00", periods=4)

        result = resample_ohlcv(df, "4h", completed_only=False)

        assert result.iloc[0]["high"] == df["high"].max()
        assert result.iloc[0]["low"] == df["low"].min()
        assert result.iloc[0]["volume"] == df["volume"].sum()


class TestResampleMultiindex:
    """Tests for resample_multiindex function."""

    def test_resample_multiindex_4h(self):
        """Test 4h resample on MultiIndex data."""
        df = create_multiindex_candles(["AAPL", "GOOG"], "2023-01-01 00:00", 10)

        result = resample_multiindex(df, "4h", completed_only=False)

        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["symbol", "timestamp"]
        assert "open" in result.columns

    def test_resample_multiindex_1d(self):
        """Test 1D resample on MultiIndex data."""
        df = create_multiindex_candles(["AAPL", "GOOG"], "2023-01-01 00:00", 48)

        result = resample_multiindex(df, "1d", completed_only=False)

        symbols = result.index.get_level_values("symbol").unique()
        assert len(symbols) == 2

    def test_multiindex_completed_only(self):
        """Test completed_only filtering on MultiIndex."""
        df = create_multiindex_candles(["AAPL"], "2023-01-01 00:00", 10)
        current_ts = pd.Timestamp("2023-01-01 09:30")

        result_completed = resample_multiindex(
            df, "4h", completed_only=True, current_ts=current_ts
        )

        if not result_completed.empty:
            timestamps = result_completed.index.get_level_values("timestamp")
            assert all(ts < current_ts.floor("4h") for ts in timestamps)

    def test_multiindex_empty(self):
        """Test resample on empty MultiIndex DataFrame."""
        df = pd.DataFrame(
            columns=list_to_axes(["open", "high", "low", "close", "volume"]),
            index=pd.MultiIndex.from_tuples([], names=["symbol", "timestamp"]),
        )

        result = resample_multiindex(df, "1h")

        assert result.empty


class TestResampleCache:
    """Tests for ResampleCache and update_resample_cache."""

    def test_cache_creation(self):
        """Test creating empty cache."""
        cache = ResampleCache()

        assert cache.cache == {}
        assert cache.anchor == {}

    def test_cache_with_data(self):
        """Test creating cache with data."""
        df = create_hourly_data("2023-01-01 00:00", 10)
        cache = ResampleCache(
            cache={"1h": df}, anchor={"1h": pd.Timestamp("2023-01-01 04:00")}
        )

        assert "1h" in cache.cache
        assert "1h" in cache.anchor

    def test_update_resample_cache_new_frequency(self):
        """Test adding new frequency to cache."""
        df = create_hourly_data("2023-01-01 00:00", 10)
        cache = ResampleCache()

        new_cache = update_resample_cache(
            cache, df, ["4h"], pd.Timestamp("2023-01-01 09:00")
        )

        assert "4h" in new_cache.cache
        assert "4h" in new_cache.anchor

    def test_update_resample_cache_no_recompute_same_bucket(self):
        """Test that same bucket doesn't trigger recompute."""
        df = create_hourly_data("2023-01-01 00:00", 10)
        cache = ResampleCache(
            cache={"4h": df},
            anchor={"4h": pd.Timestamp("2023-01-01 04:00")},
        )

        new_cache = update_resample_cache(
            cache, df, ["4h"], pd.Timestamp("2023-01-01 06:00")
        )

        assert new_cache.cache["4h"] is df

    def test_update_resample_cache_new_bucket_triggers_recompute(self):
        """Test that new bucket triggers recompute."""
        df = create_hourly_data("2023-01-01 00:00", 10)
        old_anchor = pd.Timestamp("2023-01-01 00:00")
        cache = ResampleCache(
            cache={"4h": df.iloc[:5]},
            anchor={"4h": old_anchor},
        )

        new_cache = update_resample_cache(
            cache, df, ["4h"], pd.Timestamp("2023-01-01 05:00")
        )

        assert new_cache.anchor["4h"] == pd.Timestamp("2023-01-01 04:00")
        assert new_cache.anchor["4h"] != old_anchor

    def test_update_multiple_frequencies(self):
        """Test updating multiple frequencies at once."""
        df = create_hourly_data("2023-01-01 00:00", 50)
        cache = ResampleCache()

        new_cache = update_resample_cache(
            cache, df, ["4h", "1d"], pd.Timestamp("2023-01-01 10:00")
        )

        assert "4h" in new_cache.cache
        assert "1d" in new_cache.cache


class TestGetFromCache:
    """Tests for get_from_cache function."""

    def test_get_from_cache_basic(self):
        """Test basic get from cache."""
        df = create_hourly_data("2023-01-01 00:00", 10)
        cache = ResampleCache(
            cache={"4h": df}, anchor={"4h": pd.Timestamp("2023-01-01 04:00")}
        )

        result = get_from_cache(cache, "4h")

        assert not result.empty

    def test_get_from_cache_with_symbol(self):
        """Test get from cache with symbol filter."""
        df = create_multiindex_candles(["AAPL", "GOOG"], "2023-01-01 00:00", 10)
        cache = ResampleCache(
            cache={"4h": df}, anchor={"4h": pd.Timestamp("2023-01-01 04:00")}
        )

        result = get_from_cache(cache, "4h", symbol="AAPL")

        if not result.empty:
            assert "AAPL" not in result.columns

    def test_get_from_cache_completed_only(self):
        """Test completed_only filtering when reading."""
        df = create_hourly_data("2023-01-01 00:00", 10)
        cache = ResampleCache(
            cache={"4h": df}, anchor={"4h": pd.Timestamp("2023-01-01 04:00")}
        )

        result = get_from_cache(
            cache,
            "4h",
            completed_only=True,
            current_ts=pd.Timestamp("2023-01-01 09:30"),
        )

        if not result.empty:
            assert all(idx < pd.Timestamp("2023-01-01 08:00") for idx in result.index)

    def test_get_from_cache_missing_frequency(self):
        """Test get from cache for missing frequency."""
        cache = ResampleCache()

        result = get_from_cache(cache, "1h")

        assert result.empty

    def test_get_from_cache_empty_cache(self):
        """Test get from cache when cache is empty."""
        df = pd.DataFrame(
            columns=list_to_axes(["open", "high", "low", "close", "volume"])
        )
        cache = ResampleCache(
            cache={"1h": df}, anchor={"1h": pd.Timestamp("2023-01-01 01:00")}
        )

        result = get_from_cache(
            cache,
            "1h",
            completed_only=True,
            current_ts=pd.Timestamp("2023-01-01 02:00"),
        )

        assert result.empty


class TestIntegration:
    """Integration tests combining resample and cache."""

    def test_full_resample_workflow(self):
        """Test full workflow: create data -> update cache -> read from cache."""
        df = create_multiindex_candles(["AAPL"], "2023-01-01 00:00", 10)
        cache = ResampleCache()

        cache = update_resample_cache(
            cache, df, ["4h"], pd.Timestamp("2023-01-01 09:00")
        )

        result = get_from_cache(
            cache,
            "4h",
            completed_only=True,
            current_ts=pd.Timestamp("2023-01-01 09:30"),
        )

        assert isinstance(result, pd.DataFrame)

    def test_multiple_updates_across_buckets(self):
        """Test cache updates across multiple bucket transitions."""
        dfs = [
            create_hourly_data("2023-01-01 00:00", 4),
            create_hourly_data("2023-01-01 00:00", 8),
            create_hourly_data("2023-01-01 00:00", 12),
        ]

        cache = ResampleCache()

        cache = update_resample_cache(
            cache, dfs[0], ["4h"], pd.Timestamp("2023-01-01 03:00")
        )
        assert cache.anchor["4h"] == pd.Timestamp("2023-01-01 00:00")

        cache = update_resample_cache(
            cache, dfs[1], ["4h"], pd.Timestamp("2023-01-01 05:00")
        )
        assert cache.anchor["4h"] == pd.Timestamp("2023-01-01 04:00")

        cache = update_resample_cache(
            cache, dfs[2], ["4h"], pd.Timestamp("2023-01-01 10:00")
        )
        assert cache.anchor["4h"] == pd.Timestamp("2023-01-01 08:00")

    def test_no_lookahead_guarantee(self):
        """Test that completed_only guarantees no lookahead."""
        df = create_hourly_data("2023-01-01 00:00", 10)
        current_ts = pd.Timestamp("2023-01-01 09:30")

        cache = update_resample_cache(
            cache=ResampleCache(),
            candles=df,
            frequencies=["4h"],
            current_ts=current_ts,
        )

        result = get_from_cache(cache, "4h", completed_only=True, current_ts=current_ts)

        bucket_end = current_ts.floor("4h")
        if not result.empty:
            assert all(idx < bucket_end for idx in result.index)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
