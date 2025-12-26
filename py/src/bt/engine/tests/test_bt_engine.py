import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.bt.engine.bt_engine import DataFeed, BTEngine
from src.bt.types import (
    Tick,
    BacktestResult,
    TradeSignal,
    ActionType,
)
from src.bt.portfolio.portfolio import Portfolio, PortfolioProps
from src.utils import get_ts


@pytest.fixture
def sample_ticks():
    """Sample ticks for testing."""
    return [
        Tick(
            timestamp=get_ts("2025-01-01"),
            symbol="AAPL",
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000,
        ),
        Tick(
            timestamp=get_ts("2025-01-01"),
            symbol="GOOGL",
            open=200.0,
            high=210.0,
            low=195.0,
            close=205.0,
            volume=500,
        ),
        Tick(
            timestamp=get_ts("2025-01-02"),
            symbol="AAPL",
            open=102.0,
            high=110.0,
            low=98.0,
            close=108.0,
            volume=1200,
        ),
    ]


@pytest.fixture
def mock_strategy(sample_ticks):
    """Mock strategy that returns signals based on ticks."""
    strategy = MagicMock()
    strategy.on_tick = MagicMock(
        return_value=[
            TradeSignal(
                action=ActionType.long,
                symbol=tick.symbol,
                z_score=2.0,
                timestamp=tick.timestamp,
                price=tick.close,
            )
            for tick in sample_ticks
            if tick.symbol == "AAPL"
        ]
    )
    return strategy


@pytest.fixture
def portfolio():
    """Real portfolio for testing."""
    return Portfolio(
        PortfolioProps(
            stop_loss=0.1,
            take_profit=1.5,
            initial_capital=10000,
            position_size=0.1,
            commission=0.0001,
            start_date=get_ts("2025-01-01"),
        )
    )


@pytest.fixture
def mock_data_feed(sample_ticks):
    """Mock data feed."""
    data_feed = MagicMock(spec=DataFeed)

    async def mock_get_data_stream():
        for tick in sample_ticks:
            yield tick

    data_feed.get_data_stream = mock_get_data_stream
    data_feed.symbols = ["AAPL", "GOOGL"]
    data_feed.start_date = "2025-01-01"
    data_feed.end_date = "2025-01-02"
    return data_feed


@pytest.mark.asyncio
async def test_get_data_stream(sample_ticks):
    """Test DataFeed.get_data_stream yields sorted ticks."""
    # Mock read_candles to return sample dataframes
    sample_df1 = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [95.0],
            "Close": [102.0],
            "Volume": [1000],
        },
        index=pd.Index([get_ts("2025-01-01")], dtype="datetime64[ns]"),
    )
    sample_df2 = pd.DataFrame(
        {
            "Open": [200.0],
            "High": [210.0],
            "Low": [195.0],
            "Close": [205.0],
            "Volume": [500],
        },
        index=pd.Index([get_ts("2025-01-01")], dtype="datetime64[ns]"),
    )

    with patch(
        "src.bt.engine.bt_engine.read_candles", side_effect=[sample_df1, sample_df2]
    ):
        data_feed = DataFeed(
            symbols=["AAPL", "GOOGL"], start_date="2025-01-01", end_date="2025-01-02"
        )
        ticks = []
        async for tick in data_feed.get_data_stream():
            ticks.append(tick)
        # Should be sorted by timestamp
        assert len(ticks) == 2
        assert ticks[0].symbol == "AAPL"
        assert ticks[1].symbol == "GOOGL"


@pytest.mark.asyncio
async def test_run(mock_strategy, portfolio, mock_data_feed):
    """Test BTEngine.run processes ticks and returns results."""
    engine = BTEngine(
        strategy=mock_strategy, portfolio=portfolio, data_feed=mock_data_feed
    )
    # Mock self.data
    engine.data = {
        "AAPL": pd.DataFrame(
            {"Close": [108.0]},
            index=pd.Index([get_ts("2025-01-02")], dtype="datetime64[ns]"),
        ),
        "GOOGL": pd.DataFrame(
            {"Close": [205.0]},
            index=pd.Index([get_ts("2025-01-02")], dtype="datetime64[ns]"),
        ),
    }
    results, data = await engine.run()
    assert isinstance(results, BacktestResult)
    assert results.total_trades == 1  # One trade opened and closed
    assert results.profitable_trades == 1
    assert len(results.trades) == 1
    assert results.trades[0].symbol == "AAPL"
    assert results.trades[0].position == ActionType.long
    # Check that portfolio was modified
    assert len(portfolio.trades) == 1
    assert portfolio.trades[0].symbol == "AAPL"


def test_finalize_results(portfolio, mock_data_feed):
    """Test BTEngine._finalize_results constructs BacktestResult."""
    # Simulate a trade in portfolio
    signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    portfolio.on_signal(signal)
    engine = BTEngine(
        strategy=MagicMock(), portfolio=portfolio, data_feed=mock_data_feed
    )
    engine.data = {
        "AAPL": pd.DataFrame(
            {"Close": [110.0]},
            index=pd.Index([get_ts("2025-01-02")], dtype="datetime64[ns]"),
        ),
        "GOOGL": pd.DataFrame(
            {"Close": [205.0]},
            index=pd.Index([get_ts("2025-01-02")], dtype="datetime64[ns]"),
        ),
    }
    results, data = engine._finalize_results()
    assert isinstance(results, BacktestResult)
    assert results.total_trades == 1
    assert results.profitable_trades == 1
    assert len(results.trades) == 1
    assert data == engine.data
    # Check that close_all_positions was called with correct args
    # Since portfolio is real, we can't check calls easily, but assert position is closed
    assert portfolio.positions["AAPL"] == 0
