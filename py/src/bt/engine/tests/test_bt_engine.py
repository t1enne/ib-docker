import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.bt.engine.backtest_engine import DataFeed, BacktestEngine
from src.bt.types import (
    Tick,
    TradeSignal,
    ActionType,
    PortfolioResult,
    StrategyProtocol,
)
from src.bt.portfolio import Portfolio, PortfolioProps
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


class MockStrategy(StrategyProtocol):
    """Mock strategy that returns signals based on ticks."""

    def __init__(self, signals: list[TradeSignal]):
        self._signals = signals
        self._signal_idx = 0

    def set_model(self, model):
        pass

    def on_tick(self, tick: Tick) -> list[TradeSignal]:
        if tick.symbol == "AAPL" and self._signal_idx < len(self._signals):
            signal = self._signals[self._signal_idx]
            self._signal_idx += 1
            return [signal]
        return []


@pytest.fixture
def mock_strategy(sample_ticks):
    """Mock strategy that returns signals based on ticks."""
    signals = [
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
    return MockStrategy(signals)


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
def sample_df():
    """Sample dataframe with all required columns."""
    return pd.DataFrame(
        {
            "Open": [100.0, 102.0],
            "High": [105.0, 110.0],
            "Low": [95.0, 98.0],
            "Close": [102.0, 108.0],
            "Volume": [1000, 1200],
        },
        index=pd.Index(
            [get_ts("2025-01-01"), get_ts("2025-01-02")], dtype="datetime64[ns]"
        ),
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
        "src.bt.engine.backtest_engine.read_candles",
        side_effect=[sample_df1, sample_df2],
    ):
        data_feed = DataFeed(
            symbols=["AAPL", "GOOGL"], start_date="2025-01-01", end_date="2025-01-02"
        )
        ticks = []
        async for tick in data_feed.get_data_stream():
            ticks.append(tick)
        assert len(ticks) == 2
        assert ticks[0].symbol == "AAPL"
        assert ticks[1].symbol == "GOOGL"


@pytest.mark.asyncio
async def test_run(mock_strategy, sample_df):
    """Test BacktestEngine.run processes ticks and returns results."""
    with patch("src.bt.engine.backtest_engine.read_candles", return_value=sample_df):
        engine = BacktestEngine(
            strategy=mock_strategy,
            z_model=MagicMock(),
            symbols=["AAPL", "GOOGL"],
            train_start="2024-01-01",
            train_end="2024-12-31",
            test_start="2025-01-01",
            test_end="2025-01-02",
        )
        results, data, z_scores = await engine.run()
        assert len(results.trades) == 1
        assert results.trades[0].symbol == "AAPL"
        assert results.trades[0].position == ActionType.long


def test_finalize_results(sample_df):
    """Test BacktestEngine._finalize_results constructs PortfolioResult."""
    with patch("src.bt.engine.backtest_engine.read_candles", return_value=sample_df):
        engine = BacktestEngine(
            strategy=MagicMock(spec=StrategyProtocol),
            z_model=MagicMock(),
            symbols=["AAPL", "GOOGL"],
            train_start="2024-01-01",
            train_end="2024-12-31",
            test_start="2025-01-01",
            test_end="2025-01-02",
        )
        signal = TradeSignal(
            action=ActionType.long,
            symbol="AAPL",
            z_score=2.0,
            timestamp=get_ts("2025-01-01"),
            price=100.0,
        )
        engine.portfolio.on_signal(signal)

        results, data = engine._finalize_results()
        assert isinstance(results, PortfolioResult)
        assert len(results.trades) == 1
        assert results.trades[0].pnl > 1
        assert data == engine.data
