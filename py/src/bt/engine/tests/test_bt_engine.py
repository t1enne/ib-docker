import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.bt.engine.backtest_engine import DataFeed, BacktestEngine
from src.bt.execution import ExecutionHandler
from src.bt.types import (
    Tick,
    TradeSignal,
    ActionType,
    StrategyProtocol,
    FillEvent,
    ExecutionParams,
    StrategyConfig,
    EngineWindow,
)
from src.bt.portfolio import Portfolio, PortfolioProps
from src.utils import get_ts, parse_timestamp


def get_fill(s: TradeSignal, pf: Portfolio):
    return FillEvent(
        signal=s,
        filled_qty=1,  # full
        executed_price=s.price,
        commission=pf.commission,
        slippage=0.0,
    )


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
def strategy_config():
    return StrategyConfig(
        name="test",
        strategy_type="pnd",
        symbols=["AAPL", "GOOGL"],
        entry_z=2.0,
        exit_z=0.0,
        stop_loss=0.1,
        take_profit=1.5,
        initial_capital=10000,
        position_size=0.1,
        commission=0.0001,
        training_start="2024-01-01",
        training_end="2024-12-31",
        trading_start="2025-01-01",
        trading_end="2025-01-02",
        rolling_window_size=20,
        plot=False,
        bar="1d",
    )


class MockStrategy(StrategyProtocol):
    """Mock strategy that returns signals based on ticks."""

    model = None  # StrategyModel instance - set by engine

    def __init__(self, signals: list[TradeSignal]):
        self._signals = signals
        self._signal_idx = 0

    def on_tick(self, tick: Tick, open_trade=None) -> list[TradeSignal]:
        """Process tick and return signals.

        Access computed features via self.model:
            - self.model.z_score          # Current z-score
            - self.model.current_regime   # Current HMM regime (if configured)
            - self.model.market_data      # Historical OHLCV data
        """
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


@pytest.mark.asyncio
async def test_run(mock_strategy, sample_df, strategy_config):
    """Test BacktestEngine.run processes ticks and returns results."""
    with patch("src.read_candles", return_value=sample_df):
        engine = BacktestEngine(strategy_config)

        signal = TradeSignal(
            action=ActionType.long,
            symbol="AAPL",
            z_score=2.0,
            timestamp=get_ts("2025-01-01"),
            price=100.0,
        )
        engine.portfolio.on_fill(get_fill(signal, engine.portfolio))
        _r = await engine.run()
        results = _r.pf
        assert len(results.trades) == 1
        assert results.trades[0].symbol == "AAPL"
        assert results.trades[0].position == ActionType.long


def test_finalize_results(sample_df, strategy_config):
    """Test BacktestEngine._finalize_results constructs PortfolioResult."""
    with patch("src.read_candles", return_value=sample_df):
        engine = BacktestEngine(strategy_config)
        signal = TradeSignal(
            action=ActionType.long,
            symbol="AAPL",
            z_score=2.0,
            timestamp=get_ts("2025-01-01"),
            price=100.0,
        )
        tick = Tick(
            symbol="AAPL",
            timestamp=get_ts("2025-01-02"),
            open=0.0,
            high=0.0,
            low=0.0,
            close=120.0,
            volume=0.0,
        )

        engine.portfolio.on_fill(get_fill(signal, engine.portfolio))
        engine.portfolio.update_market_value(tick)

        assert len(engine.portfolio.open_trades) == 1

        results = engine._finalize_results().pf
        assert len(results.trades) == 1
        assert results.trades[0].pnl > 1


# ---------------------------------------------------------------------------
# Signal dispatch tests
# ---------------------------------------------------------------------------


def _make_engine(strategy_config):
    """Create a BacktestEngine with patched read_candles."""
    sample_df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [95.0],
            "Close": [102.0],
            "Volume": [1000],
        },
        index=pd.Index([get_ts("2025-01-01")], dtype="datetime64[ns]"),
    )
    with patch("src.read_candles", return_value=sample_df):
        return BacktestEngine(strategy_config)


def test_all_pending_signals_execute(strategy_config):
    """All pending signals for a symbol should execute — no skipped elements."""
    engine = _make_engine(strategy_config)
    exec_handler = ExecutionHandler(ExecutionParams(spread_bps=0, slippage_bps=0))
    engine.execution_handler = exec_handler

    # 3 signals for AAPL
    engine.pending_signals = [
        TradeSignal(
            action=ActionType.long,
            symbol="AAPL",
            z_score=2.0,
            timestamp=get_ts("2025-01-01"),
            price=100.0,
        ),
        TradeSignal(
            action=ActionType.close,
            symbol="AAPL",
            z_score=0.0,
            timestamp=get_ts("2025-01-01"),
            price=105.0,
        ),
        TradeSignal(
            action=ActionType.long,
            symbol="AAPL",
            z_score=2.5,
            timestamp=get_ts("2025-01-01"),
            price=103.0,
        ),
    ]

    tick = Tick(
        timestamp=get_ts("2025-01-01"),
        symbol="AAPL",
        open=100.0,
        high=106.0,
        low=99.0,
        close=105.0,
        volume=1000,
    )

    # Simulate the dispatch loop from _run_backtest
    remaining = []
    for signal in engine.pending_signals:
        if signal.symbol == tick.symbol:
            fill = engine.execution_handler.execute(signal, tick)
            engine.portfolio.on_fill(fill)
        else:
            remaining.append(signal)
    engine.pending_signals = remaining

    # All 3 signals should have been processed (open, close, re-open)
    assert len(engine.portfolio.trades) == 2  # two opens recorded
    assert engine.pending_signals == []


def test_signals_only_execute_against_matching_tick(strategy_config):
    """Signals for GOOGL should not execute on an AAPL tick."""
    engine = _make_engine(strategy_config)
    exec_handler = ExecutionHandler(ExecutionParams(spread_bps=0, slippage_bps=0))
    engine.execution_handler = exec_handler

    aapl_signal = TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )
    googl_signal = TradeSignal(
        action=ActionType.short,
        symbol="GOOGL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=200.0,
    )
    engine.pending_signals = [aapl_signal, googl_signal]

    aapl_tick = Tick(
        timestamp=get_ts("2025-01-01"),
        symbol="AAPL",
        open=100.0,
        high=106.0,
        low=99.0,
        close=105.0,
        volume=1000,
    )

    # Dispatch against AAPL tick
    remaining = []
    for signal in engine.pending_signals:
        if signal.symbol == aapl_tick.symbol:
            fill = engine.execution_handler.execute(signal, aapl_tick)
            engine.portfolio.on_fill(fill)
        else:
            remaining.append(signal)
    engine.pending_signals = remaining

    # Only AAPL signal executed; GOOGL signal deferred
    assert len(engine.portfolio.trades) == 1
    assert engine.portfolio.trades[0].symbol == "AAPL"
    assert len(engine.pending_signals) == 1
    assert engine.pending_signals[0].symbol == "GOOGL"

    # Now dispatch against GOOGL tick
    googl_tick = Tick(
        timestamp=get_ts("2025-01-01"),
        symbol="GOOGL",
        open=200.0,
        high=210.0,
        low=195.0,
        close=205.0,
        volume=500,
    )
    remaining = []
    for signal in engine.pending_signals:
        if signal.symbol == googl_tick.symbol:
            fill = engine.execution_handler.execute(signal, googl_tick)
            engine.portfolio.on_fill(fill)
        else:
            remaining.append(signal)
    engine.pending_signals = remaining

    # Both legs now filled
    assert len(engine.portfolio.trades) == 2
    assert engine.pending_signals == []
    symbols_traded = {t.symbol for t in engine.portfolio.trades}
    assert symbols_traded == {"AAPL", "GOOGL"}
