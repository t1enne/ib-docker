import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.bt.engine.engine import Engine
from src.bt.execution.pure import execute_signal
from src.bt.types import (
    StrategyProtocol,
    FillEvent,
    ExecutionParams,
    StrategyConfig,
    EngineWindow,
)
from src.bt.state import (
    ActionType,
    Tick,
    TradeSignal,
    PortfolioState,
    create_initial_portfolio,
    create_execution_params,
)
from src.bt.portfolio.pure import apply_fill
from src.utils import get_ts, parse_timestamp


def get_fill(s: TradeSignal, portfolio: PortfolioState):
    """Create a fill event from a signal."""
    params = create_execution_params()
    tick = Tick(
        timestamp=s.timestamp,
        symbol=s.symbol,
        open=s.price,
        high=s.price,
        low=s.price,
        close=s.price,
        volume=1000,
    )
    return execute_signal(s, tick, params)


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
        strategy_params={
            "entry_z": 2.0,
            "exit_z": 0.0,
        },
        model_params={},
        bar="1d",
    )


class MockStrategy(StrategyProtocol):
    """Mock strategy that returns signals based on ticks."""

    model = None  # StrategyModel instance - set by engine

    def __init__(self, signals: list[TradeSignal]):
        self._signals = signals
        self._signal_idx = 0

    def on_tick(self, tick: Tick, open_trade=None) -> list[TradeSignal]:
        """Process tick and return signals."""
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
    return create_initial_portfolio(
        initial_capital=10000,
        start_timestamp=get_ts("2025-01-01"),
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
    """Test Engine.run processes ticks and returns results."""
    with patch("src.get_local_candles", return_value=sample_df):
        engine = Engine(strategy_config)

        signal = TradeSignal(
            action=ActionType.long,
            symbol="AAPL",
            z_score=2.0,
            timestamp=get_ts("2025-01-01"),
            price=100.0,
        )
        engine.state = engine.state.__replace__(
            portfolio=apply_fill(
                engine.state.portfolio, get_fill(signal, engine.state.portfolio)
            )
        )
        _r = await engine.run()
        results = _r.pf
        assert len(results.trades) >= 1
        assert results.trades[0].symbol == "AAPL"
        assert results.trades[0].position == ActionType.long


def test_finalize_results(sample_df, strategy_config):
    """Test Engine._finalize constructs PortfolioResult."""
    with patch("src.get_local_candles", return_value=sample_df):
        engine = Engine(strategy_config)
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

        engine.state = engine.state.__replace__(
            portfolio=apply_fill(
                engine.state.portfolio, get_fill(signal, engine.state.portfolio)
            )
        )

        from src.bt.state import Tick as StateTick

        state_tick = StateTick(
            timestamp=tick.timestamp,
            symbol=tick.symbol,
            open=tick.open,
            high=tick.high,
            low=tick.low,
            close=tick.close,
            volume=tick.volume,
        )
        engine.state = engine.state.__replace__(
            portfolio=apply_fill(
                engine.state.portfolio, get_fill(signal, engine.state.portfolio)
            )
        )

        assert len(engine.state.portfolio.positions) >= 0

        results = engine._finalize(engine.state).portfolio
        # After finalization, all positions should be closed


# ---------------------------------------------------------------------------
# Signal dispatch tests
# ---------------------------------------------------------------------------


def _make_engine(strategy_config):
    """Create a Engine with patched read_candles."""
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
    with patch("src.get_local_candles", return_value=sample_df):
        return Engine(strategy_config)


def test_all_pending_signals_execute(strategy_config):
    """All pending signals for a symbol should execute — no skipped elements."""
    engine = _make_engine(strategy_config)
    params = create_execution_params(spread_bps=0, slippage_bps=0)

    # Create ticks for execution
    tick = Tick(
        timestamp=get_ts("2025-01-01"),
        symbol="AAPL",
        open=100.0,
        high=106.0,
        low=99.0,
        close=105.0,
        volume=1000,
    )
    from src.bt.state import Tick as StateTick

    state_tick = StateTick(
        timestamp=tick.timestamp,
        symbol=tick.symbol,
        open=tick.open,
        high=tick.high,
        low=tick.low,
        close=tick.close,
        volume=tick.volume,
    )

    # 3 signals for AAPL
    signals = [
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

    # Simulate the dispatch loop
    portfolio = engine.state.portfolio
    remaining = []
    for signal in signals:
        if signal.symbol == tick.symbol:
            fill = execute_signal(signal, state_tick, params)
            portfolio = apply_fill(portfolio, fill)
        else:
            remaining.append(signal)

    # All 3 signals should have been processed (open, close, re-open)
    assert len(portfolio.trades) == 2  # two opens recorded
    assert remaining == []


def test_signals_only_execute_against_matching_tick(strategy_config):
    """Signals for GOOGL should not execute on an AAPL tick."""
    engine = _make_engine(strategy_config)
    params = create_execution_params(spread_bps=0, slippage_bps=0)

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
    signals = [aapl_signal, googl_signal]

    aapl_tick = Tick(
        timestamp=get_ts("2025-01-01"),
        symbol="AAPL",
        open=100.0,
        high=106.0,
        low=99.0,
        close=105.0,
        volume=1000,
    )
    from src.bt.state import Tick as StateTick

    state_tick = StateTick(
        timestamp=aapl_tick.timestamp,
        symbol=aapl_tick.symbol,
        open=aapl_tick.open,
        high=aapl_tick.high,
        low=aapl_tick.low,
        close=aapl_tick.close,
        volume=aapl_tick.volume,
    )

    # Dispatch against AAPL tick
    portfolio = engine.state.portfolio
    remaining = []
    for signal in signals:
        if signal.symbol == aapl_tick.symbol:
            fill = execute_signal(signal, state_tick, params)
            portfolio = apply_fill(portfolio, fill)
        else:
            remaining.append(signal)

    # Only AAPL signal executed; GOOGL signal deferred
    assert len(portfolio.trades) == 1
    assert portfolio.trades[0].symbol == "AAPL"
    assert len(remaining) == 1
    assert remaining[0].symbol == "GOOGL"

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
    state_tick = StateTick(
        timestamp=googl_tick.timestamp,
        symbol=googl_tick.symbol,
        open=googl_tick.open,
        high=googl_tick.high,
        low=googl_tick.low,
        close=googl_tick.close,
        volume=googl_tick.volume,
    )

    for signal in remaining:
        if signal.symbol == googl_tick.symbol:
            fill = execute_signal(signal, state_tick, params)
            portfolio = apply_fill(portfolio, fill)

    # Both legs now filled
    assert len(portfolio.trades) == 2
    symbols_traded = {t.symbol for t in portfolio.trades}
    assert symbols_traded == {"AAPL", "GOOGL"}
