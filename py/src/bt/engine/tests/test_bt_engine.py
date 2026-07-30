import pytest
import pandas as pd
from src.bt.engine.backtest import Backtest, run_backtest, candle_generator
from src.bt.engine.handlers import (
    default_execution_handler,
    default_risk_handler,
)
from src.bt.execution.pure import execute_signal
from src.bt.types import (
    StrategyConfig,
)
from src.bt.state import (
    Candle,
    TradeSignal,
    PortfolioState,
    create_execution_params,
)
from src.utils import parse_timestamp


def get_fill(s: TradeSignal, _pf: PortfolioState):
    """Create a fill event from a signal."""
    params = create_execution_params()
    candle = Candle(
        timestamp=s.timestamp,
        symbol=s.symbol,
        open=s.price,
        high=s.price,
        low=s.price,
        close=s.price,
        volume=0.0,
    )
    return execute_signal(s, candle, params)


@pytest.fixture
def sample_df():
    """Sample OHLCV data for testing."""
    idx = pd.date_range("2025-01-01", periods=10, freq="h")
    df = pd.DataFrame(
        {"open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000},
        index=idx,
    )
    return df


@pytest.fixture
def strategy_config():
    """Sample strategy config."""
    return StrategyConfig(
        name="test",
        strategy_type="momentum_regime",
        symbols=["AAPL", "GOOGL"],
        stop_loss=0.05,
        take_profit=0.1,
        initial_capital=10000.0,
        position_size=0.2,
        commission=0.5,
        training_start="2024-01-01",
        training_end="2024-12-31",
        trading_start="2025-01-01",
        trading_end="2025-12-31",
        bar="1h",
        strategy_params={},
        model_params={},
    )


class TestCandleGenerator:
    """Test the candle_generator function."""

    def test_empty_df_returns_empty_generator(self):
        """Empty DataFrame yields no candles."""
        # Create empty DataFrame with proper MultiIndex structure
        df = pd.DataFrame(
            columns=pd.MultiIndex.from_product(
                [["AAPL"], ["open", "high", "low", "close", "volume"]]
            )
        )
        gen = candle_generator(df, ["AAPL"])
        candles = list(gen)
        assert candles == []

    def test_single_symbol(self):
        """Single symbol generates correct candles."""
        # Create DataFrame with MultiIndex like load_candles produces
        idx = pd.date_range("2025-01-01", periods=10, freq="h")
        data = {
            ("AAPL", "open"): [100] * 10,
            ("AAPL", "high"): [105] * 10,
            ("AAPL", "low"): [95] * 10,
            ("AAPL", "close"): [102] * 10,
            ("AAPL", "volume"): [1000] * 10,
        }
        df = pd.DataFrame(data, index=idx)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        gen = candle_generator(df, ["AAPL"])
        candles = list(gen)
        assert len(candles) == 10
        assert all(c.symbol == "AAPL" for c in candles)

    def test_multiple_symbols(self):
        """Multiple symbols generate candles for each."""
        idx = pd.date_range("2025-01-01", periods=5, freq="h")
        data = {
            ("AAPL", "open"): [100] * 5,
            ("AAPL", "high"): [105] * 5,
            ("AAPL", "low"): [95] * 5,
            ("AAPL", "close"): [102] * 5,
            ("AAPL", "volume"): [1000] * 5,
            ("GOOGL", "open"): [200] * 5,
            ("GOOGL", "high"): [205] * 5,
            ("GOOGL", "low"): [195] * 5,
            ("GOOGL", "close"): [202] * 5,
            ("GOOGL", "volume"): [500] * 5,
        }
        df = pd.DataFrame(data, index=idx)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        gen = candle_generator(df, ["AAPL", "GOOGL"])
        candles = list(gen)

        aapl_candles = [c for c in candles if c.symbol == "AAPL"]
        googl_candles = [c for c in candles if c.symbol == "GOOGL"]

        assert len(aapl_candles) == 5
        assert len(googl_candles) == 5


class TestBacktest:
    """Test the Backtest dataclass."""

    def test_creates_window_from_config(self, strategy_config):
        """Backtest creates EngineWindow from config."""
        bt = Backtest(strategy_config)
        assert bt.window.train_start == parse_timestamp("2024-01-01")
        assert bt.window.train_end == parse_timestamp("2024-12-31")
        assert bt.window.test_start == parse_timestamp("2025-01-01")
        assert bt.window.test_end == parse_timestamp("2025-12-31")

    def test_creates_execution_params(self, strategy_config):
        """Backtest creates ExecutionParams from config."""
        bt = Backtest(strategy_config)
        assert bt.execution_params.fixed_commission == 0.5

    def test_creates_risk_config(self, strategy_config):
        """Backtest creates RiskConfig from config."""
        bt = Backtest(strategy_config)
        assert bt.risk_config.stop_loss_pct == 0.05
        assert bt.risk_config.take_profit_pct == 0.1


class TestRunBacktest:
    """Test the run_backtest function."""

    def _create_test_df(self):
        """Create a test DataFrame with proper MultiIndex."""
        idx = pd.date_range("2025-01-01", periods=10, freq="h")
        data = {
            ("AAPL", "open"): [100] * 10,
            ("AAPL", "high"): [105] * 10,
            ("AAPL", "low"): [95] * 10,
            ("AAPL", "close"): [102] * 10,
            ("AAPL", "volume"): [1000] * 10,
        }
        df = pd.DataFrame(data, index=idx)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    def test_runs_with_default_handlers(self, strategy_config):
        """run_backtest works with default handlers."""
        bt = Backtest(strategy_config)
        df = self._create_test_df()
        gen = candle_generator(df, ["AAPL"])
        exec_handler = default_execution_handler()
        risk_handler = default_risk_handler()

        results, state = run_backtest(bt, gen, exec_handler, risk_handler)

        assert results is not None
        assert state is not None
        assert state.portfolio is not None

    def test_tracks_trades(self, strategy_config):
        """run_backtest tracks executed trades."""
        bt = Backtest(strategy_config)
        df = self._create_test_df()
        gen = candle_generator(df, ["AAPL"])
        exec_handler = default_execution_handler()
        risk_handler = default_risk_handler()

        results, state = run_backtest(bt, gen, exec_handler, risk_handler)

        # With no signals, no trades should be executed
        assert len(state.portfolio.trades) == 0
