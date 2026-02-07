import pytest
from src.bt.execution import ExecutionHandler, ExecutionParams
from src.bt.types import TradeSignal, Tick, ActionType
from src.utils import get_ts


@pytest.fixture
def execution_handler():
    params = ExecutionParams(spread_bps=10.0, slippage_bps=2.0)
    return ExecutionHandler(params)


@pytest.fixture
def long_signal():
    return TradeSignal(
        action=ActionType.long,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )


@pytest.fixture
def short_signal():
    return TradeSignal(
        action=ActionType.short,
        symbol="AAPL",
        z_score=2.0,
        timestamp=get_ts("2025-01-01"),
        price=100.0,
    )


@pytest.fixture
def tick_bullish():
    return Tick(
        timestamp=get_ts("2025-01-01"),
        symbol="AAPL",
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.5,
        volume=1000,
    )


@pytest.fixture
def tick_bearish():
    return Tick(
        timestamp=get_ts("2025-01-01"),
        symbol="AAPL",
        open=100.5,
        high=101.0,
        low=98.0,
        close=99.0,
        volume=1000,
    )


def test_long_entry_pays_spread(execution_handler, long_signal, tick_bullish):
    fill = execution_handler.execute(long_signal, tick_bullish)
    expected_spread = 100.0 * (10.0 / 10000)  # 0.10
    expected_price_base = 100.0 + expected_spread  # 100.10 (long pays spread)
    expected_slippage = 100.0 * (2.0 / 10000)  # 0.02
    expected_price = expected_price_base + expected_slippage  # ~100.12
    assert fill.executed_price > long_signal.price
    assert fill.executed_price > expected_price_base


def test_short_entry_receives_spread(execution_handler, short_signal, tick_bullish):
    fill = execution_handler.execute(short_signal, tick_bullish)
    expected_spread = 100.0 * (10.0 / 10000)  # 0.10
    expected_price_base = 100.0 - expected_spread  # 99.90 (short receives spread)
    expected_slippage = 100.0 * (2.0 / 10000)  # 0.02 (slippage is always added)
    expected_price = expected_price_base + expected_slippage  # ~99.92
    assert fill.executed_price < short_signal.price
    assert fill.executed_price > expected_price_base  # Slippage added


def test_adverse_selection_long(execution_handler, long_signal, tick_bearish):
    fill = execution_handler.execute(long_signal, tick_bearish)
    adverse_slippage = 100.0 * (2.0 * 1.5 / 10000)  # Extra 50% for adverse
    assert fill.slippage > 0


def test_adverse_selection_short(execution_handler, short_signal, tick_bullish):
    fill = execution_handler.execute(short_signal, tick_bullish)
    adverse_slippage = 100.0 * (2.0 * 1.5 / 10000)  # Extra 50% for adverse
    assert fill.slippage > 0


def test_commission_applied(execution_handler, long_signal, tick_bullish):
    fill = execution_handler.execute(long_signal, tick_bullish)
    assert fill.commission > 0


def test_no_execution_handler_backwards_compat():
    from src.bt.engine.bt_engine import BTEngine
    from src.bt.portfolio.portfolio import Portfolio, PortfolioProps
    from src.bt.engine.bt_engine import DataFeed
    from src.bt.algos.pairs_trading import PairsTradingStrategy
    from src.utils import get_ts
    import pandas as pd

    portfolio = Portfolio(
        PortfolioProps(
            stop_loss=0.1,
            take_profit=0.5,
            initial_capital=10000,
            position_size=0.1,
            commission=0.001,
            start_date=get_ts("2024-12-31"),
        )
    )

    aapl_idx = pd.DatetimeIndex([get_ts("2025-01-01")])
    msft_idx = pd.DatetimeIndex([get_ts("2025-01-01")])
    hdata = {
        "AAPL": pd.DataFrame({"Close": [100.0]}, index=aapl_idx),
        "MSFT": pd.DataFrame({"Close": [200.0]}, index=msft_idx),
    }

    strat = PairsTradingStrategy(
        symbols=["AAPL", "MSFT"],
        hdata=hdata,
        entry_z=2.0,
        rolling_window_size=20,
    )

    hdata = {
        "AAPL": pd.DataFrame(
            {"Close": [100.0]}, index=pd.DatetimeIndex([get_ts("2024-12-31")])
        ),
        "MSFT": pd.DataFrame(
            {"Close": [200.0]}, index=pd.DatetimeIndex([get_ts("2024-12-31")])
        ),
    }

    feed = DataFeed(["AAPL", "MSFT"], "2024-12-31", "2025-01-01")

    engine = BTEngine(strat, portfolio, feed)
    assert engine.execution_handler is None


def test_with_execution_params():
    from src.bt.engine.bt_engine import BTEngine
    from src.bt.portfolio.portfolio import Portfolio, PortfolioProps
    from src.bt.engine.bt_engine import DataFeed
    from src.bt.algos.pairs_trading import PairsTradingStrategy
    from src.utils import get_ts
    import pandas as pd

    portfolio = Portfolio(
        PortfolioProps(
            stop_loss=0.1,
            take_profit=0.5,
            initial_capital=10000,
            position_size=0.1,
            commission=0.001,
            start_date=get_ts("2024-12-31"),
        )
    )

    aapl_idx = pd.DatetimeIndex([get_ts("2025-01-01")])
    msft_idx = pd.DatetimeIndex([get_ts("2025-01-01")])
    hdata = {
        "AAPL": pd.DataFrame({"Close": [100.0]}, index=aapl_idx),
        "MSFT": pd.DataFrame({"Close": [200.0]}, index=msft_idx),
    }

    strat = PairsTradingStrategy(
        symbols=["AAPL", "MSFT"],
        hdata=hdata,
        entry_z=2.0,
        rolling_window_size=20,
    )

    hdata = {
        "AAPL": pd.DataFrame(
            {"Close": [100.0]}, index=pd.DatetimeIndex([get_ts("2024-12-31")])
        ),
        "MSFT": pd.DataFrame(
            {"Close": [200.0]}, index=pd.DatetimeIndex([get_ts("2024-12-31")])
        ),
    }

    feed = DataFeed(["AAPL", "MSFT"], "2024-12-31", "2025-01-01")

    exec_params = ExecutionParams(spread_bps=5.0, slippage_bps=2.0)
    engine = BTEngine(strat, portfolio, feed, execution_params=exec_params)

    assert engine.execution_handler is not None
    assert engine.execution_handler.params.spread_bps == 5.0
    assert engine.execution_handler.params.slippage_bps == 2.0
