"""Functional backtest module.

Provides a fully functional API for running backtests:
- Backtest: dataclass holding config only
- ticks_generator: creates tick generator from DataFrame
- run_backtest: runs the backtest loop with injected handlers

Usage:
    from src.bt.engine.backtest import Backtest, ticks_generator, run_backtest
    from src.bt.engine.handlers import default_execution_handler, default_risk_handler

    bt = Backtest(config)
    gen = ticks_generator(df, config.symbols)
    results, state = run_backtest(bt, gen, exec_handler, risk_handler)
"""

from src.bt.engine.utils import ticks_generator, merge_bt_state

from dataclasses import dataclass, field, replace
from typing import Generator, Tuple, Optional, Any, List, Callable, cast

import pandas as pd
from pandas import Timestamp

from src.bt.metrics import calculate_portfolio_result
from src.bt.state import (
    ActionType,
    BacktestState,
    Tick,
    ModelState,
    TradeSignal,
    FillEvent,
    ExecutionParams,
    RiskConfig,
    RiskEvent,
    create_initial_backtest_state,
    create_execution_params,
    create_risk_config,
    TradeExitReason,
    PortfolioState,
)
from src.bt.types import StrategyConfig, EngineWindow, BacktestResults, PlotConfig
from src.bt.engine.handlers import ExecutionHandler, RiskHandler
from src.utils import parse_timestamp


@dataclass
class Backtest:
    """Backtest configuration container.

    This is a pure dataclass - no methods, no state. It just holds config.
    All backtest logic is in the standalone run_backtest function.
    """

    config: StrategyConfig
    window: EngineWindow = field(init=False)
    execution_params: ExecutionParams = field(init=False)
    risk_config: RiskConfig = field(init=False)

    def __post_init__(self):
        self.window = EngineWindow(
            train_start=parse_timestamp(self.config.training_start),
            train_end=parse_timestamp(self.config.training_end),
            test_start=parse_timestamp(self.config.trading_start),
            test_end=parse_timestamp(self.config.trading_end),
        )
        self.execution_params = create_execution_params(
            fixed_commission=self.config.commission
        )
        self.risk_config = create_risk_config(
            stop_loss_pct=self.config.stop_loss,
            take_profit_pct=self.config.take_profit,
        )


def run_backtest(
    bt: Backtest,
    tick_gen: Generator[Tick, None, None],
    exec_handler: ExecutionHandler,
    risk_handler: RiskHandler,
    initial_state: Optional[BacktestState] = None,
    model_updater_fn: Any = None,
    strategy_mod: Any = None,
) -> Tuple[BacktestResults, BacktestState]:
    """Run backtest with the given tick generator and handlers.

    This is a pure function - given the same inputs, it always returns
    the same results.

    Args:
        bt: Backtest config
        tick_gen: Generator yielding ticks
        exec_handler: Execution handler with execute_signal, execute_risk_event, apply_fill
        risk_handler: Risk handler with check_risk
        initial_state: Optional initial state (default: create from config)
        model_updater_fn: Optional function to update model state
        strategy_fn: Optional strategy function to generate signals
        zscore_fn: Optional z-score calculation function

    Returns:
        Tuple of (BacktestResults, final BacktestState)
    """
    config = bt.config
    symbols = config.symbols

    def get_initial_state():
        start_date = parse_timestamp(config.trading_start)
        state = create_initial_backtest_state(
            symbols=symbols,
            initial_capital=config.initial_capital,
            start_timestamp=start_date,
            rolling_window_size=config.rolling_window_size,
        )
        return state

    # Initialize state
    state = initial_state or get_initial_state()

    for tick in tick_gen:
        can_trade = bt.window.test_start <= tick.timestamp <= bt.window.test_end
        state = _process_tick(
            state=state,
            tick=tick,
            config=config,
            exec_handler=exec_handler,
            risk_handler=risk_handler,
            model_updater_fn=model_updater_fn,
            strategy_fn=strategy_mod.on_tick if strategy_mod else None,
            can_trade=can_trade,
            exec_params=bt.execution_params,
            risk_config=bt.risk_config,
        )

    # Finalize - close all positions
    state = _finalize(state, bt.execution_params)

    # Build results
    equity_series = pd.Series(
        [p.equity for p in state.portfolio.equity_curve],
        index=[p.timestamp for p in state.portfolio.equity_curve],
    )

    pf_result = calculate_portfolio_result(
        equity_series, state.portfolio.trades, state.portfolio.initial_capital
    )

    plot_config: PlotConfig = (
        strategy_mod.plot(state, config) if strategy_mod else PlotConfig()
    )

    return (
        BacktestResults(
            pf=pf_result,
            data=state.candles,
            final_state=state,
            plot_config=plot_config,
        ),
        state,
    )


def _process_tick(
    state: BacktestState,
    tick: Tick,
    config: StrategyConfig,
    exec_handler: ExecutionHandler,
    risk_handler: RiskHandler,
    model_updater_fn: Optional[Callable],
    strategy_fn: Optional[Callable],
    can_trade: bool,
    exec_params: ExecutionParams,
    risk_config: RiskConfig,
) -> BacktestState:
    """Process a single tick through the pipeline."""

    # Handle HTF ticks: append to htf_data
    if tick.interval and tick.interval != config.bar:
        state = _append_htf_tick(state, tick)
        return state

    # Update models
    if model_updater_fn:
        state = model_updater_fn(state, tick)

    # Append candle
    state = _append_candle(state, tick)

    # Execute pending signals from previous tick
    portfolio, pending_signals = _execute_signals(
        state, tick, exec_handler, config, exec_params
    )

    state = merge_bt_state(
        state,
        dict(
            portfolio=portfolio,
            pending_signals=pending_signals,
        ),
    )

    if can_trade and strategy_fn:
        strategy_params = dict(config.strategy_params or {})
        if config.rolling_window_size is not None:
            strategy_params.setdefault(
                "rolling_window_size", config.rolling_window_size
            )
        if config.symbols:
            strategy_params.setdefault("symbols", list(config.symbols))
        new_signals = strategy_fn(state, tick, strategy_params)
    else:
        new_signals = []

    if new_signals:
        pending_signals = pending_signals + list(new_signals)

    state = merge_bt_state(
        state,
        dict(
            portfolio=portfolio,
            pending_signals=pending_signals,
        ),
    )

    # Execute any signals created on this tick
    portfolio, pending_signals = _execute_signals(
        state, tick, exec_handler, config, exec_params
    )

    state = merge_bt_state(
        state,
        dict(
            portfolio=portfolio,
            pending_signals=pending_signals,
        ),
    )

    # Check and execute risk
    state = _check_and_execute_risk(
        state, tick, exec_handler, risk_handler, exec_params, risk_config
    )

    # Update prices
    from src.bt.portfolio.pure import update_prices

    portfolio = update_prices(state.portfolio, tick)

    return merge_bt_state(
        state,
        dict(
            portfolio=portfolio,
            timestamp=tick.timestamp,
        ),
    )


def _append_candle(state: BacktestState, tick: Tick) -> BacktestState:
    """Append a candle to the per-symbol candles dict."""
    new_row = pd.DataFrame(
        {
            "open": [tick.open],
            "high": [tick.high],
            "low": [tick.low],
            "close": [tick.close],
            "volume": [tick.volume],
        },
        index=pd.Index([tick.timestamp]),
    )

    candles = dict(state.candles)
    current = candles.get(tick.symbol)
    if current is None or current.empty:
        candles[tick.symbol] = new_row
    else:
        candles[tick.symbol] = pd.concat([current, new_row])

    return merge_bt_state(state, dict(candles=candles))


def _append_htf_tick(state: BacktestState, tick: Tick) -> BacktestState:
    """Append an HTF tick to the htf_data DataFrame."""
    freq = tick.interval
    if freq is None:
        return state

    new_row = pd.DataFrame(
        {
            "open": [tick.open],
            "high": [tick.high],
            "low": [tick.low],
            "close": [tick.close],
            "volume": [tick.volume],
        },
        index=pd.MultiIndex.from_tuples(
            [(tick.symbol, tick.timestamp)], names=["symbol", "timestamp"]
        ),
    )

    current_htf = state.htf_data.get(freq)
    if current_htf is None or current_htf.empty:
        htf_data = new_row
    else:
        htf_data = pd.concat([current_htf, new_row])

    new_htf_data = dict(state.htf_data)
    new_htf_data[freq] = htf_data

    return merge_bt_state(state, dict(htf_data=new_htf_data))


def _execute_signals(
    state: BacktestState,
    tick: Tick,
    exec_handler: ExecutionHandler,
    config: StrategyConfig,
    exec_params: ExecutionParams,
) -> Tuple[PortfolioState, List[TradeSignal]]:
    """Execute all pending signals."""
    if not state.pending_signals:
        return state.portfolio, []

    portfolio = state.portfolio
    remaining_signals: List[TradeSignal] = []
    for signal in state.pending_signals:
        if signal.symbol != tick.symbol:
            remaining_signals.append(signal)
            continue

        fill = exec_handler.execute_signal(signal, tick, exec_params)
        portfolio = exec_handler.apply_fill(
            portfolio,
            fill,
            position_size_pct=config.position_size,
            stop_loss_pct=config.stop_loss,
            take_profit_pct=config.take_profit,
        )

    return portfolio, remaining_signals


def _check_and_execute_risk(
    state: BacktestState,
    tick: Tick,
    exec_handler: ExecutionHandler,
    risk_handler: RiskHandler,
    exec_params: ExecutionParams,
    risk_config: RiskConfig,
) -> BacktestState:
    """Check risk and execute closes."""
    risk_events = risk_handler.check_risk(state.portfolio, tick, risk_config)

    if not risk_events:
        return state

    portfolio = state.portfolio
    for event in risk_events:
        fill = exec_handler.execute_risk_event(event, tick, exec_params)
        portfolio = exec_handler.apply_fill(portfolio, fill)

    return merge_bt_state(state, dict(portfolio=portfolio, risk_events=risk_events))


def _finalize(state: BacktestState, exec_params: ExecutionParams) -> BacktestState:
    """Close all positions at end of backtest."""
    portfolio = state.portfolio

    for symbol, position in list(portfolio.positions.items()):
        close_signal = TradeSignal(
            action=ActionType.close,
            symbol=symbol,
            timestamp=state.timestamp or pd.Timestamp.now(),
            price=position.last_price,
            reason=TradeExitReason.end,
        )

        fill = FillEvent(
            signal=close_signal,
            filled_qty=abs(position.qty),
            executed_price=position.last_price,
            commission=exec_params.fixed_commission,
            slippage=0.0,
            timestamp=close_signal.timestamp,
        )

        from src.bt.portfolio.pure import apply_fill

        portfolio = apply_fill(portfolio, fill)

    return merge_bt_state(
        state,
        dict(
            portfolio=portfolio,
            pending_signals=[],
            risk_events=(),
        ),
    )


def run(bt: Backtest, data: pd.DataFrame, strat_mod) -> BacktestResults:
    """Convenience function for running backtest with defaults.

    This creates default handlers and runs the backtest.
    Use run_backtest() for full control over handlers.
    """
    from src.bt.engine.handlers import default_execution_handler, default_risk_handler

    gen = ticks_generator(data, bt.config)
    exec_handler = default_execution_handler()
    risk_handler = default_risk_handler()

    results, _ = run_backtest(
        bt, gen, exec_handler, risk_handler, strategy_mod=strat_mod
    )
    return results
