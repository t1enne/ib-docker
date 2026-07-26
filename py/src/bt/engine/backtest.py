"""Functional backtest module.

Candle processing pipeline (per-candle stages, in order):
  _reject              – skip HTF bars that don't match base interval
  _update_model        – run model_updater_fn if provided
  _append_candle       – stash candle row in accumulator
  _update_price_buffers – align close prices for pairs strategies
  _execute_pending     – fill any signals queued from prior candles
  _generate_signals    – run strategy on the last symbol per timestamp
  _execute_new         – immediately fill signals generated this tick
  _check_risk          – evaluate stop-loss / take-profit
  _mark_to_market      – update position prices and equity curve

Usage:
    from src.bt.engine.backtest import Backtest, candle_generator, run_backtest
    from src.bt.engine.handlers import default_execution_handler, default_risk_handler

    bt = Backtest(config)
    gen = candle_generator(df, config.symbols)
    results, state = run_backtest(bt, gen, exec_handler, risk_handler)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.bt.engine.utils import candle_generator, merge_bt_state

from dataclasses import dataclass, field, replace
from typing import Generator, Tuple, Optional, Any, List, Callable

import pandas as pd

from src.bt.metrics import calculate_portfolio_result

if TYPE_CHECKING:
    from src.bt.types import StrategyConfig
from src.bt.state import (
    ActionType,
    BacktestState,
    Candle,
    TradeSignal,
    FillEvent,
    ExecutionParams,
    RiskConfig,
    create_initial_backtest_state,
    create_execution_params,
    create_risk_config,
    TradeExitReason,
)
from src.bt.types import StrategyConfig, EngineWindow, BacktestResults, PlotConfig
from src.bt.engine.handlers import ExecutionHandler, RiskHandler
from src.utils import parse_timestamp

import numpy as np

# Per-candle row accumulator — column-major numpy arrays keyed by symbol.
# Each symbol maps to {timestamp: ndarray, open: ndarray, ...} for zero-copy DataFrame build at flush.
CandleRows = dict[str, dict[str, np.ndarray]]


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
    candle_gen: Generator[Candle, None, None],
    exec_handler: ExecutionHandler,
    risk_handler: RiskHandler,
    initial_state: Optional[BacktestState] = None,
    model_updater_fn: Any = None,
    strategy_mod: Any = None,
) -> Tuple[BacktestResults, BacktestState]:
    """Run backtest with the given candle generator and handlers.

    This is a pure function - given the same inputs, it always returns
    the same results.

    Args:
        bt: Backtest config
        candle_gen: Generator yielding Candles (OHLCV bars)
        exec_handler: Execution handler with execute_signal, execute_risk_event, apply_fill
        risk_handler: Risk handler with check_risk
        initial_state: Optional initial state (default: create from config)
        model_updater_fn: Optional function to update model state
        strategy_mod: Optional strategy module (must have on_candle(state, candle, params))

    Returns:
        Tuple of (BacktestResults, final BacktestState)
    """
    config = bt.config
    symbols = config.symbols
    strategy_fn = strategy_mod.on_candle if strategy_mod else None
    last_symbol = symbols[-1] if symbols else None

    # Resolve typed params once if strategy defines them
    from src.bt.strategies import resolve_params

    resolved_params = resolve_params(config.strategy_type, config.strategy_params)

    def get_initial_state():
        start_date = parse_timestamp(config.trading_start)
        return create_initial_backtest_state(
            symbols=symbols,
            initial_capital=config.initial_capital,
            start_timestamp=start_date,
            rolling_window_size=config.rolling_window_size,
        )

    state = initial_state or get_initial_state()
    rows: CandleRows = {}

    for candle in candle_gen:
        can_trade = bt.window.test_start <= candle.timestamp <= bt.window.test_end

        # Stage 1: reject HTF bars that don't match base interval
        if candle.interval and candle.interval != config.bar:
            state = _append_htf_bar(state, candle)
            continue

        # Stage 2: update models
        if model_updater_fn:
            state = model_updater_fn(state, candle)

        # Stage 3: stash candle row
        rows, state = _append_candle(rows, state, candle)

        # Stage 4: align close prices for pairs strategies
        state = _update_price_buffers(rows, state, candle, symbols)

        # Stage 5: execute pending signals (from prior candles)
        state = _execute_pending(
            state, candle, exec_handler, config, bt.execution_params
        )

        # Stage 6: generate new signals (only on last symbol per timestamp)
        state = _generate_signals(
            state,
            candle,
            resolved_params,
            strategy_fn,
            last_symbol,
            can_trade,
            rows,
        )

        # Stage 7: execute signals generated this tick (skip fill_at_next_open
        # signals — they fill at next bar's open via Stage 5)
        state = _execute_pending(
            state,
            candle,
            exec_handler,
            config,
            bt.execution_params,
            skip_next_open=True,
        )

        # Stage 8: check stop-loss / take-profit
        state = _check_risk(
            state,
            candle,
            exec_handler,
            risk_handler,
            bt.execution_params,
            bt.risk_config,
        )

        # Stage 9: mark to market
        state = _mark_to_market(state, candle)

    # Finalize: flush row batches, close positions, build results
    state = _flush_candle_batches(rows, state)
    state = _finalize(state, bt.execution_params)

    # Build equity series, deduplicating by timestamp (equity curve
    # accumulates one point per candle = N points per timestamp).
    # Take the last equity value per unique timestamp.
    raw_equity = pd.DataFrame(
        [(p.timestamp, p.equity) for p in state.portfolio.equity_curve],
        columns=["ts", "equity"],
    )
    equity_series = raw_equity.groupby("ts")["equity"].last()
    # Slice to trading window for metrics
    equity_series = equity_series[
        (equity_series.index >= bt.window.test_start)
        & (equity_series.index <= bt.window.test_end)
    ]

    bm_curves: dict[str, pd.Series] = _get_bench_curves(config, bt)
    pf_result = calculate_portfolio_result(
        equity_series,
        state.portfolio.trades,
        state.portfolio.initial_capital,
        benchmark_curve=bm_curves.get("SPY"),
    )

    return (
        BacktestResults(
            pf=pf_result,
            data=state.candles,
            final_state=state,
            plot_config=PlotConfig(),
            benchmark_curves=bm_curves,
        ),
        state,
    )


def _get_bench_curves(config: StrategyConfig, bt: Backtest):
    from src.bt.data_feed import load_candles

    if not config.benchmark_symbols:
        return {}

    bm_curves: dict[str, pd.Series] = {}

    try:
        bm_df = load_candles(
            config.benchmark_symbols,
            bt.window.train_start,
            bt.window.test_end,
            config.bar,
        )
        for bm_sym in config.benchmark_symbols:
            try:
                bm_close = bm_df.xs(bm_sym, axis=1, level=0)["close"]
                # Slice to trading window for comparison
                bm_close = bm_close[
                    (bm_close.index >= bt.window.test_start)
                    & (bm_close.index <= bt.window.test_end)
                ]
                if len(bm_close) < 2:
                    continue
                # Normalize to same initial capital as strategy
                bm_eq = bm_close / bm_close.iloc[0] * config.initial_capital
                bm_curves[bm_sym] = bm_eq
            except KeyError:
                pass
    except Exception:
        pass

    return bm_curves


def _execute_pending(
    state: BacktestState,
    candle: Candle,
    exec_handler: ExecutionHandler,
    config: StrategyConfig,
    exec_params: ExecutionParams,
    skip_next_open: bool = False,
) -> BacktestState:
    """Stage 5/7: Execute all pending signals for the current symbol.

    When skip_next_open is True (Stage 7, same-bar), signals with
    fill_at_next_open=True are deferred to the next bar's Stage 5 call.
    """
    if not state.pending_signals:
        return state

    portfolio = state.portfolio
    remaining: List[TradeSignal] = []
    for signal in state.pending_signals:
        if signal.symbol != candle.symbol:
            remaining.append(signal)
            continue
        if skip_next_open and signal.fill_at_next_open:
            remaining.append(signal)
            continue

        fill = exec_handler.execute_signal(signal, candle, exec_params)
        portfolio = exec_handler.apply_fill(
            portfolio,
            fill,
            position_size_pct=config.position_size,
            stop_loss_pct=config.stop_loss,
            take_profit_pct=config.take_profit,
        )

    return merge_bt_state(state, dict(portfolio=portfolio, pending_signals=remaining))


def _generate_signals(
    state: BacktestState,
    candle: Candle,
    resolved_params: object,
    strategy_fn: Optional[Callable],
    last_symbol: Optional[str],
    can_trade: bool,
    rows: CandleRows,
) -> BacktestState:
    """Stage 6: Run strategy on last symbol per timestamp only."""
    if not (can_trade and strategy_fn and candle.symbol == last_symbol):
        return state

    # Build state.candles on-demand from numpy arrays (only for this tick)
    state = merge_bt_state(state, dict(candles=_build_candles(rows)))

    new_signals = strategy_fn(state, candle, resolved_params)
    if not new_signals:
        return state

    pending = state.pending_signals + list(new_signals)
    return merge_bt_state(state, dict(pending_signals=pending))


def _check_risk(
    state: BacktestState,
    candle: Candle,
    exec_handler: ExecutionHandler,
    risk_handler: RiskHandler,
    exec_params: ExecutionParams,
    risk_config: RiskConfig,
) -> BacktestState:
    """Stage 8: Check stop-loss / take-profit and execute risk closes."""
    risk_events = risk_handler.check_risk(state.portfolio, candle, risk_config)
    if not risk_events:
        return state

    portfolio = state.portfolio
    for event in risk_events:
        fill = exec_handler.execute_risk_event(event, candle, exec_params)
        portfolio = exec_handler.apply_fill(portfolio, fill)

    return merge_bt_state(state, dict(portfolio=portfolio, risk_events=risk_events))


def _mark_to_market(state: BacktestState, candle: Candle) -> BacktestState:
    """Stage 9: Update position prices and append equity point."""
    from src.bt.portfolio.pure import update_prices

    portfolio = update_prices(state.portfolio, candle)
    return merge_bt_state(state, dict(portfolio=portfolio, timestamp=candle.timestamp))


def _update_price_buffers(
    rows: CandleRows,
    state: BacktestState,
    candle: Candle,
    symbols: list[str],
) -> BacktestState:
    """Stage 4: Append aligned {sym: close} pair when all symbols share this timestamp."""
    if candle.symbol != (symbols[-1] if symbols else None):
        return state

    candle_ts_ns = np.datetime64(candle.timestamp.to_datetime64())

    # Verify all symbols have this timestamp
    for sym in symbols:
        sym_rows = rows.get(sym)
        if sym_rows is None:
            return state
        n = int(sym_rows["_len"][0])
        if n == 0:
            return state
        if sym_rows["timestamp"][n - 1] != candle_ts_ns:
            return state

    pair: dict[str, float] = {}
    for sym in symbols:
        n = int(rows[sym]["_len"][0])
        pair[sym] = float(rows[sym]["close"][n - 1])

    new_ms = replace(
        state.model_state,
        price_buffers=state.model_state.price_buffers + (pair,),
    )
    return merge_bt_state(state, dict(model_state=new_ms))


def _append_candle(
    rows: CandleRows, state: BacktestState, candle: Candle
) -> Tuple[CandleRows, BacktestState]:
    """Stage 3: Stash candle row into numpy column arrays (fast append).

    DataFrames are built once at flush. Strategies access state.candles
    which is populated on-demand from the column arrays in _build_candles.
    """
    sym = candle.symbol
    if sym not in rows:
        # Pre-allocate column arrays with room to grow
        rows[sym] = {
            "timestamp": np.empty(256, dtype="datetime64[ms]"),
            "open": np.empty(256, dtype=np.float64),
            "high": np.empty(256, dtype=np.float64),
            "low": np.empty(256, dtype=np.float64),
            "close": np.empty(256, dtype=np.float64),
            "volume": np.empty(256, dtype=np.float64),
            "_len": np.array([0], dtype=np.int64),
        }

    cols = rows[sym]
    n = int(cols["_len"][0])

    # Grow by 2x if full
    if n >= len(cols["timestamp"]):
        new_cap = n * 2
        for key in ("timestamp", "open", "high", "low", "close", "volume"):
            new_arr = np.empty(new_cap, dtype=cols[key].dtype)
            new_arr[:n] = cols[key]
            cols[key] = new_arr

    cols["timestamp"][n] = np.datetime64(candle.timestamp.to_datetime64())
    cols["open"][n] = candle.open
    cols["high"][n] = candle.high
    cols["low"][n] = candle.low
    cols["close"][n] = candle.close
    cols["volume"][n] = candle.volume
    cols["_len"][0] = n + 1

    return rows, state


def _build_candles(rows: CandleRows) -> dict[str, pd.DataFrame]:
    """Build DataFrames from numpy column arrays (called on-demand)."""
    candles: dict[str, pd.DataFrame] = {}
    for sym, cols in rows.items():
        n = int(cols["_len"][0])
        if n == 0:
            continue
        ts_arr = cols["timestamp"][:n]
        df = pd.DataFrame(
            {
                "open": cols["open"][:n],
                "high": cols["high"][:n],
                "low": cols["low"][:n],
                "close": cols["close"][:n],
                "volume": cols["volume"][:n],
            },
            index=pd.DatetimeIndex(ts_arr),
        )
        candles[sym] = df
    return candles


def _flush_candle_batches(rows: CandleRows, state: BacktestState) -> BacktestState:
    """Build final DataFrames from numpy arrays at end of backtest."""
    if not rows:
        return state
    candles = _build_candles(rows)
    return merge_bt_state(state, dict(candles=candles))


def _append_htf_bar(state: BacktestState, candle: Candle) -> BacktestState:
    """Append an HTF bar to the htf_data DataFrame."""
    freq = candle.interval
    if freq is None:
        return state

    new_row = pd.DataFrame(
        {
            "open": [candle.open],
            "high": [candle.high],
            "low": [candle.low],
            "close": [candle.close],
            "volume": [candle.volume],
        },
        index=pd.MultiIndex.from_tuples(
            [(candle.symbol, candle.timestamp)], names=["symbol", "timestamp"]
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


def _resolve_model_updater(config: "StrategyConfig") -> Any | None:
    """Build a model_updater_fn from config.model_updater if present."""
    mu = config.model_updater
    if not mu or not isinstance(mu, dict):
        return None

    mu_type = mu.get("type")
    if mu_type == "hmm_online":
        from src.bt.regime.model_updater import create_hmm_online_updater

        hmm_cfg = mu.get("hmm_online", {})
        return create_hmm_online_updater(
            n_regimes=hmm_cfg.get("n_regimes", 3),
            window_size=hmm_cfg.get("window_size", 500),
            vol_window=hmm_cfg.get("vol_window", 20),
            momentum_window=hmm_cfg.get("momentum_window", 10),
            retrain_interval=hmm_cfg.get("retrain_interval", 50),
            random_state=hmm_cfg.get("random_state", 42),
        )

    if mu_type == "sma":
        from src.bt.regime.model_updater import create_regime_model_updater
        from src.bt.regime.detectors import create_sma_detector

        sma_cfg = mu.get("sma", {})
        detector = create_sma_detector(
            fast_window=sma_cfg.get("fast_window", 20),
            slow_window=sma_cfg.get("slow_window", 50),
            range_threshold_pct=sma_cfg.get("range_threshold_pct", 0.005),
        )
        return create_regime_model_updater(detector)

    if mu_type == "volatility":
        from src.bt.regime.model_updater import create_regime_model_updater
        from src.bt.regime.detectors import create_volatility_detector

        vol_cfg = mu.get("volatility", {})
        detector = create_volatility_detector(
            vol_window=vol_cfg.get("vol_window", 20),
            low_vol_pctile=vol_cfg.get("low_vol_pctile", 0.25),
            high_vol_pctile=vol_cfg.get("high_vol_pctile", 0.75),
            direction_window=vol_cfg.get("direction_window", 50),
        )
        return create_regime_model_updater(detector)

    return None


def run(bt: Backtest, data: pd.DataFrame, strat_mod) -> BacktestResults:
    """Convenience function for running backtest with defaults.

    This creates default handlers, resolves model_updater from config,
    and runs the backtest.  Use run_backtest() for full control.
    """
    from src.bt.engine.handlers import default_execution_handler, default_risk_handler

    gen = candle_generator(data, bt.config)
    exec_handler = default_execution_handler()
    risk_handler = default_risk_handler()
    model_updater_fn = _resolve_model_updater(bt.config)

    results, _ = run_backtest(
        bt,
        gen,
        exec_handler,
        risk_handler,
        model_updater_fn=model_updater_fn,
        strategy_mod=strat_mod,
    )
    return results
