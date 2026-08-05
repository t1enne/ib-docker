"""Functional backtest module.

Candle processing pipeline (per-candle stages, in order):
  _append_candle       – stash every candle (base + HTF) in accumulator
  _update_model        – run model_updater_fn if provided (base only)
  _execute_pending     – fill signals queued for current symbol from prior candles
  _generate_signals    – run strategy on the last symbol per timestamp,
                          bucket returned signals by symbol into pending dict
  _execute_pending     – same-bar: fill same-symbol signals (skip fill_at_next_open)
  _check_risk          – evaluate stop-loss / take-profit
  _mark_to_market      – update position prices and equity curve

Pipeline invariants:
  - Signals execute before risk on the same bar. A rebalance emitted in
    Stage 4 is filled in Stage 6 before Stage 7 risk check runs, so
    risk events always fire against the post-rebalance position state
    (no stale position_id crashes).
  - Strategies emitting multiple signals for the same symbol in one batch
    must avoid races (e.g. close+reopen): close signals fill at next bar's
    open (Stage 4), open/rebalance fill same-bar (Stage 6), so they never
    collide on the same pass.
  - Signals are bucketed by symbol into a dict. _execute_pending reads
    directly from the current symbol's bucket — no O(N) scan over all
    pending signals.

Usage:
    from src.bt.engine.backtest import Backtest, candle_generator, run_backtest
    from src.bt.engine.handlers import default_execution_handler, default_risk_handler

    bt = Backtest(config)
    gen = candle_generator(df, config.symbols)
    results, state = run_backtest(bt, gen, exec_handler, risk_handler)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.bt.engine.candle_store import CandleStore, CandleRows
from src.bt.engine.utils import candle_generator, merge_bt_state

from dataclasses import dataclass, field, replace
from typing import Generator, Tuple, Optional, Any, Callable

import numpy as np
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
from src.bt.types import StrategyConfig, EngineWindow, BacktestResults
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

    # Create CandleStore once — wraps rows by reference, mutates in-place.
    # Strategies access it as state.candles (Mapping interface) + .latest()/.count().
    store = CandleStore(rows)
    state = merge_bt_state(state, dict(candles=store))

    for candle in candle_gen:
        can_trade = bt.window.test_start <= candle.timestamp <= bt.window.test_end
        is_base = not candle.interval or candle.interval == config.bars[0]

        # Stage 1: stash EVERY candle (base + HTF) into the same accumulator
        rows, state = _append_candle(rows, state, candle, config.bars[0])

        # HTF-only candles: accumulate and skip rest of pipeline
        if not is_base:
            continue

        # Stage 2: update models
        if model_updater_fn:
            state = model_updater_fn(state, candle)

        # Stage 3: execute pending signals (from prior candles)
        state = _execute_pending(
            state, candle, exec_handler, config, bt.execution_params
        )

        # Stage 5: generate new signals (only on last symbol per timestamp)
        state = _generate_signals(
            state,
            candle,
            resolved_params,
            strategy_fn,
            last_symbol,
            can_trade,
            rows,
        )

        # Stage 6: execute signals generated this tick (skip fill_at_next_open
        # signals — they fill at next bar's open via Stage 4)
        state = _execute_pending(
            state,
            candle,
            exec_handler,
            config,
            bt.execution_params,
            skip_next_open=True,
        )

        # Stage 7: check stop-loss / take-profit
        state = _check_risk(
            state,
            candle,
            exec_handler,
            risk_handler,
            bt.execution_params,
            bt.risk_config,
        )

        # Stage 8: mark to market
        state = _mark_to_market(state, candle)

    # Finalize: close positions, build results
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
    # Use first benchmark curve for alpha/beta (typically the primary equity index).
    # If no benchmark symbols configured, alpha/beta will be 0.0/1.0.
    first_bm = next(iter(bm_curves.values()), None) if bm_curves else None
    pf_result = calculate_portfolio_result(
        equity_series,
        state.portfolio.trades,
        state.portfolio.initial_capital,
        benchmark_curve=first_bm,
        equity_points=state.portfolio.equity_curve,
    )

    return (
        BacktestResults(
            pf=pf_result,
            data=state.candles,
            final_state=state,
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
            config.bars[0],
        )
        closes: dict[str, pd.Series] = {}
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
                closes[bm_sym] = bm_eq
            except KeyError:
                pass

        # Composite 50/50 buy-and-hold rebalanced: half capital in each symbol
        if len(closes) == 2:
            sym_a, sym_b = list(closes.keys())
            aligned = pd.concat(
                [closes[sym_a].rename("a"), closes[sym_b].rename("b")],
                axis=1,
            ).dropna()
            if len(aligned) > 1:
                ret_a = aligned["a"].pct_change().fillna(0.0)
                ret_b = aligned["b"].pct_change().fillna(0.0)
                avg_ret = (ret_a + ret_b) / 2.0
                cum = (1.0 + avg_ret).cumprod()
                bm_curves["50/50"] = cum * config.initial_capital
    except Exception:
        pass

    return bm_curves


def _bucket_signals(
    signals: tuple[TradeSignal, ...],
) -> dict[str, tuple[TradeSignal, ...]]:
    """Bucket flat signal list by symbol."""
    buckets: dict[str, list[TradeSignal]] = {}
    for s in signals:
        buckets.setdefault(s.symbol, []).append(s)
    return {sym: tuple(v) for sym, v in buckets.items()}


def _execute_pending(
    state: BacktestState,
    candle: Candle,
    exec_handler: ExecutionHandler,
    config: StrategyConfig,
    exec_params: ExecutionParams,
    skip_next_open: bool = False,
) -> BacktestState:
    """Stage 4/6: Execute pending signals for the current symbol.

    Reads from state.pending_signals[symbol] directly — no filtering needed.
    When skip_next_open is True (Stage 6, same-bar), signals with
    fill_at_next_open=True are deferred to the next bar's Stage 4 call.
    """
    symbol = candle.symbol
    queued = state.pending_signals.get(symbol, ())
    if not queued:
        return state

    portfolio = state.portfolio
    deferred: list[TradeSignal] = []
    for signal in queued:
        if skip_next_open and signal.fill_at_next_open:
            deferred.append(signal)
            continue
        fill = exec_handler.execute_signal(signal, candle, exec_params)
        portfolio = exec_handler.apply_fill(portfolio, fill)

    new_pending = dict(state.pending_signals)
    if deferred:
        new_pending[symbol] = tuple(deferred)
    else:
        new_pending.pop(symbol, None)

    return merge_bt_state(state, dict(portfolio=portfolio, pending_signals=new_pending))


def _generate_signals(
    state: BacktestState,
    candle: Candle,
    resolved_params: object,
    strategy_fn: Optional[Callable],
    last_symbol: Optional[str],
    can_trade: bool,
    rows: CandleRows,
) -> BacktestState:
    """Run strategy on last symbol per timestamp, bucket signals by symbol."""
    if not (can_trade and strategy_fn and candle.symbol == last_symbol):
        return state

    # Advance cursor so CandleStore only sees data up to this timestamp
    state.candles.advance(candle.timestamp)

    new_signals = strategy_fn(state, candle, resolved_params)
    if not new_signals:
        return state

    # Merge into existing pending dict — signals for same symbol accumulate
    pending = dict(state.pending_signals)
    for sym, sigs in _bucket_signals(tuple(new_signals)).items():
        existing = pending.get(sym, ())
        pending[sym] = existing + sigs

    return merge_bt_state(state, dict(pending_signals=pending))


def _check_risk(
    state: BacktestState,
    candle: Candle,
    exec_handler: ExecutionHandler,
    risk_handler: RiskHandler,
    exec_params: ExecutionParams,
    risk_config: RiskConfig,
) -> BacktestState:
    """Stage 8: Check stop-loss / take-profit and execute risk closes.

    risk_handler.check_risk returns (events, updated_portfolio). The updated
    portfolio carries persisted SL/TP levels (initialised or trailed) even
    when no risk event fires.
    """
    risk_events, portfolio = risk_handler.check_risk(
        state.portfolio, candle, risk_config
    )

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
    base_interval: str,
) -> BacktestState:
    """Append aligned {sym: close} pair when all base-interval symbols share timestamp."""
    if candle.symbol != (symbols[-1] if symbols else None):
        return state

    candle_ts_ns = np.datetime64(candle.timestamp.to_datetime64())

    for sym in symbols:
        key = (sym, base_interval)
        sym_rows = rows.get(key)
        if sym_rows is None:
            return state
        n = int(sym_rows["_len"][0])
        if n == 0:
            return state
        if sym_rows["timestamp"][n - 1] != candle_ts_ns:
            return state

    pair: dict[str, float] = {}
    for sym in symbols:
        key = (sym, base_interval)
        n = int(rows[key]["_len"][0])
        pair[sym] = float(rows[key]["close"][n - 1])

    new_ms = replace(
        state.model_state,
        price_buffers=state.model_state.price_buffers + (pair,),
    )
    return merge_bt_state(state, dict(model_state=new_ms))


def _append_candle(
    rows: CandleRows,
    state: BacktestState,
    candle: Candle,
    base_interval: str,
) -> Tuple[CandleRows, BacktestState]:
    """Stash candle row into numpy column arrays keyed by (symbol, interval).

    All candles — base and HTF — land in the same accumulator.
    """
    key = (candle.symbol, candle.interval or base_interval)
    if key not in rows:
        rows[key] = {
            "timestamp": np.empty(256, dtype="datetime64[ms]"),
            "open": np.empty(256, dtype=np.float64),
            "high": np.empty(256, dtype=np.float64),
            "low": np.empty(256, dtype=np.float64),
            "close": np.empty(256, dtype=np.float64),
            "volume": np.empty(256, dtype=np.float64),
            "_len": np.array([0], dtype=np.int64),
        }

    cols = rows[key]
    n = int(cols["_len"][0])

    if n >= len(cols["timestamp"]):
        new_cap = n * 2
        for col_name in ("timestamp", "open", "high", "low", "close", "volume"):
            new_arr = np.empty(new_cap, dtype=cols[col_name].dtype)
            new_arr[:n] = cols[col_name]
            cols[col_name] = new_arr

    cols["timestamp"][n] = np.datetime64(candle.timestamp.to_datetime64())
    cols["open"][n] = candle.open
    cols["high"][n] = candle.high
    cols["low"][n] = candle.low
    cols["close"][n] = candle.close
    cols["volume"][n] = candle.volume
    cols["_len"][0] = n + 1

    return rows, state


def _finalize(state: BacktestState, exec_params: ExecutionParams) -> BacktestState:
    """Close all positions at end of backtest."""
    portfolio = state.portfolio

    for symbol, positions_tuple in list(portfolio.positions.items()):
        for position in positions_tuple:
            close_signal = TradeSignal(
                action=ActionType.close,
                symbol=symbol,
                timestamp=state.timestamp or pd.Timestamp.now(),
                price=position.last_price,
                reason=TradeExitReason.end,
                position_id=position.position_id,
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
            pending_signals={},
            risk_events=(),
        ),
    )


def _resolve_model_updater(config: "StrategyConfig") -> Any | None:
    """Build a model_updater_fn from config.model_updater if present."""
    mu = config.model_updater
    if not mu or not isinstance(mu, dict):
        return None

    mu_type = mu.get("type")
    if mu_type == "dual_online":
        from src.bt.regime.model_updater import create_dual_online_updater

        cfg_d = mu.get("dual_online", {})
        return create_dual_online_updater(
            n_regimes=cfg_d.get("n_regimes", 3),
            window_size=cfg_d.get("window_size", 252),
            vol_window=cfg_d.get("vol_window", 20),
            momentum_window=cfg_d.get("momentum_window", 10),
            retrain_interval=cfg_d.get("retrain_interval", 50),
            random_state=cfg_d.get("random_state", 42),
            trend_fast=cfg_d.get("trend_fast", 50),
            trend_slow=cfg_d.get("trend_slow", 200),
            range_threshold_pct=cfg_d.get("range_threshold_pct", 0.005),
            trend_bar=cfg_d.get("trend_bar"),
            vol_bar=cfg_d.get("vol_bar"),
        )

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

    if mu_type == "kalman_pairs":
        from src.indicators.kalman.model_updater import create_kalman_pairs_updater

        kp_cfg = mu.get("kalman_pairs", {})
        pair = kp_cfg.get("pair")
        if pair is not None and isinstance(pair, list) and len(pair) == 2:
            pair = (str(pair[0]), str(pair[1]))
        return create_kalman_pairs_updater(
            pair=pair,
            process_noise=kp_cfg.get("process_noise", 1e-4),
            measurement_noise=kp_cfg.get("measurement_noise", 1e-3),
            ols_warmup=kp_cfg.get("ols_warmup", 50),
            adaptive=kp_cfg.get("adaptive", True),
            vol_window=kp_cfg.get("vol_window", 20),
            z_window=kp_cfg.get("z_window", 20),
            warmup_bars=kp_cfg.get("warmup_bars", 150),
        )

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
