"""Functional backtest engine adapter.

This provides a functional interface while maintaining compatibility
with the existing backtest engine.

Design principles:
- Dependencies can be injected for testing
- Factory function creates default dependencies for production
- Core logic is as pure as possible
"""

from src.bt.metrics import calculate_portfolio_result
from src.bt.types import PlotConfig

from src.bt.zscore import calculate_rolling_z

from src.bt.data_feed import DataFeed
from src.bt.algos import (
    ema_cross,
    pairs_trading_functional,
    vol_extension_pullback,
    yesterday_high_breakout,
    breakout_ema,
)
from src.market_data.cache import update_resample_cache, ResampleCache
from src.bt.models.correlation_model import CorrelationModel

from typing import Iterator, Optional, Dict, Any, Callable, List
from collections import defaultdict
import pandas as pd
import logging
import asyncio

from src.bt.state import (
    ActionType,
    BacktestState,
    Tick,
    PortfolioState,
    ModelState,
    MarketDataState,
    TradeSignal,
    FillEvent,
    ExecutionParams,
    RiskConfig,
    create_initial_backtest_state,
    create_execution_params,
    create_risk_config,
    TradeExitReason,
)
from src.bt.portfolio.pure import apply_fill, update_prices, get_open_position
from src.bt.risk.pure import check_risk
from src.bt.execution.pure import execute_signal, execute_risk_event
from src.bt.types import StrategyConfig, PortfolioResult, EngineWindow, BacktestResults
from src.utils import parse_timestamp

logger = logging.getLogger(__name__)


class Engine:
    """Backtest engine using pure functional state transformations.

    Dependencies can be injected via constructor for testing.
    Use create_engine() for production default setup.
    """

    def __init__(
        self,
        config: StrategyConfig,
        # Injected dependencies - all optional
        initial_state: Optional[BacktestState] = None,
        execution_params: Optional[ExecutionParams] = None,
        risk_config: Optional[RiskConfig] = None,
        window: Optional[EngineWindow] = None,
        # Data feed factory - will be called if data_feed is None
        data_feed_factory: Optional[Callable[[], DataFeed]] = None,
    ):
        self.config = config
        self.symbols = config.symbols

        # Initialize or use injected dependencies
        start_date = parse_timestamp(config.trading_start)

        self._state = (
            initial_state
            if initial_state is not None
            else create_initial_backtest_state(
                symbols=config.symbols,
                initial_capital=config.initial_capital,
                start_timestamp=start_date,
                rolling_window_size=config.rolling_window_size,
            )
        )

        self._execution_params = (
            execution_params
            if execution_params is not None
            else create_execution_params(fixed_commission=config.commission)
        )

        self._risk_config = (
            risk_config
            if risk_config is not None
            else create_risk_config(
                stop_loss_pct=config.stop_loss, take_profit_pct=config.take_profit
            )
        )

        self._window = window if window is not None else self._build_window(config)
        self._data_feed_factory = data_feed_factory

        # Initialize resample cache from config
        self._init_resample_cache()

        # Data storage for results
        self.z_scores = []
        self.z_timestamps = []
        self.regime_labels = []
        self.regime_timestamps = []

    def _init_resample_cache(self) -> None:
        """Initialize resample cache with timeframes from config."""
        timeframes = self.config.strategy_params.get("resample_timeframes", [])
        if not timeframes:
            return

        cache: Dict[str, pd.DataFrame] = {}
        anchor: Dict[str, pd.Timestamp] = {}

        self._state = BacktestState(
            portfolio=self._state.portfolio,
            timestamp=self._state.timestamp,
            pending_signals=self._state.pending_signals,
            model_state=self._state.model_state.__replace__(
                resample_cache=cache,
                resample_anchor=anchor,
            ),
            risk_events=self._state.risk_events,
            candles=self._state.candles,
        )

    @property
    def state(self) -> BacktestState:
        """Current backtest state (read-only)."""
        return self._state

    @state.setter
    def state(self, value: BacktestState):
        """Allow setting state for testing."""
        self._state = value

    @property
    def execution_params(self) -> ExecutionParams:
        return self._execution_params

    @property
    def risk_config(self) -> RiskConfig:
        return self._risk_config

    @property
    def window(self) -> EngineWindow:
        return self._window

    def _build_window(self, config: StrategyConfig) -> EngineWindow:
        """Build engine window from config."""
        return EngineWindow(
            train_start=parse_timestamp(config.training_start),
            train_end=parse_timestamp(config.training_end),
            test_start=parse_timestamp(config.trading_start),
            test_end=parse_timestamp(config.trading_end),
        )

    async def run(self) -> "BacktestResults":
        """Run backtest using functional pipeline.

        Creates data feed internally or uses injected factory.
        """
        # Create data feed using factory if provided

        data_feed = (
            self._data_feed_factory()
            if self._data_feed_factory is not None
            else DataFeed(self.config, self._window)
        )
        # sync data
        await data_feed.load(self.config, self._window)

        return await self.run_with_data_feed(data_feed)

    async def run_with_data_feed(self, data_feed: DataFeed) -> "BacktestResults":
        """Run backtest with an injected data feed.

        This is the main entry point when data feed is injected for testing.
        """
        # Process all ticks
        state = self._state
        ticks_by_timestamp = defaultdict(list)

        async for tick in data_feed.get_data_stream():
            if tick is None:
                break

            # Group ticks by timestamp
            ticks_by_timestamp[tick.timestamp].append(tick)

            # Process when we have all symbols for this timestamp
            if len(ticks_by_timestamp[tick.timestamp]) == len(self.symbols):
                tick_group = {t.symbol: t for t in ticks_by_timestamp[tick.timestamp]}

                # Check if we're in trading period
                can_trade = (
                    self._window.test_start <= tick.timestamp <= self._window.test_end
                )

                if can_trade:
                    # Process tick through functional pipeline
                    state = self._process_tick(state, tick, tick_group)
                    self._state = state

                del ticks_by_timestamp[tick.timestamp]

        # Finalize - close all positions
        self._state = self._finalize(state)

        # Build results
        return self._build_results(self._state, data_feed)

    def _process_tick(
        self, state: BacktestState, tick: Tick, tick_group: Dict[str, Any]
    ) -> BacktestState:
        """Process single tick through functional pipeline."""

        # Stage 0: Update models FIRST (so z_score is current when generating signals)
        if self.config.rolling_window_size:
            state = self._update_models(state, tick_group)

        state = self._append_candle(state, tick)

        # if state.model_state.resample_cache:
        #     state = self._update_resample_cache(state, tick)

        # Stage 1: Execute pending signals from PREVIOUS tick
        state = self._execute_signals(state, tick_group)

        def _strat_wrap() -> List[TradeSignal]:
            if self.config.strategy_type == "pnd":
                return pairs_trading_functional.on_tick(
                    state, tick_group, self.config.strategy_params
                )
            elif self.config.strategy_type == "ema_cross":
                return ema_cross.on_tick(state, tick, self.config.strategy_params)
            elif (
                self.config.strategy_type
                == "volatility_expansion_pullback_continuation"
            ):
                return vol_extension_pullback.on_tick(
                    state, tick, self.config.strategy_params
                )
            elif self.config.strategy_type == "yesterday_high":
                return yesterday_high_breakout.on_tick(
                    state, tick, self.config.strategy_params
                )
            elif self.config.strategy_type == "breakout_ema":
                return breakout_ema.on_tick(state, tick, self.config.strategy_params)

            raise ValueError("Unrecognized strat")

        # Stage 2: Generate NEW signals from strategy (using updated z_score)
        # This includes BOTH exit signals (if z regressed) and entry signals

        all_signals = _strat_wrap()

        close_signals = [s for s in all_signals if s.action == ActionType.close]
        entry_signals = [s for s in all_signals if s.action != ActionType.close]

        # Execute close signals IMMEDIATELY (if any)
        if close_signals:
            portfolio = state.portfolio
            for signal in close_signals:
                sig_tick = tick_group.get(signal.symbol)
                if sig_tick is None:
                    continue
                fill = execute_signal(signal, sig_tick, self._execution_params)
                portfolio = apply_fill(
                    portfolio,
                    fill,
                    position_size_pct=self.config.position_size,
                    stop_loss_pct=self.config.stop_loss,
                    take_profit_pct=self.config.take_profit,
                )
            state = BacktestState(
                portfolio=portfolio,
                timestamp=state.timestamp,
                pending_signals=[],
                model_state=state.model_state,
                risk_events=state.risk_events,
                candles=state.candles,
            )

        # Store entry signals for next tick (don't execute immediately)
        state = state.__replace__(pending_signals=(entry_signals))

        # Stage 3: Check risk
        state = self._check_and_execute_risk(state, tick)

        # Stage 4: Update prices
        state = self._update_prices(state, tick)

        return state

    def _update_models(self, state: BacktestState, tick_group: Dict) -> BacktestState:
        """Update model state with new data."""
        if not self.config.rolling_window_size:
            return state
        # Simplified - just update price buffers
        prices = {sym: tick.close for sym, tick in tick_group.items()}
        new_buffers = state.model_state.price_buffers + (prices,)

        # Trim buffers
        if len(new_buffers) > self.config.rolling_window_size:
            new_buffers = new_buffers[-self.config.rolling_window_size :]

        # Calculate z-score using OLS regression (only for pair trading strategies)
        z_score: Optional[float] = None
        hedge_beta = 1.0
        if (
            self.config.strategy_type in ["pnd", "spread"]
            and len(new_buffers) >= self.config.rolling_window_size
        ):
            z_score, hedge_beta = self._calculate_z_score(new_buffers)

        # Calculate correlation matrix for volatility expansion strategy
        correlation_model: Optional[CorrelationModel] = None
        if (
            self.config.strategy_type == "volatility_expansion_pullback_continuation"
            and len(new_buffers) >= self.config.rolling_window_size
        ):
            existing = state.model_state.correlation_model
            if existing is not None:
                correlation_model = existing
            else:
                correlation_model = CorrelationModel(
                    self.symbols, self.config.rolling_window_size
                )
            if correlation_model is not None:
                correlation_model.calculate_correlation_matrix(list(new_buffers))

        new_model = ModelState(
            z_score=z_score,
            current_regime=state.model_state.current_regime,
            price_buffers=new_buffers,
            market_data=state.model_state.market_data,
            hedge_beta=hedge_beta,
            correlation_model=correlation_model,
        )

        # Track z-scores for results
        self.z_scores.append(z_score)
        self.z_timestamps.append(list(tick_group.values())[0].timestamp)

        return BacktestState(
            portfolio=state.portfolio,
            timestamp=list(tick_group.values())[0].timestamp,
            pending_signals=state.pending_signals,
            model_state=new_model,
            risk_events=(),
            candles=state.candles,
        )

    def _append_candle(self, state: BacktestState, tick: Tick):
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

        if state.candles.empty:
            candles = new_row
        else:
            candles = pd.concat([state.candles, new_row])

        return BacktestState(
            portfolio=state.portfolio,
            timestamp=state.timestamp,
            pending_signals=state.pending_signals,
            model_state=state.model_state,
            risk_events=state.risk_events,
            candles=candles,
        )

    def _update_resample_cache(self, state: BacktestState, tick: Tick) -> BacktestState:
        """Update resample cache when higher-timeframe bucket completes.

        Only updates when anchor changes (new bucket started), ensuring no lookahead.
        """
        timeframes = list(state.model_state.resample_cache.keys())

        if not timeframes:
            return state

        cache_obj = ResampleCache(
            cache=state.model_state.resample_cache,
            anchor=state.model_state.resample_anchor,
        )

        new_cache = update_resample_cache(
            cache_obj,
            state.candles,
            timeframes,
            tick.timestamp,
        )

        new_model_state = state.model_state.__replace__(
            resample_cache=new_cache.cache,
            resample_anchor=new_cache.anchor,
        )

        return BacktestState(
            portfolio=state.portfolio,
            timestamp=state.timestamp,
            pending_signals=state.pending_signals,
            model_state=new_model_state,
            risk_events=state.risk_events,
            candles=state.candles,
        )

    def _calculate_z_score(self, buffers):
        """Calculate z-score from price buffers using OLS regression."""
        if len(buffers) < 2:
            return 0.0, 1.0

        if not self.config.rolling_window_size:
            return 0.0, 1.0

        # Get last two symbols
        symbols = list(buffers[0].keys())
        if len(symbols) != 2:
            return 0.0, 1.0

        sym1, sym2 = symbols
        prices1 = [b[sym1] for b in buffers if sym1 in b and sym2 in b]
        prices2 = [b[sym2] for b in buffers if sym1 in b and sym2 in b]

        if len(prices1) < self.config.rolling_window_size:
            return 0.0, 1.0

        z, _, beta = calculate_rolling_z(
            pd.Series(prices1), pd.Series(prices2), self.config.rolling_window_size
        )

        if z != z:  # NaN check
            return 0.0, beta

        return z, beta

    def _execute_signals(self, state: BacktestState, tick_group: dict) -> BacktestState:
        """Execute ALL pending signals with their corresponding ticks.

        Args:
            state: Current state with pending signals
            tick_group: Dict of {symbol: Tick} for current timestamp
        """
        if not state.pending_signals:
            return state

        # Execute ALL pending signals (not filtered by tick symbol)
        portfolio = state.portfolio
        for signal in state.pending_signals:
            # Get the tick for this signal's symbol
            tick = tick_group.get(signal.symbol)
            if tick is None:
                continue

            fill = execute_signal(signal, tick, self._execution_params)
            portfolio = apply_fill(
                portfolio,
                fill,
                position_size_pct=self.config.position_size,
                stop_loss_pct=self.config.stop_loss,
                take_profit_pct=self.config.take_profit,
            )

        return BacktestState(
            portfolio=portfolio,
            timestamp=state.timestamp,
            pending_signals=[],  # All signals executed
            model_state=state.model_state,
            risk_events=state.risk_events,
            candles=state.candles,
        )

    def _check_and_execute_risk(
        self, state: BacktestState, tick: Tick
    ) -> BacktestState:
        """Check risk and execute closes."""
        risk_events = check_risk(state.portfolio, tick, self._risk_config)

        if not risk_events:
            return state

        portfolio = state.portfolio
        for event in risk_events:
            fill = execute_risk_event(event, tick, self._execution_params)
            portfolio = apply_fill(portfolio, fill)

        return BacktestState(
            portfolio=portfolio,
            timestamp=state.timestamp,
            pending_signals=state.pending_signals,
            model_state=state.model_state,
            risk_events=risk_events,
            candles=state.candles,
        )

    def _update_prices(self, state: BacktestState, tick: Tick) -> BacktestState:
        """Update portfolio prices."""
        portfolio = update_prices(state.portfolio, tick)

        return BacktestState(
            portfolio=portfolio,
            timestamp=tick.timestamp,
            pending_signals=state.pending_signals,
            model_state=state.model_state,
            risk_events=state.risk_events,
            candles=state.candles,
        )

    def _finalize(self, state: BacktestState) -> BacktestState:
        """Close all positions at end."""

        portfolio = state.portfolio

        # Create close signals for all positions
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
                commission=self._execution_params.fixed_commission,
                slippage=0.0,
                timestamp=close_signal.timestamp,
            )

            portfolio = apply_fill(portfolio, fill)

        return BacktestState(
            portfolio=portfolio,
            timestamp=state.timestamp,
            pending_signals=[],
            model_state=state.model_state,
            risk_events=(),
            candles=state.candles,
        )

    def _build_results(self, state: BacktestState, data_feed: DataFeed):
        """Build backtest results from final state."""

        # Calculate results from portfolio
        equity_series = pd.Series(
            [p.equity for p in state.portfolio.equity_curve],
            index=[p.timestamp for p in state.portfolio.equity_curve],
        )

        pf_result = calculate_portfolio_result(
            equity_series, state.portfolio.trades, state.portfolio.initial_capital
        )

        # Call strategy plot function
        plot_config = self._get_strategy_plot_fn(state) if self.config.plot else None

        return BacktestResults(
            pf=pf_result,
            data=data_feed.candles_df,
            final_state=state,
            plot_config=plot_config,
        )

    def _get_strategy_plot_fn(self, state: BacktestState) -> Optional[PlotConfig]:
        """Call the strategy's plot function based on strategy type."""
        if self.config.strategy_type == "ema_cross":
            return ema_cross.plot(state, self.config)
        elif self.config.strategy_type == "pnd":
            return pairs_trading_functional.plot(state, self.config)
        elif self.config.strategy_type == "volatility_expansion_pullback_continuation":
            return vol_extension_pullback.plot(state, self.config)
        elif self.config.strategy_type == "yesterday_high":
            return yesterday_high_breakout.plot(state, self.config)
        return None


def create_engine(config: StrategyConfig) -> Engine:
    """Factory function to create engine with default production dependencies.

    This is the recommended way to create the engine in production.
    For testing, inject dependencies directly via Engine constructor.
    """
    return Engine(config=config)
