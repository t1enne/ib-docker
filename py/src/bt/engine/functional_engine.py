"""Functional backtest engine adapter.

This provides a functional interface while maintaining compatibility
with the existing backtest engine.

Design principles:
- Dependencies can be injected for testing
- Factory function creates default dependencies for production
- Core logic is as pure as possible
"""

from src.bt.data_feed import DataFeed
from src.bt.algos.pairs_trading_functional import pairs_trading_on_tick

from typing import Iterator, Optional, Dict, Any, Callable
from collections import defaultdict
import pandas as pd
import logging
import asyncio

from src.bt.state import (
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
)
from src.bt.portfolio.pure import apply_fill, update_prices, get_open_position
from src.bt.risk.pure import check_risk
from src.bt.execution.pure import execute_signal, execute_risk_event
from src.bt.types import StrategyConfig, PortfolioResult, EngineWindow, BacktestResults
from src.utils import parse_timestamp

logger = logging.getLogger(__name__)


class FunctionalBacktestEngine:
    """Backtest engine using pure functional state transformations.

    Dependencies can be injected via constructor for testing.
    Use create_functional_engine() for production default setup.
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

        # Data storage for results
        self.z_scores = []
        self.z_timestamps = []
        self.regime_labels = []
        self.regime_timestamps = []

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
        if self._data_feed_factory is not None:
            data_feed = self._data_feed_factory()
        else:
            data_feed = DataFeed(self.config, self._window)

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
        final_state = self._finalize(state)
        self._state = final_state

        # Build results
        return self._build_results(final_state, data_feed)

    def _process_tick(
        self, state: BacktestState, tick: Tick, tick_group: Dict[str, Any]
    ) -> BacktestState:
        """Process single tick through functional pipeline."""

        # Stage 0: Update models FIRST (so z_score is current when generating signals)
        state = self._update_models(state, tick_group)

        # Stage 1: Execute pending signals from PREVIOUS tick
        state = self._execute_signals(state, tick_group)

        # Stage 2: Generate NEW signals from strategy (using updated z_score)
        # This includes BOTH exit signals (if z regressed) and entry signals
        all_signals = pairs_trading_on_tick(
            state, tick_group, self.config.entry_z, self.config.exit_z
        )

        # Separate close and entry signals
        from src.bt.state import ActionType

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
                pending_signals=(),
                model_state=state.model_state,
                risk_events=state.risk_events,
            )

        # Store entry signals for next tick (don't execute immediately)
        state = state.__replace__(pending_signals=tuple(entry_signals))

        # Stage 3: Check risk
        state = self._check_and_execute_risk(state, tick)

        # Stage 4: Update prices
        state = self._update_prices(state, tick)

        return state

    def _update_models(self, state: BacktestState, tick_group: Dict) -> BacktestState:
        """Update model state with new data."""
        # Simplified - just update price buffers
        prices = {sym: tick.close for sym, tick in tick_group.items()}
        new_buffers = state.model_state.price_buffers + (prices,)

        # Trim buffers
        if len(new_buffers) > self.config.rolling_window_size:
            new_buffers = new_buffers[-self.config.rolling_window_size :]

        # Calculate z-score using OLS regression
        z_score = 0.0
        hedge_beta = 1.0
        if len(new_buffers) >= self.config.rolling_window_size:
            z_score, hedge_beta = self._calculate_z_score(new_buffers)

        new_model = ModelState(
            z_score=z_score,
            current_regime=state.model_state.current_regime,
            price_buffers=new_buffers,
            market_data=state.model_state.market_data,
            hedge_beta=hedge_beta,
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
        )

    def _calculate_z_score(self, buffers):
        """Calculate z-score from price buffers using OLS regression."""
        if len(buffers) < 2:
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

        from src.bt.zscore import calculate_rolling_z

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
            pending_signals=(),  # All signals executed
            model_state=state.model_state,
            risk_events=state.risk_events,
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
        )

    def _finalize(self, state: BacktestState) -> BacktestState:
        """Close all positions at end."""
        from src.bt.state import ActionType

        portfolio = state.portfolio

        # Create close signals for all positions
        for symbol, position in portfolio.positions.items():
            close_signal = TradeSignal(
                action=ActionType.close,
                symbol=symbol,
                timestamp=state.timestamp or pd.Timestamp.now(),
                price=position.last_price,
                z_score=0.0,
                reason=None,
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
            pending_signals=(),
            model_state=state.model_state,
            risk_events=(),
        )

    def _build_results(self, state: BacktestState, data_feed: DataFeed):
        """Build backtest results from final state."""
        from src.bt.types import BacktestResults
        from src.bt.metrics import calculate_portfolio_result

        # Calculate results from portfolio
        equity_series = pd.Series(
            [p.equity for p in state.portfolio.equity_curve],
            index=[p.timestamp for p in state.portfolio.equity_curve],
        )

        pf_result = calculate_portfolio_result(
            equity_series, state.portfolio.trades, state.portfolio.initial_capital
        )

        # Build z-scores dataframe
        z_df = pd.DataFrame(
            {"z": self.z_scores}, index=pd.DatetimeIndex(self.z_timestamps)
        )

        return BacktestResults(
            pf=pf_result,
            data=data_feed.candles_df,
            z_scores=z_df,
            regimes=None,
            final_state=state,
        )


def create_functional_engine(config: StrategyConfig) -> FunctionalBacktestEngine:
    """Factory function to create engine with default production dependencies.

    This is the recommended way to create the engine in production.
    For testing, inject dependencies directly via FunctionalBacktestEngine constructor.
    """
    return FunctionalBacktestEngine(config=config)
