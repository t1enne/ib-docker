from collections import defaultdict
from dataclasses import dataclass
from typing import (
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
    Union,
    LiteralString,
)

import logging
import pandas as pd

from src.bt.algos.pairs_trading import PairsTradingStrategy, StrategyParams
from src.bt.execution import ExecutionHandler
from src.bt.models.strategy_model import StrategyModel
from src.bt.portfolio import Portfolio, PortfolioProps
from src.bt.risk import RiskManager, RiskManagerProps, TakeProfitEvent
from src.hmm import get_regime_df
from src.utils import parse_timestamp
from src.bt.data_feed import DataFeed
from src.bt.types import (
    ExecutionParams,
    PortfolioResult,
    RegimeState,
    Tick,
    TradeSignal,
    ZScoreState,
    StrategyConfig,
    BacktestResults,
    EngineWindow,
)

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Unified backtesting engine with rolling z-score and optional HMM."""

    def __init__(
        self,
        config: StrategyConfig,
        strategy_class=PairsTradingStrategy,
    ):
        self.strategy_config = config
        self.symbols = config.symbols
        assert len(self.symbols) == 2, "Pairs trading requires exactly 2 symbols"

        self.window = self._build_window(config)
        self.execution_handler = ExecutionHandler(self._execution_params(config))
        self.portfolio = Portfolio(
            self._portfolio_props(config, self.window.test_start)
        )
        self.risk_manager = RiskManager(self._risk_props(config))

        self.data = pd.DataFrame()

        self.pending_signals: List[TradeSignal] = []

        self.model = StrategyModel(
            symbols=self.symbols,
            rolling_window_size=config.rolling_window_size,
            hmm_floating_window=config.hmm_floating_window,
            hmm_retrain_interval=config.hmm_retrain_interval,
        )

        self.strategy = strategy_class(
            symbols=self.symbols,
            strategy_params=self._strategy_params(config),
        )
        self.strategy.model = self.model

        self.z_score_state = ZScoreState(
            scores=[], timestamps=[], scores_synced=[], timestamps_synced=[]
        )
        self.regime_state = RegimeState(labels=[], probs=[], timestamps=[])

        self.data_feed = DataFeed(self.strategy_config, self.window)

    async def run(
        self,
    ) -> BacktestResults:
        """Run the backtest with rolling z-score calculation."""

        await self._run_backtest(self.data_feed)

        results = self._finalize_results()
        return results

    def _build_window(self, strategy: StrategyConfig) -> EngineWindow:
        train_start = parse_timestamp(strategy.training_start)
        train_end = parse_timestamp(strategy.training_end)
        test_start = parse_timestamp(strategy.trading_start)
        test_end = parse_timestamp(strategy.trading_end)

        assert train_start <= train_end, "Training start must precede training end"
        assert test_start <= test_end, "Trading start must precede trading end"

        return EngineWindow(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

    async def _run_backtest(
        self,
        feed: DataFeed,
    ) -> None:
        """Run backtest on test data with rolling z-score and optional HMM."""

        # switch when timestamp > trading_end
        CAN_TRADE = False
        ticks_by_timestamp = defaultdict(list)

        async for tick in feed.get_data_stream():
            if tick is None:
                break

            CAN_TRADE = (
                self.window.train_start <= tick.timestamp <= self.window.test_end
            )

            ticks_by_timestamp[tick.timestamp].append(tick)
            has_ticks = len(ticks_by_timestamp[tick.timestamp]) == len(self.symbols)

            # handle training
            if has_ticks:
                ticks_by_symbol = {
                    t.symbol: t for t in ticks_by_timestamp[tick.timestamp]
                }
                self.model.update(tick.timestamp, ticks_by_symbol)
                self.z_score_state.scores.append(self.model.z_score)
                self.z_score_state.timestamps.append(tick.timestamp)
                self._update_regime_state(tick.timestamp)

            if has_ticks and not CAN_TRADE:
                del ticks_by_timestamp[tick.timestamp]

            if not CAN_TRADE:
                continue

            # handle trading
            for signal in self.pending_signals:
                fill = self.execution_handler.execute(signal, tick)
                self.portfolio.on_fill(fill)

            self.risk_manager.update_trades(self.portfolio.open_trades)
            risk_events = self.risk_manager.on_tick(tick)

            for event in risk_events:
                fill = self.execution_handler.close_order(event, tick)
                self.portfolio.on_fill(fill)

            open_trade = self.portfolio.open_trades.get(tick.symbol)
            self.pending_signals = self.strategy.on_tick(tick, open_trade)
            self.portfolio.update_market_value(tick)

    def _close_open_position(self, tick: Tick) -> None:
        tpe = TakeProfitEvent(
            symbol=tick.symbol,
            trigger_price=tick.close,
            timestamp=tick.timestamp,
        )
        ev = self.execution_handler.close_order(tpe, tick)
        self.portfolio.on_fill(ev)

    def _finalize_results(
        self,
    ) -> BacktestResults:
        """Finalize and return results."""
        last_timestamp = self.window.test_end

        positions = self.portfolio.open_trades.copy().values()
        for pos in positions:
            t = Tick(
                timestamp=last_timestamp,
                symbol=pos.symbol,
                open=pos.last_price,
                high=pos.last_price,
                low=pos.last_price,
                close=pos.last_price,
                volume=0.00,
            )
            self._close_open_position(t)

        z_scores_df = pd.DataFrame(
            {"z": self.z_score_state.scores},
            index=pd.Index(self.z_score_state.timestamps),
        )
        return BacktestResults(
            pf=self.portfolio.get_results(),
            data=self.data_feed.candles_df,
            z_scores=z_scores_df,
            regimes=get_regime_df(self.regime_state),
        )

        # return , self.data, z_scores_df

    def _strategy_params(self, strategy: StrategyConfig) -> StrategyParams:
        return StrategyParams(entry_z=strategy.entry_z, exit_z=strategy.exit_z)

    def _execution_params(self, strategy: StrategyConfig) -> ExecutionParams:
        return ExecutionParams(
            fixed_commission=strategy.commission,
        )

    def _portfolio_props(
        self,
        strategy: StrategyConfig,
        start_date: pd.Timestamp,
    ) -> PortfolioProps:
        return PortfolioProps(
            stop_loss=strategy.stop_loss,
            take_profit=strategy.take_profit,
            initial_capital=strategy.initial_capital,
            position_size=strategy.position_size,
            commission=strategy.commission,
            start_date=start_date,
        )

    def _risk_props(self, strategy: StrategyConfig) -> RiskManagerProps:
        return RiskManagerProps(
            stop_loss_pct=strategy.stop_loss,
            take_profit_pct=strategy.take_profit,
        )

    def _update_regime_state(self, timestamp: pd.Timestamp) -> None:
        self.regime_state.labels.append(self.model.current_regime)
        probs = self.model.get_regime_probability()
        self.regime_state.probs.append(probs.tolist() if probs is not None else None)
        self.regime_state.timestamps.append(timestamp)
