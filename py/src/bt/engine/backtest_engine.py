from collections import defaultdict
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, AsyncGenerator, Tuple, Dict, Optional
from src.bt.portfolio import Portfolio, PortfolioProps
from src.bt.risk import RiskManager, RiskManagerProps, TakeProfitEvent
from src.bt.models.strategy_model import StrategyModel
from src.utils import read_candles
from src.hmm import get_regime_df
from src.bt.algos.pairs_trading import PairsTradingStrategy, StrategyParams
from src.bt.types import (
    Tick,
    PortfolioResult,
    TradeSignal,
    ExecutionParams,
    ZScoreState,
    RegimeState,
)
from src.bt.execution import ExecutionHandler


logger = logging.getLogger(__name__)


class DataFeed:
    """Async data feed for n symbols from SQLite."""

    def __init__(
        self,
        symbols: List[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date

    async def get_data_stream(self) -> AsyncGenerator[Optional[Tick]]:
        """Returns an async generator that yields market data ticks for all symbols."""
        candles_list = [
            read_candles(x, self.start_date, self.end_date) for x in self.symbols
        ]

        ticks = []
        for i in range(len(candles_list)):
            df = candles_list[i]
            symbol = self.symbols[i]
            for idx, row in df.iterrows():
                ticks.append(
                    Tick(
                        timestamp=idx,
                        symbol=symbol,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row["Volume"]),
                    )
                )

        ticks.sort(key=lambda x: x.timestamp)

        for tick in ticks:
            yield tick


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_data: Dict[str, pd.DataFrame]
    test_data: Dict[str, pd.DataFrame]


class BacktestEngine:
    """Unified backtesting engine with rolling z-score calculation and optional HMM regime detection."""

    def __init__(
        self,
        strategy: StrategyParams,
        symbols: List[str],
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        rolling_window_size: int = 20,
        initial_capital: float = 10000,
        position_size: float = 0.1,
        commission: float = 0.001,
        stop_loss: float = 0.10,
        take_profit: float = 1.0,
        execution_params: ExecutionParams = ExecutionParams(),
        hmm_floating_window: Optional[int] = None,
        hmm_retrain_interval: Optional[int] = None,
    ):
        self.strategy = PairsTradingStrategy(
            symbols=symbols,
            strategy_params=strategy,
        )

        self.symbols = symbols
        self.rolling_window_size = rolling_window_size
        self.execution_handler = ExecutionHandler(execution_params)

        self.train_start = pd.Timestamp(train_start)
        self.train_end = pd.Timestamp(train_end)
        self.test_start = pd.Timestamp(test_start)
        self.test_end = pd.Timestamp(test_end)

        self.portfolio = Portfolio(
            PortfolioProps(
                stop_loss=stop_loss,
                take_profit=take_profit,
                initial_capital=initial_capital,
                position_size=position_size,
                commission=commission,
                start_date=self.test_start,
            )
        )

        self.risk_manager = RiskManager(
            RiskManagerProps(
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
            )
        )

        self.data = {
            symbol: read_candles(
                symbol,
                self.test_start.strftime("%Y-%m-%d"),
                self.test_end.strftime("%Y-%m-%d"),
            )
            for symbol in symbols
        }

        self.pending_signals: List[TradeSignal] = []

        # Initialize strategy model with HMM config
        self.model = StrategyModel(
            symbols=symbols,
            rolling_window_size=rolling_window_size,
            hmm_floating_window=hmm_floating_window,
            hmm_retrain_interval=hmm_retrain_interval,
        )

        # Wire the model into the strategy
        self.strategy.model = self.model

        # Track z-scores for results/plotting
        self.z_score_state = ZScoreState(
            scores=[], timestamps=[], scores_synced=[], timestamps_synced=[]
        )

        # Track HMM regime for results/plotting
        self.regime_state = RegimeState(labels=[], probs=[], timestamps=[])

    def _compute_windows(self) -> List[WalkForwardWindow]:
        """Compute walk-forward windows."""
        train_data = {
            symbol: read_candles(
                symbol,
                self.train_start.strftime("%Y-%m-%d"),
                self.train_end.strftime("%Y-%m-%d"),
            )
            for symbol in self.symbols
        }
        test_data = {
            symbol: read_candles(
                symbol,
                self.test_start.strftime("%Y-%m-%d"),
                self.test_end.strftime("%Y-%m-%d"),
            )
            for symbol in self.symbols
        }

        return [
            WalkForwardWindow(
                train_start=self.train_start,
                train_end=self.train_end,
                test_start=self.test_start,
                test_end=self.test_end,
                train_data=train_data,
                test_data=test_data,
            )
        ]

    async def run(
        self,
    ) -> Tuple[
        PortfolioResult, Dict[str, pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame]
    ]:
        """Run the backtest with rolling z-score calculation."""
        windows = self._compute_windows()

        for window in windows:
            await self._run_backtest(window.train_data, window.test_data)

        results, data, _ = self._finalize_results()
        z_scores_df = pd.DataFrame(
            {"z": self.z_score_state.scores},
            index=pd.Index(self.z_score_state.timestamps),
        )

        # Build regime dataframe if HMM was enabled
        regime_df = get_regime_df(self.regime_state)

        return (results, data, z_scores_df, regime_df)

    async def _run_backtest(
        self,
        train_data: Dict[str, pd.DataFrame],
        test_data: Dict[str, pd.DataFrame],
    ):
        """Run backtest on test data with rolling z-score and optional HMM.

        Args:
            train_data: Training period data (used to pre-seed the model)
            test_data: Trading period data (signals generated here)
        """
        # Pre-seed model with training period data
        await self._preseed_model(train_data)

        # Now run trading period
        feed = DataFeed(self.symbols, str(self.test_start), str(self.test_end))
        tick_groups = defaultdict(list)

        async for tick in feed.get_data_stream():
            if tick is None:
                break

            for signal in self.pending_signals:
                fill = self.execution_handler.execute(signal, tick)
                self.portfolio.on_fill(fill)

            tick_groups[tick.timestamp].append(tick)

            if len(tick_groups[tick.timestamp]) == len(self.symbols):
                # Build tick group for this timestamp
                tick_group = {t.symbol: t for t in tick_groups[tick.timestamp]}

                # Update the model (computes z-score, updates market data, regime, etc.)
                self.model.update(tick.timestamp, tick_group)

                # Track z-score for results/plotting
                if len(self.model.market_data) >= self.rolling_window_size:
                    self.z_score_state.scores.append(self.model.z_score)
                    self.z_score_state.timestamps.append(tick.timestamp)

                # Track regime for plotting (always record for alignment)
                self.regime_state.labels.append(self.model.current_regime)
                probs = self.model.get_regime_probability()
                self.regime_state.probs.append(
                    probs.tolist() if probs is not None else None
                )
                self.regime_state.timestamps.append(tick.timestamp)

                del tick_groups[tick.timestamp]

            self.risk_manager.update_trades(self.portfolio.open_trades)
            risk_events = self.risk_manager.on_tick(tick)

            for event in risk_events:
                fill = self.execution_handler.close_order(event, tick)
                self.portfolio.on_fill(fill)

            open_trade = self.portfolio.open_trades.get(tick.symbol)
            # Strategy now accesses z_score via self.model.z_score
            self.pending_signals = self.strategy.on_tick(tick, open_trade)
            self.portfolio.update_market_value(tick)

    async def _preseed_model(self, train_data: Dict[str, pd.DataFrame]):
        """Pre-seed the model with training period data.

        This populates market_data with training period bars so that both
        the z-score model and the HMM have a full lookback window available
        from the start of trading.
        """
        # Build ticks from training data
        all_ticks = []
        for symbol, df in train_data.items():
            for idx, row in df.iterrows():
                all_ticks.append(
                    Tick(
                        timestamp=idx,
                        symbol=symbol,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row["Volume"]),
                    )
                )

        # Sort by timestamp
        all_ticks.sort(key=lambda x: x.timestamp)

        # Group by timestamp and feed to model
        tick_groups = defaultdict(list)
        for tick in all_ticks:
            tick_groups[tick.timestamp].append(tick)

        for timestamp in sorted(tick_groups.keys()):
            if len(tick_groups[timestamp]) == len(self.symbols):
                tick_group = {t.symbol: t for t in tick_groups[timestamp]}
                self.model.update(timestamp, tick_group)

    def _close_open_position(self, tick: Tick):
        """Close open positions at the end of the BT"""
        tpe = TakeProfitEvent(
            symbol=tick.symbol,
            trigger_price=tick.close,
            timestamp=tick.timestamp,
        )
        ev = self.execution_handler.close_order(tpe, tick)
        self.portfolio.on_fill(ev)

    def _finalize_results(
        self,
    ) -> Tuple[PortfolioResult, Dict[str, pd.DataFrame], pd.DataFrame]:
        """Finalize and return results."""
        last_timestamp = max(df.index[-1] for df in self.data.values())
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
            self._close_open_position(t)  # mock a tick to close the position

        _ = {symbol: df["Close"].iloc[-1] for symbol, df in self.data.items()}

        z_scores_df = pd.DataFrame(
            {"z": self.z_score_state.scores},
            index=pd.Index(self.z_score_state.timestamps),
        )

        return self.portfolio.get_results(), self.data, z_scores_df
