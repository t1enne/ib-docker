import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from src.bt.algos.base_pairs_strategy import BasePairsStrategy
from src.bt.types import StrategyProtocol, Tick, TradeSignal
from src.utils import get_ols_fit_model


@dataclass
class PositionState:
    entry_timestamp: pd.Timestamp
    entry_z: float
    side: str
    symbols: Tuple[str, str]
    bars_held: int = 0


class PairsTradingStrategy(StrategyProtocol):
    """Pairs trading strategy with z-score recalc at intervals."""

    bps: BasePairsStrategy
    ema9d: dict[pd.Timestamp, float] = dict()

    def __init__(
        self,
        symbols: List[str],
        hdata: dict[str, pd.DataFrame],
        entry_z: float,
        rolling_window_size: int,
        exit_threshold: float = 0.5,
        time_decay_bars: int = 20,
    ):
        self.bps = BasePairsStrategy(
            symbols,
            hdata,
            entry_threshold=entry_z,
            rolling_window_size=rolling_window_size,
        )
        self.beta = self._estimate_beta()
        self.spread_buffer = deque(maxlen=rolling_window_size)
        self.retrain_counter = 0
        self.retrain_interval = rolling_window_size
        self.exit_threshold = exit_threshold
        self.time_decay_bars = time_decay_bars
        self.position: Optional[PositionState] = None

    def on_tick(self, tick: Tick) -> List[TradeSignal]:
        """Process data from signal_queue and put signals into order_queue."""
        symbol = tick.symbol
        timestamp = tick.timestamp
        close = tick.close
        # .ewm(span=9, adjust=False).mean().iloc(-1))

        # Add to historical data
        self.bps.hdata[symbol].loc[timestamp, "Close"] = close
        # Add to pending ticks
        self.bps.pending_ticks[timestamp][symbol] = close

        # If we have data for both symbols at this timestamp, process
        if len(self.bps.pending_ticks[timestamp]) != 2:
            return []
        # Calculate z-score
        z_score = self._calculate_zscore()
        if z_score is None:
            # Clear pending for this timestamp
            del self.bps.pending_ticks[timestamp]
            return []
        # Generate signal
        self.bps.z_scores.loc[timestamp, "z"] = z_score

        signals = self._calculate_signal(timestamp)
        if len(signals) > 0:
            return signals

        return []

    def _estimate_beta(self) -> float:
        """Estimate beta from the last rolling_window_size points."""
        s1_closes = self.bps.hdata[self.bps.symbols[0]]["Close"].dropna()
        s2_closes = self.bps.hdata[self.bps.symbols[1]]["Close"].dropna()
        if (
            len(s1_closes) < self.bps.rolling_window_size
            or len(s2_closes) < self.bps.rolling_window_size
        ):
            # Fallback to all data
            tail1 = s1_closes
            tail2 = s2_closes
        else:
            tail1 = s1_closes.tail(self.bps.rolling_window_size)
            tail2 = s2_closes.tail(self.bps.rolling_window_size)
        if len(tail1) < 2:
            return 1.0  # default
        model = get_ols_fit_model(tail1, tail2)
        _, beta = model.params
        return beta

    def _calculate_zscore(self) -> float | None:
        """Calculate z-score using fixed beta and rolling spread statistics."""
        s1_closes = self.bps.hdata[self.bps.symbols[0]]["Close"].dropna()
        s2_closes = self.bps.hdata[self.bps.symbols[1]]["Close"].dropna()
        if len(s1_closes) != len(s2_closes) or len(s1_closes) < 2:
            return None

        # Get latest prices
        latest_s1 = s1_closes.iloc[-1]
        latest_s2 = s2_closes.iloc[-1]

        # Periodic beta re-estimation
        self.retrain_counter += 1
        if self.retrain_counter % self.retrain_interval == 0:
            self.beta = self._estimate_beta()

        # Calculate spread
        spread = latest_s1 - self.beta * latest_s2
        self.spread_buffer.append(spread)

        if len(self.spread_buffer) < 2:
            return None

        # Calculate rolling z-score
        spreads = np.array(self.spread_buffer)
        mean = np.mean(spreads)
        std = np.std(spreads, ddof=1)
        if std == 0:
            return 0.0
        z_score = (spread - mean) / std
        return round(z_score, 2)

    def _calculate_signal(self, timestamp: pd.Timestamp) -> List[TradeSignal]:
        """Generate buy/sell/close signal based on z-score."""
        z_score = self.bps._get_z(timestamp)
        sym1 = self.bps.symbols[0]
        sym2 = self.bps.symbols[1]

        if self.position is not None:
            self.position.bars_held += 1

            exit_signals = self._check_exit_conditions(timestamp, z_score)
            if exit_signals:
                return exit_signals

        if abs(z_score) > self.bps.entry_threshold:
            if z_score < -self.bps.entry_threshold:
                self.position = PositionState(
                    entry_timestamp=timestamp,
                    entry_z=z_score,
                    side="long_spread",
                    symbols=(sym1, sym2),
                    bars_held=0,
                )
                return [
                    self.bps._long(sym1, timestamp),
                    self.bps._short(sym2, timestamp),
                ]
            elif z_score > self.bps.entry_threshold:
                self.position = PositionState(
                    entry_timestamp=timestamp,
                    entry_z=z_score,
                    side="short_spread",
                    symbols=(sym1, sym2),
                    bars_held=0,
                )
                return [
                    self.bps._long(sym2, timestamp),
                    self.bps._short(sym1, timestamp),
                ]

        return []

    def _check_exit_conditions(
        self, timestamp: pd.Timestamp, z_score: float
    ) -> List[TradeSignal]:
        """Check if position should exit via z-score reversion or time decay."""
        if self.position is None:
            return []

        if self.position.side == "long_spread" and z_score > -self.exit_threshold:
            return self._close_position(timestamp)
        elif self.position.side == "short_spread" and z_score < self.exit_threshold:
            return self._close_position(timestamp)

        if self.position.bars_held >= self.time_decay_bars:
            return self._close_position(timestamp)

        return []

    def _close_position(self, timestamp: pd.Timestamp) -> List[TradeSignal]:
        """Generate close signals for both legs of the spread."""
        if self.position is None:
            return []

        sym1, sym2 = self.position.symbols
        self.position = None
        return [
            self.bps._close(sym1, timestamp),
            self.bps._close(sym2, timestamp),
        ]
