import pandas as pd
import numpy as np
from typing import List
from collections import deque
from src.bt.algos.base_pairs_strategy import BasePairsStrategy
from src.bt.types import StrategyProtocol, Tick, TradeSignal
from src.utils import get_ols_fit_model


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
    ):
        self.bps = BasePairsStrategy(
            symbols,
            hdata,
            entry_threshold=entry_z,
            rolling_window_size=rolling_window_size,
        )
        # Initialize beta estimation
        self.beta = self._estimate_beta()
        self.spread_buffer = deque(maxlen=rolling_window_size)
        self.retrain_counter = 0
        self.retrain_interval = (
            rolling_window_size  # re-estimate every rolling_window_size ticks
        )

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
        last_z_scores = self.bps.z_scores.tail(9)
        # ema9d = last_z_scores["z"].ewm(span=9).mean().loc[timestamp]
        sym1 = self.bps.symbols[0]
        sym2 = self.bps.symbols[1]

        if abs(z_score) > self.bps.entry_threshold:
            if z_score < -self.bps.entry_threshold:  # and z_score >= ema9d:
                # print(f"ts: {timestamp.date()} long {sym1}, short {sym2}. z: {z_score}")
                return [
                    self.bps._long(self.bps.symbols[0], timestamp),
                    self.bps._short(self.bps.symbols[1], timestamp),
                ]
            elif z_score > self.bps.entry_threshold:  # and z_score <= ema9d:
                # print(f"ts: {timestamp.date()} long {sym2}, short {sym1}. z: {z_score}")
                return [
                    self.bps._long(self.bps.symbols[1], timestamp),
                    self.bps._short(self.bps.symbols[0], timestamp),
                ]

        return []
