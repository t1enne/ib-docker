from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd

import src.bt.indicators as ta
from src.bt.state import BacktestState, TradeSignal, Candle, ActionType, Position
from src.bt.strategies.types import StrategyParams
from src.bt.strategies.utils import open, close
from src.bt.types import PlotConfig, StrategyConfig

STRATEGY_TYPE = "ema_cross"


@dataclass(frozen=True)
class Params(StrategyParams):
    fast: int = 9
    slow: int = 14
    vol_window: int = 20
    vol_multiplier: float = 1.5
    atr_period: int = 14
    atr_threshold: float = 0.5
    ema_threshold: float = 0.05


def is_ranging(
    closes: pd.Series, high: pd.Series, low: pd.Series, params: Params
) -> bool:
    """Detect ranging market using EMA convergence + ATR."""
    if len(closes) < max(params.slow, params.atr_period):
        return True

    ema_fast = ta.ema(closes, params.fast).iloc[-1]
    ema_slow = ta.ema(closes, params.slow).iloc[-1]
    ema_spread = round(abs(ema_fast - ema_slow) / closes.iloc[-1], 4)

    atr_val = ta.atr(high, low, closes, params.atr_period).iloc[-1]
    return atr_val < params.atr_threshold or ema_spread < params.ema_threshold


def volume_confirmed(volume: pd.Series, params: Params) -> bool:
    """Check if current volume is above threshold * average."""
    if len(volume) < params.vol_window:
        return True
    avg_vol = volume.iloc[-params.vol_window :].mean()
    return volume.iloc[-1] > avg_vol * params.vol_multiplier


def on_candle(
    state: BacktestState, candle: Candle, params: Params
) -> List[TradeSignal]:
    symbol = candle.symbol
    position = state.portfolio.positions.get(symbol)
    candles = state.candles[symbol]
    closes = candles["close"]
    high = candles["high"]
    low = candles["low"]

    if len(closes) < params.slow:
        return []

    ema_fast = ta.ema(closes, params.fast).iloc[-1]
    ema_slow = ta.ema(closes, params.slow).iloc[-1]

    regime: str
    if is_ranging(closes, high, low, params):
        regime = "RANGE"
    elif ema_fast > ema_slow:
        regime = "BULL"
    else:
        regime = "BEAR"

    if regime == "RANGE":
        return _handle_ranging(candle, position)

    return _handle_trending(state, candle, params, candles, regime)


def _handle_ranging(tick: Candle, position: Optional[Position]) -> List[TradeSignal]:
    if not position:
        return []
    if position.type == ActionType.short:
        return close(tick, position, "[range] close short")
    return close(tick, position, "[range] close long")


def _handle_trending(
    state: BacktestState,
    tick: Candle,
    params: Params,
    candles: pd.DataFrame,
    regime: str,
) -> List[TradeSignal]:
    closes = candles["close"]
    volumes = candles["volume"]

    ema_slow = ta.ema(closes, params.slow).iloc[-1]
    last_close = closes.iloc[-1]
    position = state.portfolio.positions.get(tick.symbol)

    with_volume = volume_confirmed(volumes, params)
    fast_signal = last_close  # use close instead of ema when volatile
    crossed_above = fast_signal > ema_slow
    crossed_below = ema_slow > fast_signal

    if not position:
        if regime == "BULL" and crossed_above and with_volume:
            return open(tick, ActionType.long, f"[{regime.lower()}] enter")
        if regime == "BEAR" and crossed_below and with_volume:
            return open(tick, ActionType.short, f"[{regime.lower()}] enter")
        return []

    is_long = position.type == ActionType.long
    if is_long and crossed_below:
        return close(tick, position, f"[{regime.lower()}] ema cross below")
    if not is_long and crossed_above:
        return close(tick, position, f"[{regime.lower()}] ema cross above")

    return []


def plot(state: BacktestState, config: StrategyConfig) -> PlotConfig:
    """Calculate EMAs from candles and return as price overlays."""
    params = Params.from_dict(config.strategy_params)

    price_overlays: Dict[str, Dict[str, pd.Series]] = {}

    for symbol in config.symbols:
        closes = state.candles[symbol]["close"]
        volumes = state.candles[symbol]["volume"]
        if len(closes) < params.slow:
            continue

        ema_fast = ta.ema(closes, params.fast)
        ema_slow = ta.ema(closes, params.slow)
        ema_spread = abs(ema_fast - ema_slow) / closes.iloc[-1]

        high = state.candles[symbol]["high"]
        low = state.candles[symbol]["low"]
        atr = ta.atr(high, low, closes, params.atr_period)
        obv = ta.obv(closes, volumes)

        price_overlays[symbol] = {
            f"ema_{params.fast}": ema_fast,
            f"ema_{params.slow}": ema_slow,
        }

        subplots: List[tuple[str, pd.Series]] = [
            (f"ema_spread {params.fast}/{params.slow}", ema_spread),
            ("atr", atr),
            ("obv", obv),
        ]

    return PlotConfig(price_overlays=price_overlays, subplots=subplots)
