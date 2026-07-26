from typing import Optional
import src.indicators.ta as ta
from src.bt.strategies.utils import open, close
from typing import List, Dict, TYPE_CHECKING
from src.bt.state import BacktestState, TradeSignal, Candle, ActionType, Position
from src.bt.types import PlotConfig
import pandas as pd

STRATEGY_TYPE = "breakout_ema"


if TYPE_CHECKING:
    from src.bt.types import StrategyConfig


def is_squeeze(
    closes: pd.Series,
    high: pd.Series,
    low: pd.Series,
    fast: int,
    slow: int,
    atr_period: int = 14,
    ema_threshold: float = 0.01,
    atr_threshold: float = 0.01,
) -> bool:
    """Detect squeeze (volatility contraction) using EMA convergence + ATR."""
    if len(closes) < slow + atr_period:
        return False

    ema_f = ta.ema(closes, fast).iloc[-1]
    ema_s = ta.ema(closes, slow).iloc[-1]
    ema_spread = abs(ema_f - ema_s) / closes.iloc[-1]

    atr_val = ta.atr(high, low, closes, atr_period).iloc[-1]
    atr_ratio = atr_val / closes.iloc[-1]

    return ema_spread < ema_threshold and atr_ratio < atr_threshold


def was_squeezing(
    closes: pd.Series, fast: int, slow: int, threshold: float = 0.01
) -> bool:
    """Check if EMA spread was below threshold in recent bars (indicating squeeze was forming)."""
    if len(closes) < slow + 5:
        return False

    ema_f = ta.ema(closes, fast)
    ema_s = ta.ema(closes, slow)
    spreads = abs(ema_f - ema_s) / closes

    recent_spreads = spreads.iloc[-5:-1]
    return (recent_spreads < threshold).all()


def volume_spike(volume: pd.Series, window: int = 20, threshold: float = 1.5) -> bool:
    """Check if current volume is above threshold * average."""
    if len(volume) < window:
        return False
    avg_vol = volume.iloc[-window:].mean()
    return volume.iloc[-1] > avg_vol * threshold


def on_candle(
    state: BacktestState, candle: Candle, strategy_params: dict
) -> List[TradeSignal]:
    symbol = candle.symbol
    position = state.portfolio.positions.get(symbol)
    candles = state.candles[symbol]
    closes = candles["close"]
    candles["volume"]
    high = candles["high"]
    low = candles["low"]
    fast = strategy_params.get("fast", 9)
    slow = strategy_params.get("slow", 14)
    strategy_params.get("vol_window", 20)
    strategy_params.get("vol_multiplier", 1.5)
    ema_th = strategy_params.get("ema_th", 0.05)
    strategy_params.get("atr_th", 0.05)

    if len(closes) < slow:
        return []

    ema_fast = ta.ema(closes, fast).iloc[-1]
    ema_slow = ta.ema(closes, slow).iloc[-1]
    ema_spread = abs(ema_fast - ema_slow) / closes.iloc[-1]

    if is_squeeze(closes, high, low, fast, slow):
        return handle_squeeze(candle, position)

    if was_squeezing(closes, fast, slow, ema_th):
        return handle_breakout(
            state, candle, strategy_params, candles, ema_fast, ema_slow, ema_spread
        )

    regime = "BULL" if ema_fast > ema_slow else "BEAR"
    return handle_trend(state, candle, strategy_params, candles, regime)


def handle_squeeze(tick: Candle, position: Optional[Position] = None):
    if not position:
        return []

    if position.type == ActionType.short:
        return close(tick, position, "[squeeze] close short")
    if position.type == ActionType.long:
        return close(tick, position, "[squeeze] close long")

    return []


def handle_breakout(
    state: BacktestState,
    tick: Candle,
    strategy_params: dict,
    candles: pd.DataFrame,
    ema_fast: float,
    ema_slow: float,
    ema_spread: float,
):
    strategy_params.get("fast", 9)
    strategy_params.get("slow", 14)
    vol_window = strategy_params.get("vol_window", 20)
    vol_multiplier = strategy_params.get("vol_multiplier", 1.5)

    closes = candles["close"]
    volumes = candles["volume"]

    position = state.portfolio.positions.get(tick.symbol)
    closes.iloc[-1]

    with_volume = volume_spike(volumes, vol_window, vol_multiplier)
    crossed_above = ema_fast > ema_slow
    crossed_below = ema_slow > ema_fast

    if not position:
        if crossed_above and with_volume:
            return open(tick, ActionType.long, "[breakout] enter long")
        if crossed_below and with_volume:
            return open(tick, ActionType.short, "[breakout] enter short")
        return []

    is_long = position.type == ActionType.long
    if is_long and crossed_below:
        return close(tick, position, "[breakout] ema cross below")

    if not is_long and crossed_above:
        return close(tick, position, "[breakout] ema cross above")

    return []


def handle_trend(
    state: BacktestState,
    tick: Candle,
    strategy_params: dict,
    candles: pd.DataFrame,
    regime: str,
):
    fast = strategy_params.get("fast", 9)
    slow = strategy_params.get("slow", 14)
    vol_window = strategy_params.get("vol_window", 20)
    vol_multiplier = strategy_params.get("vol_multiplier", 1.5)

    closes = candles["close"]
    volumes = candles["volume"]

    ema_fast = ta.ema(closes, fast).iloc[-1]
    ema_slow = ta.ema(closes, slow).iloc[-1]

    position = state.portfolio.positions.get(tick.symbol)
    closes.iloc[-1]

    with_volume = volume_spike(volumes, vol_window, vol_multiplier)
    crossed_above = ema_fast > ema_slow
    crossed_below = ema_slow > ema_fast

    if not position:
        if regime == "BULL":
            if crossed_above and with_volume:
                return open(tick, ActionType.long, f"[{regime.lower()}] enter long")
        else:
            if crossed_below and with_volume:
                return open(tick, ActionType.short, f"[{regime.lower()}] enter short")
        return []

    is_long = position.type == ActionType.long
    if is_long and crossed_below:
        return close(tick, position, f"[{regime.lower()}] ema cross below")

    if not is_long and crossed_above:
        return close(tick, position, f"[{regime.lower()}] ema cross above")

    return []


def plot(state: BacktestState, config: StrategyConfig) -> PlotConfig:
    """Calculate EMAs from candles and return as price overlays."""
    fast = config.strategy_params.get("fast", 9)
    slow = config.strategy_params.get("slow", 14)

    price_overlays: Dict[str, Dict[str, pd.Series]] = {}
    subplots = []

    for symbol in config.symbols:
        closes = state.candles[symbol]["close"]
        if len(closes) < slow:
            continue

        ema_fast = ta.ema(closes, fast)
        ema_slow = ta.ema(closes, slow)
        ema_spread = abs(ema_fast - ema_slow) / closes.iloc[-1]

        price_overlays[symbol] = {
            f"ema_{fast}": ema_fast,
            f"ema_{slow}": ema_slow,
        }
        subplots = [("ema_spread", ema_spread)]

    return PlotConfig(price_overlays=price_overlays, subplots=subplots)
