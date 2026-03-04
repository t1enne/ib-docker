import src.bt.indicators as ta
from src.bt.portfolio import TradeExitReason
from src.bt.algos.utils import open, close
from typing import List, Dict, Optional
from src.bt.state import BacktestState, TradeSignal, Tick, ActionType, Position
from src.bt.types import PlotConfig, StrategyConfig
import pandas as pd

regime = None


def is_ranging(
    closes: pd.Series, high: pd.Series, low: pd.Series, strategy_params: dict
) -> bool:
    """Detect ranging market using EMA convergence + ATR."""
    fast = strategy_params.get("fast", 9)
    slow = strategy_params.get("slow", 14)
    # atr
    atr_threshold = strategy_params.get("atr_threshold", 0.5)
    atr_period = strategy_params.get("atr_period", 14)

    # ema
    ema_threshold = strategy_params.get("ema_threshold", 0.05)
    ema_fast = ta.ema(closes, fast).iloc[-1]
    ema_slow = ta.ema(closes, slow).iloc[-1]
    ema_spread = round(abs(ema_fast - ema_slow) / closes.iloc[-1], 4)

    if len(closes) < max(slow, atr_period):
        return True

    atr_val = ta.atr(high, low, closes, atr_period).iloc[-1]
    atr_ranging = atr_val < atr_threshold
    ema_ranging = ema_spread < ema_threshold

    if atr_ranging:
        return True

    return False


def handle_ranging(tick: Tick, position: Optional[Position] = None):
    if not position:
        return []

    if position and position.type == ActionType.short:
        return close(tick, position, "[bear] ranging - short")
    if position and position.type == ActionType.long:
        return close(tick, position, "[bear] ranging - long")

    return []


def volume_confirmed(
    volume: pd.Series, window: int = 20, threshold: float = 1.5
) -> bool:
    """Check if current volume is above threshold * average."""
    if len(volume) < window:
        return True
    avg_vol = volume.iloc[-window:].mean()
    return volume.iloc[-1] > avg_vol * threshold


def on_tick(
    state: BacktestState, tick: Tick, strategy_params: dict
) -> List[TradeSignal]:
    symbol = tick.symbol
    position = state.portfolio.positions.get(symbol)
    candles = state.candles.xs(symbol)
    closes = candles["close"]
    volumes = candles["volume"]
    high = candles["high"]
    low = candles["low"]
    fast = strategy_params.get("fast", 9)
    slow = strategy_params.get("slow", 14)
    vol_window = strategy_params.get("vol_window", 20)
    vol_multiplier = strategy_params.get("vol_multiplier", 1.5)

    if len(closes) < slow:
        return []

    ema_fast = ta.ema(closes, fast).iloc[-1]
    ema_slow = ta.ema(closes, slow).iloc[-1]
    regime = "BULL" if ema_fast > ema_slow else "BEAR"
    if is_ranging(closes, high, low, strategy_params):
        regime = "RANGE"

    if regime == "RANGE":
        return handle_ranging(tick, position)

    if regime == "BULL":
        return handle_bull(state, tick, strategy_params, candles)

    return handle_bear(state, tick, strategy_params, candles)


def handle_bear(
    state: BacktestState, tick: Tick, strategy_params: dict, candles: pd.DataFrame
):
    return handle_entry_exit(state, tick, strategy_params, candles, "BEAR")


def handle_bull(
    state: BacktestState, tick: Tick, strategy_params: dict, candles: pd.DataFrame
):
    return handle_entry_exit(state, tick, strategy_params, candles, "BULL")


def handle_entry_exit(
    state: BacktestState,
    tick: Tick,
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
    last_close = closes.iloc[-1]
    position = state.portfolio.positions.get(tick.symbol)

    with_volume = volume_confirmed(volumes, vol_window, vol_multiplier)
    hedge = 1.0
    # if extremely volatile, use the close instead of the ema
    fast_signal = last_close
    crossed_above = fast_signal > ema_slow
    crossed_below = ema_slow > fast_signal

    if not position:
        if regime == "BULL":
            if crossed_above and with_volume:
                if hedge != 1.0:
                    print("Hedging!")
                return open(tick, ActionType.long, f"emaspread: {0}", hedge)
        else:
            if crossed_below and with_volume:
                if hedge != 1.0:
                    print("Hedging!")
                return open(tick, ActionType.short, f"emaspread: {0}", hedge)
        return []

    if not position:
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
    atr_period = config.strategy_params.get("atr_period", 14)

    price_overlays: Dict[str, Dict[str, pd.Series]] = {}

    for symbol in config.symbols:
        closes = state.candles.xs(symbol)["close"]
        volumes = state.candles.xs(symbol)["volume"]
        if len(closes) < slow:
            continue

        ema_fast = ta.ema(closes, fast)
        ema_slow = ta.ema(closes, slow)
        ema_spread = abs(ema_fast - ema_slow) / closes.iloc[-1]

        high = state.candles.xs(symbol)["high"]
        low = state.candles.xs(symbol)["low"]
        atr = ta.atr(high, low, closes, atr_period)

        obv = ta.obv(closes, volumes)

        price_overlays[symbol] = {
            f"ema_{fast}": ema_fast,
            f"ema_{slow}": ema_slow,
        }

        sublots = [
            (f"ema_spread {fast}/{slow}", ema_spread),
            ("atr", atr),
            ("obv", obv),
        ]

    return PlotConfig(price_overlays=price_overlays, subplots=sublots)
