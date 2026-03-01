from src.bt.portfolio import TradeExitReason
import src.bt.indicators as ta
from src.bt.algos.utils import open, close
from typing import List, Dict, TYPE_CHECKING
from src.bt.state import BacktestState, TradeSignal, Tick, ActionType
from src.bt.types import PlotConfig
import pandas as pd

if TYPE_CHECKING:
    from src.bt.types import StrategyConfig

regime = None


def on_tick(
    state: BacktestState, tick: Tick, strategy_params: dict
) -> List[TradeSignal]:
    symbol = tick.symbol
    position = state.portfolio.positions.get(symbol)
    closes = state.candles.xs(tick.symbol)["close"]
    fast = strategy_params.get("fast", 9)
    slow = strategy_params.get("slow", 14)
    snail = strategy_params.get("snail", 50)
    closes = state.candles.xs(tick.symbol)["close"]
    ema_fast = ta.ema(closes, fast).iloc[-1]
    ema_slow = ta.ema(closes, slow).iloc[-1]
    ema_snail = ta.ema(closes, snail).iloc[-1]

    if len(closes) < snail:
        return []

    regime = "BULL" if ema_fast > ema_slow else "BEAR"

    print(regime)
    if regime == "BULL":
        return handle_bull(state, tick, strategy_params)

    return handle_bear(state, tick, strategy_params)


def handle_bear(state: BacktestState, tick: Tick, strategy_params: dict):
    fast = strategy_params.get("fast", 9)
    slow = strategy_params.get("slow", 14)
    snail = strategy_params.get("super_slow", 50)
    closes = state.candles.xs(tick.symbol)["close"]
    ema_fast = ta.ema(closes, fast).iloc[-1]
    ema_slow = ta.ema(closes, slow).iloc[-1]
    ema_snail = ta.ema(closes, snail).iloc[-1]
    position = state.portfolio.positions.get(tick.symbol)

    crossed_above = ema_fast > ema_slow
    crossed_below = ema_slow > ema_fast
    if not position:
        if crossed_below:
            return open(tick, ActionType.short, 0)
        return []

    # has position
    is_long = position.type == ActionType.long
    if is_long and crossed_below:
        return close(tick, position, "ema cross below")
    if not is_long and crossed_above:
        return open(tick, ActionType.short, 0)

    return []


def handle_bull(state: BacktestState, tick: Tick, strategy_params: dict):
    fast = strategy_params.get("fast", 9)
    slow = strategy_params.get("slow", 14)
    snail = strategy_params.get("super_slow", 50)
    closes = state.candles.xs(tick.symbol)["close"]
    ema_fast = ta.ema(closes, fast).iloc[-1]
    ema_slow = ta.ema(closes, slow).iloc[-1]
    ema_snail = ta.ema(closes, snail).iloc[-1]
    position = state.portfolio.positions.get(tick.symbol)

    crossed_above = ema_fast > ema_slow
    crossed_below = ema_slow > ema_fast

    if not position:
        if crossed_above:
            return open(tick, ActionType.long, 0)
        return []

    # has position

    is_long = position.type == ActionType.long
    if is_long and crossed_below:
        return close(tick, position, "ema cross below")
    if not is_long and crossed_above:
        return close(tick, position, "ema cross above")

    return []


def plot(state: BacktestState, config: "StrategyConfig") -> PlotConfig:
    """Calculate EMAs from candles and return as price overlays."""
    fast = config.strategy_params.get("fast", 9)
    slow = config.strategy_params.get("slow", 14)

    price_overlays: Dict[str, Dict[str, pd.Series]] = {}

    for symbol in config.symbols:
        closes = state.candles.xs(symbol)["close"]
        if len(closes) < slow:
            continue

        price_overlays[symbol] = {
            f"ema_{fast}": ta.ema(closes, fast),
            f"ema_{slow}": ta.ema(closes, slow),
        }

    return PlotConfig(price_overlays=price_overlays)
