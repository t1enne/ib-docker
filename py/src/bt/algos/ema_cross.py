from src.bt.portfolio import TradeExitReason
import src.bt.indicators as ta
from src.bt.algos.utils import open, close
from typing import List, Dict, TYPE_CHECKING
from src.bt.state import BacktestState, TradeSignal, Tick, ActionType
from src.bt.types import PlotConfig
import pandas as pd

if TYPE_CHECKING:
    from src.bt.types import StrategyConfig


def on_tick(
    state: BacktestState, tick: Tick, strategy_params: dict
) -> List[TradeSignal]:
    symbol = tick.symbol
    position = state.portfolio.positions.get(symbol)

    fast = strategy_params.get("fast", 9)
    slow = strategy_params.get("slow", 14)
    closes = state.candles.xs(tick.symbol)["close"]
    if len(closes) < slow:
        return []

    ema_fast = ta.ema(closes, fast)
    ema_slow = ta.ema(closes, slow)

    crossed_below = ema_slow > ema_fast
    if position and crossed_below:
        return close(tick, position, "ema cross below", 0)

    if ema_fast > ema_slow and not position:
        return open(tick, ActionType.long, 0)

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

        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()

        price_overlays[symbol] = {
            f"ema_{fast}": ema_fast,
            f"ema_{slow}": ema_slow,
        }

    return PlotConfig(price_overlays=price_overlays)
