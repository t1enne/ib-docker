import src.bt.indicators as ta
from src.bt.algos.utils import open, close, get_resampled_candles
from typing import List, Dict, TYPE_CHECKING, Optional
from src.bt.state import BacktestState, TradeSignal, Tick, ActionType
from src.bt.types import PlotConfig
import pandas as pd

if TYPE_CHECKING:
    from src.bt.types import StrategyConfig


_signal_state: Dict[str, dict] = {}


def get_signal_state() -> Dict[str, dict]:
    return _signal_state


def reset_signal_state():
    global _signal_state
    _signal_state = {}


def on_tick(
    state: BacktestState, tick: Tick, strategy_params: dict
) -> List[TradeSignal]:
    symbol = tick.symbol
    signal_state = get_signal_state()

    if symbol not in signal_state:
        signal_state[symbol] = {
            "yesterday_high": 0.0,
            "yesterday_low": 0.0,
        }

    symbol_state = signal_state[symbol]
    position = state.portfolio.positions.get(symbol)

    if position:
        return handle_exit(state, tick, strategy_params, symbol_state, position)

    return handle_entry(state, tick, strategy_params, symbol_state)


def handle_entry(
    state: BacktestState,
    tick: Tick,
    strategy_params: dict,
    symbol_state: dict,
) -> List[TradeSignal]:
    symbol = tick.symbol

    gap = strategy_params.get("gap", 1.0)
    stop_loss = strategy_params.get("stop_loss", 3.0)
    take_profit = strategy_params.get("take_profit", 9.0)

    roc_cfg = strategy_params.get("roc_filter", {})
    roc_enabled = roc_cfg.get("enabled", False)
    roc_threshold = roc_cfg.get("threshold", 1.0)

    daily = state.candles[symbol]
    if daily.empty or len(daily) < 2:
        return []

    yesterday_high = daily["high"].iloc[-2]
    yesterday_low = daily["low"].iloc[-2]

    if pd.isna(yesterday_high) or pd.isna(yesterday_low):
        return []

    symbol_state["yesterday_high"] = yesterday_high
    symbol_state["yesterday_low"] = yesterday_low

    if roc_enabled:
        closes = state.candles[symbol]["close"]
        if len(closes) < 2:
            return []

        prev_close = closes.iloc[-2]
        current_close = closes.iloc[-1]

        if prev_close <= 0:
            return []

        roc = ((current_close - prev_close) / prev_close) * 100
        if roc <= roc_threshold:
            return []

    gap_offset = (yesterday_high * gap) / 100
    entry_price = yesterday_high + gap_offset

    if tick.close < yesterday_high:
        return open(tick, ActionType.long, "tick below YH")

    return []


def handle_exit(
    state: BacktestState,
    tick: Tick,
    strategy_params: dict,
    symbol_state: dict,
    position,
) -> List[TradeSignal]:
    symbol = tick.symbol

    stop_loss = strategy_params.get("stop_loss", 3.0)
    take_profit = strategy_params.get("take_profit", 9.0)

    ts_cfg = strategy_params.get("trailing_stop", {})
    ts_enabled = ts_cfg.get("enabled", True)
    ts_activation = ts_cfg.get("activation_pct", 2.0)
    ts_distance = ts_cfg.get("distance_pct", 1.0)

    ema_cfg = strategy_params.get("ema_exit", {})
    ema_enabled = ema_cfg.get("enabled", False)
    ema_period = ema_cfg.get("period", 10)

    entry_price = symbol_state.get("entry_price", position.entry_price)
    current_price = tick.close

    pnl_pct = (
        ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
    )

    # if ts_enabled and pnl_pct >= ts_activation:
    #     trailing_stop_price = current_price * (1 - ts_distance / 100)
    #     if current_price < trailing_stop_price:
    #         return close(tick, position, "trailing_stop", 0)

    if ema_enabled:
        candles = state.candles[symbol]
        if len(candles) >= ema_period:
            ema_val = ta.ema(candles["close"], ema_period).iloc[-1]
            if tick.close < ema_val:
                return close(tick, position, "ema_exit", 0)

    if pnl_pct <= -stop_loss:
        return close(tick, position, "stop_loss", 0)

    if pnl_pct >= take_profit:
        return close(tick, position, "take_profit", 0)

    return []


def plot(state: BacktestState, config: "StrategyConfig") -> PlotConfig:
    strategy_params = config.strategy_params
    gap = strategy_params.get("gap", 1.0)
    stop_loss = strategy_params.get("stop_loss", 3.0)
    take_profit = strategy_params.get("take_profit", 9.0)

    ts_cfg = strategy_params.get("trailing_stop", {})
    ts_enabled = ts_cfg.get("enabled", True)
    ts_activation = ts_cfg.get("activation_pct", 2.0)

    ema_cfg = strategy_params.get("ema_exit", {})
    ema_enabled = ema_cfg.get("enabled", False)
    ema_period = ema_cfg.get("period", 10)

    price_overlays: Dict[str, Dict[str, pd.Series]] = {}
    subplots: List[tuple[str, pd.Series]] = []

    for symbol in config.symbols:
        daily = get_resampled_candles(state, "1D", symbol, completed_only=False)
        if daily.empty:
            continue

        overlays = {}

        if "high" in daily.columns and len(daily) > 0:
            yesterday_high = daily["high"].iloc[-1] if len(daily) >= 1 else None
            if yesterday_high and not pd.isna(yesterday_high):
                gap_offset = (yesterday_high * gap) / 100
                entry_level = yesterday_high + gap_offset
                overlays["yesterday_high"] = pd.Series(
                    entry_level, index=[daily.index[-1]]
                )

        candles = state.candles[symbol]
        if len(candles) >= ema_period:
            ema_val = ta.ema(candles["close"], ema_period)
            overlays[f"ema_{ema_period}"] = ema_val

        if overlays:
            price_overlays[symbol] = overlays

    return PlotConfig(price_overlays=price_overlays, subplots=subplots)
