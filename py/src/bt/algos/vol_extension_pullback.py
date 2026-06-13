from src.bt.portfolio import TradeExitReason
import src.bt.indicators as ta
from src.bt.algos.utils import open, close
from typing import List, Dict, TYPE_CHECKING, Optional
from src.bt.state import BacktestState, TradeSignal, Candle, ActionType
from src.bt.types import PlotConfig
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from src.bt.types import StrategyConfig


class SignalPhase:
    COMPRESSION = "COMPRESSION"
    BREAKOUT = "BREAKOUT"
    PULLBACK = "PULLBACK"
    ENTERED = "ENTERED"
    CLOSED = "CLOSED"


_signal_state: Dict[str, dict] = {}


def get_signal_state() -> Dict[str, dict]:
    return _signal_state


def reset_signal_state():
    global _signal_state
    _signal_state = {}


def on_candle(
    state: BacktestState, candle: Candle, strategy_params: dict
) -> List[TradeSignal]:
    symbol = candle.symbol
    signal_state = get_signal_state()

    if symbol not in signal_state:
        signal_state[symbol] = {
            "phase": SignalPhase.COMPRESSION,
            "breakout_high": 0.0,
            "breakout_date": None,
            "pullback_start": None,
            "atr_at_entry": 0.0,
            "entry_price": 0.0,
            "breakout_candle_range": 0.0,
        }

    symbol_state = signal_state[symbol]
    position = state.portfolio.positions.get(symbol)

    if position:
        return handle_exit(state, candle, strategy_params, symbol_state, position)

    if symbol_state["phase"] == SignalPhase.ENTERED:
        return []

    max_positions = strategy_params.get("max_concurrent_positions", 5)
    if len(state.portfolio.positions) >= max_positions:
        return []

    return handle_entry(state, candle, strategy_params, symbol_state)


def handle_entry(
    state: BacktestState,
    tick: Candle,
    strategy_params: dict,
    symbol_state: dict,
) -> List[TradeSignal]:
    symbol = tick.symbol

    comp = strategy_params.get("compression", {})
    atr_short = comp.get("atr_short", 14)
    atr_long = comp.get("atr_long", 100)
    compression_threshold = comp.get("compression_threshold", 0.6)

    brk = strategy_params.get("breakout", {})
    breakout_lookback = brk.get("breakout_lookback_days", 20)
    tr_multiplier = brk.get("true_range_multiplier", 1.5)
    vol_multiplier = brk.get("volume_multiplier", 1.5)
    vol_lookback = brk.get("volume_lookback_days", 30)

    pull = strategy_params.get("pullback", {})
    ema_short = pull.get("ema_short_period", 5)
    ema_long = pull.get("ema_long_period", 10)
    retracement_min = pull.get("retracement_min", 0.3)
    retracement_max = pull.get("retracement_max", 0.5)
    max_pullback_days = pull.get("max_pullback_days", 4)

    cont = strategy_params.get("continuation_trigger", {})
    require_close_above_prev_high = cont.get("require_close_above_prev_high", True)
    allow_bullish_engulfing = cont.get("allow_bullish_engulfing", True)

    trend = strategy_params.get("trend_filter", {})
    ma_fast = trend.get("ma_fast", 20)
    ma_slow = trend.get("ma_slow", 50)
    require_positive_slope = trend.get("require_positive_slope", True)
    momentum_lookback = trend.get("momentum_lookback_days", 63)
    require_positive_return = trend.get("require_positive_return", True)

    risk = strategy_params.get("risk_management", {})
    risk_per_trade = risk.get("risk_per_trade", 0.01)

    corr = strategy_params.get("correlation_filter", {})
    corr_enabled = corr.get("enabled", True)
    max_corr = corr.get("max_pairwise_correlation", 0.8)

    candles = state.candles[symbol]
    if len(candles) < atr_long + breakout_lookback + vol_lookback:
        return []

    closes = candles["close"]
    highs = candles["high"]
    lows = candles["low"]
    volumes = candles["volume"]

    atr_14 = ta.atr(highs, lows, closes, atr_short)
    atr_100 = ta.atr(highs, lows, closes, atr_long)

    if pd.isna(atr_14.iloc[-1]) or pd.isna(atr_100.iloc[-1]):
        return []

    compression_ratio = atr_14.iloc[-1] / atr_100.iloc[-1]

    if symbol_state["phase"] == SignalPhase.COMPRESSION:
        if compression_ratio < compression_threshold:
            symbol_state["phase"] = SignalPhase.BREAKOUT
            symbol_state["breakout_high"] = highs.iloc[-breakout_lookback:].max()
            symbol_state["breakout_date"] = tick.timestamp

    if symbol_state["phase"] != SignalPhase.BREAKOUT:
        return []

    current_close = closes.iloc[-1]
    current_volume = volumes.iloc[-1]
    avg_volume = volumes.iloc[-vol_lookback:].mean()
    true_range = max(
        highs.iloc[-1] - lows.iloc[-1],
        abs(highs.iloc[-1] - closes.iloc[-2]),
        abs(lows.iloc[-1] - closes.iloc[-2]),
    )

    breakout_triggered = (
        current_close > symbol_state["breakout_high"]
        and true_range > tr_multiplier * atr_14.iloc[-1]
        and current_volume > vol_multiplier * avg_volume
    )

    if not breakout_triggered:
        return []

    symbol_state["phase"] = SignalPhase.PULLBACK
    symbol_state["pullback_start"] = tick.timestamp
    symbol_state["breakout_candle_range"] = highs.iloc[-1] - lows.iloc[-1]
    symbol_state["breakout_high"] = highs.iloc[-1]

    return []


def handle_pullback_entry(
    state: BacktestState,
    tick: Candle,
    strategy_params: dict,
    symbol_state: dict,
) -> List[TradeSignal]:
    symbol = tick.symbol

    comp = strategy_params.get("compression", {})
    atr_short = comp.get("atr_short", 14)

    pull = strategy_params.get("pullback", {})
    ema_short = pull.get("ema_short_period", 5)
    ema_long = pull.get("ema_long_period", 10)
    retracement_min = pull.get("retracement_min", 0.3)
    retracement_max = pull.get("retracement_max", 0.5)
    max_pullback_days = pull.get("max_pullback_days", 4)

    cont = strategy_params.get("continuation_trigger", {})
    require_close_above_prev_high = cont.get("require_close_above_prev_high", True)
    allow_bullish_engulfing = cont.get("allow_bullish_engulfing", True)

    trend = strategy_params.get("trend_filter", {})
    ma_fast = trend.get("ma_fast", 20)
    ma_slow = trend.get("ma_slow", 50)
    require_positive_slope = trend.get("require_positive_slope", True)
    momentum_lookback = trend.get("momentum_lookback_days", 63)
    require_positive_return = trend.get("require_positive_return", True)

    risk = strategy_params.get("risk_management", {})
    risk_per_trade = risk.get("risk_per_trade", 0.01)

    corr = strategy_params.get("correlation_filter", {})
    corr_enabled = corr.get("enabled", True)
    max_corr = corr.get("max_pairwise_correlation", 0.8)

    candles = state.candles[symbol]
    closes = candles["close"]
    highs = candles["high"]
    lows = candles["low"]
    opens = candles["open"]
    volumes = candles["volume"]

    if len(candles) < ema_long + ma_slow + momentum_lookback:
        return []

    ema_5 = ta.ema(closes, ema_short)
    ema_10 = ta.ema(closes, ema_long)
    ma_20 = ta.sma(closes, ma_fast)
    ma_50 = ta.sma(closes, ma_slow)

    if pd.isna(ema_5) or pd.isna(ema_10) or pd.isna(ma_20) or pd.isna(ma_50):
        return []

    current_close = closes.iloc[-1]
    prev_high = highs.iloc[-2]
    current_open = opens.iloc[-1]
    current_low = lows.iloc[-1]
    prev_close = closes.iloc[-2]
    prev_open = opens.iloc[-2]

    pullback_days = (tick.timestamp - symbol_state["pullback_start"]).days

    breakout_high = symbol_state["breakout_high"]
    breakout_range = symbol_state.get("breakout_candle_range", 0)

    price_at_ema = (closes.iloc[-1] >= ema_5 and closes.iloc[-1] <= ema_10) or (
        closes.iloc[-1] >= ema_10 and closes.iloc[-1] <= ema_5
    )

    retracement_pct = 0.0
    if breakout_range > 0:
        retracement_pct = (breakout_high - current_close) / breakout_range

    in_retracement_zone = retracement_min <= retracement_pct <= retracement_max

    consolidation = False
    if pullback_days >= 2:
        recent_volumes = volumes.iloc[-pullback_days:]
        if all(
            recent_volumes.iloc[i] > recent_volumes.iloc[i + 1]
            for i in range(len(recent_volumes) - 1)
        ):
            consolidation = True

    is_pulling_back = price_at_ema or in_retracement_zone or consolidation

    if not is_pulling_back:
        return []

    entry_triggered = False
    if require_close_above_prev_high:
        if current_close > prev_high:
            entry_triggered = True

    if not entry_triggered and allow_bullish_engulfing:
        if (
            current_close > prev_open
            and current_open < prev_close
            and current_close > prev_close
        ):
            entry_triggered = True

    if not entry_triggered:
        return []

    ma_20_val = ma_20.iloc[-1]
    ma_50_val = ma_50.iloc[-1]
    ma_20_prev = ma_20.iloc[-2] if len(ma_20) > 1 else ma_20_val
    ma_50_prev = ma_50.iloc[-2] if len(ma_50) > 1 else ma_50_val

    trend_ok = True
    if ma_20_val <= ma_50_val:
        trend_ok = False

    if require_positive_slope:
        if ma_20_val <= ma_20_prev:
            trend_ok = False

    if require_positive_return:
        if len(closes) < momentum_lookback:
            trend_ok = False
        else:
            momentum_return = (closes.iloc[-1] / closes.iloc[-momentum_lookback]) - 1
            if momentum_return <= 0:
                trend_ok = False

    if not trend_ok:
        return []

    if corr_enabled and state.model_state.correlation_model is not None:
        corr_model = state.model_state.correlation_model
        for existing_sym in state.portfolio.positions:
            corr = corr_model.get_correlation(symbol, existing_sym)
            if corr > max_corr:
                return []

    capital = state.portfolio.initial_capital
    atr_14 = ta.atr(highs, lows, closes, atr_short)
    atr_val = atr_14.iloc[-1]

    if pd.isna(atr_val) or atr_val == 0:
        return []

    initial_stop_multiplier = strategy_params.get("stops", {}).get(
        "initial_atr_multiple", 2.5
    )
    stop_distance = initial_stop_multiplier * atr_val
    shares = int((capital * risk_per_trade) / stop_distance)

    if shares <= 0:
        return []

    symbol_state["phase"] = SignalPhase.ENTERED
    symbol_state["atr_at_entry"] = atr_val
    symbol_state["entry_price"] = current_close

    return [
        TradeSignal(
            action=ActionType.long,
            symbol=symbol,
            timestamp=tick.timestamp,
            price=current_close,
            qty=shares,
            reason="vol_ext_entry",
        )
    ]


def handle_exit(
    state: BacktestState,
    tick: Candle,
    strategy_params: dict,
    symbol_state: dict,
    position,
) -> List[TradeSignal]:
    symbol = tick.symbol

    stops = strategy_params.get("stops", {})
    trailing_atr_multiple = stops.get("trailing_atr_multiple", 3.0)
    time_stop_days = stops.get("time_stop_days", 20)
    ema_exit_period = stops.get("ema_exit_period", 20)

    candles = state.candles[symbol]
    closes = candles["close"]
    highs = candles["high"]
    lows = candles["low"]

    if len(candles) < ema_exit_period + 10:
        return []

    current_close = closes.iloc[-1]
    low_10 = lows.iloc[-10:].min()
    ema_20 = ta.ema(closes, ema_exit_period)

    atr_14 = ta.atr(highs, lows, closes, 14)
    atr_val = atr_14.iloc[-1]

    entry_price = symbol_state.get("entry_price", position.last_price)
    atr_at_entry = symbol_state.get("atr_at_entry", atr_val)

    trailing_stop = entry_price - trailing_atr_multiple * atr_at_entry
    current_stop = max(trailing_stop, current_close - trailing_atr_multiple * atr_val)

    days_in_trade = (
        tick.timestamp - symbol_state.get("entry_date", tick.timestamp)
    ).days

    exit_reasons = []

    if current_close < low_10:
        exit_reasons.append("10_day_low")
    elif current_close < ema_20:
        exit_reasons.append("ema_exit")
    elif days_in_trade >= time_stop_days:
        exit_reasons.append("time_stop")

    if not exit_reasons:
        return []

    reason = exit_reasons[0]

    return close(tick, position, reason, 0)


def plot(state: BacktestState, config: "StrategyConfig") -> PlotConfig:
    """Calculate indicators from candles and return as price overlays."""
    strategy_params = config.strategy_params

    comp = strategy_params.get("compression", {})
    atr_short = comp.get("atr_short", 14)
    atr_long = comp.get("atr_long", 100)

    trend = strategy_params.get("trend_filter", {})
    ma_fast = trend.get("ma_fast", 20)
    ma_slow = trend.get("ma_slow", 50)

    exits = strategy_params.get("stops", {})
    ema_exit_period = exits.get("ema_exit_period", 20)

    price_overlays: Dict[str, Dict[str, pd.Series]] = {}

    for symbol in config.symbols:
        try:
            candles = state.candles[symbol]
        except KeyError:
            continue

        if len(candles) < max(atr_long, ma_slow, ema_exit_period):
            continue

        closes = candles["close"]
        highs = candles["high"]
        lows = candles["low"]

        overlays = {}

        atr_14 = ta.atr(highs, lows, closes, atr_short)
        atr_100 = ta.atr(highs, lows, closes, atr_long)
        overlays[f"atr_{atr_short}"] = atr_14
        overlays[f"atr_{atr_long}"] = atr_100

        ma_20 = ta.sma(closes, ma_fast)
        ma_50 = ta.sma(closes, ma_slow)
        overlays[f"ma_{ma_fast}"] = ma_20
        overlays[f"ma_{ma_slow}"] = ma_50

        ema_exit = closes.ewm(span=ema_exit_period, adjust=False).mean()
        overlays[f"ema_{ema_exit_period}"] = ema_exit

        price_overlays[symbol] = overlays

    return PlotConfig(price_overlays=price_overlays)
