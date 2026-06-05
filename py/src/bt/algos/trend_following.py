import src.bt.indicators as ta
from src.bt.algos.utils import open, close, htf_candles
from typing import List, Dict, TYPE_CHECKING, Optional, cast
from src.bt.state import BacktestState, TradeSignal, Tick, ActionType, Position
from src.bt.types import PlotConfig
import pandas as pd

if TYPE_CHECKING:
    from src.bt.types import StrategyConfig


def volume_confirmed(
    volume: pd.Series, window: int = 20, threshold: float = 1.0
) -> bool:
    if len(volume) < window:
        return True
    avg_vol = volume.iloc[-window:].mean()
    return volume.iloc[-1] > avg_vol * threshold


def di_rising(series: pd.Series) -> bool:
    if len(series) < 2:
        return False
    return series.iloc[-1] > series.iloc[-2]


def adx_falling(series: pd.Series) -> bool:
    if len(series) < 2:
        return False
    return series.iloc[-1] < series.iloc[-2]


def adx_flat(series: pd.Series, slope_threshold: float = 0.5) -> bool:
    if len(series) < 2:
        return False
    return abs(series.iloc[-1] - series.iloc[-2]) < slope_threshold


def risk_hedge_multiplier(adx_value: float, base: float = 1.0) -> float:
    if adx_value > 25:
        return base + 0.5
    if adx_value < 20:
        return base - 0.5
    return base


def on_tick(
    state: BacktestState, tick: Tick, strategy_params: dict
) -> List[TradeSignal]:
    symbol = tick.symbol
    position = state.portfolio.positions.get(symbol)
    candles = state.candles[symbol]

    window = strategy_params.get("window", 14)
    vol_window = strategy_params.get("vol_window", 20)
    vol_multiplier = strategy_params.get("vol_multiplier", 1.0)
    mfi_overbought = strategy_params.get("mfi_overbought", 80)
    mfi_oversold = strategy_params.get("mfi_oversold", 20)
    adx_no_trade = strategy_params.get("adx_no_trade", 15)
    adx_flat_slope = strategy_params.get("adx_flat_slope", 0.5)
    news_blackout = strategy_params.get("news_blackout", False)

    if len(candles) < max(window, vol_window) + 2:
        return []

    closes = cast(pd.Series, candles["close"])
    highs = cast(pd.Series, candles["high"])
    lows = cast(pd.Series, candles["low"])
    volumes = cast(pd.Series, candles["volume"])

    lsma = ta.lsma(closes, window)
    ema = ta.ema(closes, window)
    lsma_low = ta.lsma(lows, window)
    lsma_high = ta.lsma(highs, window)
    mfi = ta.mfi(highs, lows, closes, volumes, window)
    adx = ta.adx(highs, lows, closes, window)
    plus_di = ta.plus_di(highs, lows, closes, window)
    minus_di = ta.minus_di(highs, lows, closes, window)

    if pd.isna(lsma.iloc[-1]) or pd.isna(ema.iloc[-1]):
        return []

    htf = htf_candles(state, "4h", tick)

    if htf.empty or len(htf) < window + 1:
        return []

    htf_closes = cast(pd.Series, htf["close"])
    htf_highs = cast(pd.Series, htf["high"])
    htf_lows = cast(pd.Series, htf["low"])

    htf_lsma = ta.lsma(htf_closes, window).iloc[-1]
    htf_ema = ta.ema(htf_closes, window).iloc[-1]
    htf_plus = ta.plus_di(htf_highs, htf_lows, htf_closes, window).iloc[-1]
    htf_minus = ta.minus_di(htf_highs, htf_lows, htf_closes, window).iloc[-1]
    htf_adx = ta.adx(htf_highs, htf_lows, htf_closes, window).iloc[-1]

    htf_bull = htf_lsma > htf_ema and htf_plus > htf_minus
    htf_bear = htf_lsma < htf_ema and htf_minus > htf_plus

    current_close = closes.iloc[-1]
    prior_high = highs.iloc[-2]
    prior_low = lows.iloc[-2]

    with_volume = volume_confirmed(volumes, vol_window, vol_multiplier)
    adx_last = adx.iloc[-1]
    adx_prev = adx.iloc[-2]
    no_trade = news_blackout or (
        adx_last < adx_no_trade and adx_flat(adx, adx_flat_slope)
    )

    if not position and no_trade:
        return []

    if not position:
        if htf_bull:
            long_ok = (
                lsma.iloc[-1] > ema.iloc[-1]
                and lsma.iloc[-1] > lsma_low.iloc[-1]
                and plus_di.iloc[-1] > minus_di.iloc[-1]
                and di_rising(plus_di)
                and with_volume
                and mfi.iloc[-1] < mfi_overbought
                and current_close > prior_high
            )
            if long_ok and mfi.iloc[-1] <= mfi_overbought:
                hedge = risk_hedge_multiplier(htf_adx)
                hedge = max(min(hedge, 2.0), 0.25)
                return open(tick, ActionType.long, "[trend] enter long", hedge)

        if htf_bear:
            short_ok = (
                lsma.iloc[-1] < ema.iloc[-1]
                and lsma.iloc[-1] < lsma_high.iloc[-1]
                and minus_di.iloc[-1] > plus_di.iloc[-1]
                and di_rising(minus_di)
                and with_volume
                and mfi.iloc[-1] > mfi_oversold
                and current_close < prior_low
            )
            if short_ok and mfi.iloc[-1] >= mfi_oversold:
                hedge = risk_hedge_multiplier(htf_adx)
                hedge = max(min(hedge, 2.0), 0.25)
                return open(tick, ActionType.short, "[trend] enter short", hedge)

        return []

    is_long = position.type == ActionType.long
    if is_long:
        if ema.iloc[-1] > lsma.iloc[-1]:
            return close(tick, position, "[trend] ema above lsma")
        if mfi.iloc[-1] > mfi_overbought and adx_falling(adx):
            return close(tick, position, "[trend] mfi overbought + adx falling")
        if minus_di.iloc[-1] > plus_di.iloc[-1]:
            return close(tick, position, "[trend] di flip")
        if current_close < lsma_low.iloc[-1]:
            return close(tick, position, "[trend] close below lsma low")
    else:
        if ema.iloc[-1] < lsma.iloc[-1]:
            return close(tick, position, "[trend] ema below lsma")
        if mfi.iloc[-1] < mfi_oversold and adx_falling(adx):
            return close(tick, position, "[trend] mfi oversold + adx falling")
        if plus_di.iloc[-1] > minus_di.iloc[-1]:
            return close(tick, position, "[trend] di flip")
        if current_close > lsma_high.iloc[-1]:
            return close(tick, position, "[trend] close above lsma high")

    return []


def plot(state: BacktestState, config: "StrategyConfig") -> PlotConfig:
    window = config.strategy_params.get("window", 14)

    price_overlays: Dict[str, Dict[str, pd.Series]] = {}
    subplots: List[tuple[str, pd.Series]] = []

    for symbol in config.symbols:
        candles = state.candles[symbol]
        if len(candles) < window:
            continue

        closes = cast(pd.Series, candles["close"])
        highs = cast(pd.Series, candles["high"])
        lows = cast(pd.Series, candles["low"])
        volumes = cast(pd.Series, candles["volume"])

        ema = ta.ema(closes, window)
        lsma = ta.lsma(closes, window)
        lsma_low = ta.lsma(lows, window)
        lsma_high = ta.lsma(highs, window)
        adx = ta.adx(highs, lows, closes, window)
        plus_di = ta.plus_di(highs, lows, closes, window)
        minus_di = ta.minus_di(highs, lows, closes, window)
        mfi = ta.mfi(highs, lows, closes, volumes, window)

        price_overlays[symbol] = {
            f"lsma_{window}": lsma,
            f"ema_{window}": ema,
            f"lsma_low_{window}": lsma_low,
            f"lsma_high_{window}": lsma_high,
        }

        subplots = [
            ("adx", adx),
            ("plus_di", plus_di),
            ("minus_di", minus_di),
            ("mfi", mfi),
        ]

    return PlotConfig(price_overlays=price_overlays, subplots=subplots)
