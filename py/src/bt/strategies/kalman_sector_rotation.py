"""Sector rotation: rank ETFs by Kalman velocity, rotate into top N.

Core idea: run an online constant-velocity Kalman filter on each sector ETF.
The smoothed velocity (trend strength) is the ranking signal. Each bar,
rank all symbols, go long the top N, exit when a position drops below
the bottom-K threshold or velocity turns strongly negative.

Entry gated by: regime filter (SPY > SMA50), volatility spike filter.
Exits: rank drop, velocity collapse (asymmetric threshold), trailing stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.bt.state import ActionType, BacktestState, Candle, TradeSignal
from src.bt.strategies.types import StrategyParams

STRATEGY_TYPE = "kalman_sector_rotation"

# ---------------------------------------------------------------------------
# typed params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params(StrategyParams):
    process_noise: float = 1e-4
    measurement_noise: float = 1e-3
    velocity_smooth: int = 8  # bars to smooth velocity before ranking
    top_n: int = 2  # number of top-ranked sectors to hold
    drop_rank: int = 4  # exit if position falls to this rank or below (alias for exit_rank_threshold)
    warmup_bars: int = 50  # minimum bars before trading
    cooldown_bars: int = 3  # bars to wait after rank-based exit before re-entry
    cooldown_vel_exit_bars: int = 24  # bars to wait after velocity-based exit
    exit_velocity_threshold: float = (
        -2.0
    )  # exit if velocity below this (negative = strong momentum collapse)
    exit_rank_threshold: int = (
        5  # exit if position rank > this (default: same as drop_rank)
    )
    max_positions: int = 3  # hard cap on concurrent positions (bull regime)
    max_positions_bear: int = 1  # hard cap during bear regime
    regime_filter: str = ""  # symbol for regime detection (e.g., "SPY")
    bar: str = "1h"  # bar size (needed for lazy-loading regime data)
    vol_filter_window: int = 20  # window for volatility spike filter
    vol_spike_mult: float = 2.5  # skip entry if recent vol > mult * long vol

    @classmethod
    def from_dict(cls, d: dict) -> Params:
        """Merge drop_rank into exit_rank_threshold if only former provided."""
        d = dict(d)
        if "exit_rank_threshold" not in d and "drop_rank" in d:
            d["exit_rank_threshold"] = d["drop_rank"]
        return super().from_dict(d)


# ---------------------------------------------------------------------------
# per-symbol Kalman filter (online, no filterpy dependency in hot path)
# ---------------------------------------------------------------------------


class _KalmanOnline:
    """Online 2-state [price, velocity] Kalman filter. Lightweight, no filterpy."""

    def __init__(self, q: float, r: float) -> None:
        self._q = q
        self._r = r
        self._x: np.ndarray = np.zeros((2, 1), dtype=float)
        self._P: np.ndarray = np.eye(2, dtype=float) * 10.0
        self._initialized = False

    @property
    def velocity(self) -> float:
        return float(self._x[1, 0])

    @property
    def filtered(self) -> float:
        return float(self._x[0, 0])

    def update(self, price: float) -> tuple[float, float]:
        """Process one observation. Returns (filtered_price, velocity)."""
        z = np.array([[price]], dtype=float)
        if not self._initialized:
            self._x[0, 0] = price
            self._x[1, 0] = 0.0
            self._initialized = True
            return (price, 0.0)

        F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
        Q = self._q * np.array([[1.0 / 3, 0.5], [0.5, 1.0]], dtype=float)
        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + Q

        H = np.array([[1.0, 0.0]], dtype=float)
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + self._r
        K = P_pred @ H.T / S
        self._x = x_pred + K @ y
        self._P = P_pred - K @ H @ P_pred

        return (float(self._x[0, 0]), float(self._x[1, 0]))


# ---------------------------------------------------------------------------
# module-level state
# ---------------------------------------------------------------------------

_RANK_STATE: dict = {}
_LAST_RANKING: dict[str, float] = {}
_LAST_RANKING_TS: pd.Timestamp | None = None
_REGIME_BEAR: bool = False
_REGIME_TS: pd.Timestamp | None = None
_REGIME_CANDLES_CACHE: dict[str, pd.DataFrame | None] = {}


def _reset_state() -> None:
    global _RANK_STATE, _LAST_RANKING, _LAST_RANKING_TS, _REGIME_BEAR, _REGIME_TS
    _RANK_STATE = {}
    _LAST_RANKING = {}
    _LAST_RANKING_TS = None
    _REGIME_BEAR = False
    _REGIME_TS = None


def _is_bear_regime(
    state: BacktestState,
    regime_symbol: str,
    bar: str = "1h",
    sma_window: int = 50,
) -> bool:
    """Check if benchmark is in a bear regime (price < SMA). Cached per bar.

    If regime symbol data isn't in state.candles, lazy-loads it once from disk.
    """
    global _REGIME_BEAR, _REGIME_TS

    ts = state.timestamp
    if ts is not None and _REGIME_TS == ts:
        return _REGIME_BEAR

    candles = state.candles.get(regime_symbol)
    if candles is None:
        candles = _load_regime_candles(regime_symbol, bar)
    if candles is None or len(candles) < sma_window:
        _REGIME_BEAR = False
        _REGIME_TS = ts
        return False

    closes = cast(pd.Series, candles["close"])
    current = float(closes.iloc[-1])
    sma = float(closes.iloc[-sma_window:].mean())
    _REGIME_BEAR = current < sma
    _REGIME_TS = ts
    return _REGIME_BEAR


def _load_regime_candles(
    symbol: str,
    bar: str,
) -> pd.DataFrame | None:
    """Lazy-load regime symbol candles once. Cached at module level."""
    # ponytail: global mutable cache — single symbol, intentional, tiny
    cache_key = f"{symbol}:{bar}"
    if cache_key in _REGIME_CANDLES_CACHE:
        return _REGIME_CANDLES_CACHE[cache_key]

    try:
        from src.bt.data_feed import load_candles

        now = pd.Timestamp.now()
        end = cast(
            pd.Timestamp, now if now is not pd.NaT else pd.Timestamp("2099-01-01")
        )
        # Load all available data for this symbol (wide range)
        df = load_candles(
            [symbol], cast(pd.Timestamp, pd.Timestamp("2010-01-01")), end, bar
        )
        if df.empty:
            _REGIME_CANDLES_CACHE[cache_key] = None
            return None
        result = df.xs(symbol, axis=1, level=0)
        _REGIME_CANDLES_CACHE[cache_key] = result
        return result
    except Exception:
        _REGIME_CANDLES_CACHE[cache_key] = None
        return None


def _rank_symbols(
    state: BacktestState,
    params: Params,
) -> dict[str, float]:
    """Compute normalized momentum score for every symbol. Cached per timestamp."""
    global _LAST_RANKING, _LAST_RANKING_TS

    ts = state.timestamp
    if ts is not None and _LAST_RANKING_TS == ts and _LAST_RANKING:
        return _LAST_RANKING

    rankings: dict[str, float] = {}
    for sym in state.candles:
        candles = state.candles[sym]
        closes = cast(pd.Series, candles["close"])
        if len(closes) < params.warmup_bars:
            continue

        current_price = float(closes.iloc[-1])
        if np.isnan(current_price):
            continue  # skip symbols with missing data for this bar

        if sym not in _RANK_STATE:
            kf = _KalmanOnline(params.process_noise, params.measurement_noise)
            for p in closes.values:
                kf.update(float(p))
            _RANK_STATE[sym] = {
                "kf": kf,
                "vel_hist": [],
                "cooldown": 0,
                "cooldown_reason": "",
            }
        else:
            kf = _RANK_STATE[sym]["kf"]
            kf.update(current_price)

        vel_hist = _RANK_STATE[sym]["vel_hist"]
        vel_hist.append(kf.velocity)
        if len(vel_hist) > params.velocity_smooth:
            vel_hist.pop(0)

        if len(vel_hist) >= params.velocity_smooth:
            smooth_vel = float(np.mean(vel_hist))
        else:
            smooth_vel = kf.velocity

        returns = closes.pct_change().dropna()
        if len(returns) >= params.vol_filter_window:
            vol = float(returns.iloc[-params.vol_filter_window :].std())
            if vol > 1e-12:
                rankings[sym] = smooth_vel / vol
            else:
                rankings[sym] = smooth_vel
        else:
            rankings[sym] = smooth_vel

    _LAST_RANKING = rankings
    if ts is not None:
        _LAST_RANKING_TS = ts
    return rankings


def _rank_of(symbol: str, rankings: dict[str, float]) -> int:
    sorted_syms = sorted(rankings, key=lambda s: rankings[s], reverse=True)
    try:
        return sorted_syms.index(symbol) + 1
    except ValueError:
        return 999


# ---------------------------------------------------------------------------
# on_candle
# ---------------------------------------------------------------------------


def on_candle(
    state: BacktestState,
    candle: Candle,
    params: Params,
) -> list[TradeSignal]:
    rankings = _rank_symbols(state, params)
    if not rankings:
        return []

    signals: list[TradeSignal] = []
    sorted_ranked = sorted(rankings, key=lambda s: rankings[s], reverse=True)
    exiting: set[str] = set()

    # --- Exit checks for open positions ---
    for sym, position in list(state.portfolio.positions.items()):
        if sym not in rankings:
            continue
        vel = rankings[sym]
        rank = _rank_of(sym, rankings)
        sym_closes = cast(pd.Series, state.candles[sym]["close"])
        sym_price = float(sym_closes.iloc[-1])
        if np.isnan(sym_price):
            continue  # skip if price data is missing

        exited = False
        if position.type == ActionType.long and vel < params.exit_velocity_threshold:
            _RANK_STATE[sym]["cooldown"] = params.cooldown_vel_exit_bars
            _RANK_STATE[sym]["cooldown_reason"] = "vel"
            exited = True
            signals.append(
                TradeSignal(
                    action=ActionType.close,
                    symbol=sym,
                    timestamp=candle.timestamp,
                    price=sym_price,
                    qty=abs(position.qty),
                    reason=f"[krot] vel collapse {vel:.3f} < {params.exit_velocity_threshold}",
                )
            )
        elif position.type == ActionType.long and rank > params.exit_rank_threshold:
            _RANK_STATE[sym]["cooldown"] = params.cooldown_bars
            _RANK_STATE[sym]["cooldown_reason"] = "rank"
            exited = True
            signals.append(
                TradeSignal(
                    action=ActionType.close,
                    symbol=sym,
                    timestamp=candle.timestamp,
                    price=sym_price,
                    qty=abs(position.qty),
                    reason=f"[krot] rank {rank} > {params.exit_rank_threshold}",
                )
            )
        if exited:
            exiting.add(sym)

    # --- Entry checks ---
    effective_positions = len(state.portfolio.positions) - len(exiting)

    # Determine max positions based on regime
    if params.regime_filter:
        bear = _is_bear_regime(state, params.regime_filter, params.bar)
        max_pos = params.max_positions_bear if bear else params.max_positions
    else:
        max_pos = params.max_positions

    for sym in sorted_ranked:
        if (
            effective_positions
            + len([s for s in signals if s.action == ActionType.long])
            >= max_pos
        ):
            break

        if sym in state.portfolio.positions and sym not in exiting:
            continue

        if any(s.symbol == sym for s in signals):
            continue

        rank = _rank_of(sym, rankings)
        if rank > params.top_n:
            continue

        # Cooldown: decrement each bar, allow entry only at zero
        sym_state = _RANK_STATE.get(sym, {})
        cooldown = sym_state.get("cooldown", 0)
        if cooldown > 0:
            _RANK_STATE[sym]["cooldown"] -= 1
            continue

        vel = rankings[sym]
        if vel <= 0:
            continue

        candles = state.candles.get(sym)
        if candles is None or len(candles) < params.warmup_bars:
            continue
        closes = cast(pd.Series, candles["close"])

        if not _vol_filter(closes, params):
            continue

        entry_price = float(closes.iloc[-1])
        if np.isnan(entry_price):
            continue

        # Set cooldown post-entry (rank-based, short — prevents double-entry)
        _RANK_STATE[sym]["cooldown"] = params.cooldown_bars
        _RANK_STATE[sym]["cooldown_reason"] = "rank"
        signals.append(
            TradeSignal(
                action=ActionType.long,
                symbol=sym,
                timestamp=candle.timestamp,
                price=entry_price,
                reason=f"[krot] rank {rank}/{len(rankings)} vel={vel:.3f}",
            )
        )

    return signals


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _vol_filter(closes: pd.Series, params: Params) -> bool:
    if len(closes) < params.vol_filter_window * 2:
        return True
    returns = closes.pct_change().dropna()
    recent_vol = float(returns.iloc[-params.vol_filter_window :].std())
    long_vol = float(
        returns.iloc[-params.vol_filter_window * 3 :].std()
        if len(returns) >= params.vol_filter_window * 3
        else returns.std()
    )
    if long_vol <= 0:
        return True
    return recent_vol <= params.vol_spike_mult * long_vol


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def plot(state: BacktestState, config: object) -> object:
    from src.bt.types import PlotConfig

    params_raw = getattr(config, "strategy_params", {})
    params = Params.from_dict(params_raw)

    price_overlays: dict[str, dict[str, pd.Series]] = {}
    for symbol in getattr(config, "symbols", []):
        candles = state.candles.get(symbol)
        if candles is None or len(candles) < params.warmup_bars:
            continue
        closes = cast(pd.Series, candles["close"])
        kf = _KalmanOnline(params.process_noise, params.measurement_noise)
        filtered = np.empty(len(closes))
        for i, p in enumerate(closes.values):
            f, _ = kf.update(float(p))
            filtered[i] = f
        price_overlays[symbol] = {
            "kalman_filtered": pd.Series(filtered, index=closes.index),
        }

    return PlotConfig(price_overlays=price_overlays, subplots=[])
