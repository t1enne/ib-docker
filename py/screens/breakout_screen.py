"""Breakout screen — institutional-grade breakout detection and ranking system.

Two-stage process:
  Stage 1 — Hard filters: only stocks meeting entry criteria survive.
  Stage 2 — Weighted ranking: survivors are scored on RS, breakout
  strength, volume, trend, volatility contraction, ATR expansion,
  ADX trend strength, base quality, and VCP patterns.

Approach blends Minervini trend templates, VCP setups, and classic
Turtle breakout concepts rather than a generic momentum rank.

Usage:
    python main.py screen breakout_screen universe.yml \\
        --param benchmark_symbol=SPY

Benchmark data is fetched automatically from the same candle source.
If unavailable, RS-related filters and scores are skipped gracefully.
"""

from __future__ import annotations
from src.market_data import resample_ohlcv
from typing import Any, cast

import numpy as np
import pandas as pd

from src.bt.indicators import sma, atr, adx
from src.screen.types import ScreenFn, ScreenResult


# ── Default parameters ─────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    # Trend
    "fast_sma": 20,
    "mid_sma": 50,
    "slow_sma": 200,
    "sma50_slope_window": 10,
    "sma200_slope_window": 20,
    "adx_window": 14,
    "adx_min": 15,
    # 52-week high
    "pct_off_high_max": 0.10,
    # Volume
    "rvol_window": 50,
    "rvol_min": 1.5,
    "rvol_peek_window": 5,
    "accumulation_short": 20,
    "accumulation_long": 100,
    # Relative strength
    "benchmark_symbol": "SPY",
    "rs_window": 63,
    # Breakout
    "breakout_window_20": 20,
    "breakout_window_55": 55,
    # Volatility contraction
    "vol_window_short": 20,
    "vol_window_mid": 40,
    "vol_window_long": 100,
    # ATR expansion
    "atr_window_short": 14,
    "atr_window_long": 50,
    # Base quality
    "base_lookback": 90,
    "base_max_depth": 0.50,
}


# ── Helpers ────────────────────────────────────────────────────


def _highest(series: pd.Series, window: int) -> pd.Series:
    """Rolling maximum (same as Pine Script's highest())."""
    result: pd.Series = series.rolling(window=window, min_periods=1).max()
    return result


def _lowest(series: pd.Series, window: int) -> pd.Series:
    """Rolling minimum (same as Pine Script's lowest())."""
    result: pd.Series = series.rolling(window=window, min_periods=1).min()
    return result


def _safe_float(series: pd.Series, idx: int = -1) -> float:
    """Extract a float from a Series, returning 0.0 if NaN or empty."""
    if len(series) == 0:
        return 0.0
    v = series.iloc[idx]
    return 0.0 if pd.isna(v) else float(v)


def _realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Realized volatility as annualized std of log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=window).std() * np.sqrt(252)


def _rvol(volume: pd.Series, avg_window: int) -> float:
    """Relative Volume for current bar."""
    avg_vol = float(volume.iloc[-avg_window:].mean())
    last_vol = float(volume.iloc[-1])
    return last_vol / avg_vol if avg_vol > 0 else 0.0


def _rs_value(close: pd.Series, bench_close: pd.Series, window: int) -> float:
    """Return stock_return - benchmark_return over window bars."""
    if len(close) < window + 1 or len(bench_close) < window + 1:
        return 0.0
    stock_ret = float(close.iloc[-1]) / float(close.iloc[-window - 1]) - 1.0
    bench_ret = float(bench_close.iloc[-1]) / float(bench_close.iloc[-window - 1]) - 1.0
    return stock_ret - bench_ret


# ── Screen implementation ──────────────────────────────────────


class BreakoutScreen:
    """Two-stage breakout screener: hard filters then weighted ranking.

    Upgrades over base version:
      - SMA slope filters (50/200 rising)
      - Base depth analysis
      - Weighted multi-period RS (0.4*63 + 0.3*126 + 0.3*252)
      - New 52w high bonus
      - Breakout-volume (max RVOL over N days)
      - Accumulation ratio (20d avg / 100d avg)
      - Multi-window VCP contraction (vol20 < vol40 < vol80)
      - Pullback contraction detection (VCP lite)
      - Enhanced breakout strength (distance + volume + new-high)
    """

    __slots__ = ("symbols", "params", "_benchmark_candles")

    def __init__(self, symbols: list[str], params: dict[str, Any]) -> None:
        self.symbols = symbols
        self.params = params

        # Pre-load benchmark candles once so each compute() call doesn't
        # hit the database. The benchmark symbol is known at construction.
        self._benchmark_candles: pd.DataFrame | None = None
        bench = params.get("benchmark_symbol", "SPY")
        if bench:
            try:
                from src.utils import get_local_candles

                self._benchmark_candles = get_local_candles(symbol=bench, bar="1d")
            except Exception:
                self._benchmark_candles = None

    # ── Stage 1 helpers ────────────────────────────────────────

    def _min_bars(self, p: dict[str, Any]) -> int:
        """Longest lookback needed."""
        return (
            max(
                p["slow_sma"],
                p["vol_window_long"],
                p["atr_window_long"],
                p["breakout_window_55"],
                p["base_lookback"],
                p["accumulation_long"],
                252,
            )
            + 1
        )

    def _check_trend(
        self, close: pd.Series, p: dict[str, Any]
    ) -> tuple[bool, bool, bool, bool]:
        """Check trend conditions: SMA position + SMA slopes.

        Hard filters:
          - close > SMA200
          - SMA50 > SMA200
          - SMA200 rising (200 > 200 shifted by 20)

        Returns (hard_pass, sma50_rising, sma200_rising, close_above_sma20).
        """
        sma_20 = _safe_float(sma(close, p["fast_sma"]))
        sma_50 = _safe_float(sma(close, p["mid_sma"]))
        sma_200 = _safe_float(sma(close, p["slow_sma"]))
        sma_50_prev = _safe_float(
            sma(close, p["mid_sma"]), -p["sma50_slope_window"] - 1
        )
        sma_200_prev = _safe_float(
            sma(close, p["slow_sma"]), -p["sma200_slope_window"] - 1
        )
        last_close = float(close.iloc[-1])

        close_above_200 = last_close > sma_200 if sma_200 > 0 else False
        sma_50_above_200 = sma_50 > sma_200 if sma_200 > 0 and sma_50 > 0 else False
        sma_200_rising = (
            sma_200 > sma_200_prev if sma_200 > 0 and sma_200_prev > 0 else False
        )
        sma_50_rising = (
            sma_50 > sma_50_prev if sma_50 > 0 and sma_50_prev > 0 else False
        )
        close_above_20 = last_close > sma_20 if sma_20 > 0 else False

        hard_pass = close_above_200 and sma_50_above_200 and sma_200_rising
        return hard_pass, sma_50_rising, sma_200_rising, close_above_20

    def _check_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, p: dict[str, Any]
    ) -> tuple[bool, float]:
        """ADX(14) > min threshold. Returns (passed, adx_value)."""
        adx_series = adx(high, low, close, p["adx_window"])
        adx_val = _safe_float(adx_series)
        return adx_val > p["adx_min"], adx_val

    def _check_52w_high(
        self, high: pd.Series, close: pd.Series, p: dict[str, Any]
    ) -> tuple[bool, bool, float, float]:
        """52-week high proximity and new-high detection.

        Returns (passed, new_52w_high, distance_pct, prior_52w_high).
        """
        high_252 = _safe_float(_highest(high, 252))
        high_shifted: pd.Series = cast(pd.Series, high.shift(1))  # type: ignore[assignment]
        prior_52w_high = _safe_float(_highest(high_shifted, 252))
        last_high = float(high.iloc[-1])
        last_close = float(close.iloc[-1])

        if high_252 <= 0 or last_high <= 0:
            return False, False, 0.0, 0.0

        distance = (high_252 - last_high) / high_252
        new_52w_high = last_close > prior_52w_high if prior_52w_high > 0 else False

        return distance <= p["pct_off_high_max"], new_52w_high, distance, prior_52w_high

    def _check_base_depth(
        self, high: pd.Series, low: pd.Series, p: dict[str, Any]
    ) -> tuple[bool, float]:
        """Base depth over base_lookback days.

        base_depth = (highest_high - lowest_low) / highest_high
        Hard reject: depth > 50%.

        Returns (passed, depth_pct).
        """
        lookback = p["base_lookback"]
        hh = _safe_float(_highest(high, lookback))
        ll = _safe_float(_lowest(low, lookback))
        if hh <= 0:
            return True, 0.0
        depth = (hh - ll) / hh
        return depth <= p["base_max_depth"], depth

    def _check_rvol(self, volume: pd.Series, p: dict[str, Any]) -> tuple[bool, float]:
        """Relative Volume = current volume / 50d avg volume.

        Returns (passed, rvol).
        """
        rvol_val = _rvol(volume, p["rvol_window"])
        return rvol_val >= p["rvol_min"], rvol_val

    def _check_relative_strength(
        self,
        close: pd.Series,
    ) -> tuple[bool, float]:
        """RS check. Returns (passed, weighted_rs).

        weighted_rs = 0.4*rs63 + 0.3*rs126 + 0.3*rs252
        """
        if self._benchmark_candles is None:
            return True, 0.0

        bench_close = cast(pd.Series, self._benchmark_candles["close"])

        rs63 = _rs_value(close, bench_close, 63)
        rs126 = _rs_value(close, bench_close, 126)
        rs252 = _rs_value(close, bench_close, 252)

        weighted_rs = 0.4 * rs63 + 0.3 * rs126 + 0.3 * rs252

        return weighted_rs > 0, weighted_rs

    def _check_breakout(
        self, high: pd.Series, close: pd.Series, p: dict[str, Any]
    ) -> tuple[bool, bool, bool, float, float]:
        """Check 20d and 55d breakouts using prior highs (shift(1)).

        Uses close to test resistance.

        Returns (passed, breakout_20d, breakout_55d,
                 prior_20d_high, prior_55d_high).
        """
        hh_20_prior = _safe_float(
            cast(pd.Series, _highest(high, p["breakout_window_20"]).shift(1))
        )
        hh_55_prior = _safe_float(
            cast(pd.Series, _highest(high, p["breakout_window_55"]).shift(1))
        )
        last_close = float(close.iloc[-1])

        b20 = last_close >= hh_20_prior if hh_20_prior > 0 else False
        b55 = last_close >= hh_55_prior if hh_55_prior > 0 else False

        return (b20 or b55), b20, b55, hh_20_prior, hh_55_prior

    def _check_volatility_contraction(
        self, close: pd.Series, p: dict[str, Any]
    ) -> bool:
        """Check whether volatility is contracting (20d RV < 100d RV)."""
        vol_20 = _realized_vol(close, p["vol_window_short"])
        vol_100 = _realized_vol(close, p["vol_window_long"])

        v20 = _safe_float(vol_20)
        v100 = _safe_float(vol_100)

        return (v20 < v100) if v100 > 0 else True

    # ── Stage 2 scoring helpers ────────────────────────────────

    def _score_weighted_rs(self, close: pd.Series, p: dict[str, Any]) -> float:
        """Score 0-1 for weighted multi-period RS.

        weighted_rs = 0.4*rs63 + 0.3*rs126 + 0.3*rs252
        clamped to [-0.5, 0.5], mapped 0→1.
        """
        if self._benchmark_candles is None:
            return 0.5

        bench_close = cast(pd.Series, self._benchmark_candles["close"])

        rs63 = _rs_value(close, bench_close, 63)
        rs126 = _rs_value(close, bench_close, 126)
        rs252 = _rs_value(close, bench_close, 252)

        weighted = 0.4 * rs63 + 0.3 * rs126 + 0.3 * rs252
        clipped = max(-0.5, min(0.5, weighted))
        return (clipped + 0.5) / 1.0

    def _score_breakout_strength(
        self,
        volume: pd.Series,
        bko_20: bool,
        bko_55: bool,
        prior_20d_high: float,
        prior_55d_high: float,
        last_high: float,
        new_52w_high: bool,
        p: dict[str, Any],
    ) -> float:
        """Enhanced breakout strength score.

        Components:
          0.5 * breakout_distance
          0.3 * breakout_volume
          0.2 * new_high_status
        """
        # Distance component
        if bko_55 and prior_55d_high > 0:
            resistance = prior_55d_high
        elif bko_20 and prior_20d_high > 0:
            resistance = prior_20d_high
        else:
            resistance = prior_20d_high

        if resistance > 0:
            breakout_distance = (last_high - resistance) / resistance
        else:
            breakout_distance = 0.0

        distance_score = min(1.0, max(0.0, breakout_distance * 10))

        # Volume component — max RVOL over peek window
        max_rvol = 0.0
        peek = p["rvol_peek_window"]
        avg_vol = float(volume.iloc[-p["rvol_window"] :].mean())
        if avg_vol > 0:
            recent_rvols = [
                float(volume.iloc[-i]) / avg_vol
                for i in range(1, min(peek, len(volume)) + 1)
            ]
            max_rvol = max(recent_rvols) if recent_rvols else 0.0

        if max_rvol >= 3.0:
            vol_score = 1.0
        elif max_rvol >= 2.0:
            vol_score = 0.75
        elif max_rvol >= 1.5:
            vol_score = 0.5
        elif max_rvol >= 1.0:
            vol_score = 0.25
        else:
            vol_score = 0.0

        # New-high bonus
        new_high_bonus = 0.15 if new_52w_high else 0.0

        if bko_55:
            base = 0.7
        elif bko_20:
            base = 0.5
        else:
            base = 0.0

        combined = base + 0.5 * distance_score + 0.3 * vol_score + 0.2 * new_high_bonus
        return min(1.0, combined)

    def _score_rvol(self, volume: pd.Series, p: dict[str, Any]) -> float:
        """Score RVOL: 1.5->0.5, 2.0->0.75, 3.0->1.0."""
        rvol_val = _rvol(volume, p["rvol_window"])
        if rvol_val >= 3.0:
            return 1.0
        if rvol_val >= 1.5:
            return 0.5 + (rvol_val - 1.5) / 3.0
        if rvol_val >= 1.0:
            return (rvol_val - 1.0) * 1.0
        return max(0.0, rvol_val / 2.0)

    def _score_trend_alignment(
        self,
        close: pd.Series,
        sma50_rising: bool,
        sma200_rising: bool,
        close_above_sma20: bool,
        p: dict[str, Any],
    ) -> float:
        """Score trend quality: position + SMA slopes.

        Returns 0-1.
        """
        sma_20 = _safe_float(sma(close, p["fast_sma"]))
        sma_50 = _safe_float(sma(close, p["mid_sma"]))
        sma_200 = _safe_float(sma(close, p["slow_sma"]))
        last_close = float(close.iloc[-1])

        if sma_200 <= 0:
            return 0.0

        score = 0.0
        if last_close > sma_200:
            score += 0.15
        if sma_50 > sma_200:
            score += 0.15
        if sma_20 > sma_50:
            score += 0.15
        if close_above_sma20:
            score += 0.10

        if sma50_rising:
            score += 0.20
        if sma200_rising:
            score += 0.25

        return min(1.0, score)

    def _score_volatility_contraction(
        self, close: pd.Series, high: pd.Series, low: pd.Series, p: dict[str, Any]
    ) -> float:
        """Score 0-1 for VCP progression + tightness."""
        vol_20 = _realized_vol(close, p["vol_window_short"])
        vol_40 = _realized_vol(close, p["vol_window_mid"])
        vol_80 = _realized_vol(close, 80)

        v20 = _safe_float(vol_20)
        v40 = _safe_float(vol_40)
        v80 = _safe_float(vol_80)

        cont_score = 0.0
        if v80 > 0:
            step1 = v40 / v80
            step2 = v20 / v40
            cont_score = max(
                0.0,
                0.5 * (1.0 - step1) + 0.5 * (1.0 - step2),
            )
            cont_score = min(1.0, cont_score)

        h20 = _safe_float(_highest(high, 20))
        l20 = _safe_float(_lowest(low, 20))
        last_close = float(close.iloc[-1])
        if last_close > 0:
            tightness = (h20 - l20) / last_close
            tight_score = max(0.0, 1.0 - tightness / 0.20)
        else:
            tight_score = 0.0

        return 0.6 * cont_score + 0.4 * tight_score

    def _score_atr_expansion(
        self, high: pd.Series, low: pd.Series, close: pd.Series, p: dict[str, Any]
    ) -> float:
        """Score 0-1 for ATR expansion."""
        atr_14 = _safe_float(atr(high, low, close, p["atr_window_short"]))
        atr_50 = _safe_float(atr(high, low, close, p["atr_window_long"]))
        if atr_50 <= 0:
            return 0.5

        ratio = atr_14 / atr_50
        if ratio >= 1.2:
            return 1.0
        if ratio >= 1.0:
            return 0.6 + (ratio - 1.0) / 0.5
        if ratio >= 0.7:
            return (ratio - 0.7) * 2.0
        return 0.0

    def _score_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, p: dict[str, Any]
    ) -> float:
        """Score 0-1 for ADX trend strength."""
        adx_val = _safe_float(adx(high, low, close, p["adx_window"]))
        if adx_val >= 40:
            return 1.0
        if adx_val >= 25:
            return 0.7 + (adx_val - 25) / 50
        if adx_val >= 20:
            return 0.5
        if adx_val >= 15:
            return 0.25
        return 0.0

    def _score_base_quality(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        base_depth: float,
        p: dict[str, Any],
    ) -> float:
        """Score base quality: depth, tightness, position in range.

        Returns 0-1.
        """
        # Depth score (40%)
        if base_depth <= 0.0:
            depth_score = 0.5
        elif base_depth <= 0.10:
            depth_score = 1.0
        elif base_depth <= 0.25:
            depth_score = 1.0 - (base_depth - 0.10) / 0.15 * 0.3
        elif base_depth <= 0.35:
            depth_score = 0.7 - (base_depth - 0.25) / 0.10 * 0.3
        elif base_depth <= 0.50:
            depth_score = 0.4 - (base_depth - 0.35) / 0.15 * 0.4
        else:
            depth_score = 0.0

        # Tightness (30%): recent 20d range / base range
        lookback = p["base_lookback"]
        base_range = _safe_float(_highest(high, lookback)) - _safe_float(
            _lowest(low, lookback)
        )
        recent_range = _safe_float(_highest(high, 20)) - _safe_float(_lowest(low, 20))
        if base_range > 0:
            tightness_ratio = recent_range / base_range
            tightness_score = max(0.0, 1.0 - tightness_ratio * 2.0)
        else:
            tightness_score = 0.5

        # Position in base (30%): close near top
        lookback_high = _safe_float(_highest(high, lookback))
        lookback_low = _safe_float(_lowest(low, lookback))
        if lookback_high > lookback_low and lookback_high > 0:
            position_ratio = (float(close.iloc[-1]) - lookback_low) / (
                lookback_high - lookback_low
            )
            position_score = min(1.0, max(0.0, position_ratio))
        else:
            position_score = 0.5

        return 0.4 * depth_score + 0.3 * tightness_score + 0.3 * position_score

    def _score_accumulation(
        self, volume: pd.Series, p: dict[str, Any]
    ) -> tuple[float, float]:
        """Accumulation ratio and its score.

        accumulation_ratio = volume_20d_avg / volume_100d_avg

        Returns (ratio, score_0_to_1).
        """
        vol_20_avg = float(volume.iloc[-p["accumulation_short"] :].mean())
        vol_100_avg = float(volume.iloc[-p["accumulation_long"] :].mean())
        if vol_100_avg <= 0:
            return 0.0, 0.5

        ratio = vol_20_avg / vol_100_avg
        if ratio >= 1.20:
            score = 1.0
        elif ratio >= 1.00:
            score = 0.5 + (ratio - 1.00) / 0.20 * 0.5
        else:
            score = ratio / 1.00 * 0.5

        return ratio, min(1.0, score)

    def _score_vcp_pullback(
        self, high: pd.Series, low: pd.Series, close: pd.Series, p: dict[str, Any]
    ) -> float:
        """VCP lite: detect shrinking pullbacks (Minervini).

        Splits base into thirds, measures range contraction.
        Returns 0-1.
        """
        lookback = p["base_lookback"]
        n = min(lookback, len(high) - 1)
        if n < 30:
            return 0.5

        third = n // 3
        seg1_high = _safe_float(_highest(high.iloc[-n : -2 * third], third))
        seg1_low = _safe_float(_lowest(low.iloc[-n : -2 * third], third))
        seg2_high = _safe_float(_highest(high.iloc[-2 * third : -third], third))
        seg2_low = _safe_float(_lowest(low.iloc[-2 * third : -third], third))
        seg3_high = _safe_float(_highest(high.iloc[-third:], third))
        seg3_low = _safe_float(_lowest(low.iloc[-third:], third))

        last_close = float(close.iloc[-1])
        if last_close <= 0:
            return 0.5

        swing1 = (seg1_high - seg1_low) / last_close if seg1_high > 0 else 0.0
        swing2 = (seg2_high - seg2_low) / last_close if seg2_high > 0 else 0.0
        swing3 = (seg3_high - seg3_low) / last_close if seg3_high > 0 else 0.0

        ideal_s1 = 0.18
        ideal_s2 = 0.12
        ideal_s3 = 0.08

        def _err(val: float, ideal: float, cap: float) -> float:
            return min(1.0, abs(val - ideal) / cap)

        err1 = _err(swing1, ideal_s1, 0.20)
        err2 = _err(swing2, ideal_s2, 0.15)
        err3 = _err(swing3, ideal_s3, 0.10)

        contraction_bonus = 0.0
        if swing1 > 0 and swing2 > 0 and swing3 > 0:
            if swing3 < swing2 < swing1:
                contraction_bonus = 0.20
            elif swing3 < swing2 or swing3 < swing1:
                contraction_bonus = 0.10

        raw = 1.0 - (err1 * 0.33 + err2 * 0.33 + err3 * 0.34)
        return min(1.0, max(0.0, raw + contraction_bonus))

    # ── Public API ─────────────────────────────────────────────

    def compute(self, symbol: str, hourly_candles: pd.DataFrame) -> ScreenResult:
        """Score a single symbol — Stage 1 filters, Stage 2 ranks."""
        p = self.params
        candles = resample_ohlcv(hourly_candles, "1d")
        close = cast(pd.Series, candles["close"])
        high = cast(pd.Series, candles["high"])
        low = cast(pd.Series, candles["low"])
        volume = cast(pd.Series, candles["volume"])
        last_close_val = float(close.iloc[-1])
        last_high = float(high.iloc[-1])

        min_bars = self._min_bars(p)
        if len(close) < min_bars:
            return ScreenResult(
                symbol=symbol,
                signal="neutral",
                score=0.0,
                price=last_close_val,
                metadata={"reason": "insufficient_data"},
            )

        # ── Stage 1: Hard Filters ──────────────────────────────

        trend_ok, sma50_rising, sma200_rising, close_above_sma20 = self._check_trend(
            close, p
        )
        # adx_ok, adx_val = self._check_adx(high, low, close, p)
        high_ok, new_52w_high, high_dist, _ = self._check_52w_high(high, close, p)
        base_ok, base_depth = self._check_base_depth(high, low, p)
        rvol_ok, rvol_val = self._check_rvol(volume, p)
        rs_ok, weighted_rs = self._check_relative_strength(close)
        bko_ok, bko_20, bko_55, hh_20_prior, hh_55_prior = self._check_breakout(
            high, close, p
        )
        # Hard filter pass count (allow 1 miss)
        filter_flags = [
            trend_ok,
            # adx_ok,
            high_ok,
            base_ok,
            rvol_ok,
            bko_ok,
        ]
        if self._benchmark_candles is not None:
            filter_flags.append(rs_ok)

        hard_filter_passes = sum(filter_flags)
        hard_filter_total = len(filter_flags)

        # ── Stage 2: Weighted Ranking ──────────────────────────

        s_rs = self._score_weighted_rs(close, p)
        s_bo = self._score_breakout_strength(
            volume,
            bko_20,
            bko_55,
            hh_20_prior,
            hh_55_prior,
            last_high,
            new_52w_high,
            p,
        )
        s_rv = self._score_rvol(volume, p)
        s_tr = self._score_trend_alignment(
            close,
            sma50_rising,
            sma200_rising,
            close_above_sma20,
            p,
        )
        s_vc = self._score_volatility_contraction(close, high, low, p)
        s_ad = self._score_adx(high, low, close, p)
        s_bq = self._score_base_quality(high, low, close, base_depth, p)
        # acc_ratio, s_acc = self._score_accumulation(volume, p)
        s_vcp = self._score_vcp_pullback(high, low, close, p)

        # Raw weighted score
        raw_score = (
            0.25 * s_rs
            + 0.25 * s_bo
            + 0.10 * s_rv
            + 0.15 * s_tr
            + 0.10 * s_vc
            + 0.05 * s_vcp
            + 0.05 * s_ad
            + 0.05 * s_bq
        )

        # Scale by filter pass rate
        pass_ratio = hard_filter_passes / hard_filter_total
        if pass_ratio < 0.6:
            score = raw_score * pass_ratio * 0.5
        elif pass_ratio < 0.8:
            score = raw_score * pass_ratio
        else:
            score = raw_score

        score = round(float(score), 4)

        # Signal determination
        if score >= 0.5 and bko_ok and rvol_ok:
            signal: str = "long"
        elif score >= 0.3 and bko_ok:
            signal = "long"
        else:
            signal = "neutral"

        # ── Metadata ───────────────────────────────────────────
        metadata: dict[str, Any] = {
            "rvol": round(rvol_val, 2),
            "distance_52w": round(high_dist * 100, 2),
            "w_rs": round(weighted_rs, 4),
            "new_high": int(new_52w_high),
            "bo_vol": round(s_bo, 4),
            "vcp_score": round(s_vcp, 4),
        }

        return ScreenResult(
            symbol=symbol,
            signal=signal,  # type: ignore[arg-type]
            score=score,
            price=round(last_close_val, 2),
            metadata=metadata,
        )

    def rank(self, results: list[ScreenResult]) -> list[ScreenResult]:
        """Sort by score descending. Insufficient-data results sink."""
        return sorted(
            results,
            key=lambda r: (
                r.score,
                r.metadata.get("reason") is None,
            ),
            reverse=True,
        )


# ── Factory (required export) ──────────────────────────────────


def make(symbols: list[str], params: dict[str, Any]) -> BreakoutScreen:
    """Construct a BreakoutScreen instance."""
    return BreakoutScreen(symbols=symbols, params=params)
