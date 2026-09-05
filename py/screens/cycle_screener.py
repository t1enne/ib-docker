"""cycle_screener.py — Macro cycle regime detection screen.

Operates on 40+ cross-asset tickers (equities, sectors, fixed income,
commodities, currencies, volatility) to answer:

  "What macro regime is the market pricing today,
   and how is that regime changing?"

This is a standalone screen (NOT a backtest strategy). It follows the
same pattern as ``screens/breakout_screen.py`` — exports ``DEFAULTS`` +
``make()`` and conforms to ``ScreenFn``.

Usage:

    python main.py screen cycle_screener universe.yml

The screen computes a single macro result (pseudo-symbol ``"MACRO"``)
with rich metadata showing composite regime scores, cross-asset ratios,
and regime classification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.market_data import resample_ohlcv
from src.screen.types import ScreenFn, ScreenResult


# ── Default parameters ─────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "lookback_days": 252,  # ~1 year of daily data
    "momentum_windows": [20, 50, 100, 200],
    "rs_window": 63,  # relative strength window
    "dma_fast": 50,
    "dma_slow": 200,
    "vol_window": 20,
}


# ── Cross-asset universe (hardcoded) ──────────────────────────

EQUITY_INDICES: list[str] = ["SPY", "QQQ", "IWM", "VWO", "SMH"]
SECTORS: list[str] = [
    "XLF",
    "XLK",
    "XLI",
    "XLB",
    "XLE",
    "XLV",
    "XLP",
    "XLU",
    "XLRE",
    "XLC",
]
TREASURIES: list[str] = ["SHY", "IEI", "IEF", "TLT"]
CREDIT: list[str] = ["HYG", "JNK", "LQD"]
COMMODITIES: list[str] = ["CPER", "GLD", "SLV", "USO", "UNG"]
CURRENCY: list[str] = ["UUP"]
VOLATILITY: list[str] = ["VIX"]

ALL_ASSETS: list[str] = (
    EQUITY_INDICES + SECTORS + TREASURIES + CREDIT + COMMODITIES + CURRENCY + VOLATILITY
)

# Primary benchmark for relative-strength comparisons
_BENCHMARK: str = "SPY"

# ── Regime labels ──────────────────────────────────────────────

_REGIME_LABELS: list[str] = [
    "Crisis",
    "Recovery",
    "Expansion",
    "Late_Cycle",
    "Inflationary_Boom",
]

# ── Ratio definitions ──────────────────────────────────────────

_RATIO_DEFS: list[tuple[str, str, str]] = [
    # (label, numerator, denominator)
    ("xly_xlp", "XLY", "XLP"),  # consumer discretion / staples
    ("xli_xlu", "XLI", "XLU"),  # industrials / utilities
    ("xlf_xlu", "XLF", "XLU"),  # financials / utilities
    ("iwm_spy", "IWM", "SPY"),  # small-cap / large-cap
    ("cper_gld", "CPER", "GLD"),  # copper / gold (growth vs safety)
    ("uso_gld", "USO", "GLD"),  # oil / gold
]


# ================================================================
#  Layer helpers — pure functions, each independently testable
# ================================================================


def _safe_float(series: pd.Series, idx: int = -1) -> float:
    """Extract a float from a Series, returning 0.0 if NaN or empty."""
    if len(series) == 0:
        return 0.0
    v = series.iloc[idx]
    return 0.0 if pd.isna(v) else float(v)


def _highest(series: pd.Series, window: int) -> pd.Series:
    """Rolling maximum."""
    return series.rolling(window=window, min_periods=1).max()


def _lowest(series: pd.Series, window: int) -> pd.Series:
    """Rolling minimum."""
    return series.rolling(window=window, min_periods=1).min()


def _returns(series: pd.Series, periods: int) -> float:
    """Return over N periods: (close[t] / close[t-N]) - 1."""
    if len(series) < periods + 1:
        return 0.0
    denom = float(series.iloc[-periods - 1])
    if denom == 0.0:
        return 0.0
    return float(series.iloc[-1]) / denom - 1.0


# ── Layer 3: Momentum features ─────────────────────────────────


def _calc_momentum_scores(closes: dict[str, pd.Series]) -> dict[str, float]:
    """Compute multi-period momentum scores per asset.

    For each asset, compute returns over configurable windows and
    aggregate into a single momentum score (0-100).

    Returns:
        dict[asset -> momentum_score]
    """
    windows = [20, 50, 100, 200]
    weights = [0.4, 0.3, 0.2, 0.1]

    scores: dict[str, float] = {}
    for ticker, series in closes.items():
        if len(series) < max(windows) + 1:
            scores[ticker] = 0.0
            continue
        total = 0.0
        for w, wt in zip(windows, weights, strict=False):
            r = _returns(series, w)
            # Clamp return to [-0.5, 0.5] and map to [0, 100]
            clipped = max(-0.5, min(0.5, r))
            total += wt * ((clipped + 0.5) / 1.0) * 100.0
        scores[ticker] = round(total, 2)
    return scores


# ── Layer 1: Relative strength vs benchmark ────────────────────


def _relative_strength(
    closes: dict[str, pd.Series],
    bench_close: pd.Series,
    window: int = 63,
) -> dict[str, float]:
    """RS = asset_return - benchmark_return over *window* days.

    Returns:
        dict[asset -> rs_score]
    """
    rs_scores: dict[str, float] = {}
    for ticker, series in closes.items():
        if ticker == _BENCHMARK:
            continue
        asset_ret = _returns(series, window)
        bench_ret = _returns(bench_close, window)
        rs_scores[ticker] = round(asset_ret - bench_ret, 4)
    return rs_scores


# ── Layer 2: Leadership ranking ────────────────────────────────


def _leadership_ranking(rs_scores: dict[str, float]) -> list[tuple[str, float]]:
    """Rank assets by RS descending. Returns list of (ticker, rs)."""
    return sorted(rs_scores.items(), key=lambda x: x[1], reverse=True)


# ── Layer 4: Ratios ────────────────────────────────────────────


def _compute_ratios(
    closes: dict[str, pd.Series],
) -> dict[str, float]:
    """Compute cross-asset ratio values.

    Each ratio is computed as (numerator_close / denominator_close).

    Returns:
        dict[ratio_name -> value]
    """
    ratios: dict[str, float] = {}
    for name, num, den in _RATIO_DEFS:
        num_s = closes.get(num)
        den_s = closes.get(den)
        if num_s is None or den_s is None or len(num_s) < 2 or len(den_s) < 2:
            ratios[name] = 0.0
            continue
        num_val = float(num_s.iloc[-1])
        den_val = float(den_s.iloc[-1])
        if den_val <= 0:
            ratios[name] = 0.0
            continue
        ratios[name] = round(num_val / den_val, 4)
    return ratios


def _ratio_trends(
    closes: dict[str, pd.Series],
) -> dict[str, float]:
    """63-day % change for each ratio, capturing directional trend.

    Returns:
        dict[ratio_name -> pct_change]
    """
    trends: dict[str, float] = {}
    for name, num, den in _RATIO_DEFS:
        num_s = closes.get(num)
        den_s = closes.get(den)
        if num_s is None or den_s is None or len(num_s) < 64 or len(den_s) < 64:
            trends[f"{name}_trend"] = 0.0
            continue
        ratio_now = float(num_s.iloc[-1]) / float(den_s.iloc[-1])
        ratio_63 = float(num_s.iloc[-64]) / float(den_s.iloc[-64])
        if ratio_63 <= 0:
            trends[f"{name}_trend"] = 0.0
            continue
        trends[f"{name}_trend"] = round((ratio_now / ratio_63 - 1.0) * 100.0, 2)
    return trends


# ── Layer 5: Breadth ───────────────────────────────────────────


def _calc_breadth(closes: dict[str, pd.Series]) -> dict[str, float]:
    """Percentage of assets above 50-day and 200-day SMA.

    Returns:
        dict with keys "above_50dma" and "above_200dma"
    """
    above_50 = 0.0
    above_200 = 0.0
    count = 0

    for ticker, series in closes.items():
        if len(series) < 201:
            continue
        count += 1
        last_close = float(series.iloc[-1])
        sma_50 = float(series.rolling(50).mean().iloc[-1])
        sma_200 = float(series.rolling(200).mean().iloc[-1])
        if last_close > sma_50:
            above_50 += 1.0
        if last_close > sma_200:
            above_200 += 1.0

    total = count if count > 0 else 1
    return {
        "above_50dma": round(above_50 / total * 100.0, 1),
        "above_200dma": round(above_200 / total * 100.0, 1),
    }


# ── Layer 6: Credit ────────────────────────────────────────────


def _calc_credit_scores(closes: dict[str, pd.Series]) -> dict[str, float]:
    """Compute credit market scores.

    Returns:
        dict with keys:
            "hyg_tlt_ratio": HYG/TLT price ratio
            "lqd_tlt_ratio": LQD/TLT price ratio
            "credit_spread_trend": HYG/TLT trend over 63d
    """
    hyg = closes.get("HYG")
    lqd = closes.get("LQD")
    tlt = closes.get("TLT")

    result: dict[str, float] = {}

    if hyg is not None and tlt is not None and len(hyg) > 1 and len(tlt) > 1:
        hyg_now = float(hyg.iloc[-1])
        tlt_now = float(tlt.iloc[-1])
        result["hyg_tlt_ratio"] = round(hyg_now / tlt_now, 4)
        if len(hyg) >= 64 and len(tlt) >= 64:
            hyg_63 = float(hyg.iloc[-64])
            tlt_63 = float(tlt.iloc[-64])
            ratio_now = hyg_now / tlt_now
            ratio_63 = hyg_63 / tlt_63
            if ratio_63 > 0:
                result["hyg_tlt_trend"] = round((ratio_now / ratio_63 - 1.0) * 100.0, 2)
    if lqd is not None and tlt is not None and len(lqd) > 1 and len(tlt) > 1:
        lqd_now = float(lqd.iloc[-1])
        tlt_now = float(tlt.iloc[-1])
        result["lqd_tlt_ratio"] = round(lqd_now / tlt_now, 4)

    return result


# ── Layer 7: Rates ─────────────────────────────────────────────


def _calc_rates_scores(closes: dict[str, pd.Series]) -> dict[str, float]:
    """Compute interest-rate-related scores.

    Uses Treasury ETFs to proxy the yield curve.

    Returns:
        dict with keys:
            "spread_2s10s": SHY(1-3Y) vs IEF(7-10Y) ratio
            "yield_trend_10y": IEF price trend (inverted = rising yields)
    """
    shy = closes.get("SHY")
    ief = closes.get("IEF")

    result: dict[str, float] = {}

    if shy is not None and ief is not None and len(shy) > 1 and len(ief) > 1:
        shy_now = float(shy.iloc[-1])
        ief_now = float(ief.iloc[-1])
        result["spread_2s10s"] = round(shy_now / ief_now, 4)

        if len(ief) >= 64:
            ief_63 = float(ief.iloc[-64])
            if ief_63 > 0:
                ief_trend = (ief_now / ief_63 - 1.0) * 100.0
                # Invert: negative IEF trend = rising yields = negative
                result["yield_trend_10y"] = round(-ief_trend, 2)

    return result


# ── Layer 8: Regime scoring ────────────────────────────────────


def _score_risk(
    credit_scores: dict[str, float],
    volatility_px: float | None,
) -> float:
    """Risk score (0-100). High = risk-on appetite.

    Components:
      - HYG/TLT trend positive -> +risk (risk-on)
      - VIX low -> +risk

    Returns 0-100.
    """
    score = 50.0  # neutral baseline

    hyg_trend = credit_scores.get("hyg_tlt_trend", 0.0)
    if hyg_trend > 5:
        score += 20.0
    elif hyg_trend > 2:
        score += 10.0
    elif hyg_trend < -5:
        score -= 20.0
    elif hyg_trend < -2:
        score -= 10.0

    if volatility_px is not None:
        if volatility_px < 15:
            score += 15.0
        elif volatility_px < 20:
            score += 5.0
        elif volatility_px > 30:
            score -= 20.0
        elif volatility_px > 25:
            score -= 10.0

    return round(max(0.0, min(100.0, score)), 1)


def _score_growth(
    ratio_trends: dict[str, float],
    leadership: list[tuple[str, float]],
) -> float:
    """Growth score (0-100). High = pro-cyclical.

    Components:
      - XLY/XLP trend positive -> growth
      - IWM/SPY trend positive -> small-cap leadership
      - XLI/XLU trend positive -> industrials leading
      - Sector leaders are cyclical -> growth
    """
    score = 50.0

    xly_xlp = ratio_trends.get("xly_xlp_trend", 0.0)
    if xly_xlp > 5:
        score += 15.0
    elif xly_xlp > 2:
        score += 7.0
    elif xly_xlp < -5:
        score -= 15.0
    elif xly_xlp < -2:
        score -= 7.0

    iwm_spy = ratio_trends.get("iwm_spy_trend", 0.0)
    if iwm_spy > 3:
        score += 10.0
    elif iwm_spy < -3:
        score -= 10.0

    xli_xlu = ratio_trends.get("xli_xlu_trend", 0.0)
    if xli_xlu > 3:
        score += 10.0
    elif xli_xlu < -3:
        score -= 10.0

    # Cyclical sectors in top 3 leaders -> bonus
    cyclical = {"XLF", "XLK", "XLI", "XLB", "XLE"}
    top3 = {sym for sym, _ in leadership[:3]}
    cycle_count = len(top3 & cyclical)
    score += cycle_count * 5.0

    return round(max(0.0, min(100.0, score)), 1)


def _score_inflation(
    ratio_trends: dict[str, float],
    closes: dict[str, pd.Series],
) -> float:
    """Inflation score (0-100). High = rising inflation expectations.

    Components:
      - Copper/Gold rising -> inflation
      - Oil/Gold rising -> inflation
      - Commodity-sector (XLE) momentum
    """
    score = 50.0

    cper_gld = ratio_trends.get("cper_gld_trend", 0.0)
    if cper_gld > 5:
        score += 15.0
    elif cper_gld > 2:
        score += 7.0
    elif cper_gld < -5:
        score -= 15.0
    elif cper_gld < -2:
        score -= 7.0

    uso_gld = ratio_trends.get("uso_gld_trend", 0.0)
    if uso_gld > 5:
        score += 10.0
    elif uso_gld > 2:
        score += 5.0
    elif uso_gld < -5:
        score -= 10.0
    elif uso_gld < -2:
        score -= 5.0

    # XLE energy sector momentum
    xle = closes.get("XLE")
    if xle is not None and len(xle) >= 64:
        xle_ret = _returns(xle, 63)
        if xle_ret > 0.15:
            score += 10.0
        elif xle_ret > 0.05:
            score += 5.0
        elif xle_ret < -0.10:
            score -= 10.0

    return round(max(0.0, min(100.0, score)), 1)


def _score_breadth(breadth: dict[str, float]) -> float:
    """Breadth score (0-100). High = broad participation.

    More assets above their 50 and 200 DMA -> higher score.
    """
    above_200 = breadth.get("above_200dma", 0.0)
    above_50 = breadth.get("above_50dma", 0.0)

    # Blend: 200 DMA is heavier weight (longer-term breadth)
    score = 0.4 * above_50 + 0.6 * above_200
    return round(max(0.0, min(100.0, score)), 1)


def _score_liquidity(
    rates_scores: dict[str, float],
    credit_scores: dict[str, float],
) -> float:
    """Liquidity / monetary conditions score (0-100).

    High = accommodative (low yields, narrow credit spreads).
    Low = tight (rising yields, widening credit spreads).
    """
    score = 50.0

    yield_trend = rates_scores.get("yield_trend_10y", 0.0)
    if yield_trend < -5:  # Yields rising sharply -> tightening
        score -= 15.0
    elif yield_trend < -2:
        score -= 7.0
    elif yield_trend > 5:  # Yields falling -> easing
        score += 10.0
    elif yield_trend > 2:
        score += 5.0

    hyg_trend = credit_scores.get("hyg_tlt_trend", 0.0)
    if hyg_trend > 5:  # Credit spreads narrowing -> good liquidity
        score += 10.0
    elif hyg_trend < -5:  # Credit spreads widening -> liquidity stress
        score -= 10.0

    return round(max(0.0, min(100.0, score)), 1)


# ── Layer 9: Regime classification ─────────────────────────────


def _compute_regime_likelihoods(
    growth_score: float,
    inflation_score: float,
    risk_score: float,
    breadth_score: float,
    liquidity_score: float,
) -> dict[str, float]:
    """Compute likelihood scores for each regime category.

    Higher = more likely. Each score is a weighted blend of the
    five dimension scores, following the regime mapping:

    +=================+=========+===========+============+
    | Regime           | Growth  | Inflation | Risk/Liq   |
    +=================+=========+===========+============+
    | Crisis           | low     | any       | risk-off   |
    | Recovery         | rising  | low/stable| improving  |
    | Expansion        | high    | moderate  | risk-on    |
    | Late_Cycle       | slowing | rising    | tightening |
    | Inflationary_Boom| high    | high      | mixed      |
    +=================+=========+===========+============+
    """
    rc = 100.0 - risk_score
    gc = 100.0 - growth_score
    bc = 100.0 - breadth_score
    lc = 100.0 - liquidity_score
    ic = 100.0 - inflation_score

    return {
        "Crisis": gc * 0.3 + rc * 0.3 + bc * 0.2 + lc * 0.2,
        "Recovery": growth_score * 0.25
        + ic * 0.2
        + risk_score * 0.2
        + breadth_score * 0.15
        + liquidity_score * 0.2,
        "Expansion": growth_score * 0.30
        + max(0.0, 100.0 - abs(inflation_score - 50.0)) * 0.15
        + risk_score * 0.2
        + breadth_score * 0.2
        + liquidity_score * 0.15,
        "Late_Cycle": gc * 0.2
        + inflation_score * 0.25
        + lc * 0.25
        + rc * 0.15
        + breadth_score * 0.15,
        "Inflationary_Boom": growth_score * 0.25
        + inflation_score * 0.30
        + risk_score * 0.1
        + bc * 0.1
        + liquidity_score * 0.25,
    }


def _compute_confidence(
    likelihoods: dict[str, float],
    best_regime: str,
) -> float:
    """Compute regime confidence from likelihood margin."""
    best_score = likelihoods[best_regime]
    sorted_scores = sorted(likelihoods.values(), reverse=True)
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    margin = best_score - second_score

    if margin > 20:
        confidence = min(100.0, best_score * 0.7 + margin * 0.3)
    elif margin > 10:
        confidence = min(90.0, best_score * 0.6 + margin * 0.4)
    else:
        confidence = best_score * 0.5 + margin

    return round(min(100.0, max(0.0, confidence)), 1)


def _classify_regime(
    risk_score: float,
    growth_score: float,
    inflation_score: float,
    breadth_score: float,
    liquidity_score: float,
) -> tuple[str, float]:
    """Map composite scores to a regime label with confidence."""
    likelihoods = _compute_regime_likelihoods(
        growth_score,
        inflation_score,
        risk_score,
        breadth_score,
        liquidity_score,
    )
    best_regime: str = max(likelihoods, key=lambda k: likelihoods[k])  # type: ignore[arg-type]
    confidence = _compute_confidence(likelihoods, best_regime)
    return best_regime, confidence


# ── Layer 10: Cross-asset confirmation ─────────────────────────


def _direction(score: float, lower: float = 45, upper: float = 55) -> str:
    """Classify a score as up, down, or flat."""
    return "up" if score > upper else "down" if score < lower else "flat"


def _signals_agree(a: float, b: float, threshold: float = 2.0) -> bool:
    """Both signals are non-trivial and point the same direction."""
    return abs(a) > threshold and abs(b) > threshold and (a > 0) == (b > 0)


def _cross_asset_confirmation(
    ratio_trends: dict[str, float],
    risk_score: float,
    growth_score: float,
) -> float:
    """Cross-asset confirmation score (0-100).

    How many independent asset classes agree on the dominant
    macro direction. Higher = stronger conviction.
    """
    agreements = 0.0
    total_signals = 0

    eq_signal = ratio_trends.get("iwm_spy_trend", 0.0)
    cons_signal = ratio_trends.get("xly_xlp_trend", 0.0)
    comm_signal = ratio_trends.get("cper_gld_trend", 0.0)
    credit_signal = ratio_trends.get("hyg_tlt_trend", 0.0)

    # Growth vs risk direction
    gd = _direction(growth_score)
    rd = _direction(risk_score)
    if gd != "flat" and rd != "flat":
        total_signals += 1
        if gd == rd:
            agreements += 1.0

    # Equity vs cyclical consumption
    if _signals_agree(eq_signal, cons_signal):
        total_signals += 1
        agreements += 1.0

    # Commodities vs equities
    if _signals_agree(comm_signal, eq_signal):
        total_signals += 1
        agreements += 1.0

    # Credit vs equities
    if _signals_agree(credit_signal, eq_signal):
        total_signals += 1
        agreements += 1.0

    if total_signals == 0:
        return 50.0

    return round(agreements / total_signals * 100.0, 1)


# ================================================================
#  Screen implementation
# ================================================================


class CycleScreen:
    """Macro cycle regime detection screen.

    Analyzes the full cross-asset universe as a single entity and
    produces ONE result (pseudo-symbol ``"MACRO"``) with rich metadata.
    """

    __slots__ = ("symbols", "params", "_computed")

    def __init__(self, symbols: list[str], params: dict[str, Any]) -> None:
        self.symbols = symbols
        self.params = params
        self._computed: bool = False

    # ── Data loading ────────────────────────────

    def _fetch_universe_data(self) -> dict[str, pd.Series]:
        """Fetch and resample daily close data for all cross-asset tickers.

        Returns:
            dict[ticker -> daily close Series]
        """
        from src.utils import get_local_candles

        closes: dict[str, pd.Series] = {}
        for ticker in ALL_ASSETS:
            try:
                df = get_local_candles(symbol=ticker, bar="1h")
                if df.empty:
                    continue
                daily = resample_ohlcv(df, "1d")
                if daily.empty:
                    continue
                closes[ticker] = daily["close"]
            except Exception:
                continue

        return closes

    # ── compute ─────────────────────────────────

    def compute(self, symbol: str, candles: pd.DataFrame) -> ScreenResult:
        """Compute macro regime result.

        Args:
            symbol: Ignored -- the screen always produces a single
                    ``"MACRO"`` result from the full universe.
            candles: Ignored -- the screen fetches its own cross-asset data.

        Returns:
            ScreenResult with ``"MACRO"`` symbol or a dummy neutral
            result for any other symbol.
        """
        # Only produce the macro result on the first call
        if self._computed:
            return ScreenResult(
                symbol=symbol,
                signal="neutral",
                score=0.0,
                price=0.0,
                metadata={"reason": "macro_result_already_produced"},
            )

        self._computed = True

        closes = self._fetch_universe_data()
        if not closes:
            return ScreenResult(
                symbol=symbol,
                signal="neutral",
                score=0.0,
                price=0.0,
                metadata={"reason": "no_universe_data"},
            )

        bench_close = closes.get(_BENCHMARK)
        if bench_close is None:
            return ScreenResult(
                symbol=symbol,
                signal="neutral",
                score=0.0,
                price=0.0,
                metadata={"reason": "no_benchmark_data"},
            )

        # ── Compute all layers ──────────────────

        # Momentum
        momentum_scores = _calc_momentum_scores(closes)

        # Relative strength
        rs_window = int(self.params.get("rs_window", 63))
        rs_scores = _relative_strength(closes, bench_close, rs_window)

        # Leadership ranking
        leadership = _leadership_ranking(rs_scores)

        # Ratios
        ratio_trends = _ratio_trends(closes)

        # Breadth
        breadth = _calc_breadth(closes)

        # Credit
        credit_scores = _calc_credit_scores(closes)

        # Rates
        rates_scores = _calc_rates_scores(closes)

        # Volatility
        vix_close = closes.get("VIX")
        vol_px = _safe_float(vix_close) if vix_close is not None else None

        # Regime scoring
        risk_score = _score_risk(credit_scores, vol_px)
        growth_score = _score_growth(ratio_trends, leadership)
        inflation_score = _score_inflation(ratio_trends, closes)
        breadth_score = _score_breadth(breadth)
        liquidity_score = _score_liquidity(rates_scores, credit_scores)

        # Regime classification
        regime, confidence = _classify_regime(
            risk_score,
            growth_score,
            inflation_score,
            breadth_score,
            liquidity_score,
        )

        # Cross-asset confirmation
        confirmation = _cross_asset_confirmation(
            ratio_trends,
            risk_score,
            growth_score,
        )

        # ── Composite score ─────────────────────

        # Weighted blend of all scores
        composite = (
            0.20 * (risk_score / 100.0)
            + 0.20 * (growth_score / 100.0)
            + 0.15 * (1.0 - abs(inflation_score - 50.0) / 50.0)
            + 0.20 * (breadth_score / 100.0)
            + 0.15 * (liquidity_score / 100.0)
            + 0.10 * (confirmation / 100.0)
        )
        # Boost by regime confidence
        score = round(composite * (confidence / 100.0), 4)

        # Signal: long = pro-risk, short = defensive, neutral = mixed
        if regime in ("Expansion", "Recovery") and confidence > 50:
            signal: str = "long"
        elif regime in ("Crisis",) and confidence > 50:
            signal = "short"
        else:
            signal = "neutral"

        # ── Metadata ─────────────────────────────

        # Top 5 leadership
        leaders_str = ",".join(sym for sym, _ in leadership[:5])

        # Asset class average momentum
        avg_mom = np.mean(list(momentum_scores.values())) if momentum_scores else 0.0

        metadata: dict[str, Any] = {
            # Regime
            "regime": regime,
            "confidence": confidence,
            # Composite scores
            "risk_score": risk_score,
            "growth_score": growth_score,
            "inflation_score": inflation_score,
            "breadth_score": breadth_score,
            "liquidity_score": liquidity_score,
            "confirmation": confirmation,
            # Breadth
            "above_50dma": breadth.get("above_50dma", 0.0),
            "above_200dma": breadth.get("above_200dma", 0.0),
            # Leadership
            "leadership": leaders_str,
            # Ratios
            "spread_2s10s": rates_scores.get("spread_2s10s", 0.0),
            "yield_trend_10y": rates_scores.get("yield_trend_10y", 0.0),
            "hyg_tlt_trend": credit_scores.get("hyg_tlt_trend", 0.0),
            "xly_xlp_trend": ratio_trends.get("xly_xlp_trend", 0.0),
            "iwm_spy_trend": ratio_trends.get("iwm_spy_trend", 0.0),
            "cper_gld_trend": ratio_trends.get("cper_gld_trend", 0.0),
            "uso_gld_trend": ratio_trends.get("uso_gld_trend", 0.0),
            # Momentum
            "avg_momentum": round(float(avg_mom), 1),
        }

        return ScreenResult(
            symbol=symbol,
            signal=signal,  # type: ignore[arg-type]
            score=score,
            price=0.0  # No single price for a macro result
            if regime != "Expansion"
            else round(float(bench_close.iloc[-1]), 2),
            metadata=metadata,
        )

    def rank(self, results: list[ScreenResult]) -> list[ScreenResult]:
        """Sort by score descending. No-op results sink."""
        return sorted(
            results,
            key=lambda r: (
                r.score,
                r.metadata.get("reason") is None,
            ),
            reverse=True,
        )


# ── Factory (required export) ──────────────────────────────────


def make(symbols: list[str], params: dict[str, Any]) -> CycleScreen:
    """Construct a CycleScreen instance."""
    return CycleScreen(symbols=symbols, params=params)
