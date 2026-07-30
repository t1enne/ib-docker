"""Tests for AEGIS pure functions: VAM, anchor selection, minimax, SLSQP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bt.strategies.aegis import (
    _vam,
    _log_returns,
    _select_anchors,
    _momentum_gate,
    _minimax_diversifiers,
    _slsqp_optimise,
    SECTOR_MAP,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic price series
# ---------------------------------------------------------------------------


def _make_prices(
    n: int = 300,
    drift: float = 0.0005,
    vol: float = 0.01,
    seed: int = 42,
) -> pd.Series:
    """Synthetic price series with log-normal returns."""
    rng = np.random.default_rng(seed)
    log_ret = drift + vol * rng.standard_normal(n - 1)
    prices = 100.0 * np.exp(np.cumsum(log_ret))
    return pd.Series(np.insert(prices, 0, 100.0))


def _make_prices_high_drift(n: int = 300, seed: int = 42) -> pd.Series:
    """Higher drift for anchor-worthy assets."""
    return _make_prices(n=n, drift=0.001, vol=0.008, seed=seed)


def _make_prices_low_vol(n: int = 300, seed: int = 99) -> pd.Series:
    """Lower volatility for high VAM."""
    return _make_prices(n=n, drift=0.0005, vol=0.005, seed=seed)


def _make_prices_negative(n: int = 300, seed: int = 7) -> pd.Series:
    """Negative drift — should fail momentum gate."""
    return _make_prices(n=n, drift=-0.0005, vol=0.01, seed=seed)


# ---------------------------------------------------------------------------
# log_returns
# ---------------------------------------------------------------------------


def test_log_returns_shape():
    prices = _make_prices(100)
    lr = _log_returns(prices)
    assert len(lr) == len(prices)
    assert pd.isna(lr.iloc[0])  # first NaN
    assert not pd.isna(lr.iloc[1])


def test_log_returns_time_additivity():
    """Log returns are additive: Σlog_r = ln(P_t / P_0)."""
    prices = _make_prices(100)
    lr = _log_returns(prices)
    total = lr.dropna().sum()
    expected = np.log(prices.iloc[-1] / prices.iloc[0])
    assert np.isclose(total, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# VAM
# ---------------------------------------------------------------------------


def test_vam_positive_drift():
    prices = _make_prices(300, drift=0.001, vol=0.005)
    score = _vam(prices, mom_lb=252, skip=21, vol_w=252)
    assert score > 0.5  # decent risk-adjusted return


def test_vam_negative_drift():
    prices = _make_prices(300, drift=-0.001, vol=0.01)
    score = _vam(prices, mom_lb=252, skip=21, vol_w=252)
    assert score < 0.0


def test_vam_insufficient_data():
    prices = _make_prices(50)
    score = _vam(prices, mom_lb=252, skip=21, vol_w=252)
    assert score == -np.inf


def test_vam_low_vol_beats_high_vol():
    """Same return, lower vol → higher VAM."""
    # Same drift, different vol
    high_vol = _make_prices(300, drift=0.0005, vol=0.02, seed=1)
    low_vol = _make_prices(300, drift=0.0005, vol=0.005, seed=1)
    score_high = _vam(high_vol, mom_lb=252, skip=21, vol_w=252)
    score_low = _vam(low_vol, mom_lb=252, skip=21, vol_w=252)
    assert score_low > score_high


# ---------------------------------------------------------------------------
# Anchor selection
# ---------------------------------------------------------------------------


def test_select_anchors_returns_correct_count():
    symbols = ["A1", "A2", "B1", "B2", "C1", "C2"]
    # Assign sectors
    old_map = dict(SECTOR_MAP)
    try:
        SECTOR_MAP.update(
            {
                "A1": "SectorA",
                "A2": "SectorA",
                "B1": "SectorB",
                "B2": "SectorB",
                "C1": "SectorC",
                "C2": "SectorC",
            }
        )
        closes = {}
        for i, sym in enumerate(symbols):
            closes[sym] = _make_prices(300, drift=0.0005 + i * 0.0001, vol=0.01, seed=i)

        anchors = _select_anchors(
            symbols=symbols,
            closes_map=closes,
            mom_lb=252,
            skip=21,
            vol_w=252,
            n_anchors=3,
        )
        assert len(anchors) == 3
        assert all(a in symbols for a in anchors)
    finally:
        SECTOR_MAP.clear()
        SECTOR_MAP.update(old_map)


def test_select_anchors_insufficient_candidates():
    """Only 1 symbol in universe → fewer than n_anchors."""
    symbols = ["A1"]
    closes = {"A1": _make_prices(300)}
    anchors = _select_anchors(
        symbols=symbols,
        closes_map=closes,
        mom_lb=252,
        skip=21,
        vol_w=252,
        n_anchors=3,
    )
    assert len(anchors) <= 1


# ---------------------------------------------------------------------------
# Momentum gate
# ---------------------------------------------------------------------------


def test_momentum_gate_positive_pass():
    prices = _make_prices(300, drift=0.001)
    result = _momentum_gate(["A"], {"A": prices}, mom_lb=252, skip=21)
    assert "A" in result


def test_momentum_gate_negative_fail():
    prices = _make_prices_negative(300)
    result = _momentum_gate(["A"], {"A": prices}, mom_lb=252, skip=21)
    assert "A" not in result


# ---------------------------------------------------------------------------
# Minimax diversifiers
# ---------------------------------------------------------------------------


def test_minimax_adds_diversifiers():
    """Diversifiers should be uncorrelated with anchors."""
    rng = np.random.default_rng(42)
    n = 300

    # Anchor: positive trend
    anchor = pd.Series(rng.standard_normal(n).cumsum() * 0.01 + 100.0)
    # Candidate A: correlated with anchor (same noise)
    candidate_a = anchor + pd.Series(rng.standard_normal(n) * 0.001)
    # Candidate B: uncorrelated (independent noise)
    candidate_b = pd.Series(rng.standard_normal(n).cumsum() * 0.005 + 50.0)

    rets = {
        "A": _log_returns(anchor).dropna(),
        "X": _log_returns(candidate_a).dropna(),
        "Y": _log_returns(candidate_b).dropna(),
    }
    corr = pd.DataFrame(rets).corr()

    divers = _minimax_diversifiers(
        anchors=["A"],
        candidates=["X", "Y"],
        corr=corr,
        n_slots=1,
    )
    # Should pick Y (lower correlation with anchor)
    assert divers[0] == "Y"


def test_minimax_no_candidates():
    corr = pd.DataFrame({"A": [1.0, 0.5], "B": [0.5, 1.0]}, index=["A", "B"])
    divers = _minimax_diversifiers(
        anchors=["A", "B"],
        candidates=[],
        corr=corr,
        n_slots=5,
    )
    assert divers == []


# ---------------------------------------------------------------------------
# SLSQP optimisation
# ---------------------------------------------------------------------------


def test_slsqp_returns_valid_weights():
    """Weights sum to 1 and are in [0, max_weight].

    Need ≥20 assets with max_weight=0.05 for feasibility (20 × 0.05 = 1.0).
    """
    rng = np.random.default_rng(0)
    n_assets, n_days = 25, 100
    data = {}
    for i in range(n_assets):
        data[f"S{i}"] = rng.standard_normal(n_days) * 0.01 + 0.0005 + i * 0.0002

    returns_df = pd.DataFrame(data)
    weights = _slsqp_optimise(
        symbols=list(returns_df.columns),
        returns_df=returns_df,
        risk_free_rate=0.04,
        max_weight=0.05,
    )

    assert len(weights) == n_assets
    assert abs(sum(weights.values()) - 1.0) < 1e-8
    for _, w in weights.items():
        assert -1e-8 <= w <= 0.05 + 1e-8


def test_slsqp_single_asset():
    returns_df = pd.DataFrame(
        {"A": np.random.default_rng(1).standard_normal(100) * 0.01 + 0.0005}
    )
    weights = _slsqp_optimise(
        symbols=["A"],
        returns_df=returns_df,
        risk_free_rate=0.04,
        max_weight=0.05,
    )
    assert weights == {"A": 1.0}


def test_slsqp_empty():
    weights = _slsqp_optimise([], pd.DataFrame(), 0.04, 0.05)
    assert weights == {}


def test_slsqp_prefers_high_return_low_vol():
    """Asset with higher Sharpe gets higher weight."""
    rng = np.random.default_rng(2)
    n = 200
    good = pd.Series(rng.standard_normal(n) * 0.005 + 0.0015)  # high return, low vol
    bad = pd.Series(rng.standard_normal(n) * 0.02 + 0.0002)  # low return, high vol

    df = pd.DataFrame({"good": good, "bad": bad})
    weights = _slsqp_optimise(
        symbols=["good", "bad"],
        returns_df=df,
        risk_free_rate=0.04,
        max_weight=0.95,
    )
    assert weights["good"] > weights["bad"]
