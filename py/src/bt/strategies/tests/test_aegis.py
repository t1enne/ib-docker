"""Tests for AEGIS pure functions — one test per function."""

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


def _make_prices(
    n: int = 300, drift: float = 0.0005, vol: float = 0.01, seed: int = 42
) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_ret = drift + vol * rng.standard_normal(n - 1)
    return pd.Series(np.insert(100.0 * np.exp(np.cumsum(log_ret)), 0, 100.0))


# ── log_returns ─────────────────────────────────────────────────


def test_log_returns_additivity():
    prices = _make_prices(100)
    lr = _log_returns(prices)
    assert np.isclose(
        lr.dropna().sum(), np.log(prices.iloc[-1] / prices.iloc[0]), atol=1e-10
    )


# ── VAM ─────────────────────────────────────────────────────────


def test_vam_positive_on_uptrend():
    prices = _make_prices(300, drift=0.001, vol=0.005)
    assert _vam(prices, mom_lb=252, skip=21, vol_w=252) > 0.5


def test_vam_negative_on_downtrend():
    prices = _make_prices(300, drift=-0.001, vol=0.01)
    assert _vam(prices, mom_lb=252, skip=21, vol_w=252) < 0.0


# ── Anchor selection ────────────────────────────────────────────


def test_select_anchors_count():
    symbols = ["A1", "A2", "B1", "B2", "C1", "C2"]
    old_map = dict(SECTOR_MAP)
    try:
        SECTOR_MAP.update({s: f"Sector{s[0]}" for s in symbols})
        closes = {
            sym: _make_prices(300, drift=0.0005 + i * 0.0001, seed=i)
            for i, sym in enumerate(symbols)
        }
        anchors = _select_anchors(
            symbols=symbols,
            closes_map=closes,
            mom_lb=252,
            skip=21,
            vol_w=252,
            n_anchors=3,
        )
        assert len(anchors) == 3
    finally:
        SECTOR_MAP.clear()
        SECTOR_MAP.update(old_map)


# ── Momentum gate ───────────────────────────────────────────────


def test_momentum_gate_passes_positive():
    prices = _make_prices(300, drift=0.001)
    assert "A" in _momentum_gate(["A"], {"A": prices}, mom_lb=252, skip=21)


def test_momentum_gate_fails_negative():
    prices = _make_prices(300, drift=-0.0005)
    assert "A" not in _momentum_gate(["A"], {"A": prices}, mom_lb=252, skip=21)


# ── Minimax ─────────────────────────────────────────────────────


def test_minimax_picks_uncorrelated():
    rng = np.random.default_rng(42)
    n = 300
    anchor = pd.Series(rng.standard_normal(n).cumsum() * 0.01 + 100.0)
    candidate_a = anchor + pd.Series(rng.standard_normal(n) * 0.001)
    candidate_b = pd.Series(rng.standard_normal(n).cumsum() * 0.005 + 50.0)
    rets = {
        "A": _log_returns(anchor).dropna(),
        "X": _log_returns(candidate_a).dropna(),
        "Y": _log_returns(candidate_b).dropna(),
    }
    corr = pd.DataFrame(rets).corr()
    divers = _minimax_diversifiers(
        anchors=["A"], candidates=["X", "Y"], corr=corr, n_slots=1
    )
    assert divers[0] == "Y"  # uncorrelated, not X


# ── SLSQP ───────────────────────────────────────────────────────


def test_slsqp_weights_sum_to_one():
    rng = np.random.default_rng(0)
    n_assets, n_days = 25, 100
    returns_df = pd.DataFrame(
        {
            f"S{i}": rng.standard_normal(n_days) * 0.01 + 0.0005 + i * 0.0002
            for i in range(n_assets)
        }
    )
    weights = _slsqp_optimise(
        symbols=list(returns_df.columns),
        returns_df=returns_df,
        risk_free_rate=0.04,
        max_weight=0.05,
    )
    assert abs(sum(weights.values()) - 1.0) < 1e-8
    for w in weights.values():
        assert 0 <= w <= 0.05 + 1e-8
