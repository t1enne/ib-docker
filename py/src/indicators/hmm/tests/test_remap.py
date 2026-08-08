"""Tests for vol-ranked HMM component remap (regime label stability).

Regression for the gauge-instability bug: sorting raw HMM states by the
empirical mean of a predict() assignment was nondeterministic across refits,
so `current_vol`/`current_regime` could jump spuriously. The remap is now
anchored to the fitted emission covariance, which is deterministic per fit
and canonicalises the component-permutation gauge.
"""

from __future__ import annotations

import io
import sys
import warnings

import numpy as np

from src.indicators.hmm.hmm import rank_states_by_vol
from src.indicators.hmm.online import MarketRegimeHMMOnline


def _fit_noisy(model):
    old = sys.stderr
    sys.stderr = io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(_feature_matrix())
    finally:
        sys.stderr = old
    return model


def _feature_matrix(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(size=n)
    # Scale-misaligned features, as in production: momentum dominates.
    vol = np.abs(np.convolve(returns, np.ones(10) / 10, mode="same"))
    momentum = np.convolve(returns, np.ones(5) / 5, mode="same") * 252
    return np.column_stack([returns, vol, momentum])


def test_remap_is_bijective():
    from hmmlearn import hmm

    m = _fit_noisy(hmm.GaussianHMM(n_components=3, random_state=42))
    state_to_regime, regime_to_state = rank_states_by_vol(m, 3)
    assert set(state_to_regime) == {0, 1, 2}
    assert set(state_to_regime.values()) == {0, 1, 2}
    # Inverse is bijectively consistent.
    for raw, ranked in state_to_regime.items():
        assert regime_to_state[ranked] == raw


def test_remap_ranks_by_fitted_return_variance():
    from hmmlearn import hmm

    m = _fit_noisy(hmm.GaussianHMM(n_components=3, random_state=42))
    cov = np.asarray(m.covars_)
    state_to_regime, _ = rank_states_by_vol(m, 3)
    # Return-var must be monotone increasing in the ranked order.
    var_in_rank_order = [cov[state_to_regime[r], 0, 0] for r in range(3)]
    assert all(var_in_rank_order[i] <= var_in_rank_order[i + 1] for i in range(2))


def _deterministic(seed: int, n_comp: int = 3):
    from hmmlearn import hmm

    m1 = _fit_noisy(hmm.GaussianHMM(n_components=n_comp, random_state=seed))
    s2r1, _ = rank_states_by_vol(m1, n_comp)
    vals1 = tuple(s2r1[i] for i in sorted(s2r1))
    return vals1


def test_remap_deterministic_for_same_fit():
    """Same fit -> identical remap (no coin-flip between predict runs)."""
    assert _deterministic(42) == _deterministic(42)


def test_online_remap_does_not_flag_calm_as_high_vol():
    """With a deterministic vol-ranked remap, calm markets must not be
    persistently rolled into the untradeable high-vol regime (rank 2).

    The old empirical predict-mean sort let the label permutation shuffle the
    vol ranking between refits, so ``current_vol`` could flip to the
    untradeable regime during calm markets without a real vol change (measured
    up to ~25-30% of the calm window pre-fix). Anchoring the rank to the fitted
    emission variance removes that gauge flip.

    The sliding-window HMM still re-anchors to the current window's vol level
    (a genuine model recency limit, not a bug), so we assert the *gross* case:
    calm data is almost never flagged as high-vol.
    """
    rng = np.random.default_rng(0)
    returns = np.concatenate([rng.normal(0, 0.004, 700), rng.normal(0, 0.03, 400)])
    prices = np.exp(np.cumsum(returns))

    hmm = MarketRegimeHMMOnline(
        n_regimes=3,
        window_size=500,
        vol_window=15,
        momentum_window=8,
        retrain_interval=50,
        random_state=42,
    )
    seq = np.array([hmm.update(float(p)) for p in prices])

    calm = seq[:700]
    calm = calm[calm >= 0]  # drop pre-fit sentinel
    assert len(calm) > 0
    # Before the deterministic remap, calm bled into the untradeable high-vol
    # regime ~25-30% of the time. After it, essentially never on clean data.
    assert (calm == 2).mean() < 0.10
