"""Unit tests for the stable anchor remaps in scripts/rdd_lead_2004_anchorab.py.

Covers the semantic mapping (higher z_acc emission mean → regime1, lowest
variance → regime0), determinism, tie-breaks, both hmmlearn covars layouts, and
an end-to-end tiny fit with known column-0 separation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from hmmlearn import hmm

from scripts.rdd_lead_2004_anchorab import (
    _FittedHMM,
    _remap_by_emission_mean,
    _remap_by_emission_var,
)


@dataclass
class StubHMM:
    """Minimal ``_FittedHMM`` stand-in exposing means_/covars_ only."""

    means_: np.ndarray
    covars_: np.ndarray


# --------------------------------------------------------------------------- #
# _remap_by_emission_mean
# --------------------------------------------------------------------------- #


def test_emission_mean_lower_acc_is_regime0() -> None:
    # state 1 has the lower z_acc emission mean → cost-push easing → regime0.
    model = StubHMM(
        means_=np.array([[1.0, 0.0], [-1.0, 0.0]]),
        covars_=np.zeros((2, 2)),
    )
    assert _remap_by_emission_mean(model, 2) == {0: 1, 1: 0}


def test_emission_mean_permutation_preserves_semantic() -> None:
    # Swap component order: the mapping is a pure function of the means, so the
    # high-acc state must still map to regime1 regardless of raw indices.
    model = StubHMM(
        means_=np.array([[-1.0, 0.0], [1.0, 0.0]]),
        covars_=np.zeros((2, 2)),
    )
    # state 1 now holds the high acc mean → still regime1; state 0 → regime0.
    assert _remap_by_emission_mean(model, 2) == {0: 0, 1: 1}


def test_emission_mean_uses_acc_column_only() -> None:
    # Separation lives in column 1 (z_real); col 0 is equal → index tie-break.
    model = StubHMM(
        means_=np.array([[0.0, 5.0], [0.0, -5.0]]),
        covars_=np.zeros((2, 2)),
    )
    assert _remap_by_emission_mean(model, 2) == {0: 0, 1: 1}


def test_emission_mean_tie_breaks_by_index() -> None:
    model = StubHMM(
        means_=np.array([[0.5, 0.0], [0.5, 0.0]]),
        covars_=np.zeros((2, 2)),
    )
    assert _remap_by_emission_mean(model, 2) == {0: 0, 1: 1}


def test_emission_mean_deterministic() -> None:
    means = np.array([[0.3, -1.0], [-0.7, 2.0], [1.2, 0.5]])
    model = StubHMM(means_=means, covars_=np.zeros((3, 2)))
    first = _remap_by_emission_mean(model, 3)
    second = _remap_by_emission_mean(model, 3)
    assert first == second
    # state with lowest acc mean → regime0; highest → regime2
    assert first[int(np.argmin(means[:, 0]))] == 0
    assert first[int(np.argmax(means[:, 0]))] == 2


# --------------------------------------------------------------------------- #
# _remap_by_emission_var
# --------------------------------------------------------------------------- #


def test_emission_var_lowest_var_is_regime0() -> None:
    # hmmlearn 0.3.x diag covars are an (n_components, n_dim, n_dim) stack.
    covars = np.array(
        [
            [[0.5, 0.0], [0.0, 0.1]],
            [[0.2, 0.0], [0.0, 0.1]],
        ]
    )
    model = StubHMM(means_=np.zeros((2, 2)), covars_=covars)
    assert _remap_by_emission_var(model, 2) == {0: 1, 1: 0}


def test_emission_var_2d_row_profile() -> None:
    # older/dot-style layout: (n_components, n_features) row of variances.
    covars = np.array([[0.3, 0.1], [0.1, 0.1]])
    model = StubHMM(means_=np.zeros((2, 2)), covars_=covars)
    assert _remap_by_emission_var(model, 2) == {0: 1, 1: 0}


def test_emission_var_tie_breaks_by_index() -> None:
    covars = np.array(
        [
            [[0.4, 0.0], [0.0, 0.1]],
            [[0.4, 0.0], [0.0, 0.1]],
        ]
    )
    model = StubHMM(means_=np.zeros((2, 2)), covars_=covars)
    assert _remap_by_emission_var(model, 2) == {0: 0, 1: 1}


# --------------------------------------------------------------------------- #
# end-to-end tiny fit with known column-0 separation
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def _separable_model() -> hmm.GaussianHMM:
    """A real 2-state fit with clear col-0 (z_acc) separation at ±2."""
    rng = np.random.default_rng(0)
    x_high = rng.normal(2.0, 0.5, (300, 2))
    x_low = rng.normal(-2.0, 0.5, (300, 2))
    X = np.vstack([x_high, x_low])
    model = hmm.GaussianHMM(
        n_components=2,
        covariance_type="diag",
        random_state=42,
        n_iter=200,
        tol=1e-3,
    )
    model.fit(X)
    return model


def test_tiny_fit_emission_mean_semantic(_separable_model: hmm.GaussianHMM) -> None:
    remap = _remap_by_emission_mean(_separable_model, 2)
    high_state = int(np.argmax(_separable_model.means_[:, 0]))
    # the high-acc state must be regime1 (risk-off), the low-acc state regime0
    assert remap[high_state] == 1
    assert remap[1 - high_state] == 0


def test_tiny_fit_emission_mean_deterministic(
    _separable_model: hmm.GaussianHMM,
) -> None:
    assert _remap_by_emission_mean(_separable_model, 2) == _remap_by_emission_mean(
        _separable_model, 2
    )


def test_tiny_fit_emission_var_accepts_fitted_model(
    _separable_model: hmm.GaussianHMM,
) -> None:
    # must not raise on a real fitted model and returns a full 2-state bijection
    remap = _remap_by_emission_var(_separable_model, 2)
    assert set(remap.values()) == {0, 1}
    assert set(remap.keys()) == {0, 1}


def test_protocol_matches_real_model(_separable_model: hmm.GaussianHMM) -> None:
    # structural check: the real model satisfies the _FittedHMM protocol
    hmm_typed: _FittedHMM = _separable_model
    assert hmm_typed.means_.shape[0] == 2
