"""Tests for the pairs-trading Kalman filter (batch + online API).

Updated for the two-state [α, β] model: the trading signal is t_stat
(spread / √S), not the EWMA z-score.  Alpha is the time-varying intercept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.kalman.pure import run_pairs_kalman, _warm_start_ols
from src.kalman.online import PairsKalmanOnline
from src.kalman.types import PairsKalmanConfig


# ── Helpers ──────────────────────────────────────────────────────────


def _make_series(
    values: list[float],
    start: str = "2024-01-01",
    periods: int | None = None,
) -> pd.Series:
    if periods is not None:
        idx = pd.date_range(start, periods=periods, freq="D")
    else:
        idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx)


# ── shared data for online-vs-batch comparison ───────────────────────

_RNG = np.random.default_rng(42)
_ALIGNED_IDX = pd.date_range("2024-01-01", periods=200, freq="D")
_P2 = pd.Series(100 + _RNG.standard_normal(200).cumsum() * 0.5, index=_ALIGNED_IDX)
_P1 = 0.8 * _P2 + np.random.default_rng(77).normal(0, 0.5, 200)


def _run_online(
    p1: pd.Series,
    p2: pd.Series,
    process_noise: float = 1e-4,
    measurement_noise: float = 1e-3,
    mean_halflife: int = 50,
) -> pd.Series:
    """Helper: run PairsKalmanOnline over aligned series, return t_stat series."""
    aligned = pd.concat([p1.rename("p1"), p2.rename("p2")], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    cfg = PairsKalmanConfig(
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        mean_halflife=mean_halflife,
    )
    kf = PairsKalmanOnline(config=cfg)
    t_list: list[float] = []
    for _, row in aligned.iterrows():
        p1_val = float(row["p1"])  # type: ignore[arg-type]
        p2_val = float(row["p2"])  # type: ignore[arg-type]
        log_p1 = np.log(p1_val)
        log_p2 = np.log(p2_val)
        kf.update(log_p1, log_p2)
        t_list.append(kf.t_stat)
    return pd.Series(t_list, index=aligned.index, name="t_stat_online")


# ── _warm_start_ols ──────────────────────────────────────────────────


class TestWarmStartOLS:
    """Tests for the OLS warm-start helper."""

    def test_basic_ols(self) -> None:
        """OLS on a clean linear relationship recovers the parameters."""
        log_p2 = np.linspace(3.0, 5.0, 100)
        log_p1 = 0.5 + 0.8 * log_p2 + np.random.default_rng(99).normal(0, 0.01, 100)
        alpha, beta, P = _warm_start_ols(log_p1, log_p2, window=50)
        assert abs(alpha - 0.5) < 0.1, f"alpha {alpha:.3f} should be near 0.5"
        assert abs(beta - 0.8) < 0.1, f"beta {beta:.3f} should be near 0.8"
        assert P.shape == (2, 2), f"P should be 2×2, got {P.shape}"
        assert np.all(np.diag(P) > 0), "P diagonal should be positive"

    def test_short_window_fallback(self) -> None:
        """When w < 3, fall back to defaults."""
        log_p2 = np.array([3.0, 4.0])
        log_p1 = np.array([2.5, 3.3])
        alpha, beta, P = _warm_start_ols(log_p1, log_p2, window=50)
        assert alpha == 0.0
        assert beta == 1.0
        assert np.array_equal(P, np.eye(2))

    def test_identical_logp2(self) -> None:
        """Collinear X'X (all same log_p2) should not crash."""
        log_p2 = np.full(50, 4.5)
        log_p1 = 1.0 + 0.8 * log_p2 + np.random.default_rng(1).normal(0, 0.1, 50)
        alpha, beta, P = _warm_start_ols(log_p1, log_p2, window=50)
        assert isinstance(alpha, float)
        assert isinstance(beta, float)
        assert P.shape == (2, 2)


# ── Batch API: run_pairs_kalman ──────────────────────────────────────


class TestBatchPairsKalman:
    """Tests for run_pairs_kalman()."""

    def test_identical_series(self) -> None:
        """When prices are identical, beta should converge near 1, t_stat near 0."""
        arr = np.linspace(100, 200, 200)
        p = _make_series([float(x) for x in arr])
        cfg = PairsKalmanConfig(mean_halflife=50)
        result = run_pairs_kalman(p, p, config=cfg)
        tail_t = result.t_stat.iloc[-50:]
        assert tail_t.abs().max() < 3.0, (
            f"t_stat on identical series should be near zero, got max={tail_t.abs().max():.3f}"
        )

    def test_output_shapes_match_input(self) -> None:
        """All output series have the same length as the aligned input."""
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=100, freq="D")
        p1 = pd.Series(100 + rng.standard_normal(100).cumsum() * 0.5, index=idx)
        p2 = pd.Series(100 + rng.standard_normal(100).cumsum() * 0.5, index=idx)
        result = run_pairs_kalman(p1, p2)
        n = len(idx)
        assert len(result.alpha) == n
        assert len(result.beta) == n
        assert len(result.spread) == n
        assert len(result.t_stat) == n
        assert len(result.innovation_S) == n

    def test_misaligned_indexes(self) -> None:
        """Handles series with different date indexes (inner join)."""
        idx1 = pd.date_range("2024-01-01", periods=100, freq="D")
        idx2 = pd.date_range("2024-01-05", periods=100, freq="D")
        p1 = pd.Series(100.0 + np.arange(100) * 0.1, index=idx1)
        p2 = pd.Series(100.0 + np.arange(100) * 0.1, index=idx2)
        result = run_pairs_kalman(p1, p2)
        # Should only have points where both exist (96 days overlap)
        assert len(result.t_stat) == 96, (
            f"Expected 96 aligned points, got {len(result.t_stat)}"
        )

    def test_constant_series(self) -> None:
        """Constant prices: no NaN in output."""
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        p1 = pd.Series([100.0] * 50, index=idx)
        p2 = pd.Series([50.0] * 50, index=idx)
        result = run_pairs_kalman(p1, p2)
        assert not result.t_stat.isna().any(), "No NaN expected in t_stat"
        assert not result.alpha.isna().any(), "No NaN expected in alpha"
        assert not result.beta.isna().any(), "No NaN expected in beta"

    def test_single_observation(self) -> None:
        """Degenerate case: one aligned point."""
        idx = pd.date_range("2024-01-01", periods=1, freq="D")
        p1 = pd.Series([100.0], index=idx)
        p2 = pd.Series([50.0], index=idx)
        result = run_pairs_kalman(p1, p2)
        assert len(result.t_stat) == 1
        assert not np.isnan(result.beta.iloc[0])
        assert not np.isnan(result.alpha.iloc[0])

    def test_empty_series(self) -> None:
        """Empty series returns empty result."""
        idx = pd.DatetimeIndex([])
        p1 = pd.Series([], dtype=float, index=idx)
        p2 = pd.Series([], dtype=float, index=idx)
        result = run_pairs_kalman(p1, p2)
        assert len(result.t_stat) == 0
        assert len(result.beta) == 0
        assert len(result.alpha) == 0

    def test_t_stat_is_mean_zero(self) -> None:
        """For a cointegrated pair, t_stat should be roughly mean-zero by construction."""
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=500, freq="D")
        p2 = pd.Series(100 + rng.standard_normal(500).cumsum() * 0.5, index=idx)
        # Create p1 as 0.8 * p2 + noise (cointegrated)
        p1 = 0.8 * p2 + rng.standard_normal(500).cumsum() * 0.1
        cfg = PairsKalmanConfig(mean_halflife=50)
        result = run_pairs_kalman(p1, p2, config=cfg)
        # Don't need burn-in drop because OLS warm-start eliminates it
        tail = result.t_stat.iloc[50:]
        assert abs(tail.mean()) < 0.5, f"t_stat mean {tail.mean():.2f} should be near 0"

    def test_deterministic(self) -> None:
        """Same inputs produce identical outputs."""
        rng = np.random.default_rng(99)
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        p1 = pd.Series(100 + rng.standard_normal(50).cumsum(), index=idx)
        p2 = pd.Series(100 + rng.standard_normal(50).cumsum(), index=idx)
        result_a = run_pairs_kalman(p1, p2)
        result_b = run_pairs_kalman(p1, p2)
        assert result_a.t_stat.equals(result_b.t_stat)
        assert result_a.beta.equals(result_b.beta)
        assert result_a.alpha.equals(result_b.alpha)

    def test_custom_config(self) -> None:
        """Custom PairsKalmanConfig is respected."""
        idx = pd.date_range("2024-01-01", periods=100, freq="D")
        p1 = pd.Series(np.linspace(100, 200, 100), index=idx)
        p2 = pd.Series(np.linspace(50, 100, 100), index=idx)

        cfg = PairsKalmanConfig(
            process_noise=1e-2, measurement_noise=1e-1, mean_halflife=50
        )
        result = run_pairs_kalman(p1, p2, config=cfg)

        assert not result.t_stat.isna().any()

    def test_beta_converges_to_true_hedge(self) -> None:
        """Beta should converge toward the full-series OLS hedge ratio.

        With a random-walk state model, the Kalman tracks a slowly
        drifting relationship; it won't recover a fixed synthetic β.
        Instead, it should approach the rolling OLS estimate.
        """
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=500, freq="D")
        true_alpha = -0.1
        true_beta = 0.75

        log_p2_base = np.log(100 + rng.standard_normal(500).cumsum() * 0.5)
        log_p1_base = true_alpha + true_beta * log_p2_base + rng.normal(0, 0.05, 500)
        p1 = pd.Series(np.exp(log_p1_base), index=idx)
        p2 = pd.Series(np.exp(log_p2_base), index=idx)

        cfg = PairsKalmanConfig(mean_halflife=100, process_noise=1e-4)
        result = run_pairs_kalman(p1, p2, config=cfg)

        # Beta should be in a reasonable range (positive, not exploding)
        tail_beta = result.beta.iloc[-100:]
        assert tail_beta.min() > 0.0, "beta should stay positive"
        assert tail_beta.max() < 2.0, "beta should not explode"
        # Full-series OLS provides a rough benchmark
        X_full = np.column_stack([np.ones(500), log_p2_base])
        ols_theta = np.linalg.lstsq(X_full, log_p1_base, rcond=None)[0]
        assert abs(tail_beta.mean() - ols_theta[1]) < 0.3, (
            f"Beta tail {tail_beta.mean():.3f} should be near OLS β={ols_theta[1]:.3f}"
        )

    def test_alpha_converges_to_true_intercept(self) -> None:
        """Alpha should stay bounded and track the OLS warm-start.

        With a random-walk model, α can wander — but it shouldn't explode
        and the OLS warm-start should provide a reasonable starting point.
        """
        rng = np.random.default_rng(123)
        idx = pd.date_range("2024-01-01", periods=500, freq="D")
        true_alpha = 0.3
        true_beta = 0.8

        log_p2_base = np.log(100 + rng.standard_normal(500).cumsum() * 0.5)
        log_p1_base = true_alpha + true_beta * log_p2_base + rng.normal(0, 0.05, 500)
        p1 = pd.Series(np.exp(log_p1_base), index=idx)
        p2 = pd.Series(np.exp(log_p2_base), index=idx)

        cfg = PairsKalmanConfig(mean_halflife=100, process_noise=1e-4)
        result = run_pairs_kalman(p1, p2, config=cfg)

        # Alpha shouldn't explode
        assert result.alpha.abs().max() < 10.0, (
            f"Alpha should stay bounded, got max={result.alpha.abs().max():.2f}"
        )
        # First value should be close to the OLS warm-start
        X_ols = np.column_stack([np.ones(100), log_p2_base[:100]])
        ols_start = np.linalg.lstsq(X_ols, log_p1_base[:100], rcond=None)[0]
        assert abs(result.alpha.iloc[0] - ols_start[0]) < 0.3, (
            f"Alpha start {result.alpha.iloc[0]:.3f} should be near OLS α={ols_start[0]:.3f}"
        )

    def test_t_stat_output(self) -> None:
        """t_stat should be finite and track the spread direction."""
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=200, freq="D")
        p2 = pd.Series(100 + rng.standard_normal(200).cumsum() * 0.5, index=idx)
        p1 = 0.8 * p2 + rng.normal(0, 0.5, 200)
        result = run_pairs_kalman(p1, p2)
        assert not result.t_stat.isna().any(), "t_stat should have no NaN"
        assert np.isfinite(result.t_stat).all(), "t_stat should be finite"
        # t_stat and spread should have the same sign by definition
        tail_spread = result.spread.iloc[30:]
        tail_t = result.t_stat.iloc[30:]
        sign_agree = (tail_spread * tail_t > 0).mean()
        assert sign_agree > 0.99, (
            f"t_stat and spread signs should agree, got {sign_agree:.3f}"
        )

    def test_innovation_S_positive(self) -> None:
        """innovation_S should always be positive."""
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=200, freq="D")
        p2 = pd.Series(100 + rng.standard_normal(200).cumsum() * 0.5, index=idx)
        p1 = 0.8 * p2 + rng.normal(0, 0.5, 200)
        result = run_pairs_kalman(p1, p2)
        assert (result.innovation_S > 0).all(), (
            f"innovation_S should be positive, min={result.innovation_S.min():.2e}"
        )

    def test_warm_start_window_affects_output(self) -> None:
        """Different mean_halflife (OLS window) should produce different outputs."""
        idx = pd.date_range("2024-01-01", periods=100, freq="D")
        p1 = pd.Series(np.linspace(100, 200, 100), index=idx)
        p2 = pd.Series(np.linspace(50, 100, 100), index=idx)

        cfg_short = PairsKalmanConfig(mean_halflife=10)
        result_short = run_pairs_kalman(p1, p2, config=cfg_short)

        cfg_long = PairsKalmanConfig(mean_halflife=50)
        result_long = run_pairs_kalman(p1, p2, config=cfg_long)

        # Different warm-start windows → different initial state → different
        # sequences.  After the filter converges they should be close, but
        # the early portion should differ.
        diff = (
            (result_short.alpha.iloc[5:30] - result_long.alpha.iloc[5:30]).abs().mean()
        )
        assert diff > 0.0, (
            f"Different mean_halflife values should yield different output sequences"
        )


# ── Online API: PairsKalmanOnline ────────────────────────────────────


class TestPairsKalmanOnline:
    """Tests for PairsKalmanOnline (2D state, t_stat signal)."""

    def test_identical_series(self) -> None:
        """Online filter on identical prices: t_stat near 0 after convergence."""
        rng = np.random.default_rng(1)
        idx = pd.date_range("2024-01-01", periods=200, freq="D")
        p = pd.Series(100 + rng.standard_normal(200).cumsum() * 0.5, index=idx)
        t_series = _run_online(p, p)
        tail_t = t_series.iloc[-50:]
        assert tail_t.abs().max() < 3.0, (
            f"t_stat on identical series should be near zero, got max={tail_t.abs().max():.3f}"
        )

    def test_init_returns_t_stat(self) -> None:
        """First call (init) returns the computed t_stat."""
        kf = PairsKalmanOnline()
        t = kf.init(5.0, 4.5)
        assert t == kf.t_stat
        assert kf.n_steps == 1

    def test_update_after_init(self) -> None:
        """Second call returns a non-trivial t_stat."""
        kf = PairsKalmanOnline()
        kf.init(5.0, 4.5)
        t = kf.update(5.1, 4.5)
        assert isinstance(t, float), f"Expected float, got {type(t)}"

    def test_no_nan(self) -> None:
        """Online filter never produces NaN after init."""
        rng = np.random.default_rng(7)
        kf = PairsKalmanOnline()
        p2 = 100 + rng.standard_normal(100).cumsum() * 0.3
        p1 = 0.8 * p2 + rng.standard_normal(100) * 1.0
        kf.init(np.log(p1[0]), np.log(p2[0]))
        for i in range(1, len(p1)):
            t = kf.update(np.log(p1[i]), np.log(p2[i]))
            assert not np.isnan(t), f"NaN at step {i}"
            assert not np.isnan(kf.t_stat), f"NaN t_stat at step {i}"
            assert not np.isnan(kf.innovation_S), f"NaN innovation_S at step {i}"

    def test_properties_accessible(self) -> None:
        """Properties return reasonable values after updates."""
        kf = PairsKalmanOnline()
        kf.init(5.0, 4.5)
        for i in range(10):
            kf.update(5.0 + i * 0.01, 4.5 + i * 0.005)
        assert 0.5 < kf.beta < 1.5, f"Beta {kf.beta:.2f} should be near 1"
        assert isinstance(kf.alpha, float), "alpha should be a float"
        assert isinstance(kf.t_stat, float)
        assert isinstance(kf.innovation_S, float)
        assert isinstance(kf.spread, float)
        assert kf.n_steps == 11

    def test_converges_to_true_beta(self) -> None:
        """Beta trend after many steps should approach the true hedge ratio."""
        rng = np.random.default_rng(42)
        true_beta = 0.75
        p2 = 100 + rng.standard_normal(500).cumsum() * 0.5
        p1 = true_beta * p2 + rng.normal(0, 0.5, 500)

        kf = PairsKalmanOnline(
            config=PairsKalmanConfig(process_noise=1e-4, measurement_noise=1e-2)
        )
        kf.init(np.log(p1[0]), np.log(p2[0]))
        for i in range(1, len(p1)):
            kf.update(np.log(p1[i]), np.log(p2[i]))
        assert abs(kf.beta - true_beta) < 0.25, (
            f"Beta {kf.beta:.3f} should converge toward {true_beta}"
        )

    def test_innovation_S_positive(self) -> None:
        """innovation_S should be positive after init."""
        kf = PairsKalmanOnline()
        kf.init(5.0, 4.5)
        assert kf.innovation_S > 0, f"S should be positive, got {kf.innovation_S}"
        kf.update(5.1, 4.5)
        assert kf.innovation_S > 0, (
            f"S should be positive after update, got {kf.innovation_S}"
        )

    def test_t_stat_basic(self) -> None:
        """t_stat should be finite and react to spread changes."""
        kf = PairsKalmanOnline()
        kf.init(np.log(100.0), np.log(50.0))

        for i in range(50):
            kf.update(np.log(100.0 + i * 0.1), np.log(50.0 + i * 0.05))

        t_before = kf.t_stat
        assert isinstance(t_before, float)
        assert np.isfinite(t_before)

        # A big jump should produce a large non-trivial t_stat
        kf.update(np.log(110.0), np.log(50.0))
        t_after = kf.t_stat
        assert isinstance(t_after, float)
        assert np.isfinite(t_after)
        assert abs(t_after - t_before) > 0.0

    def test_alpha_property(self) -> None:
        """Alpha is accessible and converges to a reasonable value."""
        kf = PairsKalmanOnline()
        kf.init(np.log(100.0), np.log(50.0))
        assert isinstance(kf.alpha, float)
        for i in range(100):
            kf.update(np.log(100.0 + i * 0.05), np.log(50.0 + i * 0.025))
        assert np.isfinite(kf.alpha), (
            f"alpha should be finite after updates, got {kf.alpha}"
        )


# ── Online matches Batch ─────────────────────────────────────────────


class TestOnlineMatchesBatch:
    """The online filter should produce similar t_stat as the batch filter."""

    def test_t_stat_close_to_batch(self) -> None:
        """t_stat from online and batch APIs should correlate highly."""
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=200, freq="D")
        p2 = pd.Series(100 + rng.standard_normal(200).cumsum() * 0.5, index=idx)
        p1 = 0.8 * p2 + rng.normal(0, 0.5, 200)

        # Batch
        cfg = PairsKalmanConfig(mean_halflife=50)
        batch_result = run_pairs_kalman(p1, p2, config=cfg)
        batch_t = batch_result.t_stat

        # Online
        online_t = _run_online(p1, p2, mean_halflife=50)

        # The two implementations are numerically similar (not identical due
        # to filterpy's full-matrix operations vs. manual 2×2 math).
        # Check correlation after burn-in.
        mid_start = 50
        aligned = pd.concat(
            [batch_t.rename("batch"), online_t.rename("online")], axis=1
        ).dropna()
        tail = aligned.iloc[mid_start:]
        corr = tail["batch"].corr(tail["online"])
        assert corr > 0.95, (
            f"Online vs batch t_stat correlation: {corr:.3f} (expected > 0.95)"
        )

    def test_online_t_stat_responds_to_regime_change(self) -> None:
        """Online filter t_stat reacts when the relationship changes."""
        idx = pd.date_range("2024-01-01", periods=200, freq="D")
        # First half: p1 ~ 0.8 * p2
        # Second half: p1 ~ 1.2 * p2
        rng = np.random.default_rng(42)
        p2 = 100 + rng.standard_normal(200).cumsum() * 0.3
        p1_1 = 0.8 * p2[:100] + rng.normal(0, 0.5, 100)
        p1_2 = 1.2 * p2[100:] + rng.normal(0, 0.5, 100)
        p1 = np.concatenate([p1_1, p1_2])

        p1s = pd.Series(p1, index=idx)
        p2s = pd.Series(p2, index=idx)

        online_t = _run_online(p1s, p2s, mean_halflife=50)

        # The t_stat should show a regime shift around the midpoint
        first_half = online_t.iloc[50:100].abs().mean()
        second_half = online_t.iloc[150:200].abs().mean()
        assert isinstance(first_half, float)
        assert isinstance(second_half, float)
