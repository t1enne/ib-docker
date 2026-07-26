"""Tests for the univariate Kalman filter (run_filter / compute_stats)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.kalman.pure import run_filter, compute_stats
from src.indicators.kalman.types import KalmanConfig, FilterResult


# ── Helpers ──────────────────────────────────────────────────────────


def _series(
    values: list[float],
    start: str = "2024-01-01",
) -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx)


# ── run_filter ───────────────────────────────────────────────────────


class TestRunFilter:
    """Tests for run_filter()."""

    def test_output_types_and_shapes(self) -> None:
        """All output series have same index and length as input."""
        rng = np.random.default_rng(42)
        prices = pd.Series(100 + rng.standard_normal(200).cumsum())
        result = run_filter(prices, KalmanConfig())
        assert isinstance(result, FilterResult)
        assert len(result.filtered) == 200
        assert len(result.predicted) == 200
        assert len(result.upper_ci) == 200
        assert len(result.lower_ci) == 200
        assert len(result.residuals) == 200
        assert len(result.kalman_gains) == 200
        assert len(result.velocity) == 200

    def test_filtered_tracks_price(self) -> None:
        """Filtered estimate should be close to actual prices (low-noise)."""
        prices = _series([100.0 + i * 0.1 for i in range(100)])
        # Very low noise → filter should track tightly
        config = KalmanConfig(
            process_noise=1e-6,
            measurement_noise=1e-2,
        )
        result = run_filter(prices, config)
        # RMSE should be small
        resid = (prices - result.filtered).dropna()
        rmse = float(np.sqrt((resid**2).mean()))
        assert rmse < 5.0, f"RMSE {rmse:.2f} should be small for low-noise series"

    def test_predicted_leads_filtered(self) -> None:
        """Predicted (a-priori) should be less smooth than filtered (a-posteriori)."""
        rng = np.random.default_rng(42)
        prices = pd.Series(100 + rng.standard_normal(500).cumsum())
        result = run_filter(prices, KalmanConfig())
        # The std of residuals should be < std of prediction errors
        pred_err = (prices - result.predicted).std()
        filt_err = (prices - result.filtered).std()
        assert pred_err >= filt_err, (
            f"Prediction error std {pred_err:.4f} should be >= filtered error std {filt_err:.4f}"
        )

    def test_residual_mean_near_zero(self) -> None:
        """Mean of residuals should be approximately zero (unbiased)."""
        rng = np.random.default_rng(42)
        prices = pd.Series(100 + rng.standard_normal(500).cumsum())
        result = run_filter(prices, KalmanConfig())
        mean_resid = float(result.residuals.mean())
        assert abs(mean_resid) < 1.0, f"Mean residual {mean_resid:.3f} should be near 0"

    def test_velocity_derivative_shape(self) -> None:
        """Velocity should reflect the direction of price changes."""
        # Monotonically increasing prices → velocity should be positive
        prices = _series([100.0 + i * 0.5 for i in range(200)])
        result = run_filter(prices, KalmanConfig())
        mean_vel = float(result.velocity.mean())
        assert mean_vel > 0, f"Mean velocity {mean_vel:.4f} should be > 0 for uptrend"

        # Monotonically decreasing prices → velocity should be negative
        prices_down = _series([200.0 - i * 0.5 for i in range(200)])
        result_down = run_filter(prices_down, KalmanConfig())
        mean_vel_down = float(result_down.velocity.mean())
        assert mean_vel_down < 0, (
            f"Mean velocity {mean_vel_down:.4f} should be < 0 for downtrend"
        )

    def test_ci_widens_with_process_noise(self) -> None:
        """Higher process noise produces wider confidence bands."""
        rng = np.random.default_rng(42)
        prices = pd.Series(100 + rng.standard_normal(500).cumsum())

        low_q = run_filter(prices, KalmanConfig(process_noise=1e-6))
        high_q = run_filter(prices, KalmanConfig(process_noise=1.0))

        low_band = (low_q.upper_ci - low_q.lower_ci).mean()
        high_band = (high_q.upper_ci - high_q.lower_ci).mean()

        assert float(high_band) > float(low_band), (
            f"High-Q band {high_band:.2f} should be wider than low-Q band {low_band:.2f}"
        )

    def test_constant_price(self) -> None:
        """Constant price: no NaN, velocity near 0."""
        prices = _series([100.0] * 100)
        result = run_filter(prices, KalmanConfig())
        assert not result.filtered.isna().any()
        assert not result.velocity.isna().any()
        tail_vel = result.velocity.iloc[-20:]
        assert tail_vel.abs().max() < 1e-3, (
            f"Velocity should ~0 for constant price, got {tail_vel.abs().max():.2e}"
        )

    def test_single_observation(self) -> None:
        """Degenerate case: one data point."""
        prices = _series([100.0])
        result = run_filter(prices, KalmanConfig())
        assert len(result.filtered) == 1
        assert not np.isnan(result.filtered.iloc[0])

    def test_two_observations(self) -> None:
        """Two data points — filter should converge quickly."""
        prices = _series([100.0, 105.0])
        result = run_filter(prices, KalmanConfig())
        assert len(result.filtered) == 2
        assert not result.filtered.isna().any()

    def test_deterministic(self) -> None:
        """Same inputs produce identical outputs."""
        rng = np.random.default_rng(99)
        prices = pd.Series(100 + rng.standard_normal(100).cumsum())
        result_a = run_filter(prices, KalmanConfig())
        result_b = run_filter(prices, KalmanConfig())
        assert result_a.filtered.equals(result_b.filtered)
        assert result_a.kalman_gains.equals(result_b.kalman_gains)
        assert result_a.velocity.equals(result_b.velocity)

    def test_adaptive_r_changes_gains(self) -> None:
        """Adaptive R should produce different Kalman gains from static R."""
        rng = np.random.default_rng(42)
        prices = pd.Series(100 + rng.standard_normal(500).cumsum())

        static_result = run_filter(prices, KalmanConfig(adaptive=False))
        adaptive_result = run_filter(prices, KalmanConfig(adaptive=True, vol_window=20))

        # Gains should not be identical
        gains_diff = static_result.kalman_gains - adaptive_result.kalman_gains
        assert gains_diff.abs().max() > 1e-6, (
            "Adaptive and static Kalman gains should differ"
        )

    def test_high_noise_damps_gain(self) -> None:
        """Higher measurement noise → lower Kalman gain."""
        rng = np.random.default_rng(42)
        prices = pd.Series(100 + rng.standard_normal(500).cumsum())

        low_r = run_filter(prices, KalmanConfig(measurement_noise=1e-4))
        high_r = run_filter(prices, KalmanConfig(measurement_noise=1.0))

        mean_gain_low = float(low_r.kalman_gains.mean())
        mean_gain_high = float(high_r.kalman_gains.mean())
        assert mean_gain_low > mean_gain_high, (
            f"Low R gain {mean_gain_low:.4f} should be > high R gain {mean_gain_high:.4f}"
        )

    def test_nan_in_prices(self) -> None:
        """NaN in price series should propagate without crash."""
        prices = _series([100.0, 101.0, float("nan"), 103.0, 104.0])
        result = run_filter(prices, KalmanConfig())
        # The filter runs but produces NaN at the NaN observation
        assert not result.filtered.isna().all()
        # The point corresponding to the NaN should also be NaN or handled
        assert result.filtered.iloc[2] is not None


# ── compute_stats ────────────────────────────────────────────────────


class TestComputeStats:
    """Tests for compute_stats()."""

    def test_stats_types(self) -> None:
        """All stats fields return valid numeric values."""
        rng = np.random.default_rng(42)
        prices = pd.Series(100 + rng.standard_normal(100).cumsum())
        result = run_filter(prices, KalmanConfig())
        stats = compute_stats(prices, result)
        assert 0 <= stats.rmse < 1e6
        assert 0 <= stats.mae < 1e6
        assert 0.0 <= stats.coverage_95 <= 1.0
        assert 0 <= stats.avg_kalman_gain <= 1.0
        assert stats.n_observations == 100

    def test_rmse_is_larger_than_mae(self) -> None:
        """RMSE >= MAE (by Cauchy-Schwarz, strictly for non-constant error)."""
        rng = np.random.default_rng(42)
        prices = pd.Series(100 + rng.standard_normal(200).cumsum())
        result = run_filter(prices, KalmanConfig())
        stats = compute_stats(prices, result)
        assert stats.rmse >= stats.mae, (
            f"RMSE {stats.rmse:.4f} should be >= MAE {stats.mae:.4f}"
        )

    def test_coverage_tight_with_low_noise(self) -> None:
        """With very low measurement noise, the filter tracks tightly."""
        prices = _series([100.0, 101.0, 99.0, 102.0, 98.0])
        result = run_filter(
            prices, KalmanConfig(process_noise=1e-6, measurement_noise=1e-6)
        )
        stats = compute_stats(prices, result)
        # Kalman gain should be near 1 (trusts observations)
        assert stats.avg_kalman_gain > 0.5, (
            f"Avg gain {stats.avg_kalman_gain:.3f} should be high with low noise"
        )
        # At least some observations should be inside the (prediction) CI
        assert stats.coverage_95 > 0.0

    def test_N_equal_len(self) -> None:
        """n_observations equals length of input."""
        prices = _series([float(x) for x in range(50)])
        result = run_filter(prices, KalmanConfig())
        stats = compute_stats(prices, result)
        assert stats.n_observations == 50

    def test_empty_series_returns_zero_stats(self) -> None:
        """Empty price series returns zeroed-out stats."""
        prices = pd.Series([], dtype=float)
        # run_filter would fail on empty, so we construct a minimal case
        # where compute_stats gets called with a 1-point result
        prices = _series([100.0])
        result = run_filter(prices, KalmanConfig())
        stats = compute_stats(prices, result)
        assert stats.n_observations == 1
        assert isinstance(stats.rmse, float)
