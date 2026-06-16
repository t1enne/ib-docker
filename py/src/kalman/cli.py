"""
Kalman filter CLI — load data → run filter → print stats → save plot.

Moved out of __init__.py so that __init__.py can stay a pure export hub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, cast

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.kalman.pure import run_filter, compute_stats
from src.kalman.types import KalmanConfig, KalmanStats, FilterResult
from src.utils import get_local_candles, parse_timestamp


# ---------------------------------------------------------------------------
# console output
# ---------------------------------------------------------------------------


def _print_statistics(
    stats: KalmanStats,
    symbol: str,
    start_date: str,
    end_date: str,
    config: KalmanConfig,
) -> None:
    print(f"\n{'=' * 60}")
    print("KALMAN FILTER ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Symbol:             {symbol}")
    print(f"Period:             {start_date} to {end_date}")
    print(f"Process noise (Q):  {config.process_noise:.2e}")
    print(f"Measurement noise:  {config.measurement_noise:.2e}")
    print(f"Adaptive R:         {'yes' if config.adaptive else 'no'}")
    print(f"\n{'─' * 60}")
    print("STATISTICS")
    print(f"{'─' * 60}")
    print(f"  Observations:      {stats.n_observations}")
    print(f"  RMSE:              {stats.rmse:.4f}")
    print(f"  MAE:               {stats.mae:.4f}")
    print(f"  95% CI coverage:   {stats.coverage_95:.1%}")
    print(f"  Avg Kalman gain:   {stats.avg_kalman_gain:.4f}")
    print(f"{'─' * 60}\n")


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


def _build_figure(
    prices: pd.Series,
    result: FilterResult,
    symbol: str,
    config: KalmanConfig,
) -> go.Figure:  # type: ignore[name-defined]
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            f"{symbol} Price — Filtered Estimate + 95% CI",
            "Residuals (actual − filtered)",
            "Kalman Gain",
            "Estimated Velocity (trend)",
        ),
        row_heights=[0.4, 0.2, 0.2, 0.2],
    )

    # Row 1 — price + filtered + CI
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            mode="lines",
            name="Close",
            line=dict(color="rgba(0,0,0,0.35)", width=1),
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=result.filtered.index,
            y=result.filtered.values,
            mode="lines",
            name="Filtered",
            line=dict(color="royalblue", width=2),
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=result.predicted.index,
            y=result.predicted.values,
            mode="lines",
            name="Predicted",
            line=dict(color="orange", width=1, dash="dash"),
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    # CI band
    fig.add_trace(
        go.Scatter(
            x=result.upper_ci.index,
            y=result.upper_ci.values,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="upper_ci",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=result.lower_ci.index,
            y=result.lower_ci.values,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(65,105,225,0.15)",
            line=dict(width=0),
            showlegend=False,
            name="lower_ci",
        ),
        row=1,
        col=1,
    )

    # Row 2 — residuals
    fig.add_trace(
        go.Scatter(
            x=result.residuals.index,
            y=result.residuals.values,
            mode="lines",
            name="Residual",
            line=dict(color="crimson", width=1),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    # ±2σ reference
    r_std = result.residuals.std()
    hline_row: int = 2
    hline_col: int = 1
    fig.add_hline(
        y=2 * r_std,
        line_dash="dash",
        line_color="gray",
        annotation_text="+2σ",
        row=hline_row,  # type: ignore[arg-type]
        col=hline_col,  # type: ignore[arg-type]
    )
    fig.add_hline(
        y=-2 * r_std,
        line_dash="dash",
        line_color="gray",
        annotation_text="−2σ",
        row=hline_row,  # type: ignore[arg-type]
        col=hline_col,  # type: ignore[arg-type]
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=hline_row, col=hline_col)  # type: ignore[arg-type]

    # Row 3 — Kalman gain
    fig.add_trace(
        go.Scatter(
            x=result.kalman_gains.index,
            y=result.kalman_gains.values,
            mode="lines",
            name="Kalman Gain",
            line=dict(color="seagreen", width=1.5),
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    # Row 4 — velocity
    fig.add_trace(
        go.Scatter(
            x=result.velocity.index,
            y=result.velocity.values,
            mode="lines",
            name="Velocity",
            line=dict(color="darkorange", width=1.5),
            showlegend=False,
        ),
        row=4,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=4, col=1)  # type: ignore[arg-type]

    # Layout
    mode_label = "adaptive" if config.adaptive else "static"
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Kalman Filter — {symbol}</b><br>"
                f"<sup>Q={config.process_noise:.1e}  "
                f"R={config.measurement_noise:.1e}  "
                f"({mode_label})</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=900,
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.06,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    fig.update_yaxes(title_text="Gain", row=3, col=1)
    fig.update_yaxes(title_text="Velocity", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)

    return fig


# ---------------------------------------------------------------------------
# public entry point (called from main.py click command)
# ---------------------------------------------------------------------------


def kalman(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    process_noise: float = 1e-5,
    measurement_noise: float = 1e-3,
    adaptive: bool = False,
    vol_window: int = 20,
    plot: bool = True,
) -> None:
    """Run Kalman filter on *symbol* and optionally generate a plot."""

    # Load data
    print(f"Loading data for {symbol}…")
    _start = parse_timestamp(start) if start else None
    _end = parse_timestamp(end) if end else None
    df = get_local_candles(symbol.upper(), _start, _end)

    if df.empty:
        print(f"No data found for {symbol}")
        return

    idx = cast(pd.DatetimeIndex, df.index)
    start_date = str(idx[0])[:10]  # "YYYY-MM-DD"
    end_date = str(idx[-1])[:10]  # "YYYY-MM-DD"
    print(f"Loaded {len(df)} observations from {start_date} to {end_date}")

    prices = cast(pd.Series, df["close"])

    # Build config & run
    config = KalmanConfig(
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        adaptive=adaptive,
        vol_window=vol_window,
    )

    print("Running Kalman filter…")
    result = run_filter(prices, config)
    stats = compute_stats(prices, result)

    # Console output
    _print_statistics(stats, symbol, start_date, end_date, config)

    # Persist
    out = Path("./plots")
    out.mkdir(parents=True, exist_ok=True)

    # Plot
    if plot:
        print("Generating plot…")
        fig = _build_figure(prices, result, symbol, config)
        plot_path = out / f"kalman_{symbol}_{start_date}_{end_date}.html"
        fig.write_html(str(plot_path))
        print(f"Plot saved to: {plot_path}")

    print(f"\n{'=' * 60}")
    print("Kalman Analysis Complete")
    print(f"{'=' * 60}\n")
