"""
HMM CLI module for market regime visualization and analysis.

Provides a command-line interface similar to the spread module for
visualizing market regimes detected by Hidden Markov Models.
"""

from src.hmm.types import RegimeStats

from src.bt.types import RegimeState

from pathlib import Path
from typing import Optional, cast, Union
import warnings

import numpy as np
import pandas as pd
from pandas import Timestamp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.hmm.hmm import MarketRegimeHMM
from src.utils import get_local_candles, parse_timestamp


# Regime colors for visualization
REGIME_COLORS = {
    0: "rgba(0, 200, 0, 0.3)",  # Low vol - green
    1: "rgba(255, 200, 0, 0.3)",  # Medium vol - yellow/orange
    2: "rgba(255, 50, 50, 0.3)",  # High vol - red
}

REGIME_COLORS_OPAQUE = {
    0: "rgb(0, 200, 0)",
    1: "rgb(255, 200, 0)",
    2: "rgb(255, 50, 50)",
}

REGIME_LABELS = {
    0: "Low Vol",
    1: "Medium Vol",
    2: "High Vol",
}


def _format_pct(value: float) -> str:
    """Format value as percentage with sign."""
    if np.isnan(value):
        return "N/A"
    return f"{value * 100:+.2f}%"


def _format_decimal(value: float) -> str:
    """Format value as decimal percentage."""
    if np.isnan(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def _print_statistics(
    stats: RegimeStats, symbol: str, start_date: str, end_date: str
) -> None:
    """Print regime statistics to console in a formatted table."""
    print(f"\n{'=' * 60}")
    print(f"HMM MARKET REGIME ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Number of Regimes: {stats.n_regimes}")
    print(f"\n{'─' * 60}")
    print("REGIME STATISTICS")
    print(f"{'─' * 60}")
    print(f"{'Regime':<15} {'Mean Return':<15} {'Volatility':<15} {'Frequency':<10}")
    print(f"{'─' * 60}")

    for regime in range(stats.n_regimes):
        label = REGIME_LABELS.get(regime, f"Regime {regime}")
        mean_ret = stats.mean_return.get(regime, np.nan)
        vol = stats.volatility.get(regime, np.nan)
        freq = stats.frequency.get(regime, 0.0)

        print(
            f"{label:<15} {_format_pct(mean_ret):<15} {_format_decimal(vol):<15} {_format_decimal(freq):<10}"
        )

    print(f"{'─' * 60}\n")


def _print_transition_matrix(transmat: pd.DataFrame) -> None:
    """Print transition matrix to console in a formatted table."""
    print(f"{'─' * 60}")
    print("TRANSITION MATRIX")
    print(f"{'─' * 60}")

    # Get column names for header
    col_names = [REGIME_LABELS.get(i, f"R{i}") for i in range(len(transmat.columns))]
    header = f"{'From/To':<15}" + "".join([f"{name:<12}" for name in col_names])
    print(header)
    print(f"{'─' * 60}")

    # Print each row
    for i in range(len(transmat)):
        row_name = REGIME_LABELS.get(i, f"Regime {i}")
        values = " ".join(
            [f"{transmat.iloc[i, j]:>10.1%}  " for j in range(len(transmat.columns))]
        )
        print(f"{row_name:<15} {values}")

    print(f"{'─' * 60}\n")


def _add_regime_backgrounds_to_subplot(
    fig: go.Figure,
    dates: pd.DatetimeIndex,
    regimes: pd.Series,
    xref: str = "x",
    yref: str = "y",
) -> None:
    """
    Add colored vertical spans for each regime period to a subplot.
    """
    if len(regimes) == 0:
        return

    # Skip initial NaN values (training period)
    valid_mask = ~pd.isna(regimes)
    if not valid_mask.any():
        return

    first_valid_idx = valid_mask.idxmax()
    first_valid_pos = regimes.index.get_loc(first_valid_idx)

    if isinstance(first_valid_pos, slice):
        return

    # Find contiguous regime blocks starting from first valid index
    current_regime = int(regimes.iloc[int(first_valid_pos)])
    block_start_idx = int(first_valid_pos)

    for i in range(block_start_idx + 1, len(regimes)):
        if pd.isna(regimes.iloc[i]):
            continue

        regime = int(regimes.iloc[i])

        if regime != current_regime:
            # End of block - add vrect
            block_end_idx = i - 1
            fig.add_vrect(
                x0=dates[block_start_idx],
                x1=dates[block_end_idx],
                fillcolor=REGIME_COLORS.get(current_regime, "rgba(128, 128, 128, 0.1)"),
                opacity=1.0,
                line_width=0,
                xref=xref,
                yref=yref,
                layer="below",
            )

            # Start new block
            current_regime = regime
            block_start_idx = i

    # Add final block
    if block_start_idx < len(regimes) - 1:
        fig.add_vrect(
            x0=dates[block_start_idx],
            x1=dates[-1],
            fillcolor=REGIME_COLORS.get(current_regime, "rgba(128, 128, 128, 0.1)"),
            opacity=1.0,
            line_width=0,
            xref=xref,
            yref=yref,
            layer="below",
        )


def _add_price_trace(
    fig: go.Figure,
    df: pd.DataFrame,
    symbol: str,
    row: int = 1,
    col: int = 1,
) -> None:
    """Add price line trace to the figure."""
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name=f"{symbol} Price",
            line=dict(color="black", width=1.5),
            hovertemplate=f"{symbol}: %{{y:.2f}}<br>Date: %{{x}}<extra></extra>",
        ),
        row=row,
        col=col,
    )


def _add_regime_probability_traces(
    fig: go.Figure,
    probabilities: pd.DataFrame,
    row: int = 2,
    col: int = 1,
) -> None:
    """Add stacked area traces for regime probabilities."""
    colors = [REGIME_COLORS_OPAQUE[i] for i in range(len(probabilities.columns))]

    for i, col_name in enumerate(probabilities.columns):
        regime_num = int(col_name.split("_")[-1])
        label = REGIME_LABELS.get(regime_num, f"Regime {regime_num}")
        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=probabilities.index,
                y=probabilities[col_name],
                mode="lines",
                name=label,
                line=dict(width=0),
                stackgroup="one",
                fillcolor=color,
                hovertemplate=f"{label}: %{{y:.1%}}<extra></extra>",
                showlegend=True,
            ),
            row=row,
            col=col,
        )


def _create_regime_legend_annotations(n_regimes: int) -> list:
    """Create annotation dicts for regime legend boxes."""
    annotations = []

    for i in range(n_regimes):
        label = REGIME_LABELS.get(i, f"Regime {i}")
        color = REGIME_COLORS.get(i, "rgba(128, 128, 128, 0.3)")

        annotations.append(
            dict(
                x=0.02 + i * 0.12,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"<b>{label}</b>",
                showarrow=False,
                bgcolor=color.replace("0.3", "0.7"),
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10, color="black"),
                align="center",
            )
        )

    return annotations


def _plot_hmm_analysis(
    df: pd.DataFrame,
    regimes: pd.Series,
    probabilities: pd.DataFrame,
    symbol: str,
) -> go.Figure:
    """Create combined plot with price+regimes and probabilities."""
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"{symbol} Price with Market Regimes",
            "Regime Probabilities",
        ),
        vertical_spacing=0.08,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
    )

    # Add regime backgrounds to top subplot
    _add_regime_backgrounds_to_subplot(
        fig,
        cast(pd.DatetimeIndex, df.index),
        regimes,
        xref="x",
        yref="y",
    )

    # Add price trace to top subplot
    _add_price_trace(fig, df, symbol, row=1, col=1)

    # Add probability traces to bottom subplot
    _add_regime_probability_traces(fig, probabilities, row=2, col=1)

    # Add regime legend annotations
    annotations = _create_regime_legend_annotations(len(probabilities.columns))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>HMM Market Regime Analysis - {symbol}</b>",
            x=0.5,
            xanchor="center",
        ),
        height=800,
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
        annotations=annotations,
        template="plotly_white",
    )

    # Update axes
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Probability", range=[0, 1], row=2, col=1)

    return fig


def hmm(
    symbol: str,
    start: Optional[Timestamp] = None,
    end: Optional[Timestamp] = None,
    n_regimes: int = 3,
    vol_window: int = 20,
    momentum_window: int = 10,
    min_train_size: int = 252,
    update_interval: int = 50,
    output_dir: str = "./hmm_models",
    plot: bool = True,
) -> None:
    """Analyze market regimes using Hidden Markov Model and generate visualization."""
    # Load data
    print(f"Loading data for {symbol}...")
    _start = parse_timestamp(start) if start else None
    _end = parse_timestamp(end) if end else None
    df = get_local_candles(symbol.upper(), _start, _end)

    if df.empty:
        print(f"No data found for {symbol}")
        return

    start_date = df.index[0].strftime("%Y-%m-%d")
    end_date = df.index[-1].strftime("%Y-%m-%d")

    print(f"Loaded {len(df)} observations from {start_date} to {end_date}")

    # Check if we have enough data
    if len(df) <= min_train_size:
        print(
            f"Warning: Only {len(df)} observations. Using {len(df) // 2} as min_train_size."
        )
        min_train_size = max(len(df) // 2, 50)

    # Initialize and fit HMM
    print(f"\nFitting HMM with {n_regimes} regimes...")
    model = MarketRegimeHMM(
        n_regimes=n_regimes,
        vol_window=vol_window,
        momentum_window=momentum_window,
        min_train_size=min_train_size,
        update_interval=update_interval,
    )

    try:
        model.fit(df["Close"])
    except Exception as e:
        print(f"Error fitting model: {e}")
        return

    # Predict regimes and probabilities
    print("Predicting regimes...")
    regimes = model.predict(df["Close"])
    probabilities = model.predict_proba(df["Close"])

    # Calculate statistics
    print("Calculating regime statistics...")
    stats = model.get_regime_statistics(df["Close"])
    transmat = model.get_transition_matrix()

    # Print statistics to console
    _print_statistics(stats, symbol, start_date, end_date)
    _print_transition_matrix(transmat)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_filename = f"hmm_{symbol}_{n_regimes}regimes_{end_date}.pkl"
    model_path = output_path / model_filename
    model.save(str(model_path))
    print(f"Model saved to: {model_path}")

    # Generate plots if requested
    if plot:
        print("\nGenerating plot...")
        fig = _plot_hmm_analysis(df, regimes, probabilities, symbol)

        plot_filename = f"hmm_{symbol}_{start_date}_{end_date}.html"
        plot_path = output_path / plot_filename
        fig.write_html(str(plot_path))
        print(f"Plot saved to: {plot_path}")

    print(f"\n{'=' * 60}")
    print("HMM Analysis Complete")
    print(f"{'=' * 60}\n")


def get_regime_df(rstate: RegimeState) -> Union[pd.DataFrame, None]:
    probs = rstate.probs
    labels = rstate.labels
    timestamps = rstate.timestamps
    first_prob = next((p for p in probs if p is not None), None)
    if not first_prob:
        return None

    n_regimes = len(first_prob)
    regime_data = {"regime": labels}
    for i in range(n_regimes):
        regime_data[f"prob_{i}"] = [p[i] if p is not None else None for p in probs]

    return pd.DataFrame(regime_data, index=pd.Index(timestamps))
