import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Optional
from src.bt.types import ActionType, PortfolioResult


def plot_backtest_results(
    results: PortfolioResult,
    symbols: list[str],
    strategy: str,
    data: Dict[str, pd.DataFrame],
    z_scores: Optional[pd.DataFrame] = None,
    regime_df: Optional[pd.DataFrame] = None,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
):
    """Plot backtest performance with price charts, volume, z-score, HMM regime, and entries/exits using Plotly"""
    has_z_scores = z_scores is not None and not z_scores.empty
    has_regime = regime_df is not None and not regime_df.empty

    # Calculate row count: symbols (each with price+volume) + z-score + regime + equity + drawdown
    num_rows = (
        len(symbols) + 1 + (1 if has_z_scores else 0) + (1 if has_regime else 0) + 1
    )

    subplot_titles = []
    for symbol in symbols:
        subplot_titles.append(f"{symbol} Price + Volume")
    if has_z_scores:
        subplot_titles.append(f"Z-Score (Entry: ±{entry_z}, Exit: ±{exit_z})")
    if has_regime:
        subplot_titles.append("HMM Regime Probabilities")
    subplot_titles.extend(
        [
            f"Equity Curve - {strategy.upper()} Strategy",
            "Drawdown",
        ]
    )

    # Create specs for secondary y-axis on price charts
    specs = []
    for _ in symbols:
        specs.append([{"secondary_y": True}])  # Price chart with volume
    if has_z_scores:
        specs.append([None])
    if has_regime:
        specs.append([None])
    specs.append([None])  # Equity
    specs.append([None])  # Drawdown

    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04,
        horizontal_spacing=0.05,
        specs=specs,
    )

    # Price charts for each symbol with volume
    for i, symbol in enumerate(symbols):
        row = i + 1
        price_data = data[symbol]["Close"]
        volume_data = data[symbol]["Volume"]

        # Price line
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data.values,
                mode="lines",
                name=f"{symbol} Price",
                line=dict(color="orange"),
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        # Volume bars (secondary y-axis)
        fig.add_trace(
            go.Bar(
                x=volume_data.index,
                y=volume_data.values,
                name=f"{symbol} Volume",
                marker_color="rgba(100, 100, 200, 0.3)",
                showlegend=False,
            ),
            row=row,
            col=1,
            secondary_y=True,
        )

        # Scale volume axis so bars sit in bottom ~25%
        max_vol = volume_data.max()
        if max_vol > 0:
            fig.update_yaxes(
                range=[0, max_vol * 4.5],
                showticklabels=False,
                showgrid=False,
                row=row,
                col=1,
                secondary_y=True,
            )

        # Trade entry/exit markers
        long_entries = []
        long_entries_text = []
        long_exits = []
        long_exits_text = []
        short_entries = []
        short_entries_text = []
        short_exits = []
        short_exits_text = []

        for trade in results.trades:
            if trade.symbol == symbol:
                if trade.position == ActionType.long:
                    long_entries.append((trade.entry_time, trade.entry_price))
                    long_entries_text.append(f"Z: {trade.z_score:.2f}")
                    if trade.exit_time:
                        long_exits.append((trade.exit_time, trade.exit_price))
                        long_exits_text.append(
                            f"Reason: {trade.close_reason or 'unknown'}, PnL: {trade.pnl:.2f}"
                        )
                else:
                    short_entries.append((trade.entry_time, trade.entry_price))
                    short_entries_text.append(f"Z: {trade.z_score:.2f}")
                    if trade.exit_time:
                        short_exits.append((trade.exit_time, trade.exit_price))
                        short_exits_text.append(
                            f"Reason: {trade.close_reason or 'unknown'}, PnL: {trade.pnl:.2f}"
                        )

        if long_entries:
            times, prices = zip(*long_entries)
            t = pd.DatetimeIndex(times)
            fig.add_trace(
                go.Scatter(
                    x=list(t),
                    y=list(prices),
                    mode="markers",
                    name="Long Entry",
                    marker=dict(symbol="triangle-up", color="green", size=8),
                    text=long_entries_text,
                    textposition="top center",
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
        if long_exits:
            times, prices = zip(*long_exits)
            fig.add_trace(
                go.Scatter(
                    x=list(times),
                    y=list(prices),
                    mode="markers",
                    name="Long Exit",
                    marker=dict(symbol="triangle-down", color="red", size=8),
                    text=long_exits_text,
                    textposition="bottom center",
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
        if short_entries:
            times, prices = zip(*short_entries)
            t = pd.DatetimeIndex(times)
            fig.add_trace(
                go.Scatter(
                    x=list(t),
                    y=list(prices),
                    mode="markers",
                    name="Short Entry",
                    marker=dict(symbol="triangle-down", color="yellow", size=8),
                    text=short_entries_text,
                    textposition="top center",
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
        if short_exits:
            times, prices = zip(*short_exits)
            t = pd.DatetimeIndex(times)
            fig.add_trace(
                go.Scatter(
                    x=list(t),
                    y=list(prices),
                    mode="markers",
                    name="Short Exit",
                    marker=dict(symbol="triangle-up", color="blue", size=8),
                    text=short_exits_text,
                    textposition="bottom center",
                    showlegend=False,
                ),
                row=row,
                col=1,
            )

        fig.update_yaxes(title_text="Price", row=row, col=1, secondary_y=False)
        fig.update_xaxes(title_text="Date", row=row, col=1)

    # Z-Score subplot
    row_offset = len(symbols)
    if has_z_scores:
        assert z_scores
        row_z = row_offset + 1
        fig.add_trace(
            go.Scatter(
                x=z_scores.index,
                y=z_scores["z"].values,
                mode="lines",
                name="Z-Score",
                line=dict(color="blue"),
                showlegend=False,
            ),
            row=row_z,
            col=1,
        )
        # Entry threshold lines (solid)
        fig.add_hline(
            y=entry_z,
            line_dash="solid",
            line_color="green",
            row=row_z,
            col=1,
            annotation_text=f"+{entry_z} entry",
        )
        fig.add_hline(
            y=-entry_z,
            line_dash="solid",
            line_color="red",
            row=row_z,
            col=1,
            annotation_text=f"-{entry_z} entry",
        )
        # Exit threshold lines (dashed)
        fig.add_hline(
            y=exit_z,
            line_dash="dash",
            line_color="lightgreen",
            row=row_z,
            col=1,
            annotation_text=f"+{exit_z} exit",
        )
        fig.add_hline(
            y=-exit_z,
            line_dash="dash",
            line_color="lightcoral",
            row=row_z,
            col=1,
            annotation_text=f"-{exit_z} exit",
        )
        # Zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=row_z, col=1)
        fig.update_yaxes(title_text="Z-Score", row=row_z, col=1)
        row_offset = row_z

    # HMM Regime Probabilities subplot
    if has_regime:
        assert regime_df
        row_hmm = row_offset + 1

        # Regime colors and labels
        colors = {
            0: "rgba(76, 175, 80, 0.5)",
            1: "rgba(255, 193, 7, 0.5)",
            2: "rgba(244, 67, 54, 0.5)",
        }
        labels = {0: "Low Vol", 1: "Med Vol", 2: "High Vol"}

        # Stacked area chart - add traces bottom to top
        for i in [2, 1, 0]:
            col = f"prob_{i}"
            if col in regime_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=regime_df.index,
                        y=regime_df[col].values,
                        name=labels[i],
                        fill="tonexty" if i < 2 else "tozeroy",
                        line=dict(width=0.5, color=colors[i].replace("0.5", "1")),
                        fillcolor=colors[i],
                        showlegend=False,
                    ),
                    row=row_hmm,
                    col=1,
                )

        # Annotate regime transitions
        regimes = regime_df["regime"].dropna()
        prev_regime = None
        for ts, regime in regimes.items():
            if regime != prev_regime and prev_regime is not None:
                prob_col = f"prob_{int(regime)}"
                prob = (
                    regime_df.loc[ts, prob_col]
                    if prob_col in regime_df.columns
                    else None
                )
                prob_text = f" ({prob:.0%})" if prob is not None else ""
                fig.add_annotation(
                    x=ts,
                    y=1.0,
                    text=f"{labels.get(int(regime), '?')}{prob_text}",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-25,
                    font=dict(size=9),
                    row=row_hmm,
                    col=1,
                )
            prev_regime = regime

        fig.update_yaxes(title_text="Regime Prob", range=[0, 1], row=row_hmm, col=1)
        row_offset = row_hmm

    # Equity curve
    row_eq = row_offset + 1
    equity_data = results.equity_curve
    if isinstance(equity_data, dict):
        equity_series = pd.Series(equity_data)
    else:
        equity_series = equity_data
    fig.add_trace(
        go.Scatter(
            x=equity_series.index,
            y=equity_series.values,
            mode="lines",
            name="Equity Curve",
            line=dict(color="green"),
            showlegend=False,
        ),
        row=row_eq,
        col=1,
    )
    fig.update_yaxes(title_text="Portfolio Value ($)", row=row_eq, col=1)

    # Drawdown
    row_dd = row_eq + 1
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            name="Drawdown",
            line=dict(color="red"),
            fill="tozeroy",
            showlegend=False,
        ),
        row=row_dd,
        col=1,
    )
    fig.update_yaxes(title_text="Drawdown", row=row_dd, col=1)
    fig.update_xaxes(title_text="Date", row=row_dd, col=1)

    # Update layout
    fig.update_layout(
        height=300 * num_rows,
        title_text=f"Backtest Results - {strategy.upper()} Strategy",
        showlegend=False,
    )
    fig.update_xaxes(type="date")

    output_file = f"backtest_results_{strategy.lower().replace(' ', '_')}.html"
    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")
