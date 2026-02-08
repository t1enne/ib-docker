import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict
from src.bt.types import ActionType, PortfolioResult


def plot_backtest_results(
    results: PortfolioResult,
    symbols: list[str],
    strategy: str,
    data: Dict[str, pd.DataFrame],
    z_scores: pd.DataFrame,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
):
    """Plot backtest performance with price charts, z-score, and entries/exits using Plotly"""
    num_rows = len(symbols) + 3  # +1 for z-score, +1 for equity, +1 for drawdown
    has_z_scores = z_scores is not None and not z_scores.empty

    subplot_titles = [f"{symbol} Price Chart" for symbol in symbols]
    if has_z_scores:
        subplot_titles.append(f"Z-Score (Entry: ±{entry_z}, Exit: ±{exit_z})")
    subplot_titles.extend(
        [
            f"Equity Curve - {strategy.upper()} Strategy",
            "Drawdown",
        ]
    )

    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04,
        horizontal_spacing=0.05,
    )

    # Price charts for each symbol
    for i, symbol in enumerate(symbols):
        row = i + 1
        price_data = data[symbol]["Close"]
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data.values,
                mode="lines",
                name=f"{symbol} Price",
                line=dict(color="orange"),
            ),
            row=row,
            col=1,
        )

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
                ),
                row=row,
                col=1,
            )

        fig.update_yaxes(title_text="Price", row=row, col=1)
        fig.update_xaxes(title_text="Date", row=row, col=1)

    # Z-Score subplot
    if has_z_scores:
        row_z = len(symbols) + 1
        fig.add_trace(
            go.Scatter(
                x=z_scores.index,
                y=z_scores["z"].values,
                mode="lines",
                name="Z-Score",
                line=dict(color="blue"),
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

    # Equity curve
    row_eq = len(symbols) + 2 if has_z_scores else len(symbols) + 1
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
        ),
        row=row_eq,
        col=1,
    )
    fig.update_yaxes(title_text="Portfolio Value ($)", row=row_eq, col=1)

    # Drawdown
    row_dd = len(symbols) + 3 if has_z_scores else len(symbols) + 2
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
