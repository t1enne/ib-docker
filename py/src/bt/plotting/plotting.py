import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict
from src.bt.types import ActionType, BacktestResult


def plot_backtest_results(
    results: BacktestResult,
    symbols: list[str],
    strategy: str,
    data: Dict[str, pd.DataFrame],
):
    """Plot backtest performance with price charts and entries/exits using Plotly"""
    num_rows = len(symbols) + 2
    subplot_titles = [f"{symbol} Price Chart" for symbol in symbols] + [
        f"Equity Curve - {strategy.upper()} Strategy",
        "Drawdown",
    ]
    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,  # Reduce vertical spacing
        horizontal_spacing=0.05,  # Reduce horizontal spacing
    )

    # Price charts for each symbol
    for i, symbol in enumerate(symbols):
        row = i + 1
        price_data = data[symbol]["Close"]
        # Add price line
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

        # Collect entry/exit points
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
                else:  # short
                    short_entries.append((trade.entry_time, trade.entry_price))
                    short_entries_text.append(f"Z: {trade.z_score:.2f}")
                    if trade.exit_time:
                        short_exits.append((trade.exit_time, trade.exit_price))
                        short_exits_text.append(
                            f"Reason: {trade.close_reason or 'unknown'}, PnL: {trade.pnl:.2f}"
                        )

        # Add markers
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
                    marker=dict(symbol="triangle-down", color="magenta", size=8),
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

    # Equity curve
    row_eq = len(symbols) + 1
    fig.add_trace(
        go.Scatter(
            x=results.equity_curve.index,
            y=results.equity_curve.values,
            mode="lines",
            name="Equity Curve",
            line=dict(color="green"),
        ),
        row=row_eq,
        col=1,
    )
    fig.update_yaxes(title_text="Portfolio Value ($)", row=row_eq, col=1)

    # Drawdown
    row_dd = len(symbols) + 2
    rolling_max = results.equity_curve.expanding().max()
    drawdown = (results.equity_curve - rolling_max) / rolling_max
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
    fig.show()
