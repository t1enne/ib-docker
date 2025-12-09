import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List


def plot_equity_curve(equity_curve: List[float], title: str = "Equity Curve"):
    """Plot the equity curve using plotly."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=equity_curve,
            mode="lines",
            name="Equity",
            line=dict(color="blue", width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Equity",
        template="plotly_white",
        showlegend=False,
    )

    fig.show()


def plot_trades(trades: List[Dict], title: str = "Trades"):
    """Plot trades over time using plotly."""
    if not trades:
        print("No trades to plot")
        return

    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Create color mapping
    colors = []
    for action in df["action"]:
        if action == "BUY":
            colors.append("green")
        elif action == "SELL":
            colors.append("red")
        else:
            colors.append("blue")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["z_score"],
            mode="markers",
            marker=dict(color=colors, size=8, symbol="circle"),
            name="Trades",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Z-Score",
        template="plotly_white",
        showlegend=False,
    )

    fig.show()


def plot_walk_forward_results(wf_results: Dict):
    """Plot walk-forward analysis results."""
    # Plot equity curve
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Equity Curve", "Window Returns"),
        vertical_spacing=0.1,
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            y=wf_results["equity_curve"],
            mode="lines",
            name="Equity",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    # Window returns
    window_returns = wf_results["window_results"]
    window_nums = [r.window.window_id for r in window_returns]
    returns = [r.performance_metrics["total_return"] for r in window_returns]

    fig.add_trace(
        go.Bar(
            x=window_nums, y=returns, name="Window Returns", marker_color="lightblue"
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Walk-Forward Analysis Results", template="plotly_white", showlegend=False
    )

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Window", row=2, col=1)
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Return", row=2, col=1)

    fig.show()


def plot_results(results: Dict, is_walk_forward: bool = False):
    """Plot comprehensive backtest results."""
    if is_walk_forward:
        plot_walk_forward_results(results)
    else:
        plot_equity_curve(results["equity_curve"], "Backtest Equity Curve")
        plot_trades(results["trades"], "Trading Signals")
