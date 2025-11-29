from datetime import date
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, cast
from dataclasses import dataclass
from enum import Enum
import click
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import logging


from src.utils import (
    calculate_zscore_spread,
    get_ols_fit_model,
    get_returns,
    read_candles,
)

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    z_score: float
    symbol: str
    position: str  # "long" or "short"
    pnl: float = 0.0
    status: str = "open"  # "open", "closed", "stopped"
    close_reason: Optional[str] = None


@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    trades: List[Trade]
    equity_curve: pd.Series


@dataclass
class Strategy:
    name: str
    strategy_type: str
    symbols: list[str]
    ma_period: int
    entry_z: float
    stop_loss: float
    take_profit: float
    initial_capital: float
    position_size: float
    commission: float
    start_date: str
    end_date: str
    plot: bool


def load_strategy(path: str) -> Strategy:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        for key in ["start_date", "end_date"]:
            if key in data and isinstance(data[key], date):
                data[key] = data[key].isoformat()

    return Strategy(**data)


class StrategyType(Enum):
    PND = "pnd"
    SPREAD = "spread"


def backtest(strategy: Strategy):
    """
    Backtest a trading strategy.
    """
    try:
        # Validate inputs
        if (
            strategy.strategy_type
            in [StrategyType.PND.value, StrategyType.SPREAD.value]
            and len(strategy.symbols) != 2
        ):
            raise click.BadParameter(
                f"{strategy.strategy_type.upper()} strategy requires exactly 2 symbols"
            )

        # Load and prepare data
        train_data = load_backtest_data(
            strategy.symbols, strategy.start_date, strategy.end_date
        )

        # Generate signals based on strategy
        signals = generate_signals(
            train_data, strategy.strategy_type, strategy.ma_period
        )

        data = load_backtest_data(strategy.symbols, strategy.end_date, None)

        # Run backtest
        results = run_backtest(
            data=data,
            signals=signals,
            initial_capital=strategy.initial_capital,
            position_size=strategy.position_size,
            entry_threshold=strategy.entry_z,
            stop_loss=strategy.stop_loss,
            take_profit=strategy.take_profit,
            commission=strategy.commission,
        )

        # Display results
        display_results(results, strategy.symbols, strategy.strategy_type)

        # Plot results if requested
        if strategy.plot:
            plot_backtest_results(
                results, strategy.symbols, strategy.strategy_type, train_data
            )

    except Exception as e:
        print(e)
        click.echo(f"Error in backtest: {str(e)}", err=True)
        raise click.Abort()


def load_backtest_data(
    symbols: list[str], start_date: str | None, end_date: str | None
) -> Dict[str, pd.DataFrame]:
    """Load and prepare data for backtesting"""
    print(start_date, end_date)
    data = {}
    for symbol in symbols:
        df = get_returns(read_candles(symbol.upper(), start_date, end_date))
        df.index = pd.to_datetime(df.index)
        data[symbol] = df

    # Align all dataframes on common index
    if len(symbols) > 1:
        common_index = data[symbols[0]].index
        for symbol in symbols[1:]:
            common_index = common_index.intersection(data[symbol].index)
        for symbol in symbols:
            data[symbol] = data[symbol].loc[common_index]

    return data


def generate_signals(
    data: Dict[str, pd.DataFrame], strategy: str, ma_period: int
) -> pd.DataFrame:
    """Generate trading signals based on the selected strategy"""
    if strategy == StrategyType.PND.value:
        return generate_pnd_signals(data, ma_period)
    elif strategy == StrategyType.SPREAD.value:
        return generate_spread_signals(data, ma_period)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def generate_pnd_signals(data: Dict[str, pd.DataFrame], ma_period: int) -> pd.DataFrame:
    """Generate signals for Pairs Normalized Deviation strategy"""
    sym1, sym2 = list(data.keys())
    col = "Close"
    s1 = data[sym1][col]
    s2 = data[sym2][col]
    model = get_ols_fit_model(s1, s2)
    alpha, beta = model.params
    z_score = calculate_zscore_spread(s1, s2)

    df = pd.DataFrame(
        {
            sym1: s1,
            f"{sym2}_scaled": alpha + beta * s2,
            "z_score": z_score,
        }
    )
    df = df.dropna()  # Remove NaNs from rolling calc

    signals = pd.DataFrame(
        {
            "z_score": z_score,
            "z_1": s1,
            "z_2": s2,
            "symbol_1": sym1,
            "symbol_2": sym2,
            "close_1": data[sym1]["Close"],
            "close_2": data[sym2]["Close"],
        },
        index=s1.index,
    )

    return signals.dropna()


def generate_spread_signals(
    data: Dict[str, pd.DataFrame], ma_period: int
) -> pd.DataFrame:
    """Generate signals for Spread strategy"""
    sym1, sym2 = list(data.keys())
    col = "Close"
    s1 = cast(pd.Series, data[sym1][col])
    s2 = cast(pd.Series, data[sym2][col])
    signals = pd.DataFrame(
        {
            "Date": data[sym1].index,
            "z_score": None,
            "z_1": s1,
            "z_2": s2,
            "symbol_1": sym1,
            "symbol_2": sym2,
            "close_1": data[sym1]["Close"],
            "close_2": data[sym2]["Close"],
        },
        index=s1.index,
    )

    return signals.dropna()


def run_backtest(
    data: Dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    initial_capital: float,
    position_size: float,
    entry_threshold: float,
    stop_loss: float,
    take_profit: float,
    commission: float,
) -> BacktestResult:
    """Execute the backtest with given parameters"""

    capital = initial_capital
    trades = []
    current_trades: List[Trade] = []
    equity_curve = pd.Series(dtype=float)

    logger.info(f"Starting backtest with {len(signals)} signals")

    for i, (timestamp, row) in enumerate(signals.iterrows()):
        # Check for trade exits
        if current_trades:
            current_trades = check_exit_conditions(
                current_trades,
                row,
                cast(pd.Timestamp, timestamp),
                stop_loss,
                take_profit,
            )

            closed_trades = [t for t in current_trades if t.status == "closed"]
            for trade in closed_trades:
                # Calculate PnL and update capital
                if trade.position == "long":
                    pnl = (trade.exit_price - trade.entry_price) / trade.entry_price
                else:  # short
                    pnl = (trade.entry_price - trade.exit_price) / trade.entry_price

                # Apply commission and position sizing
                trade_value = (
                    initial_capital * position_size / 2
                )  # Split between two trades
                pnl_amount = trade_value * pnl - (trade_value * commission * 2)
                capital += pnl_amount
                trade.pnl = pnl_amount
                trades.append(trade)

            current_trades = [t for t in current_trades if t.status == "open"]

        # Check for new trade entries
        if not current_trades:
            if abs(row["z_score"]) > entry_threshold:
                new_trades = enter_trade(
                    row, timestamp, entry_threshold, position_size, initial_capital
                )
                current_trades.extend(new_trades)

        # Record equity
        equity_curve.loc[timestamp] = capital

    # Close any open trades at the end
    for trade in current_trades:
        if trade.status == "open":
            trade.exit_price = get_current_price(trade.symbol, signals.iloc[-1])
            trade.exit_time = signals.index[-1]
            trade.status = "closed"
            trades.append(trade)

    # Calculate performance metrics
    return calculate_performance_metrics(trades, equity_curve, initial_capital)


def enter_trade(
    row: pd.Series,
    timestamp: pd.Timestamp,
    entry_threshold: float,
    position_size: float,
    initial_capital: float,
) -> List[Trade]:
    """Enter two new trades based on pairs signal"""
    z_score = row["z_score"]
    sym1, sym2 = str(row["symbol_1"]), str(row["symbol_2"])
    price1, price2 = float(row["close_1"]), float(row["close_2"])

    trades = []
    if z_score < -entry_threshold:
        # Long sym1, Short sym2
        trades.append(
            Trade(
                entry_time=timestamp,
                exit_time=None,
                entry_price=price1,
                exit_price=None,
                symbol=sym1,
                position="long",
                status="open",
                z_score=float(z_score),
            )
        )
        trades.append(
            Trade(
                entry_time=timestamp,
                exit_time=None,
                entry_price=price2,
                exit_price=None,
                symbol=sym2,
                position="short",
                status="open",
                z_score=float(z_score),
            )
        )
    elif z_score > entry_threshold:
        # Short sym1, Long sym2
        trades.append(
            Trade(
                entry_time=timestamp,
                exit_time=None,
                entry_price=price1,
                exit_price=None,
                symbol=sym1,
                position="short",
                status="open",
                z_score=float(z_score),
            )
        )
        trades.append(
            Trade(
                entry_time=timestamp,
                exit_time=None,
                entry_price=price2,
                exit_price=None,
                symbol=sym2,
                position="long",
                status="open",
                z_score=float(z_score),
            )
        )
    return trades


def check_exit_conditions(
    trades: List[Trade],
    current_row: pd.Series,
    current_time: pd.Timestamp,
    stop_loss: float,
    take_profit: float,
) -> List[Trade]:
    """Check if current conditions warrant exiting the trades"""
    z = current_row["z_score"]

    def close_trade(trade: Trade):
        if trade.status == "open":
            trade.exit_price = get_current_price(trade.symbol, current_row)
            trade.exit_time = current_time
            trade.status = "closed"

    # Z-score exit condition: exit when Z crosses 0
    for trade in trades:
        if trade.status != "open":
            continue

        current_price = get_current_price(trade.symbol, current_row)
        ret = (current_price - trade.entry_price) / trade.entry_price
        if trade.position == "short":
            ret = (trade.entry_price - current_price) / trade.entry_price

        # Stop loss and take profit per trade
        if trade.position == "long":
            should_close_long = ret <= -stop_loss or ret >= take_profit or z >= 0
            if not should_close_long:
                continue
            reason = ""
            if ret <= -stop_loss:
                reason = "SL"
            elif ret >= take_profit:
                reason = "TP"
            elif z >= 0:
                reason = "Z"
            trade.close_reason = (
                f"{ret:.2f} {reason} SL:{stop_loss} TP:{take_profit} Z:{z:.2f}"
            )
            close_trade(trade)

        else:  # short
            should_close_short = ret <= -stop_loss or ret >= take_profit or z <= 0
            if not should_close_short:
                continue
            reason = ""
            if ret <= -stop_loss:
                reason = "SL"
            elif ret >= take_profit:
                reason = "TP"
            elif z <= 0:
                reason = "Z"
            else:
                raise Exception("Unknown reasons for closing trade")
            trade.close_reason = (
                f"{ret:.2f} {reason} SL:{stop_loss} TP:{take_profit} Z:{z:.2f}"
            )
            close_trade(trade)

    return trades


def get_current_price(sym: str, current_row: pd.Series) -> float:
    """Get current price based on trade type"""
    if sym == current_row["symbol_1"]:
        return float(current_row["close_1"])
    if sym == current_row["symbol_2"]:
        return float(current_row["close_2"])

    raise ValueError(f"Symbol {sym} not present in row")


def calculate_performance_metrics(
    trades: List[Trade], equity_curve: pd.Series, initial_capital: float
) -> BacktestResult:
    """Calculate performance metrics from trades and equity curve"""
    if not trades:
        return BacktestResult(0, 0, 0, 0, 0, 0, [], pd.Series())

    closed_trades = [t for t in trades if t.status == "closed"]
    profitable_trades = [t for t in closed_trades if t.pnl > 0]

    total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
    win_rate = len(profitable_trades) / len(closed_trades) if closed_trades else 0

    # Calculate Sharpe ratio (simplified)
    returns = equity_curve.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0

    # Calculate max drawdown
    equity_series = equity_curve
    rolling_max = equity_series.expanding().max()
    drawdowns = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    return BacktestResult(
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_trades=len(closed_trades),
        profitable_trades=len(profitable_trades),
        trades=closed_trades,
        equity_curve=equity_series,
    )


def display_results(results: BacktestResult, symbols: list[str], strategy: str):
    """Display backtest results in a formatted table"""
    click.echo("\n" + "=" * 50)
    click.echo(f"BACKTEST RESULTS - {strategy.upper()} Strategy")
    click.echo(f"Symbols: {', '.join(symbols)}")
    click.echo("=" * 50)

    click.echo(f"Total Return: {results.total_return:.2%}")
    click.echo(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    click.echo(f"Max Drawdown: {results.max_drawdown:.2%}")
    click.echo(f"Win Rate: {results.win_rate:.2%}")
    click.echo(f"Total Trades: {results.total_trades}")
    click.echo(f"Profitable Trades: {results.profitable_trades}")

    if not results.trades:
        return

    click.echo("\nTrades:")
    for trade in results.trades:  # Show last 5 trades
        click.echo(
            f"${trade.pnl:.2f} / {trade.entry_time}-{trade.exit_time}: {trade.position.upper()} ({trade.symbol}) at Z: {trade.z_score:.2f} "
        )


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
                if trade.position == "long":
                    long_entries.append((trade.entry_time, trade.entry_price))
                    long_entries_text.append(f"Z: {trade.z_score:.2f}")
                    if trade.exit_time:
                        long_exits.append((trade.exit_time, trade.exit_price))
                        long_exits_text.append(trade.close_reason or "")
                else:  # short
                    short_entries.append((trade.entry_time, trade.entry_price))
                    short_entries_text.append(f"Z: {trade.z_score:.2f}")
                    if trade.exit_time:
                        short_exits.append((trade.exit_time, trade.exit_price))
                        short_exits_text.append(trade.close_reason or "")

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
                    marker=dict(symbol="triangle-down", color="blue", size=8),
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
                    marker=dict(symbol="triangle-up", color="orange", size=8),
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


__all__ = ["backtest", "load_strategy", "Strategy"]
