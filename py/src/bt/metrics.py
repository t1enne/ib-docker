from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, cast
from scipy import stats
from io import StringIO
import sys

from src.bt.types import PortfolioResult, ActionType


@dataclass
class PerformanceMetrics:
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    stability: float
    max_drawdown: float
    omega_ratio: float
    skewness: float
    kurtosis: float
    alpha: float
    beta: float


@dataclass
class DrawdownPeriod:
    peak_date: Optional[pd.Timestamp]
    valley_date: pd.Timestamp
    recovery_date: Optional[pd.Timestamp]
    duration: int
    net_drawdown_pct: float


def _get_returns(equity_curve: pd.Series) -> pd.Series:
    returns = equity_curve.pct_change().dropna()
    return returns


def _equity_curve_time_info(equity_curve: pd.Series) -> tuple[float, float]:
    if len(equity_curve) < 2:
        return 0.0, 0.0

    index = equity_curve.index
    if isinstance(index, pd.DatetimeIndex):
        start = index[0]
        end = index[-1]
        try:
            elapsed_seconds = float((end - start).total_seconds())
        except AttributeError:
            elapsed_seconds = 0.0

        if elapsed_seconds > 0:
            elapsed_years = elapsed_seconds / (365.25 * 24 * 60 * 60)
            periods = len(equity_curve) - 1
            periods_per_year = periods / elapsed_years if elapsed_years > 0 else 0.0
            return elapsed_years, periods_per_year

    periods = len(equity_curve) - 1
    if periods <= 0:
        return 0.0, 0.0
    elapsed_years = periods / 252.0
    return elapsed_years, 252.0


def periods_per_year(equity_curve: pd.Series) -> float:
    _, ppy = _equity_curve_time_info(equity_curve)
    return ppy


def annual_return(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    years, _ = _equity_curve_time_info(equity_curve)
    if years <= 0:
        return 0.0
    cagr = ((1 + total_return) ** (1 / years)) - 1
    return cagr


def annual_volatility(equity_curve: pd.Series) -> float:
    returns = _get_returns(equity_curve)
    if len(returns) == 0:
        return 0.0
    ppy = periods_per_year(equity_curve)
    if ppy <= 0:
        return 0.0
    return float(returns.std() * np.sqrt(ppy))


def max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min())


def drawdown_periods(
    equity_curve: pd.Series, min_drawdown: float = 0.05
) -> List[DrawdownPeriod]:
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max

    periods: List[DrawdownPeriod] = []
    in_drawdown = False
    peak_idx: Optional[pd.Timestamp] = None

    dates = list(drawdown.index)
    values = drawdown.values

    for i, (date, value) in enumerate(zip(dates, values)):
        if not in_drawdown and value < -min_drawdown:
            in_drawdown = True
            peak_idx = dates[0] if i == 0 else dates[i - 1]

        elif in_drawdown and value >= 0:
            in_drawdown = False
            if peak_idx is not None:
                valley_idx = i - 1
                valley_date = dates[valley_idx]
                recovery_date: Optional[pd.Timestamp] = date
                net_dd_pct = abs(values[valley_idx]) * 100

                try:
                    duration = (date - peak_idx).days
                except TypeError, AttributeError:
                    duration = i - dates.index(peak_idx) if peak_idx in dates else 0

                periods.append(
                    DrawdownPeriod(
                        peak_date=peak_idx,
                        valley_date=valley_date,
                        recovery_date=recovery_date,
                        duration=duration,
                        net_drawdown_pct=net_dd_pct,
                    )
                )
            peak_idx = None

    if in_drawdown and peak_idx is not None:
        valley_idx = np.argmin(values)
        valley_date = dates[valley_idx]
        net_dd_pct = abs(values[valley_idx]) * 100

        try:
            duration = (dates[-1] - peak_idx).days
        except TypeError, AttributeError:
            duration = len(dates) - dates.index(peak_idx) if peak_idx in dates else 0

        periods.append(
            DrawdownPeriod(
                peak_date=peak_idx,
                valley_date=valley_date,
                recovery_date=None,
                duration=duration,
                net_drawdown_pct=net_dd_pct,
            )
        )

    return sorted(periods, key=lambda x: x.net_drawdown_pct, reverse=True)


def sortino_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    returns = _get_returns(equity_curve)
    if len(returns) == 0:
        return 0.0
    downside_returns = returns[returns < 0]
    downside_std = float(downside_returns.std()) if len(downside_returns) > 0 else 0.0
    if downside_std == 0:
        return 0.0
    ret = annual_return(equity_curve)
    ppy = periods_per_year(equity_curve)
    if ppy <= 0:
        return 0.0
    downside_std_annual = downside_std * np.sqrt(ppy)
    return (ret - risk_free_rate) / downside_std_annual
    # return (ret - risk_free_rate) / downside_std


def omega_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    returns = _get_returns(equity_curve)
    if len(returns) == 0:
        return 0.0
    ppy = periods_per_year(equity_curve)
    per_period_rf = risk_free_rate / ppy if ppy > 0 else 0.0
    gain = float(returns[returns > per_period_rf].sum())
    loss = float(-returns[returns < per_period_rf].sum())
    if loss == 0:
        return 1.0
    return gain / loss


def skewness(equity_curve: pd.Series) -> float:
    returns = _get_returns(equity_curve)
    if len(returns) < 3:
        return 0.0
    return float(returns.skew())


def kurtosis(equity_curve: pd.Series) -> float:
    returns = _get_returns(equity_curve)
    if len(returns) < 4:
        return 0.0
    return float(returns.kurt())


def stability(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    x = np.arange(len(equity_curve))
    y = equity_curve.values
    result = stats.linregress(x, y)
    r_value: Any = getattr(result, "rvalue", None)
    if r_value is None:
        r_value = result[2] if isinstance(result, tuple) else 0.0
    return float(r_value) ** 2


def alpha_beta(equity_curve: pd.Series) -> tuple[float, float]:
    returns = _get_returns(equity_curve)
    if len(returns) < 2:
        return 0.0, 1.0

    market_return = returns

    if len(returns) != len(market_return):
        min_len = min(len(returns), len(market_return))
        returns = returns.iloc[:min_len]
        market_return = market_return.iloc[:min_len]

    covariance = np.cov(returns, market_return)[0][1]
    market_variance = np.var(market_return)

    if market_variance == 0:
        return 0.0, 1.0

    beta = covariance / market_variance
    period_alpha = float(returns.mean() - beta * market_return.mean())
    ppy = periods_per_year(equity_curve)
    if ppy <= 0:
        return 0.0, float(beta)

    annualized_alpha = ((1 + period_alpha) ** ppy) - 1 if period_alpha > -1 else -1

    return annualized_alpha, float(beta)


def calmar_ratio(equity_curve: pd.Series) -> float:
    ret = annual_return(equity_curve)
    max_dd = abs(max_drawdown(equity_curve))
    if max_dd == 0:
        return 0.0
    return ret / max_dd


def analyze_portfolio(result: PortfolioResult) -> PerformanceMetrics:
    equity_curve = result.equity_curve
    sharpe_ratio = result.sharpe_ratio if result.sharpe_ratio else 0.0
    alpha, beta = alpha_beta(equity_curve)

    return PerformanceMetrics(
        annual_return=annual_return(equity_curve),
        annual_volatility=annual_volatility(equity_curve),
        sharpe_ratio=sharpe_ratio,
        calmar_ratio=calmar_ratio(equity_curve),
        sortino_ratio=sortino_ratio(equity_curve),
        stability=stability(equity_curve),
        max_drawdown=max_drawdown(equity_curve),
        omega_ratio=omega_ratio(equity_curve),
        skewness=skewness(equity_curve),
        kurtosis=kurtosis(equity_curve),
        alpha=alpha,
        beta=beta,
    )


def calculate_portfolio_result(
    equity_curve: pd.Series, trades, initial_capital: float
) -> PortfolioResult:
    """Calculate portfolio result from equity curve and trades.

    Args:
        equity_curve: Equity curve as pandas Series
        trades: Iterable of Trade objects
        initial_capital: Starting capital

    Returns:
        PortfolioResult with all calculated metrics
    """
    returns = equity_curve.pct_change().dropna()
    ppy = periods_per_year(equity_curve)

    total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital

    sharpe = 0.0
    if len(returns) > 0 and returns.std() != 0 and ppy > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(ppy)

    return PortfolioResult(
        total_return=total_return,
        sharpe_ratio=sharpe,
        trades=tuple(trades),
        equity_curve=equity_curve,
        annual_return=annual_return(equity_curve),
        annual_volatility=annual_volatility(equity_curve),
        max_drawdown=max_drawdown(equity_curve),
        calmar_ratio=calmar_ratio(equity_curve),
        sortino_ratio=sortino_ratio(equity_curve),
        omega_ratio=omega_ratio(equity_curve),
        skewness=skewness(equity_curve),
        kurtosis=kurtosis(equity_curve),
        stability=stability(equity_curve),
        alpha=alpha_beta(equity_curve)[0],
        beta=alpha_beta(equity_curve)[1],
    )


def _safe_date_str(date: object | None) -> str:
    if date is None:
        return "NaT"
    if hasattr(date, "strftime"):
        return cast(Any, date).strftime("%Y-%m-%d")
    return str(date)[:10]


def get_backtest_results_analysis(
    result: PortfolioResult,
    metrics: Optional[PerformanceMetrics] = None,
    title: str = "Backtest Results",
) -> str:
    if metrics is None:
        metrics = analyze_portfolio(result)

    equity_curve = result.equity_curve

    # Capture output
    output = StringIO()
    original_stdout = sys.stdout
    sys.stdout = output

    print(f"\n{title}")
    print("=" * 80)

    trades = result.trades

    dds = drawdown_periods(equity_curve)
    if drawdown_periods:
        print(f"\n{'Worst Drawdown Periods':<75}")
        print(
            f"{'Net DD %':<15} {'Peak Date':<15} {'Valley Date':<15} {'Recovery':<15} {'Duration':<10}"
        )
        print("-" * 75)
        for dd in dds[:5]:
            print(
                f"{dd.net_drawdown_pct:<14.2f} {_safe_date_str(dd.peak_date):<15} {_safe_date_str(dd.valley_date):<15} {_safe_date_str(dd.recovery_date):<15} {dd.duration:<10}"
            )

    print(f"\n{'Trades':<75}")
    print("-" * 120)
    # Table header
    print(
        f"{'Entry Time':<20} {'Exit Time':<20} {'Entry':<10} {'Exit':<10} {'PnL':<10} {'Pos':<8} {'Reason':<15} {'SL/TP':<15}"
    )
    print("-" * 120)

    # Table rows
    for i, t in enumerate(trades, 1):
        # Format SL/TP column
        sl_tp_str = f"{t.stop_loss:.2f}/{t.take_profit:.2f}"
        exit_price_str = f"{t.exit_price:.2f}" if t.exit_price is not None else "N/A"
        print(
            f"{_safe_date_str(t.entry_time):<20} "
            f"{_safe_date_str(t.exit_time):<20} "
            f"{t.entry_price:<10.2f} "
            f"{exit_price_str:<10} "
            f"{t.pnl:<10.2f} "
            f"{t.position == ActionType.long and 'L' or 'S':<8} "
            f"{str(t.close_reason):<15} "
            f"{sl_tp_str:<15}"
        )

    closed_trades = [t for t in trades if t.status.value == "closed"]
    profitable = [t for t in closed_trades if t.pnl > 0] if closed_trades else []
    win_rate = len(profitable) / len(closed_trades) if closed_trades else 0.0

    print(f"\n{'Trading Statistics':<25}")
    print("-" * 42)
    print(f"{'Starting Capital':<25} {equity_curve.iloc[0]:>15}")
    print(f"{'Total Trades':<25} {len(trades):>15}")
    print(f"{'Closed Trades':<25} {len(closed_trades):>15}")
    print(f"{'Win Rate':<25} {win_rate:>14.2%}")
    print(f"{'Total P&L':<25} {equity_curve.iloc[-1] - equity_curve.iloc[0]:>15.2f}")

    first_date = equity_curve.index[0]
    last_date = equity_curve.index[-1]
    first_str = _safe_date_str(first_date)
    last_str = _safe_date_str(last_date)

    print(f"\nData Start Date: {first_str}")
    print(f"Data End Date: {last_str}")

    n_periods = len(equity_curve)
    duration_desc = f"{n_periods} periods"
    if isinstance(equity_curve.index, pd.DatetimeIndex) and n_periods > 1:
        start = equity_curve.index[0]
        end = equity_curve.index[-1]
        try:
            elapsed_days = (end - start).total_seconds() / (24 * 60 * 60)
        except AttributeError:
            elapsed_days = 0.0
        if elapsed_days > 0:
            months = elapsed_days / 30.44
            duration_desc = f"{n_periods} periods ({months:.1f} months)"
    print(f"\nBacktest Duration: {duration_desc}")

    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-" * 42)
    print(f"{'Annual Return':<25} {metrics.annual_return:>14.2%}")
    print(f"{'Annual Volatility':<25} {metrics.annual_volatility:>14.2%}")
    print(f"{'Sharpe Ratio':<25} {metrics.sharpe_ratio:>15.2f}")
    print(f"{'Calmar Ratio':<25} {metrics.calmar_ratio:>15.2f}")
    print(f"{'Sortino Ratio':<25} {metrics.sortino_ratio:>15.2f}")
    print(f"{'Omega Ratio':<25} {metrics.omega_ratio:>15.2f}")
    print(f"{'Max Drawdown':<25} {metrics.max_drawdown:>14.2%}")
    print(f"{'Stability':<25} {metrics.stability:>15.2f}")
    print(f"{'Skewness':<25} {metrics.skewness:>15.2f}")
    print(f"{'Kurtosis':<25} {metrics.kurtosis:>15.2f}")
    print(f"{'Alpha':<25} {metrics.alpha:>15.2f}")
    print(f"{'Beta':<25} {metrics.beta:>15.2f}")

    # Restore stdout and return output if requested
    sys.stdout = original_stdout
    return output.getvalue()


def exposure_and_turnover(
    result: PortfolioResult,
) -> Dict[str, float]:
    trades = result.trades
    closed_trades = [t for t in trades if t.status.value == "closed"]

    if closed_trades:
        turnover = sum(abs(t.qty * t.entry_price) for t in closed_trades)
        ppy = periods_per_year(result.equity_curve)
        num_periods = len(result.equity_curve) - 1
        annualized_turnover = (
            turnover * ppy / num_periods if num_periods > 0 and ppy > 0 else turnover
        )
    else:
        turnover = 0.0
        annualized_turnover = 0.0

    return {
        "gross_exposure": turnover,
        "turnover": turnover,
        "annualized_turnover": float(annualized_turnover),
    }
