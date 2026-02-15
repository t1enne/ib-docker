# IBKR PY

CLI package for testing pairs trading strategies and visualizing signals. Supports z-score based pairs trading with optional HMM-based regime detection for market regime filtering.

## What It Does

The system performs **walk-forward backtesting** on pairs trading strategies:

1. **Pairs Trading**: Trades pairs of securities (e.g., MA/V) based on z-score divergence from historical spread relationships
2. **Z-Score Model**: Computes rolling z-score of the spread to identify trading opportunities
3. **HMM Regime Detection** (optional): Uses Hidden Markov Models to detect market regimes (Low/Med/High volatility) and filter trades
4. **Risk Management**: Stop-loss, take-profit, and position sizing
5. **Visualization**: HTML charts showing prices, volume, z-score, regime probabilities, equity curve, and drawdown

## Quick Start

```bash
# Run a backtest with a strategy config
uv run main.py strategy ma_v.yaml

# Or run directly
uv run python -c "
import asyncio
from src.bt import load_strategy, backtest

async def main():
    strategy = load_strategy('ma_v.yaml')
    await backtest(strategy)

The project relies on `ty` for LSP functionality and `ruff` for formatting.
To typecheck: `uv tool run ty check`
To format: `uv tool run ruff format`

# Project Structure

- `src/bt/algos/` - Trading strategies and models
- `src/bt/engine/` - Backtesting engines
- `src/bt/portfolio/` - Portfolio management
- `src/bt/execution/` - Execution handler for spreads/slippage
