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

asyncio.run(main())
"
```

## Configuration

Strategy YAML files define the backtest parameters:

```yaml
# ma_v.yaml - Example pairs trading config
name: ma_v
strategy_type: pnd
symbols:
  - ma # Morgan Stanley
  - v # Visa

# Date ranges
training_start: 2022-01-01 # Initial training period
training_end: 2022-06-01 # End of training
trading_start: 2022-06-02 # Start of trading period
trading_end: 2026-01-01 # End of backtest

# Z-score parameters
rolling_window_size: 75 # Lookback window for z-score
entry_z: 2.5 # Enter when |z| > this
exit_z: 0.5 # Exit when |z| < this

# HMM regime detection (optional)
hmm_floating_window: 252 # Bars to use for HMM training
hmm_retrain_interval: 50 # Retrain HMM every N bars

# Risk parameters
stop_loss: 1.0 # Stop loss %
take_profit: 5.0 # Take profit %
position_size: 0.2 # Position size as % of capital

# Other
initial_capital: 10000
commission: 0.5 # Commission in $
plot: true
```

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI (main.py)                             │
│                    load_strategy() → backtest()                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   WalkForwardEngine                               │
│  - Orchestrates single walk-forward window                       │
│  - Passes config to BacktestEngine                              │
│  - Calls plotting on results                                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BacktestEngine                                │
│  1. Pre-seed model with training period data                    │
│  2. Loop through trading period ticks:                          │
│     - Update StrategyModel (z-score + HMM)                      │
│     - Get signals from strategy                                 │
│     - Execute trades, update portfolio                          │
│     - Check risk (stop-loss, take-profit)                      │
│  3. Return results + z-scores + regime data                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│      StrategyModel       │     │        Portfolio            │
│  - ZModel (z-score)    │     │  - Track positions          │
│  - MarketDataView      │     │  - PnL calculation          │
│  - RegimeModel (HMM)   │     │  - Risk management          │
│                         │     └─────────────────────────────┘
│  Exposed to strategies:  │
│  - model.z_score        │
│  - model.current_regime │
│  - model.market_data    │
│  - model.hmm           │
└─────────────────────────┘
```

### Data Flow

1. **Data Loading**: `read_candles()` fetches OHLCV data from SQLite database
2. **Pre-seeding**: Training period data (`training_start` → `training_end`) is fed into the model first to warm up both z-score and HMM
3. **Trading Loop**: For each bar in the trading period:
   - `model.update(timestamp, tick_group)` updates z-score and runs HMM training/prediction
   - `strategy.on_tick(tick, open_trade)` generates signals based on z-score
   - `portfolio.on_fill(fill)` executes trades
   - `risk_manager.on_tick(tick)` checks stop-loss/take-profit
4. **Results**: Returns portfolio results, price data, z-scores, and regime probabilities

### Strategy Access to Models

Strategies access computed features via the `model` object:

```python
class MyStrategy:
    def on_tick(self, tick, open_trade):
        # Z-score
        z = self.model.z_score

        # Regime (if HMM enabled)
        regime = self.model.current_regime  # 0=Low, 1=Med, 2=High

        # Historical data
        closes = self.model.market_data[-14:].close

        # Indicators
        from src.bt.indicators import ema, rsi
        ema_9 = ema(closes["AAPL"], 9)
        rsi_14 = rsi(closes["AAPL"], 14)

        # Should we trade based on regime?
        if self.model.should_trade():
            # Generate signals...
```

### Available Indicators

`src/bt/indicators.py` provides:

- `ema(data, *spans)` - Exponential moving average
- `sma(data, window)` - Simple moving average
- `rsi(data, window=14)` - Relative Strength Index
- `atr(high, low, close, window=14)` - Average True Range
- `bollinger_bands(data, window=20, num_std=2.0)` - Bollinger Bands
- `macd(data, fast=12, slow=26, signal=9)` - MACD
- `stochastic(high, low, close, ...)` - Stochastic Oscillator
- `momentum(data, window=10)` - Momentum
- `volatility(data, window=20, annualized=True)` - Rolling volatility

## Project Structure

```
py/
├── main.py                     # CLI entrypoint (Click)
├── ma_v.yaml                  # Example strategy config
│
├── src/
│   ├── bt/                    # Backtesting core
│   │   ├── __init__.py        # Strategy dataclass, backtest()
│   │   ├── types.py           # Type definitions (Tick, TradeSignal, etc.)
│   │   ├── indicators.py       # Technical indicator functions
│   │   │
│   │   ├── algos/
│   │   │   ├── pairs_trading.py    # PairsTradingStrategy
│   │   │   ├── base_pairs_strategy.py  # Signal factory helpers
│   │   │   └── tests/
│   │   │
│   │   ├── models/
│   │   │   ├── strategy_model.py   # StrategyModel (composite facade)
│   │   │   ├── market_data.py      # MarketDataView (historical OHLCV)
│   │   │   ├── z_model.py          # ZModel (z-score calculator)
│   │   │   ├── regime_model.py     # RegimeModel wrapper
│   │   │   └── zscore.py           # Z-score calculation logic
│   │   │
│   │   ├── engine/
│   │   │   ├── backtest_engine.py  # Core backtest loop
│   │   │   ├── walk_forward_engine.py  # Walk-forward orchestration
│   │   │   └── tests/
│   │   │
│   │   ├── portfolio/          # Position & PnL tracking
│   │   ├── risk/              # Stop-loss, take-profit
│   │   ├── execution/         # Slippage, commission
│   │   ├── metrics.py         # Performance metrics
│   │   └── plotting/          # Plotly charts
│   │
│   ├── hmm/                   # HMM regime detection
│   │   └── hmm.py            # MarketRegimeHMM class
│   │
│   ├── syncm/                 # Data sync from IBKR
│   │
│   └── db/                    # SQLite models
│
└── tests/                     # Test suite
```

## Development

### Running Tests

```bash
# All tests
uv run pytest

# Watch mode
find . -type f -name "*.py" | entr uv run pytest

# Specific test file
uv run pytest src/bt/engine/tests/test_bt_engine.py -v
```

### Type Checking & Formatting

```bash
# Type check
uv tool run ty check src/bt/

# Format
uv tool run ruff format src/bt/
```

## Current State

### Implemented Features

- [x] Z-score based pairs trading strategy
- [x] Rolling z-score calculation with configurable window
- [x] Walk-forward backtesting framework
- [x] HMM regime detection (live training during backtest)
- [x] Configurable HMM floating window and retrain interval
- [x] Historical market data access from strategies
- [x] Technical indicators (EMA, SMA, RSI, ATR, Bollinger Bands, MACD, etc.)
- [x] Risk management (stop-loss, take-profit)
- [x] Execution with slippage and commission
- [x] Portfolio tracking and PnL
- [x] Performance metrics (Sharpe, drawdown, etc.)
- [x] Plotly visualization:
  - Price charts with volume bars
  - Z-score with entry/exit thresholds
  - HMM regime probabilities (stacked area chart)
  - Equity curve and drawdown

### What's Missing / Gaps

- [ ] **Multi-window walk-forward**: Currently only runs a single train/test window. True walk-forward would roll forward multiple times (e.g., train 6mo, test 3mo, roll forward)
- [ ] **More strategy types**: Only `pnd` (pairs trading) is implemented. Stub strategies exist but aren't wired up (Kalman, Ratio, VolumeWeighted, MLSpread)
- [ ] **Position hedging**: No explicit hedge ratio calculation beyond simple price ratio
- [ ] **Transaction costs modeling**: Simple fixed commission, no slippage modeling beyond fixed bps
- [ ] **Live trading**: Backtesting only, no connection to IBKR API for live execution
- [ ] **Parameter optimization**: No grid search or optimization over entry/exit z-scores
- [ ] **Benchmark comparison**: No SPY or other benchmark for relative performance
- [ ] **Advanced regime features**: Currently uses close price only; could incorporate volume, volatility, correlations
- [ ] **Strategy YAML validation**: No schema validation; bad config leads to cryptic errors
- [ ] **Error handling**: Minimal error handling; failures often propagate as exceptions

## Extending the System

### Adding a New Indicator

Add to `src/bt/indicators.py`:

```python
def my_indicator(data: pd.Series, param: int = 14) -> pd.Series:
    """Description of what it does."""
    # Implement...
    return result
```

### Adding a New Strategy

1. Create `src/bt/algos/my_strategy.py` implementing `StrategyProtocol`:

```python
from src.bt.types import StrategyProtocol, Tick, TradeSignal

class MyStrategy(StrategyProtocol):
    model: Any  # StrategyModel instance

    def __init__(self, symbols, params):
        self.symbols = symbols
        self.params = params

    def on_tick(self, tick: Tick, open_trade) -> List[TradeSignal]:
        z = self.model.z_score
        # Generate signals...
        return signals
```

2. Wire it into the engine in `src/bt/__init__.py`

### Adding a New Model

1. Create training/inference logic in `src/hmm/` or a new module
2. Add a model wrapper class that integrates with `StrategyModel`
3. The strategy can then access it via `self.model.your_model`

### Contributing

#### Coding style

- Use negative-space programming (assertions) instead of early returns as needed.
- Use @dataclass instead of dicts.
- Use numpy for hot paths for speed
- Prefer functional patterns. return state instead of mutating it directly (where it won't degrade perf). Compose functions.
- Prefer class composition over inheritance.
- Functions should stay under 50 LOC, classes under 150 LOC.
- Test where possible

## License

MIT
