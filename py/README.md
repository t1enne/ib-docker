# IBKR PY — Composable Quantitative Trading Toolkit

A modular CLI toolkit for quantitative trading: data synchronization, indicator computation, statistical models, and a functional backtesting engine. Configure strategies in YAML, run them from the terminal, and iterate fast.

## Quickstart

```bash
uv sync
uv run py bt run strats/trend.yaml
```

## Core Concepts

### Functional Pipeline

Every component is a **pure function** operating on immutable state. The backtest loop composes these functions:

```
Data → model_updater → strategy → execution → portfolio → risk → mark-to-market
```

- **Pure functions** — same inputs, same outputs. No side effects.
- **Immutable state** — `dataclasses.replace()` returns new state, never mutates.
- **Protocol-based injection** — components implement `StrategyFn`, `ExecutionFn`, `RiskCheckFn` etc., not base classes.

### Code Organization

```
src/
├── data/             ← IBKR market data sync, DB, resampling
│   ├── ibkr/         ← IBKR REST API client (candles, lookup, rate limiter)
│   ├── sync.py       ← sync_data(), preview_sync()
│   ├── db.py         ← SQLite query helpers
│   ├── resample.py   ← OHLCV resampling
│   └── cli.py        ← `data` CLI group
├── indicators/       ← Signal processing models
│   ├── ta.py         ← Technical indicators: EMA, RSI, MACD, ATR, ADX, MFI, LSMA...
│   ├── kalman/       ← Kalman filter (univariate + pairs)
│   └── hmm/          ← Hidden Markov Model regime detection
├── bt/               ← Backtesting engine
│   ├── engine/       ← Backtest loop (pure functional)
│   ├── strategies/   ← Strategy implementations (6 built-in)
│   ├── state/        ← Immutable dataclasses
│   ├── types.py      ← Protocols, StrategyConfig
│   ├── portfolio/    ← Position sizing, fill application, mark-to-market
│   ├── execution/    ← Signal → fill with slippage/spread
│   ├── risk/         ← Stop-loss/take-profit checks
│   ├── metrics.py    ← Sharpe, Sortino, Calmar, drawdowns
│   └── data_feed/    ← Load/sync OHLCV candles
├── utils.py          ← Shared utilities (DB read, z-score, env loader)
└── main.py           ← CLI entry point
```

## Running a Backtest

### 1. Define a Strategy (YAML)

```yaml
# strats/my_strat.yaml
name: ema-cross
training_start: 2024-01-01
training_end: 2024-01-02
trading_start: 2024-01-02
trading_end: 2025-01-01
commission: 0.1
initial_capital: 10000
position_size: 0.2
strategy_type: ema_cross
stop_loss: 0.2
take_profit: 0.5
bar: 1h
htf:
  - 4h
model_params: {}
strategy_params:
  fast: 9
  slow: 30
symbols:
  - COIN
```

### 2. Run

```bash
uv run py bt run strats/my_strat.yaml
```

Output: equity curve summary, trade log, metrics table (Sharpe, Sortino, Calmar, max drawdown, win rate, etc.).

For JSON output:

```bash
uv run py bt run strats/my_strat.yaml --format jsonl
```

## Built-in Strategies

Strategies are auto-discovered from `src/bt/strategies/`. Each `.py` file with a `STRATEGY_TYPE` and `on_candle()` registers itself — no manual wiring.

| Strategy                | `strategy_type`                              | File                          | Description                                                            |
| ----------------------- | -------------------------------------------- | ----------------------------- | ---------------------------------------------------------------------- |
| EMA Cross               | `ema_cross`                                  | `ema_cross.py`                | EMA fast/slow cross with ranging detection (ATR + EMA convergence)     |
| Trend Following         | `trend_following`                            | `trend_following.py`          | LSMA+EMA with DMI, MFI, volume confirmation, HTF alignment             |
| Breakout EMA            | `breakout_ema`                               | `breakout_ema.py`             | Squeeze detection → breakout with volume spike → trend ride            |
| Pairs Trading           | `pnd`                                        | `pairs_trading_functional.py` | Z-score mean reversion on pairs with OLS hedge ratio                   |
| Vol Extension Pullback  | `volatility_expansion_pullback_continuation` | `vol_extension_pullback.py`   | ATR compression → breakout → pullback entry with continuation triggers |
| Yesterday High Breakout | `yesterday_high_breakout`                    | `yesterday_high_breakout.py`  | Breaks previous day's high with gap entry and trailing stop            |

## Writing a Custom Strategy

A strategy module needs one function: `on_candle()`.

```python
# src/bt/strategies/my_strat.py

STRATEGY_TYPE = "my_strat"

import src.indicators.ta as ta
from src.bt.strategies.utils import open, close
from src.bt.state import BacktestState, TradeSignal, Candle, ActionType

def on_candle(
    state: BacktestState, candle: Candle, strategy_params: dict
) -> list[TradeSignal]:
    """Called on every candle. Return signals."""
    symbol = candle.symbol
    position = state.portfolio.positions.get(symbol)
    candles = state.candles[symbol]
    closes = candles["close"]

    ema_fast = ta.ema(closes, 9).iloc[-1]
    ema_slow = ta.ema(closes, 21).iloc[-1]

    # No position → check entry
    if not position and ema_fast > ema_slow:
        return open(candle, ActionType.long, "golden cross")

    # In position → check exit
    if position and ema_fast < ema_slow:
        return close(candle, position, "death cross")

    return []
```

**That's it.** Drop your `.py` file in `src/bt/strategies/`, define `STRATEGY_TYPE` and `on_candle()`, and the engine finds it automatically — no registry edits needed.

### Available State in `on_candle()`

```python
state.portfolio                # PortfolioState: cash, positions, trades, equity_curve
state.portfolio.positions      # Dict[str, Position] — open positions by symbol
state.candles                  # Dict[str, pd.DataFrame] — OHLCV per symbol
state.model_state.z_score      # Current z-score (pairs strategies)
state.model_state.price_buffers# Aligned {sym: close} pairs
state.htf_data                 # Dict[str, pd.DataFrame] — higher-timeframe bars
state.timestamp                # Current timestamp
```

### Available Indicators (`src.indicators.ta`) `

`ema`, `sma`, `rsi`, `atr`, `bollinger_bands`, `macd`, `stochastic`, `momentum`, `volatility`, `vwma`, `obv`, `mfi`, `lsma`, `plus_di`, `minus_di`, `adx`

### Helper Utilities (`src/bt/strategies/utils`)

```python
open(candle, ActionType.long, "reason", hedge=1.0)    # → [TradeSignal]
close(candle, position, "reason")                       # → [TradeSignal]
htf_candles(state, "4h", tick)                          # → pd.DataFrame (no lookahead)
```

## Higher Timeframe Data

Strategies can access resampled bars from higher timeframes without lookahead bias. Enable in config:

```yaml
htf:
  - 4h
  - 1D
```

Then use in `on_candle()`:

```python
htf = htf_candles(state, "4h", candle)
if not htf.empty:
    htf_ema = ta.ema(htf["close"], 20).iloc[-1]
```

Only completed HTF bars (timestamp ≤ current candle) are returned — no forward-looking.

## Key Types

### `StrategyConfig`

```python
@dataclass
class StrategyConfig:
    name: str                    # Strategy name
    strategy_type: str           # Matches init_strat() key
    symbols: list[str]           # Tickers to trade
    stop_loss: float             # % stop loss
    take_profit: float           # % take profit
    initial_capital: float       # Starting capital
    position_size: float         # % of capital per trade
    commission: float            # Fixed commission per trade
    training_start: str          # ISO date
    training_end: str
    trading_start: str
    trading_end: str
    bar: str                     # "1h", "1D", etc.
    strategy_params: dict        # Passed to on_candle()
    model_params: dict           # For model updaters
    htf: list[str]               # Higher timeframes (["4h", "1D"])
    rolling_window_size: int | None
```

### `PortfolioResult`

```python
@dataclass(frozen=True)
class PortfolioResult:
    total_return: float
    sharpe_ratio: float
    trades: tuple[Trade, ...]
    equity_curve: pd.Series
    annual_return: float
    annual_volatility: float
    calmar_ratio: float
    sortino_ratio: float
    max_drawdown: float
    alpha: float
    beta: float
    skewness: float
    kurtosis: float
    stability: float
    omega_ratio: float
```

### Programmatic API

```python
import asyncio
from src.bt import load_strategy, Backtest, run, get_backtest_results_analysis
from src.bt.data_feed import load_candles
from src.bt.strategies import init_strat

config = load_strategy("strats/trend.yaml")
bt = Backtest(config)
df = load_candles(config.symbols, bt.window.train_start, bt.window.test_end, config.bar)
strat_mod = init_strat(config.strategy_type)
results = run(bt, df, strat_mod=strat_mod)
print(get_backtest_results_analysis(results.pf))
```

## CLI Reference

```bash
# Run a backtest
uv run py bt run <strategy.yaml> [--format jsonl]

# Analyze a strategy (detailed JSON metrics)
uv run py bt analyze <strategy.yaml>

# Data commands
uv run py data query AAPL --from 2024-01-01   # Fetch candles

# Pipe workflows
uv run py data query AAPL --from 2024-01-01 | uv run py bt run strategy.yml
```

## Toolchain

- **Python 3.14+** required
- **`uv`** — package management
- **`ty`** — type checking
- **`ruff format`** — formatting
- **`pytest`** — testing (`uv run pytest`)

## Testing

```bash
uv run pytest                                  # All tests
uv run pytest src/bt/engine/tests/ -v          # Specific module
```

Tests use `pytest-asyncio` for async and `respx` for HTTP mocking. Pure function tests are the norm — given fixed inputs, backtest results are deterministic.

## Resources

- **IBKR REST API Client**: `../ib-rest-api-client` (local dependency, path-configured in `pyproject.toml`)
- **Data sync**: `src/data/` — IBKR candle download, DB queries, resampling
- **Indicators**: `src/indicators/` — Technical indicators (`ta.py`), Kalman filters (`kalman/`), Hidden Markov Models (`hmm/`)
