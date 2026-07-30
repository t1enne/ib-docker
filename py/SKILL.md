---
name: backtester
description: Backtest quantitative trading strategies independently — create JSON configs, write custom strategies, run backtests, and interpret results. Use when asked to "backtest a trading strategy", "test a pairs trade", "run a backtest", "create a strategy", or "evaluate a trading idea".
allowed-tools: Bash(uv:*), Bash(cd:*), Bash(find:*), Bash(cat:*), Bash(ls:*), Bash(grep:*), Read, Write, Edit
---

# IBKR Backtesting — Standalone Agent Skill

Full backtesting agent for the IBKR PY quantitative trading toolkit. Design, implement, and run strategy backtests end-to-end.

## Project Location

```bash
cd /home/nasrt/Documents/code/dev/ibkr/py
```

## Prerequisites

- Python 3.14+
- `uv` package manager
- Project dependencies: `uv sync`

## Backtest a Strategy in 3 Steps

### Step 1: Pick or Write a Strategy Module

Strategies live in `src/bt/strategies/`. A strategy module must export one function:

```python
def on_candle(
    state: BacktestState, candle: Candle, strategy_params: dict
) -> list[TradeSignal]:
    """Called on every candle. Return signals to open/close positions."""
    ...
```

**The state object gives you:**

- `state.portfolio` — `PortfolioState` with `.cash`, `.positions` (dict), `.trades`, `.equity_curve`
- `state.portfolio.positions.get(symbol)` — check if already in a position
- `state.candles[symbol]` — `pd.DataFrame` of OHLCV bars for the symbol
- `state.model_state.z_score` — current z-score (pairs strategies)
- `state.model_state.price_buffers` — aligned `{sym: close}` dicts per tick
- `state.htf_data` — dict of higher-timeframe DataFrames keyed by interval string (e.g., `"4h"`)
- `state.timestamp` — current `pd.Timestamp`

**Available indicators** (`from src.indicators.ta import ...`) :
`ema`, `sma`, `rsi`, `atr`, `bollinger_bands`, `macd`, `stochastic`, `momentum`, `volatility`, `vwma`, `obv`, `mfi`, `lsma`, `plus_di`, `minus_di`, `adx`

**Signal helpers** (`from src.bt.strategies.utils import ...`):

```python
open(candle, ActionType.long, "reason", hedge=1.0)    # → [TradeSignal]
close(candle, position, "reason")                       # → [TradeSignal]
htf_candles(state, "4h", candle)                        # → pd.DataFrame (lookahead-safe)
```

**HTF access pattern:**

```python
htf = htf_candles(state, "4h", candle)
if not htf.empty:
    htf_close = htf["close"]
    htf_ema = ta.ema(htf_close, 20).iloc[-1]
```

**Existing strategies** (in `src/bt/strategies/`):
| File | `strategy_type` key | Description |
|---|---|---|
| `ema_cross.py` | `ema_cross` | EMA fast/slow cross with ranging detection |
| `trend_following.py` | `trend_following` | LSMA+EMA with DMI/MFI/volume/HTF |
| `breakout_ema.py` | `breakout_ema` | Squeeze → breakout → trend ride |
| `pairs_trading_functional.py` | `pnd` | Z-score mean reversion, hedge ratio |
| `vol_extension_pullback.py` | `volatility_expansion_pullback_continuation` | ATR compression → breakout → pullback |
| `yesterday_high_breakout.py` | `yesterday_high_breakout` | Daily breakout with trailing stop |

**To register a new strategy**, add it to the match in `src/bt/strategies/__init__.py`:

```python
import src.bt.strategies.my_strategy  # add import at top

def init_strat(strat_name: str):
    match strat_name:
        # ... existing cases ...
        case "my_strategy":
            return src.bt.strategies.my_strategy
```

### Step 2: Write the JSON Config

Create `strats/<name>.json`:

```json
{
  "name": "strategy-name",
  "training_start": "2024-01-01",
  "training_end": "2024-01-02",
  "trading_start": "2024-01-02",
  "trading_end": "2025-01-01",
  "commission": 0.1,
  "initial_capital": 10000,
  "position_size": 0.2,
  "strategy_type": "ema_cross",
  "stop_loss": 0.2,
  "take_profit": 0.5,
  "bar": "1h",
  "htf": ["4h", "1D"],
  "model_params": {},
  "strategy_params": {
    "fast": 9,
    "slow": 30
  },
  "symbols": ["COIN", "AAPL"]
}
```

**Config field reference:**

- `training_start`/`training_end` — model training window (currently unused by most strategies)
- `trading_start`/`trading_end` — actual backtest window
- `commission` — fixed commission per trade
- `position_size` — fraction of capital deployed per trade
- `stop_loss`/`take_profit` — percentage levels; engine enforces them globally
- `htf` — resampled higher-timeframe bars injected alongside base bars (lookahead-safe)
- `strategy_params` — arbitrary dict forwarded verbatim to `on_candle(state, candle, params)`

**Available tickers** — see `/home/nasrt/Documents/code/dev/ibkr/py/universes/*.json` for symbols with local data.

### Step 3: Run the Backtest

```bash
cd /home/nasrt/Documents/code/dev/ibkr/py
uv run py bt run strats/<name>.json
make run bt run strats/<name>.json   # same, via Make shortcut
```

> **Note:** Make intercepts its own flags (`--help`, `--format`). To pass them through to `main.py`, use `-- ` separator: `make run bt run strat.json -- --format jsonl`.

**For programmatic use** (if you need structured output):

```python
import asyncio
from src.bt import load_strategy, Backtest, run
from src.bt.data_feed import load_candles
from src.bt.strategies import init_strat

config = load_strategy("strats/trend.json")
bt = Backtest(config)
df = load_candles(config.symbols, bt.window.train_start, bt.window.test_end, config.bar)
strat_mod = init_strat(config.strategy_type)
results = run(bt, df, strat_mod=strat_mod)
# results.pf → PortfolioResult (total_return, sharpe_ratio, trades, equity_curve, ...)
# results.data → dict[str, pd.DataFrame] of candles
# results.final_state → BacktestState
```

## Interpreting Results

The output contains:

- **Drawdown periods** — worst 5 drawdowns with dates and duration
- **Trade log** — every trade with entry/exit times, prices, PnL, direction, reason, SL/TP levels
- **Statistics** — win rate, total trades, starting capital, total P&L, backtest duration
- **Metrics table** — annual return, volatility, Sharpe, Calmar, Sortino, Omega, max drawdown, stability, skewness, kurtosis, alpha, beta

## Design Heuristics

When building or modifying strategies, follow these patterns from the codebase:

### Keep `on_candle()` simple

One function, no classes. Read state, compute indicators, return signals. The engine handles the rest.

### Volume filtering

Volume confirmation is a common guard — see `trend_following.py` and `breakout_ema.py` for `volume_confirmed()` / `volume_spike()` patterns.

### Ranging/squeeze detection

Use EMA convergence + ATR contraction. See `ema_cross.py:is_ranging()` and `breakout_ema.py:is_squeeze()`.

### Statefulness within strategy

For multi-phase strategies (compression → breakout → pullback → entry), use a module-level dict keyed by symbol. See `vol_extension_pullback.py` and `yesterday_high_breakout.py` for the `_signal_state` pattern. Note: call `reset_signal_state()` between backtests if state persists.

### Always handle "no position" and "in position" paths

```python
position = state.portfolio.positions.get(symbol)
if not position:
    # entry logic
else:
    # exit logic
```

### Use htf_candles() for multi-timeframe

Don't access `state.htf_data` directly — use `htf_candles(state, freq, tick)` which filters completed buckets. Direct access risks lookahead bias.

## Testing a Strategy

```bash
make test                                     # all tests
make test-fast                                # quick tests (no header)
uv run pytest src/bt/engine/tests/ -v
uv run pytest src/bt/portfolio/tests/ -v
uv run pytest src/bt/risk/tests/ -v
```

## Common Gotchas

- **Data availability**: not all symbols in `universes/*.json` have backfill on disk. If `load_candles()` returns empty, data needs syncing first via `uv run py data query <SYMBOL>` (or `make run data query <SYMBOL>`).
- **Bar size**: strategies expect the bar size in config to match available data. Most data is `1h`.
- **HTF lookahead**: `htf_candles()` is safe. Direct `state.htf_data[freq]` is not — it contains all bars including those after current tick.
- **Multiple symbols**: the engine iterates all symbols per timestamp. The strategy runs on the last symbol. Entry signals for all symbols work; position management happens per-symbol.
- **Pairs strategy** (`pnd`): requires exactly 2 symbols. Uses `model_state.price_buffers` for aligned close prices.

## Module Reference

All CLI groups under the `py` root command — also callable via `make run <subcommand> <args>`:

| Group  | Commands                 | Description                                   |
| ------ | ------------------------ | --------------------------------------------- |
| `data` | `dl`, `query`, `preview` | Sync/download OHLCV from IBKR, query local DB |
| `bt`   | `run`, `analyze`         | Backtesting engine                            |
