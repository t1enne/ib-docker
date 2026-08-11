---
name: backtester
description: Backtest quantitative trading strategies independently — create JSON configs, write custom strategies, run backtests, and interpret results. Use when asked to "backtest a trading strategy", "test a pairs trade", "run a backtest", "create a strategy", or "evaluate a trading idea".
allowed-tools: Bash(uv:*), Bash(cd:*), Bash(find:*), Bash(cat:*), Bash(ls:*), Bash(grep:*), Read, Write, Edit
---

# IBKR Backtesting — Standalone Agent Skill

Full backtesting agent for the IBKR PY quantitative trading toolkit. Design, implement, and run strategy backtests end-to-end.

## Workflow

When asked to backtest, create a strategy, or evaluate a trading idea, follow this sequence:

1. **Understand the request** — what symbols, what kind of strategy, what timeframe?
2. **Pick or write a strategy module** — reuse existing if possible (see table below), otherwise write `on_candle()`.
3. **Write the JSON config** — create `strats/<name>.json` with appropriate params.
4. **Run the backtest** — `uv run ibkr bt run strats/<name>.json`.
5. **Interpret and report** — summarize equity curve, metrics, drawdowns, trade log.

## Project Location

```bash
cd /home/nasrt/Documents/code/dev/ibkr/py
```

## Prerequisites

- Python 3.14+
- `uv` package manager
- Project dependencies: `uv sync`
- **IBKR REST API client** at `../ib-rest-api-client/` (local dependency). If missing, generate it:
  ```bash
  cd /home/nasrt/Documents/code/dev/ibkr/ib-rest-api-client
  uvx openapi-python-client generate --path ../py/openapi.spec.json --output-dir .
  cd ../py && uv sync
  ```
- **Gateway session** — login required before data sync or API calls:
  ```bash
  export IBKR_USERNAME=... IBKR_PASSWORD=... TRADING_MODE=paper
  uv run python scripts/login_ibkr.py
  ```
  Gateway must be running (see `../client-portal/`).

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
- `state.candles` — `CandleStore` (Mapping keyed by `(symbol, interval)`), supports:
  - `state.candles[(sym, interval)]` — `pd.DataFrame` of OHLCV bars (cursor-truncated, lookahead-safe)
  - `state.candles.get((sym, interval))` — same, but returns `None` if missing
  - `state.candles.latest(sym, interval)` — O(1) latest close (`float | None`)
  - `state.candles.count(sym, interval)` — O(1) bar count (`int`)
- `state.model_state.z_score` — current z-score (pairs strategies)
- `state.model_state.price_buffers` — aligned `{sym: close}` dicts per tick
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

HTF candles accumulate in `state.candles` keyed by their interval. Use the same
CandleStore interface — no separate `state.htf_data` dict:

```python
# Direct CandleStore access (preferred — zero-function-call overhead)
htf_df = state.candles.get((candle.symbol, "4h"))
if htf_df is not None and len(htf_df) >= 2:
    htf_ema = ta.ema(htf_df["close"], 20).iloc[-1]

# Or via the helper (lookahead-safe wrapper)
htf = htf_candles(state, "4h", candle)
if not htf.empty:
    htf_close = htf["close"]
    htf_ema = ta.ema(htf_close, 20).iloc[-1]
```

**Existing strategies** (in `src/bt/strategies/`):

| File                             | `strategy_type` key                          | Description                                                                      |
| -------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------- |
| `aegis.py`                       | `aegis`                                      | Multi-asset adaptive equity generation (momentum/correlation/risk-free rotation) |
| `kalman_pairs.py`                | `kalman_pairs`                               | Kalman-filter pairs trading (mean-reversion, z-score of filtered spread)         |
| `momentum_regime.py`             | `momentum_regime`                            | Momentum strategy filtered by regime (HMM/trend detection)                       |
| `sector_mean_reversion.py`       | `sector_mean_reversion`                      | Sector rotation: buy worst 6-month performers, sell when they recover            |
| `sector_mean_reversion_trail.py` | `sector_mean_reversion_trail`                | Sector mean reversion + regime gating, ATR sizing, trail exit                    |
| `shannons_demon.py`              | `shannons_demon`                             | Volatility harvesting via periodic rebalancing                                   |
| `trend_pullback_atr_enhanced.py` | `trend_pullback_atr_enhanced`                | Weekly-trend-confirmed mean reversion, dual-position entry                       |
| `trend_pullback_atr_size.py`     | `trend_pullback_atr_size`                    | Weekly-trend mean reversion with ATR position sizing                             |
| `trend_pullback_atr_trail.py`    | `trend_pullback_atr_trail`                   | Weekly-trend mean reversion with progressive trail exit                          |
| `vol_extension_pullback.py`      | `volatility_expansion_pullback_continuation` | ATR compression → breakout → pullback continuation                               |

**To register a new strategy**, drop a `.py` file in `src/bt/strategies/` that declares
`STRATEGY_TYPE` and `on_candle()` — it's auto-discovered, no manual wiring:

```python
# src/bt/strategies/my_strategy.py
STRATEGY_TYPE = "my_strategy"

def on_candle(state, candle, params) -> list[TradeSignal]:
    """Called on every base-interval candle. Return signals to open/close."""
    ...
```

Optional: define a typed `Params` dataclass (see `src/bt/strategies/types.py`) and the
engine instantiates it from `strategy_params` instead of passing a raw dict.

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
  "strategy_type": "momentum_regime",
  "stop_loss": 0.2,
  "take_profit": 0.5,
  "bars": ["1h", "4h"],
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
- `bars` — list of bar sizes; `bars[0]` is the base signal interval, additional
  entries (e.g. `["1h", "4h"]`) are higher-timeframe bars injected alongside
  (lookahead-safe). There is no separate `htf` field.
- `commission` — fixed commission per trade
- `position_size` — fraction of capital deployed per trade
- `stop_loss`/`take_profit` — percentage levels; engine enforces them globally
- `strategy_params` — arbitrary dict forwarded verbatim to `on_candle(state, candle, params)`

**Available tickers** — see `/home/nasrt/Documents/code/dev/ibkr/py/universes/*.json` for symbols with local data.

### Step 3: Run the Backtest

```bash
cd /home/nasrt/Documents/code/dev/ibkr/py
uv run ibkr bt run strats/<name>.json
make run bt run strats/<name>.json   # same, via Make shortcut
```

> **Note:** Make intercepts its own flags (`--help`, `--format`). To pass them through to `ibkr`, use `-- ` separator: `make run bt run strat.json -- --format jsonl`.

**For programmatic use** (if you need structured output):

```python
import asyncio
from src.bt import load_strategy, Backtest, run
from src.bt.data_feed import load_candles
from src.bt.strategies import init_strat

config = load_strategy("strats/trend_pullback_atr_enhanced.json")
bt = Backtest(config)
df = load_candles(config.symbols, bt.window.train_start, bt.window.test_end, config.bars[0])
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

Volume confirmation is a common guard — see `vol_extension_pullback.py` for volume-confirmation patterns.

### Ranging/squeeze detection

Use EMA convergence + ATR contraction. See `vol_extension_pullback.py` for ATR-compression/squeeze detection.

### Statefulness within strategy

For multi-phase strategies (compression → breakout → pullback → entry), hold cross-call state in one module-level `GLOBAL: dict` keyed by symbol. See `vol_extension_pullback.py` and `trend_pullback_atr_*` for examples. Implement `reset_global()` that rebinds `GLOBAL` to a fresh dict — the engine calls it between split/sweep windows so state doesn't bleed across folds.

### Always handle "no position" and "in position" paths

```python
position = state.portfolio.positions.get(symbol)
if not position:
    # entry logic
else:
    # exit logic
```

### Use htf_candles() for multi-timeframe

Don't access `state.model_state.resample_cache` directly — it's internal accumulator state, not lookahead-safe. Use `state.candles.get((sym, freq))` or `htf_candles(state, freq, tick)` instead.

## Testing a Strategy

```bash
make test                                     # all tests
make test-fast                                # quick tests (no header)
uv run pytest src/bt/engine/tests/ -v
uv run pytest src/bt/portfolio/tests/ -v
uv run pytest src/bt/risk/tests/ -v
```

## Common Gotchas

- **Data availability**: not all symbols in `universes/*.json` have backfill on disk. If `load_candles()` returns empty, data needs syncing first via `uv run ibkr data query <SYMBOL>` (or `make run data query <SYMBOL>`).
- **Bar size**: strategies expect the bar size in config to match available data. Most data is `1h`.
- **HTF lookahead**: `state.candles.get((sym, freq))` and `htf_candles()` are both safe (cursor-truncated). Direct `state.model_state.resample_cache[freq]` is not — it contains all bars including those after current tick.
- **Multiple symbols**: the engine iterates all symbols per timestamp. `on_candle` fires only on the last symbol per timestamp (so `state.candles` has all symbols' data). Signals for any symbol are valid — engine routes fills by `signal.symbol`. Pending signals for non-current symbols fill when that symbol's own `_execute_pending` stage runs (same bar cycle, later in the timestamp iteration).
- **Pairs strategy** (`kalman_pairs`): requires exactly 2 symbols. Uses `model_state.price_buffers` for aligned close prices.

## Module Reference

All CLI groups under the `py` root command — also callable via `make run <subcommand> <args>`:

| Group  | Commands                   | Description                                                        |
| ------ | -------------------------- | ------------------------------------------------------------------ |
| `data` | `dl`, `query`, `preview`   | Sync/download OHLCV from IBKR, query local DB                      |
| `bt`   | `run`, `split`, `optimize` | Backtesting engine + IS/OOS validation + walk-forward param tuning |

### `bt split` — IS/OOS walk-forward validation

Evaluates a strategy's **fixed** params across in-sample/out-of-sample windows
(does **not** re-tune per fold). Two modes:

- `--is-end <date>` — single anchor split: IS=`[trading_start, is_end]`.
- `--folds <n>` — expansion-window walk-forward (IS grows from `trading_start`).

```bash
uv run ibkr bt split strats/trend_pullback_atr_enhanced.json --folds 4
uv run ibkr bt split strats/trend_pullback_atr_enhanced.json --is-end 2020-12-31 --format json
```

Reports per-fold IS/OOS ann-return, Sharpe, maxDD, calmar, win-rate, plus a
summary of mean/min OOS Sharpe and OOS→IS degradation. Useful to check whether
a strategy's edge survives out-of-sample rather than being curve-fit to the
training window.

### `bt optimize` — per-fold IS tune → OOS validate

Bridges `bt sweep` (tune params, whole window) and `bt split` (locked params).
Per fold it sweeps a param grid **on the in-sample window**, locks the best
combo, and validates it on the out-of-sample window.

```bash
uv run ibkr bt optimize strats/trend_pullback_atr_enhanced.json \
  '{"strategy_params":{"atr_mult":[1.5,2.0,2.5]}}' --folds 4
uv run ibkr bt optimize strats/trend_pullback_atr_enhanced.json \
  '{"strategy_params":{"ma_slow":[50,100,200]}}' --is-end 2020-12-31 --format json
```

Honest about overfitting: per-fold tuning curve-fits the IS window, and the OOS
result prices that cost. If mean OOS Sharpe holds up across folds the edge is
likely real; if IS is strong but OOS collapses, the grid is fitting noise.
Reports perf-fold chosen params + IS/OOS metrics, plus mean/min OOS Sharpe.
