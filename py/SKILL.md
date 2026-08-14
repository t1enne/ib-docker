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
  "strategy_type": "momentum_regime",
  "bars": ["1h", "4h"],
  "strategy_params": {
    "fast": 9,
    "slow": 30
  },
  "symbols": ["COIN", "AAPL"]
}
```

**Config field reference:**

- `training_start`/`training_end` — warmup window (data loaded for indicator/model warmup before trading)
- `trading_start`/`trading_end` — actual backtest window
- `bars` — list of bar sizes; `bars[0]` is the base signal interval, additional
  entries (e.g. `["1h", "4h"]`) are higher-timeframe bars injected alongside
  (lookahead-safe). There is no separate `htf` field.
- `commission` — fixed commission per trade
- `strategy_params` — arbitrary dict forwarded verbatim to `on_candle(state, candle, params)`;
  position sizing, stop-loss/take-profit, and any model hyper-params live here
  (strategy-owned, per-trade). There is **no** top-level `model_updater`/`model_params` —
  cross-candle model state is owned by the strategy itself

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

Write strategies that need cross-call state on the **stateful DSL**
(`@strategy(stateful=True)`): hold it in `ctx.shared`, a fresh dict the engine
mints **per run/window**, so cross-window bleed is impossible and strategies are
safe to run concurrently (including across `sweep`/`split`/`optimize` worker
processes). Model objects (e.g. `OnlinePairs`, `OnlineRegime`) live in
`ctx.shared` and are fed per candle; there is no engine `model_updater`/
`ModelState` channel.

Legacy raw `on_candle` strategies may still hold state in a module-level
`GLOBAL: dict` and expose `reset_global()` (which rebinds `GLOBAL` to a fresh
dict). The runner calls `reset_global()` defensively before each unit of work
for back-compat, but prefer the DSL — module-level state is shared across every
run in a process and is the thing the per-run `ctx.shared` holder replaces.

### Always handle "no position" and "in position" paths

```python
position = state.portfolio.positions.get(symbol)
if not position:
    # entry logic
else:
    # exit logic
```

### Use htf_candles() for multi-timeframe

Use `state.candles.get((sym, freq))` or `htf_candles(state, freq, tick)` for
higher-timeframe bars — both are cursor-truncated and lookahead-safe. There is
no `state.model_state` channel; cross-candle model/indicator state is owned by
the strategy (see `ctx.shared` for the stateful DSL earlier in this file).

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
- **HTF lookahead**: `state.candles.get((sym, freq))` and `htf_candles()` are both safe (cursor-truncated).
- **Multiple symbols**: the engine iterates all symbols per timestamp. `on_candle` fires only on the last symbol per timestamp (so `state.candles` has all symbols' data). Signals for any symbol are valid — engine routes fills by `signal.symbol`. Pending signals for non-current symbols fill when that symbol's own `_execute_pending` stage runs (same bar cycle, later in the timestamp iteration).
- **Pairs strategy** (`kalman_pairs_dsl`): requires exactly 2 symbols. The Kalman
  signal is strategy-owned — an `OnlinePairs` filter (`from src.indicators.kalman.strategy`)
  is held in `ctx.shared` and fed once per candle from `state.candles`. No engine
  `model_updater` is involved.

## Module Reference

All CLI groups under the `py` root command — also callable via `make run <subcommand> <args>`:

| Group  | Commands                   | Description                                                        |
| ------ | -------------------------- | ------------------------------------------------------------------ |
| `data` | `dl`, `query`, `preview`   | Sync/download OHLCV from IBKR, query local DB                      |
| `bt`   | `run`, `sweep`, `split`, `optimize` | Backtesting engine, hyperparam sweep, IS/OOS validation, walk-forward tuning |

### `bt sweep` — hyperparameter sweep

Sweeps a `PARAM_GRID` over a strategy and ranks every combo by a chosen metric
(`--sort-by`, default `annual_return`). Any list-valued leaf in the grid is
swept (cartesian); scalars override once. Shown `--limit` top N if given.

```bash
uv run ibkr bt sweep strat.json '{"strategy_params":{"position_size":[0.8,0.95]}}'
uv run ibkr bt sweep strat.json '{...grid...}' --sort-by sharpe_ratio --limit 5 --format json
```

Candle data loads once per distinct (symbol set, bar) and is window-sliced per
combo — no per-combo reload.

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

Run folds in parallel with `--workers N` — each IS/OOS window pair is an
independent unit of work.

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

Run folds in parallel with `--workers N` — each fold tunes its IS and
validates its OOS independently (combos inside a fold stay sequential).

### Parallelism (`--workers`)

`bt sweep`, `bt split` and `bt optimize` each accept `--workers N` (default `1` =
sequential) to parallelize their independent units of work — grid combos
(sweep) or folds (split, optimize) — over a **process pool** via
`concurrent.futures.ProcessPoolExecutor`, not threads (`src/bt/parallel.py`).
The engine is pure over immutable inputs, so separate worker processes get
isolation for free; the shared candle feed pickles once per worker, and
streaming output order is preserved.

**When to use `--workers`** (every count returns identical results; it is a
throughput knob, not a correctness one):

- There is a **fixed ~2s spawn cost** per pooled run (forkserver + workers on
  Py 3.14). The pool only helps when the units it parallelizes are expensive
  by comparison — so keep `--workers 1` for trivial runs (a few combos/folds on
  a small daily feed), where the spawn cost exceeds any savings.
- **`bt sweep`: biggest win** — parallelizes the cartesian grid. A 198-combo
  sweep measured **≈3x faster with `--workers 8`** (41s → 14s). Use it whenever
  the grid has dozens of combos or a large feed.
- **`bt split`: only with many folds / heavy feeds** — each fold is just an
  IS+OOS pair, and the spawn overhead dwarfs a few folds (a 4-fold daily split
  was *slower* pooled: 3s → 6s). Reach for `--workers` at `--folds 10+` or on
  large intraday feeds.
- **`bt optimize`: same as split** — fold-level parallelism (the in-fold IS
  sweep stays sequential). It is heavier than split (grid × folds), so it
  pays off at lower fold counts.

Effective workers are capped at the unit count, so `--workers` much larger
than the number of combos/folds wastes nothing. Rule of thumb: raise `--workers`
only once a single run would take more than a few seconds.
