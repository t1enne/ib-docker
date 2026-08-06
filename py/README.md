# IBKR PY — Composable Quantitative Trading Toolkit

Modular CLI toolkit for quantitative trading: data synchronization, indicator computation, statistical models, and a functional backtesting engine. Configure strategies in JSON, run from the terminal, iterate fast.

## Quickstart

```bash
uv sync
uv run ibkr bt run strats/trend_pullback_atr_enhanced.json
make run bt run strats/trend_pullback_atr_enhanced.json   # same, via Make shortcut
```

`make run` with no args shows available commands:

```bash
make run bt -- --help              # Click help for backtesting subgroup
```

**Make eats its own flags.** To pass CLI options through to `ibkr` (`--format`, `--help`, `-F`)
you must separate them with `--`:

```bash
make run bt run strats/trend_pullback_atr_enhanced.json -- --format jsonl
```

Run `make run <args>` for zero extra typing; use `uv run ibkr <args>` directly to avoid the `--` syntax.

## Architecture

Pure functions + immutable state. No classes, no side effects in the hot path.

### Pipeline (per candle)

```
append_candle → update_model → execute_pending → generate_signals
     → execute_pending (same-bar) → check_risk → mark_to_market
```

- Signals fill at next bar's open (fill_at_next_open) or same bar (immediate).
- Risk checks (stop-loss / take-profit) run after same-bar fills, so they always see the post-rebalance position.
- Higher-timeframe candles accumulate alongside base candles; HTF-only candles skip the pipeline.

### Composition

Strategies are plain modules with `on_candle(state, candle, params) → list[TradeSignal]`.
Handlers (`ExecutionHandler`, `RiskHandler`) are dataclasses wrapping injectable functions.
Auto-discovery scans `src/bt/strategies/` — any `.py` with `STRATEGY_TYPE` + `on_candle()` is registered.

### Multi-Symbol & Multi-Interval Data Flow

#### Per-timestamp iteration

The candle generator yields every symbol at every timestamp, then interleaves any
higher-timeframe (HTF) candles sharing the same timestamp. Base-interval candles
run through the full pipeline; HTF-only candles are merely accumulated and skip
the rest.

With `symbols: ["AAPL", "GOOGL", "MSFT"]` and `bars: ["1h", "4h"]` each
1h timestamp produces:

```
AAPL:1h  → full pipeline (append, model, signals, risk, mtm)
GOOGL:1h → full pipeline
MSFT:1h  → full pipeline
AAPL:4h  → append only (if 4h boundary)
GOOGL:4h → append only (if 4h boundary)
MSFT:4h  → append only (if 4h boundary)
```

#### `on_candle` invocation rule

`on_candle` fires **only on the last symbol per timestamp** (i.e. `candle.symbol
== symbols[-1]`). The `candle` parameter carries that symbol's OHLCV + interval.

**Why:** This guarantees `state.candles` has accumulated every symbol's data
up to the current timestamp, so the strategy can read cross-sectional (multi-symbol)
decisions in one shot. If the engine fired `on_candle` on every symbol,
the first symbol's invocation would see incomplete data for later symbols.

#### `state.candles` — the CandleStore

All candles — base and HTF — accumulate in `state.candles`, a `Mapping[(str, str),
DataFrame]` keyed by `(symbol, interval)`. It is wrapped in a `CandleStore` that
supports lookahead-safe access:

```python
# Get DataFrame up to current timestamp (safe — no future data)
df = state.candles[("AAPL", "1h")]
df = state.candles.get(("AAPL", "4h"))           # None if not present

# O(1) fast path — latest close or bar count (no DataFrame allocation)
close = state.candles.latest("AAPL", "1h")       # float | None
n     = state.candles.count("AAPL", "4h")         # int
```

Before `on_candle` fires, the engine calls `state.candles.advance(current_ts)`,
which sets a cursor. All DataFrame-building access (`__getitem__`, `get`,
iteration) truncates rows beyond the cursor. Fast-path methods (`latest`,
`count`) ignore the cursor and return the absolute latest value.

#### `candle.interval` — which bar is this?

`candle.interval` is `"1h"` for base bars, `"4h"` for HTF bars, etc. Multi-interval
strategies use this to gate logic:

```python
def on_candle(state, candle, params):
    if candle.interval != params.signal_interval:
        return []           # only act on the interval that matters

    # Read HTF data from the store
    htf_df = state.candles.get((candle.symbol, "4h"))
    if htf_df is not None and len(htf_df) > 0:
        htf_close = htf_df["close"].iloc[-1]
```

#### Multi-symbol strategy pattern

Read all symbols from `state.candles`, emit signals for any symbol. Returned
signals are bucketed by `signal.symbol` into a dict — the engine drains each
symbol's queue when its candle iteration reaches the execution stage:

```python
def on_candle(state, candle, params):
    # Rank all symbols by momentum
    closes = {}
    for sym in params.symbols:
        df = state.candles.get((sym, candle.interval or "1h"))
        if df is not None and len(df) >= params.warmup:
            closes[sym] = df["close"].iloc[-1]

    # Emit signals for any symbol — engine routes by signal.symbol
    signals = []
    for sym in sorted(closes, key=lambda s: closes[s])[:params.top_n]:
        signals.append(open(candle, ActionType.long, "[sector] top rank"))
    return signals
```

Signals carry `signal.symbol` — the engine routes each signal to its
symbol's bucket in `state.pending_signals` (a `dict[str, tuple[TradeSignal,
...]]`). Execution is O(1) per symbol, no scan over unrelated signals.

```
src/
├── data/             ← IBKR market data sync, DB, resampling
├── indicators/       ← TA, Kalman filters, HMM regime detection
├── bt/               ← Backtesting engine
│   ├── engine/       ← Backtest loop (pure functional), candle store
│   ├── strategies/   ← Strategy implementations (auto-discovered)
│   ├── state/        ← Immutable dataclasses + factories
│   ├── types.py      ← StrategyConfig, EngineWindow, Protocol types
│   ├── portfolio/    ← Position sizing, fill application, mark-to-market
│   ├── execution/    ← Signal → fill with slippage/spread
│   ├── risk/         ← Stop-loss/take-profit checks
│   ├── regime/       ← Regime detection (HMM, trend, volatility)
│   ├── metrics.py    ← Sharpe, Sortino, Calmar, drawdowns
│   └── data_feed/    ← Load/sync OHLCV candles
├── utils.py          ← Shared utilities
└── main.py           ← CLI entry point (Click)
```

## CLI Reference

```bash
# Backtesting
uv run ibkr bt run <strategy.json> [--format jsonl]
uv run ibkr bt analyze <strategy.json>
uv run ibkr bt split <strategy.json> --folds 4          # IS/OOS walk-forward
uv run ibkr bt split <strategy.json> --is-end 2020-12-31  # single anchor split

# Data
uv run ibkr data query AAPL --from 2024-01-01

# Pipe workflows
uv run ibkr data query AAPL --from 2024-01-01 | uv run ibkr bt run strategy.json
```

### `bt split` — in-sample vs out-of-sample validation

Evaluates a strategy's **fixed** parameter set across IS/OOS windows. Two modes:

- `--is-end <date>` — single anchor split: IS=`[trading_start, is_end]`, OOS=`[is_end+1d, trading_end]`.
- `--folds <n>` — expansion-window walk-forward: IS always starts at `trading_start` and grows, producing `n` non-empty folds.

```bash
uv run ibkr bt split strats/trend_pullback_atr_enhanced.json --folds 4
uv run ibkr bt split strats/trend_pullback_atr_enhanced.json --is-end 2020-12-31 --format json
```

Options: `--min-is-years` (walk-forward first-fold history floor, default 5.0),
`--train-start` (warmup override), `--format text|json`. Does **not** re-tune
params per fold — it answers *"given these locked params, how does performance
hold up out-of-sample?"*

All commands usable via `make run <subcommand> <args>`.

## Toolchain

| Tool                        | Purpose            |
| --------------------------- | ------------------ |
| Python 3.14+                | Runtime            |
| `uv`                        | Package management |
| `ty`                        | Type checking      |
| `ruff format`               | Formatting         |
| `pytest` + `pytest-asyncio` | Testing            |
| `make`                      | Script runner      |

```bash
make check       # lint + format + typecheck + tests
make test        # all tests
make test-fast   # quick tests
```

## Key Types

```python
# StrategyConfig — loaded from JSON via load_strategy()
@dataclass
class StrategyConfig:
    name: str; strategy_type: str; symbols: list[str]
    stop_loss: float; take_profit: float
    initial_capital: float; position_size: float; commission: float
    training_start: str; training_end: str
    trading_start: str; trading_end: str
    bars: list[str]; strategy_params: dict; model_params: dict
    model_updater: dict | bool; rolling_window_size: int | None
    hmm_floating_window: int | None; hmm_retrain_interval: int | None
    benchmark_symbols: list[str]

# BacktestResults — returned by run()
@dataclass(frozen=True)
class BacktestResults:
    pf: PortfolioResult         # all metrics + trades + equity curve
    data: dict                  # candles dict keyed by symbol
    final_state: BacktestState  # full state at end of backtest
    benchmark_curves: dict[str, pd.Series]
```

### Programmatic API

```python
from src.bt import load_strategy, Backtest, run, get_backtest_results_analysis
from src.bt.data_feed import load_candles
from src.bt.strategies import init_strat

config = load_strategy("strats/momentum_regime.json")
bt = Backtest(config)
df = load_candles(config.symbols, bt.window.train_start, bt.window.test_end, config.bars[0])
results = run(bt, df, strat_mod=init_strat(config.strategy_type))
print(get_backtest_results_analysis(results.pf))
```

## IBKR REST API Client

The `ib-rest-api-client` package is a local dependency (`../ib-rest-api-client`), generated from `openapi.spec.json` (IBKR's OpenAPI spec).

### Setup

Run the Gateway login script before any data sync or API call:

```bash
export IBKR_USERNAME=...
export IBKR_PASSWORD=...
export TRADING_MODE=paper          # or "live"

uv run python scripts/login_ibkr.py
```

This logs into IBKR Client Portal Gateway via Playwright. The Gateway must be running (see `../client-portal/`).

### Regenerating the Client

After an API spec update:

```bash
cd ../ib-rest-api-client
uvx openapi-python-client generate --path ../py/openapi.spec.json --output-dir .
cd ../py && uv sync
```

## Resources

- **Strategy authoring & backtesting workflow**: [SKILL.md](SKILL.md)
- **Coding standards for contributors**: [AGENTS.md](AGENTS.md)
