# IBKR PY — Composable Quantitative Trading Toolkit

Modular CLI toolkit for quantitative trading: data synchronization, indicator computation, statistical models, and a functional backtesting engine. Configure strategies in JSON, run from the terminal, iterate fast.

## Quickstart

```bash
uv sync
uv run ibkr bt run strats/trend.json
make run bt run strats/trend.json   # same, via Make shortcut
```

`make run` with no args shows available commands:

```bash
make run bt -- --help              # Click help for backtesting subgroup
```

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

# Data
uv run ibkr data query AAPL --from 2024-01-01

# Pipe workflows
uv run ibkr data query AAPL --from 2024-01-01 | uv run ibkr bt run strategy.json
```

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
