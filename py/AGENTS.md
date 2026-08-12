# AGENTS.md — IBKR PY

> AI coding agent instructions for this repository.
> Read this before generating any code.

## Alpha research

Read SKILL.md

## Development Workflow

When implementing a feature or fix:

1. **Understand** — read relevant strategy code, types, and tests. Don't guess.
2. **Plan** — state approach before writing. If unclear, ask.
3. **Implement** — minimum code that works. Pure functions, immutable state, full type annotations.
4. **Test** — every new computation gets a test. Strategies, scripts and research doesn't need a test. Run `make test` before declaring done.
5. **Verify** — `make check` must pass (lint + format + typecheck + tests).

### Running things

```bash
uv run ibkr bt run strats/trend.json       # CLI entry point
uv run ibkr data query SPY                 # Query SPY data from the local DB
make run bt run strats/trend.json          # Make shortcut
make check                                 # lint + format + typecheck + test
make test                                  # all tests
make test-fast                             # quick tests
```

## Language & Toolchain

- **Python 3.14+** (required)
- **Package manager:** `uv` (not pip)
- **Type checker:** `ty`
- **Formatter:** `ruff format`
- **Test runner:** `pytest` with `pytest-asyncio`
- **Script Runner:** `make`
- **CLI binary:** `ibkr` (installed via `uv sync` from `pyproject.toml` entry point)

## Core Principles

### 1. Type Safety (NON-NEGOTIABLE)

Every function, method, and dataclass MUST have complete type annotations.
No `Any` unless truly unavoidable — and even then, comment why.

```python
# ✅ Good — fully typed
def calculate_zscore(
    prices_a: pd.Series,
    prices_b: pd.Series,
    window: int = 75,
) -> pd.Series: ...

# ❌ Bad — missing types
def calculate_zscore(prices_a, prices_b, window=75): ...

# ❌ Bad — Any escape hatch
def process(data: Any) -> Any: ...
```

#### Rules

- **Use `Protocol` for dependency injection** — the codebase uses `StrategyFn`, `ExecutionFn`, `RiskCheckFn`, `DataLoaderFn`. Follow this pattern. Never pass raw `Callable` when a Protocol exists or should exist.
- **Use `@dataclass(frozen=True)` for state.** Immutable state makes backtesting deterministic and testable. See `Tick`, `PortfolioState`, `BacktestState`, `FillEvent`, etc.
- **Use `Literal` for enums of strings.** Prefer `Literal["long", "short", "close"]` over bare `str`.
- **Use `TypedDict`** for structured dicts when a dataclass would be overkill.
- **No `object` or bare `dict`** as parameter/return types.
- **`TYPE_CHECKING` guard** for import-only types to avoid circular imports.
- **Top level imports only** No lazy imports.

```python
# ✅ Pattern: TYPE_CHECKING for type-only imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.bt.types import PlotConfig
```

### 2. Performance

This is a quantitative finance backtesting system — hot paths matter.

```python
# ✅ Vectorized — entire Series at once
z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()

# ❌ Loop over rows
z = []
for i in range(len(spread)):
    window = spread[i-window_size:i]
    z.append((spread[i] - window.mean()) / window.std())
```

#### Rules

- **Vectorize with pandas/numpy.** A loop over ticks is acceptable in the backtest engine's main loop (necessary for event sequencing). A loop over data points inside a computation is not.
- **Avoid `pd.DataFrame.append` / `pd.concat` in loops.** See `_append_candle` — this is a known hotspot. Pre-allocate or batch.
- **Use `tuple` not `list`** for immutable sequences (see `BacktestState.risk_events: Tuple[Any, ...]`).
- **Dataclass `field(default_factory=list)`** is fine — but `Tuple[...]` is preferred for state types.
- **Lazy imports** for heavy modules (plotting, HMM) inside functions, not at module top-level.
- **Profile before optimizing.** Don't guess.
- **Avoid** writing expensive tests. `make check` should run under 10s. So no HMM, ML or other long-running computations in tests.
- **Before running** expensive tasks, verify it runs and outputs the expected data. No running a 300s task to discover that it doesn't print anything, or we truncated the needed output by misusing `head` or `tail`.

### 3. Testing

Tests live alongside code in `tests/` subdirectories, named `test_*.py`.

```bash
uv run pytest                                  # all tests
uv run pytest src/bt/engine/tests/ -v          # specific module
```

#### Rules

- **Every new feature/computation gets a test.** Only strategies, scripts and research are excluded.
- **Use `respx`** for mocking HTTP calls (IBKR API layer).
- **Use `pytest-mock`** for general mocking.
- **Test pure functions first** — they're the easiest and most valuable to test.
- **Test edge cases:** empty DataFrames, single-row windows, NaN handling, boundary timestamps.
- **Test determinism:** given the same inputs, backtest results must be identical. Use fixed seeds and fixed data.
- **Test files go in the module's `tests/` directory**, e.g., `src/bt/risk/tests/test_risk.py`.

```python
# ✅ Good test — exercises a pure function, tests edge cases
def test_zscore_empty_series():
    result = calculate_zscore(pd.Series([], dtype=float))
    assert len(result) == 0

def test_zscore_constant_series():
    s = pd.Series([5.0] * 100)
    result = calculate_zscore(s, window=20)
    assert result.dropna().abs().max() < 1e-10  # ~zero z-score
```

### 4. Functional Patterns

The codebase is built around **immutable state + pure functions**.

```python
# ✅ Pure function — returns new state
def apply_fill(portfolio: PortfolioState, fill: FillEvent) -> PortfolioState:
    new_positions = {**portfolio.positions, fill.signal.symbol: new_position}
    return replace(portfolio, positions=new_positions, cash=new_cash)

# ❌ Mutation
def apply_fill(portfolio: PortfolioState, fill: FillEvent) -> None:
    portfolio.positions[fill.signal.symbol] = new_position  # mutates!
```

#### Rules

- **`replace()` for state updates.** Use `dataclasses.replace()` when modifying frozen dataclasses.
- **Return new state, never mutate.** Every function in the pipeline takes state in, returns new state out.
- **Compose functions,** don't chain methods. The backtest engine composes `model_updater_fn → strategy_fn → exec_handler → risk_handler`.
- **Protocol-based injection** over class inheritance. Strategy logic is a plain module with `on_candle()`. The engine type-annotates it as `StrategyFn`.
- **No side effects in pure functions.** I/O (DB, HTTP, file) belongs at the edges.
- **Use `merge_bt_state`** for partial state updates — it's the established pattern.

##### Mutable strategy state: the `GLOBAL` + `reset_global()` convention

Strategies are modules, so any cross-call mutable state (cooldowns, trails,
cache dicts, sets, bar counters, last-timestamp markers) is module scope and
persists across runs. The backtest engine never resets it, so state bleeds
between walks/windows unless the strategy opts into a reset. Use this
convention everywhere strategy runtime state exists:

```python
GLOBAL: dict = {"cooldown": {}, "regime_ts": None}

def reset_global() -> None:
    global GLOBAL
    GLOBAL = {"cooldown": {}, "regime_ts": None}
```

- Keep **all** cross-call state in one module-level `GLOBAL: dict` — no
  scattered `_COOLDOWNS`/`_TRAILS`/`_last` module globals.
- `GLOBAL` may hold non-dict members (`set`, `int`, `Timestamp | None`, ...).
  That's fine — reset **rebinds** `GLOBAL` to a fresh dict rather than
  clearing members with heuristics.
- Mutate members in place inside strategy logic (`GLOBAL["cooldown"][sym] = x`,  
  `GLOBAL["bar_idx"] += 1`); no `global` statement is needed unless you rebind  
  `GLOBAL` itself (only `reset_global` does).
- Provide `reset_global()` that rebuilds `GLOBAL` with correct defaults. The  
  split engine calls it via `_reset_strategy_state()` before every IS/OOS  
  window. Stateless strategies simply omit it.
- Don't hand-roll a heuristic reset (e.g. clearing only uppercase dict-like  
  globals) — it silently misses non-dict state and lowercase names and is a  
  source of cross-fold bleed (see `split.py` history).

#### Size constraints (from README)

- Functions: **≤ 50 LOC**
- Classes: **≤ 150 LOC**
- If you're exceeding these, extract or decompose.

### 5. Code Organization

```
src/bt/
├── state/types.py      # All immutable state dataclasses
├── types.py            # Protocols, StrategyConfig, enums
├── engine/             # Backtest loop (pure functional)
├── strategies/              # Strategy implementations
├── models/             # Z-score, regime, market data models
├── portfolio/pure.py   # Pure functions for position/PnL
├── execution/pure.py   # Signal → fill execution
├── risk/pure.py        # Stop-loss / take-profit checks
├── indicators.py       # Technical indicators (pure functions)
└── metrics.py          # Performance metrics (pure functions)
```

#### Module naming

- `pure.py` = stateless functions. No classes, no side effects.
- `types.py` = type definitions, Protocols, dataclasses.
- `__init__.py` = public API exports and wiring/DI.

### 6. Negative-Space Programming

Use assertions instead of early returns where it improves clarity.

```python
# ✅ Assert preconditions
def on_tick(self, state: BacktestState, tick: Tick, params: dict) -> list[TradeSignal]:
    z = state.model_state.z_score
    assert z is not None, "Z-score must be computed before signal generation"
    assert tick.symbol in params["symbols"], f"Unexpected symbol: {tick.symbol}"
    # ... logic
```

Assertions document invariants. They're also free runtime checks during tests.

### 7. Configuration & Strategy JSON

Strategies are defined in JSON files loaded via `load_strategy()` → `StrategyConfig`.

- Add new fields to `StrategyConfig` dataclass, not as loose dict entries.
- Validate config early. The codebase currently lacks schema validation — when adding validation, use dataclass field constraints or a schema library, not ad-hoc checks scattered across the codebase.

### 8. Async I/O

- The IBKR data sync layer uses `httpx` with `asyncio`.
- The backtest engine itself is synchronous (pure computation). Async only at the I/O boundary.
- Use `pytest-asyncio` for testing async code.
- Use `respx` for mocking HTTP in async tests.

### 9. Engine Data Flow to `on_candle`

Understanding how the engine feeds data to strategies is critical for writing
correct multi-symbol and multi-interval strategies.

#### `on_candle` fires once per timestamp

The engine calls `on_candle(state, candle, params)` **only when `candle.symbol`
is the last symbol** in `config.symbols`. With `["AAPL", "GOOGL", "MSFT"]`,
the generator yields → AAPL → GOOGL → MSFT per timestamp before moving to the
next timestamp. `on_candle` fires on MSFT.

**Why:** At that point `state.candles` contains all symbols' data up to the
current timestamp. If the engine fired on every symbol, the first symbol's
invocation would see incomplete data (later symbols haven't been appended yet).

#### Parameter reference

- **`state: BacktestState`** — full snapshot (portfolio, model_state, candles, pending_signals)
- **`candle: Candle`** — the OHLCV bar for the last symbol at current timestamp. Has `.symbol`, `.interval` (`"1h"`, `"4h"`, etc.), `.open`, `.high`, `.low`, `.close`, `.volume`
- **`params`** — typed dataclass (`StrategyParams` subclass) resolved by `resolve_params(config.strategy_type, config.strategy_params)`; or raw `dict` if no typed params registered

#### CandleStore: `state.candles`

The primary data interface. A `Mapping[(str, str), DataFrame]` keyed by
`(symbol, interval)`. Backed by a cursor that ensures lookahead safety:

```python
# DataFrame access (cursor-truncated — safe, no future data)
df = state.candles[("AAPL", "1h")]              # KeyError if missing
df = state.candles.get(("AAPL", "4h"))          # None if missing

# O(1) fast path — absolute latest, no DataFrame allocation
close = state.candles.latest("AAPL", "1h")      # float | None
n     = state.candles.count("AAPL", "4h")        # int
```

**Cursor semantics:** The engine calls `state.candles.advance(ts)` before each
`on_candle` invocation. `__getitem__` and `get` build DataFrames truncated to
rows ≤ cursor. `latest()` and `count()` ignore the cursor (absolute latest).

**Do NOT** access `state.model_state.resample_cache` directly — it's internal
accumulator state and not lookahead-safe.

#### HTF (higher-timeframe) access pattern

HTF candles accumulate in `state.candles` keyed by their interval string.
The candle generator interleaves HTF candles (e.g. `"4h"`) at boundaries
after all base candles for that timestamp. Use the same CandleStore interface:

```python
def on_candle(state, candle, params):
    # Only act on signal-interval bars
    if candle.interval != params.signal_interval:
        return []

    # Read HTF trend from the store
    htf_df = state.candles.get((candle.symbol, "4h"))
    if htf_df is not None and len(htf_df) >= 2:
        htf_trend = htf_df["close"].iloc[-1] > htf_df["close"].iloc[-2]
```

HTF-only candles (where `candle.interval != base_interval`) **skip the
pipeline** — they are appended to the accumulator but never trigger
`on_candle`, model updates, signal execution, or risk checks.

#### Multi-symbol strategy pattern

Read cross-sectional data from `state.candles` and emit signals for any symbol.
Returned signals are bucketed by `signal.symbol` into `state.pending_signals`
(a `dict[str, tuple[TradeSignal, ...]]`). The engine drains each symbol's bucket
when that symbol's candle iteration reaches Stage 4.

Key rules:

- Signals for any symbol are valid — the engine dispatches fills by `signal.symbol`.
- Use `candle.interval` to gate logic when multiple intervals are configured.
- `_execute_pending` (Stage 4/6) reads directly from `state.pending_signals[symbol]`.
  No O(N) scan over all pending signals — routing is explicit and O(1).
- **Same-bar execution:** Signals emitted during `on_candle` for non-current
  symbols will fill in the same bar cycle — the corresponding symbol's
  `_execute_pending` stage runs immediately before its `_generate_signals`.

#### `qty` in TradeSignal — two modes

- **`qty = 0` (default):** For `ActionType.long`/`short`, quantity is computed
  from `config.position_size * portfolio.cash / candle.close`. For `close`,
  the full position is closed.
- **`qty > 0`:** When set explicitly (e.g. from `config.position_size` in
  `sector_mean_reversion`), the execution handler scales it:
  `qty * portfolio.cash / candle.close`. This is the **base position size**
  (0.0–1.0), not an absolute share count.

## Quick Checklist Before Committing Code

- [ ] Run `make check` for formatting, lintin, typechecking and testing
- [ ] All functions/classes fully type-annotated (no `Any` without comment)
- [ ] State changes return new objects (no mutation)
- [ ] Hot-path computation is vectorized (numpy/pandas, not Python loops)
- [ ] New logic has tests covering edge cases
- [ ] Functions ≤ 50 LOC, classes ≤ 150 LOC (or extracted)
- [ ] Protocols used for injection, not inheritance
- [ ] No dead code or commented-out blocks left behind
