# AGENTS.md — IBKR PY

> AI coding agent instructions for this repository.
> Read this before generating any code.

## Language & Toolchain

- **Python 3.14+** (required)
- **Package manager:** `uv` (not pip)
- **Type checker:** `ty`
- **Formatter:** `ruff format`
- **Test runner:** `pytest` with `pytest-asyncio`

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

### 3. Testing

Tests live alongside code in `tests/` subdirectories, named `test_*.py`.

```bash
uv run pytest                                  # all tests
uv run pytest src/bt/engine/tests/ -v          # specific module
```

#### Rules

- **Every new feature/computation gets a test.** No exceptions.
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
- **Protocol-based injection** over class inheritance. Strategy logic doesn't extend a base class; it implements `StrategyProtocol` or `StrategyFn`.
- **No side effects in pure functions.** I/O (DB, HTTP, file) belongs at the edges.
- **Use `merge_bt_state`** for partial state updates — it's the established pattern.

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
├── algos/              # Strategy implementations
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

### 7. Configuration & Strategy YAML

Strategies are defined in YAML files loaded via `load_strategy()` → `StrategyConfig`.

- Add new fields to `StrategyConfig` dataclass, not as loose dict entries.
- Validate config early. The codebase currently lacks schema validation — when adding validation, use dataclass field constraints or a schema library, not ad-hoc checks scattered across the codebase.

### 8. Async I/O

- The IBKR data sync layer uses `httpx` with `asyncio`.
- The backtest engine itself is synchronous (pure computation). Async only at the I/O boundary.
- Use `pytest-asyncio` for testing async code.
- Use `respx` for mocking HTTP in async tests.

## Quick Checklist Before Committing Code

- [ ] All functions/classes fully type-annotated (no `Any` without comment)
- [ ] State changes return new objects (no mutation)
- [ ] Hot-path computation is vectorized (numpy/pandas, not Python loops)
- [ ] New logic has tests covering edge cases
- [ ] Functions ≤ 50 LOC, classes ≤ 150 LOC (or extracted)
- [ ] `ruff format` passes
- [ ] `ty check` passes on changed files
- [ ] Protocols used for injection, not inheritance
- [ ] No dead code or commented-out blocks left behind
