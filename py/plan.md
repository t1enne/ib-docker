# Plan — Screen Module for Manual Trading Signals

> Status: Approved. Scope: build `src/screen/`, an engine-agnostic scoring/signaling
> layer for manual trading, porting existing strategy logic as the first screen.

## Context / Rationale

We want a module that scores a universe of symbols in real time (or on the latest
bar) and surfaces readable signals for a human to act on — NOT a backtest execution path.

### Why NOT call the existing strategies directly

The `src/bt/strategies/` package is a runtime payload for the backtest engine, not a
portable computation library. The evaluation shows `on_candle(state, candle, params)`
is welded to:

1. **`state: BacktestState`** — a full immutable snapshot (portfolio, pending_signals,
   CandleStore with engine-driven cursor). A screen has no portfolio and no loop cursor.
2. **`state.model_state`** — injected by a separate `model_updater_fn` running in the
   engine's Stage-2 pipeline. Strategies like `momentum_regime` (regime/vol) and
   `kalman_pairs` (kalman_z/beta) consume it. A screen has no model updater.
3. **Engine ordering** — `on_candle` only fires on the last symbol in `config.symbols`
   with the cursor pre-advanced (AGENTS.md §9). Independent iteration breaks this.
4. **Unit of output is `TradeSignal`** (a fill instruction with qty), not a ranked score.
5. **`GLOBAL`-dict + `reset_global()`** module-level mutable state (kalman `GLOBAL["kf"]`)
   — no instance scoping; live screens must reproduce reset semantics or state bleeds.

### Components that ARE reusable (reuse, don't rewrite)

- `StrategyParams` / `from_dict()` / `resolve_params()` — typed-params convention.
- `src/bt/strategies/utils.py` — `open()`/`close()` builders.
- `src/bt/regime/gates.py` — `TrendGate`, `current_trend`, `current_vol` (pure).
- `src/bt/indicators/` — underlying math (SMA, ATR, Kalman `OnlinePairs`) — already pure.
- `src/bt/state` types (`Candle`, etc.) and the `CandleStore` `Mapping[(sym, iv)] -> DataFrame`
  shape — we can build a live feed into the same shape.

No existing `screen` module exists anywhere in `src/bt/`. This is greenfield.

---

## Architecture

New package `src/screen/` — engine-agnostic, pure, Protocol-injected, tests alongside.

```
src/screen/
├── __init__.py          # public exports + auto-discovery of screens
├── types.py             # ScreenState, ScreenResult, ScreenParams, ScreenFn protocol
├── runner.py            # feed OHLCV + model fields -> run screens -> rank/merge
├── adapter.py           # data_feed / live feed -> ScreenState (I/O at the edge only)
├── screens/             # drop-file auto-discovery (mirror strategies convention)
│   ├── __init__.py      # auto-discovery (SCREEN_TYPE + on_state())
│   └── momentum.py      # port of momentum_regime (first screen, baseline)
└── tests/
    └── test_momentum_screen.py  # A/B reconciliation vs backtest momentum_regime
```

### Core types (`types.py`)

```python
@dataclass(frozen=True)
class ScreenParams(StrategyParams):
    """Reuses base from_dict(). Screen-specific scoring knobs only."""
    # ...strategy-specific knobs, e.g. momentum: fast/slow/lookback/threshold
    threshold: float = 0.8

@dataclass(frozen=True)
class ScreenState:
    """Thin, purpose-built view — NOT BacktestState."""
    ts: pd.Timestamp
    frames: tuple[tuple[str, pd.DataFrame], ...]   # (symbol, OHLCV DataFrame)
    trend: dict[str, str]       # "BULL"/"BEAR"/"RANGE"/None — from regime gates
    vol: dict[str, str]         # "LOW_VOL"/"MED_VOL"/"HIGH_VOL"/None
    # extend with z / kalman / momentum as needed per screen

@dataclass(frozen=True)
class ScreenResult:
    symbol: str
    timestamp: pd.Timestamp
    score: float                          # 0..1 signal strength, NOT a qty
    action: Literal["long", "short", "flat"]
    signals: tuple[str, ...]              # human-readable: "SMA cross up", "regime BULL"
    model_features: dict[str, float]      # verbose diagnostic; z, atr, momentum, etc.

class ScreenFn(Protocol):
    def __call__(self, state: ScreenState, params: ScreenParams) -> tuple[ScreenResult, ...]: ...
```

### Auto-discovery (`screens/__init__.py`)

Mirror `strategies/__init__.py`: scan `screens/*.py` for modules with `SCREEN_TYPE` +
`on_state()`. No manual registration. Screens return **scores**, never `TradeSignal`.

### Live feed adapter (`adapter.py`)

Reuse the existing IBKR/data_feed layer to produce `Frame`/OHLCV and precomputed model
fields (trend/vol). Build a `CandleStore`-shaped view or just the `frames` tuple. All
I/O lives here and in the CLI — nowhere in pure screen logic.

### CLI

`ibkr screen <screen> --symbols ... --interval ... [--params '{...}']` → printable ranked
table (`src/bt/table.py`). No plotting, plain-text rank output.

---

## Increment 1 — First screen: `momentum` (baseline / A/B)

Port `momentum_regime`'s scoring logic into `Screens/momentum.py`.

Rationale: its math (SMA cross + regime gate + momentum filter) is already pure, and its
only engine dependency is `model_state` (trend/vol), which a screen computes itself via
`regime.gates.current_trend/current_vol` from live frames. Cleanest A/B baseline.

- Reuse `screens.types.ScreenParams`; re-use the `momentum_regime.PParams` fields
  (fast/slow, momentum_lookback/threshold, regime flags, size multipliers) as scoring knobs.
- `on_state()` returns `ScoreResult` when a long/short entry condition triggers,
  with `score` proportionate to momentum magnitude _post_-regime-gate.
- Warmup gating identical to source (`warmup_bars`, `slow` windows) — no look-ahead.

### A/B reconciliation (the validation contract)

Screen must match the backtest strategy's _decision_ on the same symbols/window.

- Run backtest `momentum_regime` → collect entry `TradeSignal` timestamps/symbols.
- Run screen over the same updated data → collect `ScreenResult`s.
- Assert: **hit-rate parity** — every backtest entry the screen also flags non-flat, and
  screen "long/short" never contradicts backtest entry direction.
- Reasonable mismatch with the backtest's flat-stance (regime RANGE flat) must be zero on
  IS, tracked on OOS.
- Out-of-sample discipline: run across walk-forward IS/OOS windows (reuse `src/bt/split`).
  A screen that "fires" where the strategy was flat is a look-ahead symptom, not an edge.

### Cost model / honesty note

For _scoring_, a screen is pre-cost by design. State explicitly in output that a high
score ≠ profit. Only model fee+slippage if/when a screen hit is converted into an actual
execution path (future work, not in this increment).

---

## Deliverables / Definition of Done

1. `src/screen/types.py` — `ScreenState`, `ScreenResult`, `ScreenParams`, `ScreenFn`.
2. `src/screen/screens/momentum.py` — ported scoring, auto-discovered.
3. `src/screen/screens/__init__.py` — auto-discovery.
4. `src/screen/runner.py` — orchestrates feed→state→screens→rank.
5. `src/screen/adapter.py` — data_feed/IBKR → `ScreenState`.
6. CLI: `ibkr screen`
7. `src/screen/tests/test_screens.py` — edge cases (empty frame, single bar, warmup,
   boundary timestamps) + `tests/test_momentum_screen.py` A/B reconciliation vs backtest.
8. `make check` green (lint + format + typecheck `ty` + full typing).
9. `make test-fast` passes.

## Out of Scope (future increments)

- Additional screens (sector_mean_reversion, kalman_pairs, shannons_demon) — port after
  momentum A/B passes.
