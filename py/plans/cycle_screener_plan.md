# Macro Cycle Screener — Enriched Implementation Plan

> **Status:** Ready for handoff to a coding agent.
> **Context:** This is a **standalone screen** (not a backtest strategy) that follows the
> same pattern as `screens/breakout_screen.py`. It is called by the `src/screen/__init__.py`
> runner. The backtest engine is **not** involved.

---

## 1. Problem Statement

Build a market-driven cycle detection engine that answers:
_"What macro regime is the market pricing today, and how is that regime changing?"_

Operates on **40+ cross-asset tickers** (equities, sectors, fixed income, commodities, currencies, volatility). Not a backtest — a Finviz-style screener producing a ranked regime output.

---

## 2. How Screens Work in This Codebase (Architecture Contract)

### 2.1 Module Discovery

Screen modules live in `screens/` (sibling to `src/`). The runner `src/screen/__init__.py` discovers them by scanning that directory:

```python
# src/screen/__init__.py
_SCREENS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "screens"
))
```

### 2.2 Required Exports

Every screen module MUST export:

| Export | Type | Purpose |
|--------|------|---------|
| `DEFAULTS` | `dict[str, Any]` | Module-level default parameters, merged with CLI `--param` overrides |
| `make(symbols, params)` | `(list[str], dict) -> ScreenFn` | Factory that returns a `ScreenFn` instance |

No base class to extend — duck typing validated at import time.

### 2.3 ScreenFn Protocol

Defined in `src/screen/types.py`:

```python
class ScreenFn(Protocol):
    def compute(self, symbol: str, candles: pd.DataFrame) -> ScreenResult: ...
    def rank(self, results: list[ScreenResult]) -> list[ScreenResult]: ...
```

### 2.4 ScreenResult / ScreenOutput

**`ScreenResult`** (frozen dataclass, `src/screen/types.py`):

```python
@dataclass(frozen=True)
class ScreenResult:
    symbol: str
    signal: Literal["long", "short", "neutral"]
    score: float           # 0-1, higher = more attractive
    price: float
    metadata: dict[str, Any]  # key-value pairs rendered in the table
```

- `metadata` keys appear as table columns in terminal output.
- Keys in `PCT_DISPLAY_KEYS` (`src/screen/style.py`) are auto-formatted with `%` sign + green/red coloring.

**`ScreenOutput`** — returned by `run_screen()`, consumed by `print_screen_output()`.

### 2.5 Construction / Data Flow

```
make(symbols, params)
    → ScreenFn instance (pre-load benchmark data if needed)
    → instance.compute(symbol, candles_df)  ← called once per symbol
    → returns ScreenResult
    → instance.rank(all_results) → sorted list
    → print_screen_output(ScreenOutput) → terminal table
```

Key detail: `compute()` receives **hourly candles** for the symbol. If you need daily, resample:

```python
from src.market_data import resample_ohlcv
candles = resample_ohlcv(hourly_candles, "1d")
```

Benchmark data should be **pre-loaded in `__init__`** (pattern from `breakout_screen.py` lines 115-130):

```python
def __init__(self, symbols, params):
    self._benchmark_candles = None
    bench = params.get("benchmark_symbol")
    if bench:
        from src.utils import get_local_candles
        self._benchmark_candles = get_local_candles(symbol=bench, bar="1d")
```

---

## 3. Key Files to Reference

| File | What it provides |
|------|-----------------|
| `src/screen/types.py` | `ScreenFn`, `ScreenResult`, `ScreenOutput` |
| `src/screen/style.py` | `Fmt`, `Style`, `PCT_DISPLAY_KEYS` |
| `src/screen/__init__.py` | `import_screen`, `run_screen`, `print_screen_output`, `discover_screens` |
| `src/utils.py` | `get_local_candles(symbol, start=None, end=None, bar="1h")` |
| `src/market_data/__init__.py` | `resample_ohlcv(df, freq)` → resamples to daily |
| `src/bt/indicators.py` | Pure functions: `sma`, `ema`, `rsi`, `atr`, `adx` |
| `screens/breakout_screen.py` | Reference implementation — follow this pattern |
| `src/hmm/hmm.py` | `MarketRegimeHMM` + `create_regime_features` (existing HMM infrastructure) |

---

## 4. Implementation Plan (Single Screen Module)

**File:** `screens/cycle_screener.py`

### 4.1 Module Structure

```python
"""cycle_screener.py — Macro cycle regime detection screen."""

# ── Imports ──
from src.market_data import resample_ohlcv
from src.screen.types import ScreenFn, ScreenResult
from src.bt.indicators import sma, ema
import numpy as np
import pandas as pd

# ── DEFAULTS ──
DEFAULTS: dict[str, Any] = {
    "benchmark_symbol": "SPY",
    "lookback_days": 252,   # ~1 year of daily data
    "regime": "auto",        # "auto" | "recovery" | "expansion" | ...
}

# ── Asset universe (hardcoded: ~40 tickers across asset classes) ──

# ── Layer functions (pure, private) ──

# ── CycleScreen class ──

# ── make() factory ──
def make(symbols: list[str], params: dict[str, Any]) -> CycleScreen:
    return CycleScreen(symbols=symbols, params=params)
```

### 4.2 Key Design Decision: What Is the "Symbol" Being Screened?

The plan describes a **singular macro regime** score across the entire cross-asset universe — not per-ticker. However, the `ScreenFn.compute(symbol, candles)` interface expects one result per symbol.

**Recommended architecture for v1:**

The screen analyzes the full cross-asset universe as a single entity. It produces ONE result (pseudo-symbol `"MACRO"`) with rich metadata.

```python
def compute(self, symbol: str, candles: pd.DataFrame) -> ScreenResult:
    if symbol != self._primary_symbol:
        return ScreenResult(symbol=symbol, signal="neutral", score=0.0,
                          price=0.0, metadata={"reason": "not_primary"})
    # ... compute macro regime from full universe
```

Where `self._primary_symbol` is the first ticker in `symbols` (or a sentinel like `"SPY"`). All other symbols are skipped with a minimal result, OR they are used to cross-check the regime report (optional v2 enhancement).

**Alternative (simpler):** Only pass ONE symbol to the screen (e.g., `SPY`), and the screen internally fetches all 40 cross-asset tickers via `get_local_candles`. The screen then acts as a "universe aggregator" that ignores the symbol input entirely.

**Recommendation:** Use the simpler approach. `compute()` ignores its `symbol` argument and computes the macro regime from an internal hardcoded universe of 40 cross-asset tickers. This is clean and avoids sending dummy results for 39 symbols.

### 4.3 Implementation Layers

#### Layer 1: Asset Universe

Hardcode the 40+ tickers in the module as constants. Categories:

```python
EQUITY_INDICES = ["SPY", "QQQ", "IWM"]
SECTORS = ["XLF", "XLK", "XLI", "XLB", "XLE", "XLV", "XLP", "XLU", "XLRE", "XLC"]
TREASURIES = ["SHY", "IEI", "IEF", "TLT"]
CREDIT = ["HYG", "JNK", "LQD"]
COMMODITIES = ["CPER", "GLD", "SLV", "USO", "UNG"]
CURRENCY = ["UUP"]  # DXY proxy
VOLATILITY = ["VIX", "VVIX"]  # note: VIX data availability varies
```

`compute()` fetches all of these internally regardless of what symbol is passed.

#### Layer 2: Data Loading in `compute()`

```python
def compute(self, symbol: str, candles: pd.DataFrame) -> ScreenResult:
    # Fetch all cross-asset data
    asset_data: dict[str, pd.DataFrame] = {}
    for ticker in ALL_ASSETS:
        df = get_local_candles(symbol=ticker, bar="1d")
        if df.empty:
            continue
        asset_data[ticker] = df

    # Resample to daily close-only series
    closes: dict[str, pd.Series] = {}
    for ticker, df in asset_data.items():
        daily = resample_ohlcv(df, "1d")
        closes[ticker] = daily["close"]

    # ... compute layers ...
```

#### Layer 3-7: Feature Computation (Pure Functions)

Each layer should be a standalone pure function that takes a `dict[str, pd.Series]` (ticker → close prices) and returns a score or structure.

**Key computations** (from the plan):

| Layer | Description | Key inputs | Return type |
|-------|-------------|-----------|-------------|
| Relative Strength | RS vs SPY for each sector | sectors, SPY | dict[sector → rs_score] |
| Leadership Ranking | Rank sectors by RS, compute rank changes | sector RS values | list of ranked sectors |
| Momentum | 20/50/100/200d returns, DMA position | all assets | dict[asset → momentum_score] |
| Ratios | XLY/XLP, XLI/XLU, XLF/XLU, IWM/SPY, Copper/Gold, Oil/Gold | specific pairs | dict[ratio_name → value] |
| Breadth | % above 50/200 DMA (use equity index components if available, or proxies) | equity data | breadth_score |
| Credit | HYG/TLT, LQD/TLT trends | credit + treasuries | credit_score |
| Rates | 2s10s spread, 10Y yield trend | treasury ETFs | rates_score |
| Regime Scoring | Risk, Growth, Inflation, Breadth, Liquidity scores (0-100) | all above | 5 scores |
| Regime Classification | Map scores → regime label | 5 scores + ratios | regime + confidence |
| Transition | Delta of scores, leadership shifts | current vs. prior scores | transition metadata |
| Cross-Asset Confirmation | Agreement across asset classes | ratio directions | confirmation_score |

#### Layer 8: Score Consolidation

All scores feed into a final 0-1 `score` for the `ScreenResult`. This is a composite confidence of the detected regime.

#### Layer 9: Metadata

Pack ALL computed values into `ScreenResult.metadata`:

```python
metadata = {
    "regime": "Expansion",
    "confidence": 81,
    "risk_score": 78,
    "growth_score": 84,
    "inflation_score": 55,
    "breadth_score": 73,
    "liquidity_score": 67,
    "confirmation": 88,
    "leadership": "XLI,XLF,XLK,...",
    "above_200dma": 72,
    "transition_from": "Recovery",
    "transition_days": 18,
    "2s10s_spread": 0.45,
    "copper_gold_ratio": 2.1,
    ...
}
```

Keys added to `PCT_DISPLAY_KEYS` in `src/screen/style.py` (if they represent percentages) will auto-format with `%` sign and color.

#### Layer 10: Ranking

`rank()` is trivial — only one result (the macro regime). Just returns the list as-is or sorted by score:

```python
def rank(self, results: list[ScreenResult]) -> list[ScreenResult]:
    return sorted(results, key=lambda r: r.score, reverse=True)
```

---

## 5. Universe Configuration

The cycle screener requires cross-asset tickers for both data syncing (via IBKR) and runtime
fetching. Add the following to `universe.yml`:

```yaml
symbols:
  # ── Broad market indices ──
  - spy
  - qqq

  # ── Sector ETFs ──
  - xlf   # financials
  - xlk   # technology
  - xli   # industrials
  - xlb   # materials
  - xle   # energy
  - xlv   # healthcare
  - xlp   # consumer staples
  - xlu   # utilities
  - xlre  # real estate
  - xlc   # communication services

  # ── Fixed income ──
  - shy   # 1-3Y treasury
  - iei   # 3-7Y treasury
  - ief   # 7-10Y treasury
  - tlt   # 20Y+ treasury

  # ── Credit ──
  - hyg   # high yield
  - jnk   # high yield (alternative)
  - lqd   # investment grade

  # ── Commodities (ETF proxies) ──
  - gld   # gold
  - slv   # silver
  - uso   # crude oil
  - ung   # natural gas
  - cper  # copper

  # ── Currency ──
  - uup   # USD index proxy

  # ── Volatility / breadth proxies ──
  - vix   # volatility index (ETF proxy)
```

Existing tickers in `universe.yml` (e.g., `xly`, `cqqq`, `tan`, `ura`, `copx`, `reet`, `xbi`,
`ewy`, `ewj`, `vwo`, `smh`, `btc`, `spmo`) can remain — the screen only references the tickers
it needs via the internal asset universe constants.

---

## 6. Testing Strategy

Tests go in `screens/tests/test_cycle_screener.py`. Test each Layer pure function independently:

```python
def test_relative_strength():
    """RS(XLF, SPY) = XLF_return - SPY_return over window."""
    ...

def test_momentum_features():
    """20/50/100/200d returns, DMA positions."""
    ...

def test_ratio_engine():
    """XLY/XLP, IWM/SPY, Copper/Gold ratios."""
    ...

def test_regime_scoring():
    """Given known ratio values, verify score outputs."""
    ...

def test_regime_classification():
    """Given known scores, verify regime label."""
    ...
```

Use small DataFrames with hand-crafted prices to verify known outputs.

---

## 7. Integration Checklist

- [ ] `screens/cycle_screener.py` exists and exports `DEFAULTS` + `make()`
- [ ] `make(symbols, params)` returns an instance conforming to `ScreenFn`
- [ ] `compute(symbol, candles)` returns `ScreenResult` with rich metadata
- [ ] `rank(results)` sorts by score descending
- [ ] Run: `python main.py screen cycle_screener universe.yml` works
- [ ] `ruff format` passes on the module
- [ ] `ty` check passes (or `uv run ty check screens/cycle_screener.py`)
- [ ] Tests pass: `uv run pytest screens/tests/test_cycle_screener.py`
- [ ] Relevant percent metadata keys added to `PCT_DISPLAY_KEYS` in `src/screen/style.py`

---

## 8. Tradeoffs & Simplifications vs. Original Plan

The original plan is a **full engine** with transition detection, confidence scores, cross-asset confirmation, etc. For v1 handoff:

1. **Start with v0:** Compute all raw features (ratios, momentum, RS, DMA positions) and expose them as metadata. Skip regime classification and transition detection in initial implementation. The screen becomes a "macro dashboard" — all features visible as table columns.

2. **v1:** Add regime scoring and classification on top of the raw features.

3. **v2:** Add transition detection and cross-asset confirmation.

This staged approach keeps the first coding task tractable (~300-400 LOC for v0, +200 for v1, +150 for v2).

---

## 9. Notes on Data Availability

- **VIX/VVIX**: May not be available via `get_local_candles` (requires IBKR data sync). Gracefully skip if empty.
- **Commodity ETFs**: CPER, GLD, SLV, USO, UNG — standard ETFs, should work.
- **Breadth data**: % above 50/200 DMA typically requires index component data. For v0, use SPY's own DMA position as a proxy. In a later version, this could pull from a breadth data source.
- **All prices should be resampled to daily** for macro analysis (`resample_ohlcv(df, "1d")`).
