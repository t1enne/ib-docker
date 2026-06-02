# Sync Module Improvement Plan

## Current Flow (for reference)

```
main.py sync
  → load_universe_config("universe.yml")
  → asyncio.run(sync_data(symbols, from_date))

sync_data(tickers, from_date)
  → get_symbols(tickers)          # returns list of COROUTINES (not symbols!)
  → asyncio.gather(*coroutines)   # resolves tickers → ISymbol[]
  → _get_candles(symbols, from_date)
    → candles_batch(conid[], lookback, from_datetime, bar="1h")
      → Semaphore(2) → asyncio.gather(*[fetch_with_limit(c)])
        → candles(c, from_datetime, to_datetime, bar)
          → _candle_limiter (Semaphore-aware context manager)
          → get_existing_range_sync() [BLOCKING]
          → calculate_gaps()
          → _fetch_candles_iterative()
          → CandleSchema.insert_many()
```

---

## Issues Found (23 total)

### 🔴 Critical (bugs / correctness)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1 | **`get_symbols` returns coroutines, not symbols** — name is misleading, violates "function returns what it says" | `__init__.py:31-40` | Confusing API, potential misuse |
| 2 | **`_get_candles` silently swallows all errors** — `except Exception as e: print(...)` no re-raise | `__init__.py:45-51` | Silent data loss |
| 3 | **`sync_data` defaults `from_date` to `date.today()`** — means nothing to sync, always a no-op | `__init__.py:54` | Surprising behavior |
| 4 | **`get_existing_range_sync` is synchronous but called inside async** — blocks event loop | `candles.py:89-100` | Event loop stall |
| 5 | **`_fetch_candles_iterative` uses mutable default arg** `data: list[dict] = []` — Python anti-pattern | `candles.py:103` | Bugs on repeated calls |
| 6 | **`_fetch_candles_iterative` calls `get_contract_info(conid)` on every loop iteration** — wasteful API call | `candles.py:115` | Extra latency, rate limit waste |

### 🟡 Design (flow / architecture)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 7 | **Double semaphore layering** — `candles_batch` Semaphore(2) + `candles` `_candle_limiter` | `candles.py:174,215` | Confusing concurrency control |
| 8 | **`candles_batch(lookback)` parameter is dead** — passed but never used in `candles()` call | `candles.py:203` | Misleading API |
| 9 | **Module-level `_candle_limiter` is mutable global state** — not injectable, hard to test | `candles.py:34` | Poor testability |
| 10 | **`get_contract_info` in `shared.py` `pass`es on `DoesNotExist`** — API fetch fail = silent skip | `shared.py:23` | May insert contract without error, or fail silently |
| 11 | **No progress reporting** — long syncs (many symbols, large date ranges) are a black box | All modules | Poor UX |
| 12 | **`candles_batch` returns `list[int]` but never uses return value** — dead return | `__init__.py:48` | Unused |

### 🟠 Maintenance (code quality / type safety)

| # | Issue | Location |
|---|-------|----------|
| 13 | **`batch_items` defined but never called** — dead code | `rate_limiter.py:146-170` |
| 14 | **`RequestQueue` class defined but never instantiated** — dead code | `rate_limiter.py:120-144` |
| 15 | **`with_rate_limiter` decorator defined but never used** — dead code | `rate_limiter.py:92-103` |
| 16 | **No tests for `__init__.py` public functions** (`sync_data`, `load_universe_config`, `get_symbols`) | `tests/` directory |
| 17 | **`print()` everywhere instead of `logging`** — no structured output, can't suppress | All modules |
| 18 | **`load_universe_config` has no validation or helpful errors** | `__init__.py:57-60` |
| 19 | **`__all__` exports `get_symbols`** which is internal (meant for `sync_data`) | `__init__.py:63` |
| 20 | **`_fetch_candles_iterative` returns `list[dict]`** — untyped dicts, no `TypedDict` or dataclass | `candles.py:101` |

### 🟢 Functional gaps

| # | Gap |
|---|-----|
| 21 | **No `bar` parameter at top level** — `sync_data` hardcodes "1h" with no way to configure |
| 22 | **No dry-run / preview mode** — can't see what gaps exist before fetching |
| 23 | **`UniverseConf` is underpowered** — only `symbols` + `from_date`, no to_date, no bar size, no concurrency |

---

## Proposed Changes (ordered by impact)

### Phase 1: Fix bugs (critical)

#### 1. Fix `_fetch_candles_iterative` mutable default arg + repeated API call
```python
async def _fetch_candles_iterative(
    conid: int,
    bar: str,
    from_datetime: datetime,
    to_datetime: datetime,
    ticker: str,          # NEW: pass ticker directly instead of calling get_contract_info
) -> list[CandleDict]:    # Use TypedDict
    accumulated_data: list[CandleDict] = []  # No mutable default
    # ... remove get_contract_info(conid) call, use passed ticker
```

#### 2. Make `get_existing_range_sync` async
```python
# Either make it async (wrap with asyncio.to_thread) or accept it's IO and use a DB executor
async def get_existing_range(ticker: str) -> tuple[Optional[int], Optional[int]]:
    # peewee operations in thread to not block event loop
    return await asyncio.to_thread(get_existing_range_sync, ticker)
```

#### 3. Fix `_get_candles` error swallowing
```python
async def _get_candles(symbols: list[ISymbol], from_date: date) -> list[int]:
    return await candles_batch(...)  # Let errors propagate
```

#### 4. Fix `sync_data` to not silently default to today
```python
async def sync_data(tickers: list[str], from_date: date) -> SyncResult:
    # from_date is now required (no default)
    ...
```

### Phase 2: Remove dead code / simplify

#### 5. Delete dead code from `rate_limiter.py`
- Remove `batch_items` (lines 146-170)
- Remove `RequestQueue` (lines 120-144)
- Remove `with_rate_limiter` (lines 92-103)

#### 6. Collapse double semaphore layer
- Remove `_candle_limiter` from module level in candles.py
- Use only `candles_batch` semaphore (already does `Semaphore(max_concurrent)`)
- Or: inject the limiter as a parameter

#### 7. Fix `get_symbols` naming — make it return results, not coroutines
```python
async def resolve_symbols(tickers: list[str]) -> list[ISymbol]:
    """Resolve ticker strings to ISymbol objects (DB lookup + API fallback).
    This is the internal function — sync_data calls it directly.
    """
    semaphore = asyncio.Semaphore(2)
    async def bounded(ticker: str) -> Optional[ISymbol]:
        async with semaphore:
            try:
                return await _get_symbol_for_ticker(ticker)
            except Exception as e:
                print(f"Error resolving {ticker}: {e}")
                return None
    results = await asyncio.gather(*[bounded(t) for t in tickers])
    return [s for s in results if s is not None]
```

Then `sync_data` becomes:
```python
async def sync_data(tickers: list[str], from_date: date) -> SyncResult:
    symbols = await resolve_symbols(tickers)
    if not symbols:
        raise ValueError("No symbols resolved")
    result = await _get_candles(symbols, from_date)
    return SyncResult(resolved=len(symbols), fetched=result)
```

### Phase 3: Type safety & code quality

#### 8. Add `TypedDict` for candle data
```python
class CandleDict(TypedDict):
    conid: int
    ticker: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
```

#### 9. Replace `print()` with `logging`
```python
import logging
logger = logging.getLogger(__name__)
# Replace all print() with logger.info/warning/error
```

#### 10. Validate `load_universe_config`
```python
def load_universe_config(file_path: str) -> UniverseConf:
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Universe config not found: {file_path}")
    if not isinstance(data, dict):
        raise ValueError(f"Invalid universe config: expected dict, got {type(data)}")
    if "symbols" not in data:
        raise ValueError("Universe config missing required 'symbols' field")
    return UniverseConf(**data)
```

### Phase 4: Functional improvements

#### 11. Add `SyncResult` dataclass for structured output
```python
@dataclass(frozen=True)
class SyncResult:
    resolved: int       # how many tickers resolved
    fetched: list[int]  # conids that got new data
    gaps_found: int     # how many gaps were filled

    @property
    def total_fetched(self) -> int:
        return len(self.fetched)
```

#### 12. Add `bar` and `to_date` to `UniverseConf`
```python
@dataclass
class UniverseConf:
    symbols: list[str]
    from_date: Optional[date] = None
    to_date: Optional[date] = None
    bar: str = "1h"
```

#### 13. Add dry-run mode
```python
async def preview_sync(tickers: list[str], from_date: date) -> PreviewResult:
    """Show what would be fetched without making API calls."""
    ...
```

#### 14. Add progress callbacks
```python
ProgressFn = Callable[[str, int, int], None]  # (status, current, total)

async def sync_data(
    tickers: list[str],
    from_date: date,
    on_progress: Optional[ProgressFn] = None,
) -> SyncResult:
    ...
```

### Phase 5: Testing

#### 15. Add tests for `__init__.py`
- Test `load_universe_config` with valid/invalid YAML
- Test `sync_data` with mocked dependencies
- Test `resolve_symbols` with DB hits and misses
- Test error propagation paths

---

## Files to touch

| File | Changes |
|------|---------|
| `src/syncm/__init__.py` | Refactor `get_symbols` → `resolve_symbols`, fix error handling, add structured results, add progress |
| `src/syncm/ibkr_layer/candles.py` | Fix mutable default, reduce get_contract_info calls, async DB ops, remove dead `lookback` param |
| `src/syncm/ibkr_layer/rate_limiter.py` | Remove dead code (`batch_items`, `RequestQueue`, `with_rate_limiter`) |
| `src/syncm/ibkr_layer/shared.py` | Improve `get_contract_info` error handling |
| `src/syncm/ibkr_layer/types.py` | **NEW** — `CandleDict` TypedDict, `SyncResult`, `PreviewResult`, `ProgressFn` |
| `src/syncm/tests/test_init.py` | **NEW** — tests for public API |
| `src/syncm/ibkr_layer/tests/test_candles.py` | Update for refactored signatures |
| `main.py` | Accept `bar` parameter, handle `SyncResult` return |
