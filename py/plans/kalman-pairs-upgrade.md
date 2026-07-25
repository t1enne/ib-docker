# Kalman Pairs Filter — Upgrade to Practitioner Standards

> Adopt the canonical quant-industry Kalman filter for pairs trading:
> two-state `[α, β]` model with the Kalman's own standardized innovation
> as the sole trading signal. Adaptive noise is **not** in scope.

---

## 1. What Changes — Overview

| Component                | Current                               | Target                                                  |
| ------------------------ | ------------------------------------- | ------------------------------------------------------- |
| **State**                | `[β]` (1D random walk)                | `[α, β]` (2D random walk)                               |
| **Observation**          | `log(P1) = β·log(P2) + v`             | `log(P1) = α + β·log(P2) + v`                           |
| **H matrix**             | `[[log(P2)]]` (1×1)                   | `[[1, log(P2)]]` (1×2)                                  |
| **Signal**               | EWMA-MAD z-score **and** t-stat       | t-stat only: `(log(P1) - α_pred - β_pred·log(P2)) / √S` |
| **Warm start**           | `β₀ = 1.0, P₀ = 1.0`                  | OLS over initial window → `[α₀, β₀]`, narrow `P₀`       |
| **EWMA post-processing** | Full spread mean/MAD/z-score pipeline | **Removed**                                             |

Adaptive noise (`adaptive=True`, `vol_window`) is left untouched — it works
and is not required for this upgrade.

---

## 2. Why `[α, β]` Instead of `[β]` Only

Practitioners (Robot Wealth, Quantopian, Montana et al. 2009) universally
include an intercept in the state. The reason is statistical, not stylistic:

```
Spread without α:  spread_t = log(P1_t) - β_t · log(P2_t)
Spread with α:     spread_t = log(P1_t) - α_t - β_t · log(P2_t)
```

When α ≠ 0 (which is nearly always true for real pairs), the spread without α
carries a persistent level bias. The EWMA post-processing tries to absorb this
bias — but it introduces:

- **Lag** — the EWMA mean chases the spread, delaying signal reversals
- **Double filtering** — the Kalman already optimally estimates the state;
  re-smoothing the residuals with a separate EWMA loses optimality

With `[α, β]` in the state, the Kalman automatically partitions level and slope.
The innovation (spread) is then genuinely mean-zero by construction. The t-stat
`spread / √S` is the complete signal — no EWMA needed.

**What stays the same:** we keep log-prices (not raw prices).

Why not raw prices? Practitioners use both:

| Space | Observation                   | β meaning                 | Signal units  | Best for                    |
| ----- | ----------------------------- | ------------------------- | ------------- | --------------------------- |
| Log   | `log(P1) = α + β·log(P2) + v` | Elasticity (% move ratio) | Dimensionless | Long-horizon, cointegration |
| Raw   | `P1 = α + β·P2 + v`           | Hedge ratio (share ratio) | Dollars       | Short-horizon, intraday     |

The existing codebase operates in log-space throughout (`_ols_log_params`,
`calculate_zscore_spread`, the backtest strategy, the current Kalman). Switching
to raw prices would:

- Break signal scale: a $5 spread on SPY ($600) vs. KO ($70) are incomparable
- Require re-tuning all `entry_z` / `exit_z` thresholds across strategies
- Lose cross-pair comparability (can't rank multiple pairs by signal strength)
- Require separate `α` scaling per pair (α in dollars, not dimensionless)

Staying in log-space keeps signals dimensionless and pair-portable — a 2.5σ
event means the same thing for any pair.

---

## 3. Files Touched

| File                                    | Change                                                                                                                                                              | Risk                              |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `src/kalman/types.py`                   | `PairsKalmanResult` drops `spread_mean`, `spread_std`, `spread_mad`, `z_score`; keeps `t_stat`, `spread`, `beta`, `innovation_S`; adds `alpha`                      | Medium — changes public API       |
| `src/kalman/pure.py`                    | `run_pairs_kalman()`: 2D state, new H, add α to result, remove EWMA post-processing. Add `_warm_start_ols()` helper                                                 | Medium — core algorithm           |
| `src/kalman/online.py`                  | `PairsKalmanOnline`: 2D state, new H, add α property, remove EWMA accumulators, use t_stat as primary output. Add `init_from_ols()` method. Fix adaptive R ordering | High — hot-path in backtest       |
| `src/kalman/__init__.py`                | Export `alpha`-aware types, deprecate `z_score`-related attrs                                                                                                       | Low                               |
| `src/kalman/tests/test_pairs_kalman.py` | Update all tests: remove z_score assertions, add α convergence tests, update signal tests to use t_stat. Check `online == batch` correlation holds with new state   | Medium — critical for correctness |
| `src/kalman/cli.py`                     | Update output columns / plot labels                                                                                                                                 | Low                               |
| `src/spread/__init__.py`                | Update the `spread()` plotting function: remove z-score EMA subplot, add t-stat and α subplots                                                                      | Low                               |

**Not touched:** `src/bt/strategies/pairs_trading_functional.py` (it doesn't use
the Kalman), `src/bt/zscore.py`, `src/utils.py`.

---

## 4. New Types

```python
# src/kalman/types.py

@dataclass(frozen=True)
class PairsKalmanConfig:
    process_noise: float = 1e-4       # Q for [α, β]
    measurement_noise: float = 1e-3   # R
    mean_halflife: int = 50           # warm-start OLS window (repurposed from EWMA halflife)
    # adaptive + vol_window remain, untouched

@dataclass
class PairsKalmanResult:
    alpha: pd.Series                  # NEW — time-varying intercept
    beta: pd.Series                   # hedge ratio (unchanged)
    spread: pd.Series                 # innovation log(P1) - α - β·log(P2)
    t_stat: pd.Series                 # spread / √S — THE signal
    innovation_S: pd.Series           # Kalman innovation covariance

    # REMOVED: spread_mean, spread_std, spread_mad, z_score
```

The output is now **5 columns** instead of 8. Cleaner, and every column has a
clear role: α and β are the relationship parameters, spread is the raw
deviation, S is the uncertainty, t_stat is the normalized signal.

---

## 5. Batch Implementation (`run_pairs_kalman`)

### 5.1 Warm-start via OLS

```python
def _warm_start_ols(
    log_p1: np.ndarray, log_p2: np.ndarray, window: int = 50
) -> tuple[float, float, float]:
    """Estimate [α, β] and P₀ from initial OLS window."""
    w = min(window, len(log_p1))
    X = np.column_stack([np.ones(w), log_p2[:w]])
    y = log_p1[:w]
    theta = np.linalg.lstsq(X, y, rcond=None)[0]  # [α, β]
    resid = y - X @ theta
    # Covariance of estimates ≈ σ²_resid · (X'X)^(-1)
    sigma2 = resid.var(ddof=2) if w > 2 else 1.0
    P_diag = np.diag(np.linalg.inv(X.T @ X)) * sigma2
    P = np.diag(np.maximum(P_diag, 1e-6))
    return float(theta[0]), float(theta[1]), P
```

### 5.2 Kalman loop (2D state)

```python
kf = _KF(dim_x=2, dim_z=1)

# State transition: random walk on both α and β
kf.F = np.eye(2)
kf.Q = np.eye(2) * cfg.process_noise

# Initialise from OLS
alpha0, beta0, P0 = _warm_start_ols(log_p1, log_p2, cfg.mean_halflife)
kf.x = np.array([[alpha0], [beta0]])
kf.P = P0

R_base = np.array([[cfg.measurement_noise]])
kf.R = R_base

for t in range(n):
    logp2 = log_p2[t]
    logp1 = log_p1[t]

    # Observation matrix: H = [[1, log(P2_t)]]
    kf.H = np.array([[1.0, logp2]])

    # adaptive R (unchanged)

    kf.predict()
    beta_pred = kf.x[0, 0]    # β prior
    alpha_pred = kf.x[1, 0]   # α prior

    # Innovation covariance S
    S = float((kf.H @ kf.P @ kf.H.T + kf.R)[0, 0])

    # Innovation = spread
    innovation = float(logp1 - (alpha_pred + beta_pred * logp2))

    kf.update(np.array([[logp1]]))

    alpha_arr[t] = kf.x[0, 0]
    beta_arr[t] = kf.x[1, 0]
    spread_arr[t] = innovation
    innov_S_arr[t] = S

# t_stat = spread / √S  (no EWMA needed)
sqrt_S = np.sqrt(np.maximum(innov_S_arr, 1e-24))
t_stat_arr = spread_arr / sqrt_S
```

---

## 6. Online Implementation (`PairsKalmanOnline`)

Same structural changes as batch, plus:

### 6.1 Adaptive R ordering fix

Currently the online path applies adaptive R **after** the Kalman update
(so the diagnostic signals use the adaptive R but the state update doesn't).
Fix: apply adaptive R **before** `predict()`:

```python
def update(self, log_p1: float, log_p2: float) -> float:
    # Compute adaptive R first
    r_eff = self._r
    if self._adaptive and self._n > 1:
        # ... compute scale ...
        r_eff = self._r * scale

    # Predict
    beta_prior = self._beta
    alpha_prior = self._alpha
    P_prior = self._P + self._Q  # 2×2 now

    # Update
    H = np.array([[1.0, log_p2]])          # 1×2
    y = float(log_p1 - (alpha_prior + beta_prior * log_p2))
    S = float(H @ P_prior @ H.T + r_eff)
    K = (P_prior @ H.T) / S                 # 2×1, scalar division — explicit parens
    # Joseph-form P update (correct for 2D, not the scalar shortcut)
    I = np.eye(2)
    self._P = (I - K @ H) @ P_prior @ (I - K @ H).T + K @ r_eff @ K.T
    # P post-update is forced symmetric (numerical drift guard)
    self._P = (self._P + self._P.T) / 2.0
```

**Why Joseph form:** The scalar `P_post = (1 - KH) * P_prior` is correct only
for `dim_x = 1`. For 2D, the full Joseph form guarantees P remains symmetric
and positive-definite even with numerical round-off. `filterpy` uses this
internally; the online path should match.

### 6.2 `init_from_ols()` method

```python
def init_from_ols(self, log_p1_arr: np.ndarray, log_p2_arr: np.ndarray) -> float:
    """Warm-start from batch OLS over initial window. Returns initial t_stat."""
    alpha, beta, P = _warm_start_ols(log_p1_arr, log_p2_arr)
    self._alpha = alpha
    self._beta = beta
    self._P = P
    # Use last observation for spread seeding
    last_spread = float(log_p1_arr[-1] - alpha - beta * log_p2_arr[-1])
    self._spread = last_spread
    self._n = len(log_p1_arr)
    # Compute S and t_stat for the last observation
    H = np.array([[1.0, log_p2_arr[-1]]])
    S = float(H @ P @ H.T + self._r)
    self._innovation_S = S
    self._t_stat = last_spread / np.sqrt(max(S, 1e-24))
    return self._t_stat
```

### 6.3 Removed properties

- `z_score` → removed (use `t_stat` as the trading signal)
- `spread_mean`, `spread_mad` → removed (no EWMA tracking)
- `is_warm` → removed (OLS warm-start eliminates burn-in)

### 6.4 New properties

- `alpha: float` — current intercept estimate

---

## 7. Signal Change: Backtest Integration

Currently the `pairs_trading_functional.py` strategy uses the OLS z-score
(not the Kalman filter at all). No code in `src/bt/` imports from `src/kalman`.
So the Kalman types change has **zero impact on existing backtests**.

For future strategies that will use the Kalman via `model_updater_fn`:

```python
# OLD (conceptual, not in current code):
# z = state.model_state.z_score  # EWMA-MAD z-score
# if z > 2.5: ...

# NEW:
# t = state.model_state.z_score  # now holds t_stat
# if t > 2.5: ...
```

The `ModelState.z_score` field stays in place (it's `Optional[float]`) — we just
populate it with `t_stat` instead of the EWMA z-score. No schema change needed.

---

## 8. Implementation Sequence

### Phase 1 — Types & Batch (low risk, testable in isolation)

1. Update `PairsKalmanConfig`: keep all existing fields, repurpose `mean_halflife`
   as OLS warm-start window size. Add docstring explaining the change.
2. Update `PairsKalmanResult`: remove `spread_mean/spread_std/spread_mad/z_score`,
   add `alpha`.
3. Rewrite `run_pairs_kalman()`: 2D state, `_warm_start_ols()`, H = `[[1, log(P2)]]`,
   remove EWMA post-processing. Output `t_stat` and `alpha`.
4. Update `__init__.py` exports.
5. Rewrite `test_pairs_kalman.py` tests — target **all passing before Phase 2**.

**Verification gate:** `uv run pytest src/kalman/tests/ -v` — all green.

### Phase 2 — Online (medium risk, needs batch as ground truth)

1. Rewrite `PairsKalmanOnline`: 2D state, H = `[[1, log(P2)]]`, remove EWMA
   accumulators, add `alpha` property, fix adaptive R ordering.
2. Rewrite `_run_online()` test helper.
3. Update `TestOnlineMatchesBatch`: correlation check between online and batch
   `t_stat` (was `z_score`).
4. Update `TestPairsKalmanOnline` property tests to reference new fields.

**Verification gate:** `uv run pytest src/kalman/tests/ -v` — all green,
online-vs-batch t_stat correlation > 0.95.

### Phase 3 — Downstream Consumers (low risk, no existing Kalman users in bt/)

1. Update `src/kalman/cli.py` output.
2. Update `src/spread/__init__.py` plotting: show t_stat + α + β subplots
   instead of z-score EMA + MAD subplots.

**Verification gate:** `python -m src.kalman.cli --help` works,
`python -m src.spread SPY QQQ --start 2024-01-01 --end 2024-06-01` renders.

### Phase 4 — Documentation

1. Update docstrings in `pure.py` and `online.py` to document the new state model.
2. Update `AGENTS.md` if any Kalman-specific guidance exists (currently doesn't).

---

## 9. Migration: Backward Compatibility

No existing code in `src/bt/` imports from `src/kalman`. The `spread()` function
in `src/spread/__init__.py` is a standalone CLI visualizer — it's the only
consumer. So:

- **No deprecation period needed.** Remove the old fields directly.
- The `spread()` CLI will be updated in Phase 3 to use the new result shape.

If a local script or notebook outside the repo imports `PairsKalmanResult.z_score`,
it will get an `AttributeError`. This is acceptable given the module's limited
scope and the fact that the new `t_stat` field is a straight replacement.

---

## 10. Risk & Edge Cases

| Risk                                               | Mitigation                                                                                                                                                               |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| OLS warm-start on < 3 points (singular X'X)        | `_warm_start_ols` falls back to `α₀ = 0, β₀ = 1.0, P₀ = I` when window < 3                                                                                               |
| Identical log-prices (collinear X'X)               | `np.linalg.lstsq` handles rank deficiency; fallback to identity P                                                                                                        |
| t_stat explosion when S ≈ 0                        | `sqrt_S = sqrt(max(S, 1e-24))` — already in current code                                                                                                                 |
| 2D state has different convergence time than 1D    | OLS warm-start eliminates burn-in; convergence not a concern                                                                                                             |
| adaptive R ordering fix changes behavior           | The fix is a bug correction — behavior change is intended and tested in Phase 2                                                                                          |
| Online-vs-batch correlation may drop with 2D state | If < 0.95, investigate numerical precision in scalar K vs filterpy's full matrix K. Joseph-form P update (already used for scalar) may need to become Joseph-form for 2D |
