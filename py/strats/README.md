# `strats/` classifications — pass / wip / fail

Each strategy config is classified per the backtest-first honesty protocol
(Sharpe, return vs **SPY buy-and-hold** over the same window, profit factor,
expectancy, $/trade cost, trade count, regime stability, point-in-time
universe check). The benchmark column is the buy-and-hold total return of the
gate/benchmark index over the same backtest window.

## PASS — money-consistent, enough samples, beats buy-and-hold

| config | bar | n | Sharpe | tot_ret | bench | why |
|---|---|---|---|---|---|---|
| `momentum_compression_breakout_ae_gate_SPY` | 1d | 349 | 1.40 | +491% | SPY +140% | Reproducible standout. Entropy(SPY) regime gate. **SPY-gate specific** — see caveats. |
| `cup_handle_dsl` | 1d | 122 | 1.38 | +187% | SPY +80% | Beats hold 2.3x, PF 2.82, +768/trade, 5yr. Re-validated. |

> **Read this before trusting either number.** Both winners carry two standing
> caveats established by A/B stress tests (see `quant/` run configs + session
> notes):
> 1. **Survivorship-weighted universe.** Symbol lists are curated today's
>    winners. A point-in-time/survivorship-free rebuild cut the momentum
>    config's +491%→~+187%, SR 1.40→~1.0 (one-variable A/B, same engine). Any
>    live expectation should assume the *thinned* PIT numbers, not the
>    retrospective +491%.
> 2. **SPY-gate load-bearing.** A clean 2x2 (universe fixed, only
>    `regime_symbol` × `benchmark_symbols` varied) proved `regime_symbol=SPY`
>    is causal and the benchmark symbol is cosmetic:
>    gate=SPY → SR ~1.40 regardless of bench; gate=QQQ → SR ~0.77 regardless of
>    bench. Do **not** "fix" these to gate/bench the universe's own index —
>    that removes the edge. Deploy only with `regime_symbol=SPY`.

## WIP — promising metrics, but a standing hazard blocks full PASS

| config | n | Sharpe | hazard |
|---|---|---|---|
| `entropy_vp_breakdown` | 72 | 1.77 | Best risk-adjusted but the config window is **ONE YEAR (2022)** → regime-fit risk. Extend window + walk-forward. |
| `bear_breakout_dsl_2022` | 64 | 1.47 | Highest PF (4.69) but a *bear-breakout* run mostly in bull/grind → edge likely rides 2022. Needs a bear-regime split. |
| `vp_breakout_dsl_regime` | 202 | 0.91 | Positive (+59%) but trails SPY and Sharpe<1; regime filter may cost alpha. Tune. |
| `rsi_divergence_dsl` | 25 | 0.91 | PF 4.22 but n=25 < 30 → no statistical confidence yet. |

## FAIL — negative, weak, or no statistical confidence

- **Rejected A/B (documented negative):** `momentum_compression_breakout_ae_gate_flushguard_SPY` — a "book-cooling" entry-cap rule could not fix the momentum strategy's worst-month MTM losses (those come from oversized OPEN single-name positions, not concurrent-entry breadth). Aggressive cooling made results worse (SR 1.40→0.92); mild cooling could not clear its own cost. No-op guard verified byte-identical to baseline. Keep the strategy module so this can be re-run.
- **Losing / negative expectancy:** `kalman_pairs_dsl` (−13.6%, PF 0.71).
- **Underperforms buy-and-hold badly:** `trend_pullback_atr_trail_dsl_L15_r15` (+73% vs SPY +234%), `shannons_demon_dsl` (+156% vs its own 50/50 SPY+GLD hold +189%), `kalman_mr_regime_dsl` (+9.9% vs SPY +44%).
- **No statistical confidence (n<30):** `ema_cross_dsl` (7), `macd_mfi_divergence` (8), `pf_equal_weight` (4), `rsi_bearish_divergence` (1).

## Research probes — `quant/` & this session

`strats/quant/` holds A/B run configs from the momentum-gate investigation
(QQQ-gate, biotech/XBI-gate, and the gate×benchmark split). They are NOT pass
candidates; they document negative transfers:
- Momentum AE-gate on a Nasdaq/QQQ bench: SR → ~0.77 (edge was SPY-gate).
- Momentum AE-gate on biotech/XBI: near-dead (SR ~0.25, PF ~1.24) — setup does
  not fit biotech's binary-event / gap profile.

## Run any of them

```bash
uv run ibkr bt run strats/pass/momentum_compression_breakout_ae_gate_SPY.json
uv run ibkr bt run strats/pass/cup_handle_dsl.json
uv run ibkr bt run strats/wip/entropy_vp_breakdown.json
# batch all configurations in one bucket to JSON/text:
uv run ibkr bt run <config> -F json
```
