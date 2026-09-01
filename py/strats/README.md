# `strats/` classifications — pass / wip / fail

Each strategy config is classified per the backtest-first honesty protocol
(Sharpe, return vs **SPY buy-and-hold** over the same window, profit factor,
expectancy, $/trade cost, trade count, regime stability). The benchmark column
is the SPY buy-and-hold total return over the same backtest window.

## PASS — money-consistent, enough samples, beats buy-and-hold

| config | bar | n | Sharpe | tot_ret | SPY | why |
|---|---|---|---|---|---|---|
| `momentum_compression_breakout_ae_gate_SPY` | 1d | 349 | 1.40 | +491% | +140% | **repo standout.** Entropy(SPY) gate lifts the plain compression strat +174pts return, Sharpe 1.18→1.40, and *cuts* maxDD −31.8→−24.4. 6.5yr, 349 trades. |
| `momentum_compression_breakout` | 1d | 231 | 1.18 | +317% | +107% | Strong un-gated baseline; the ae-gate version above is strictly better. |
| `cup_handle_dsl` | 1d | 122 | 1.38 | +187% | +80% | Beats hold 2.3x, PF 2.82, +768/trade, 5yr. Clean. |

## WIP — promising idea/metrics, but a standing hazard blocks full PASS

| config | n | Sharpe | hazard |
|---|---|---|---|
| `entropy_vp_breakdown` | 72 | **1.77** | Best risk-adjusted (Sharpe 1.77, alpha 0.29, made +40% while SPY −20%) **but the config window is ONE YEAR (2022)** → regime-fit risk. Extend window + walk-forward before trusting. |
| `bear_breakout_dsl_2022` | 64 | 1.47 | Highest PF (4.69) but it is a *bear-breakout* strategy run mostly in a bull/grind window → edge likely rides 2022. Needs a bear-regime split. |
| `vp_breakout_dsl_regime` | 202 | 0.91 | Positive (+59%) but trails SPY (+112%) and Sharpe<1; regime filter may be costing alpha. Tune. |
| `rsi_divergence_dsl` | 25 | 0.91 | PF 4.22 but n=25 < 30 → no statistical confidence yet. Need more trades. |

## FAIL — negative, weak, or no statistical confidence

- **Losing / negative expectancy:** `kalman_pairs_dsl` (−13.6%, PF 0.71).
- **Underperforms buy-and-hold badly:** `trend_pullback_atr_trail_dsl_L15_r15` (+73% vs SPY +234%), `shannons_demon_dsl` (+156% vs its own 50/50 SPY+GLD hold +189%), `kalman_mr_regime_dsl` (+9.9% vs SPY +44%).
- **No statistical confidence (n<30):** `ema_cross_dsl` (7), `macd_mfi_divergence` (8), `pf_equal_weight` (4), `rsi_bearish_divergence` (1).
- **Range-trial experiments (hourly, dead):** `band_fade_entropy_range_1h` (−16%), `extreme_fade_range_1h` (n=16, unstable), `shannons_demon_1h_range` (underperforms its hold). See research notes — the hourly range-fade premise did not clear costs on commodities.

## Run any of them

```bash
uv run ibkr bt run strats/pass/momentum_compression_breakout_ae_gate_SPY.json
uv run ibkr bt run strats/wip/entropy_vp_breakdown.json
# batch all with a folder filter:
uv run python scripts/batch_backtest_report.py --folder pass
```

The `scripts/batch_backtest_report.py` runner now discovers configs recursively
(`strats/**/*.json`); `--folder pass|wip|fail` restricts to one bucket.
