"""Momentum compression-breakout screen — SPY adaptive-entropy gate variant.

This is the screening mirror of the validated backtest pass
``strats/pass/momentum_compression_breakout_ae_gate_SPY.json``: the same
big-move → pullback → compression → breakout scoring as the base ``momentum``
screen, but the adaptive-entropy market-regime gate observes **SPY** (the pass's
``regime_symbol``) instead of QQQ, with the pass's tuned compression knobs.

The base ``momentum`` screen hard-defaults its AE observer to ``QQQ`` and its
compression knobs to relaxed values (``body_atr_ratio 0.35``,
``min_hover_bars 3``); the validated SPY pass runs tighter
(``body_atr_ratio 0.30``, ``min_hover_bars 4``), requires a higher big-move
bar (``min_gain 0.35``), demands an AE trend-strength floor
(``regime_min_strength 0.12``) and a longer entropy lookback (``ae_lookback
40``). This screen fixes those as its defaults so ``init_screen`` /
``resolve_screen_params`` produce a SPY-gated result out of the box.

All scoring/regime logic is inherited unchanged from the base ``momentum``
screen — this module only restates the pass's parameterization, so the two
never drift on the compression-breakout math. It re-exports only the delta: a
``Params`` subclass with the pass defaults and a thin ``on_state`` delegator.

A note on placement: this is NOT wired into ``run_screens.py``'s absolute-
screen convergence tally. Momentum-compression-breakout (QQQ-gated) and its
SPY-gated twin are the SAME fresh-signal family gated on a different index — 
counting both toward "independent absolute screens converged" would double-count
one compression/breakout setup. Use it as a targeted, decision-support variant
via ``run_screen`` / ``screen_over_history`` / the direct ``init_screen`` path
(e.g. when you specifically want an SPY- (not QQQ-) confirmed bull tape before
acting on a breakout).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bt.screen.screens.momentum import Params as _MomentumParams
from src.bt.screen.screens.momentum import on_state as _momentum_on_state
from src.bt.screen.types import ScreenParams, ScreenResult, ScreenState

SCREEN_TYPE = "momentum_spy_gate"


@dataclass(frozen=True)
class Params(_MomentumParams):
    """SPY-gated compression-breakout knobs, defaulting to the validated pass.

    Overrides / additions vs the base ``momentum`` screen ``Params`` only; all
    other fields inherit the base defaults. ``from_dict`` (via
    ``StrategyParams``) still accepts a raw dict and fills these defaults, so a
    caller can override any knob or feed further trading knobs that are ignored.
    """

    #: AE market-regime observer index for this variant (SPY-gated).
    benchmark: str = "SPY"
    min_gain: float = 0.35
    body_atr_ratio: float = 0.30
    min_hover_bars: int = 4
    regime_min_strength: float = 0.12
    ae_lookback: int = 40


def on_state(state: ScreenState, params: ScreenParams) -> tuple[ScreenResult, ...]:
    """Score the state with the SPY-gated parameterization.

    Stateless delegate to the base ``momentum`` screen's scoring. Standard use
    passes a ``Params`` instance resolved from this module; the base scoring is
    agnostic to the concrete subtype as long as the ``benchmark``/regime/knob
    attributes carry the intended values.
    """
    return _momentum_on_state(state, params)
