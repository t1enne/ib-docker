"""Pure cross-timeframe trend-consensus logic.

Shared by the screen layer (``tf_divergence`` ranking) and the backtest
strategy (``tf_momentum``) so the two never drift apart. One source of truth
for how per-timeframe ``BULL``/``BEAR``/``RANGE`` labels fold into a weighted
direction, how the execution timeframe is weighted, and what counts as
divergence.
"""

from __future__ import annotations

from typing import Sequence, TypeAlias

TrendLabel: TypeAlias = str | None  # "BULL" / "BEAR" / "RANGE" / None (unresolved)

_MINUTES: dict[str, int] = {"m": 1, "h": 60, "d": 1440, "w": 10080}


def tf_minutes(iv: str) -> int | None:
    """Resolve an interval string to minutes (``1h``=60, ``1d``=1440)."""
    text = iv.strip().lower()
    digits = "".join(c for c in text if c.isdigit())
    unit = "".join(c for c in text if not c.isdigit())
    if not digits or not unit:
        return None
    n = int(digits)
    minutes = _MINUTES.get(unit)
    return n * minutes if minutes is not None else None


def is_lowest_tf(iv: str, all_ivs: Sequence[str]) -> bool:
    """True when ``iv`` has the minimum bar length among ``all_ivs``.

    The lowest bar length = the execution timeframe (shortest period has the
    most bars, highest frequency). Unparseable intervals never win.
    """
    mins = {k: tf_minutes(k) for k in all_ivs}
    resolvable = {k: m for k, m in mins.items() if m is not None}
    if not resolvable or iv not in resolvable:
        return False
    return resolvable[iv] == min(resolvable.values())


def weighted_align(
    labels: Sequence[TrendLabel],
    ivs: Sequence[str],
    lower_tf_weight: float,
    *,
    drop_unresolved: bool = True,
) -> tuple[float, float, bool]:
    """Weighted long/short alignment over per-TF labels.

    Returns ``(long_align, short_align, divergent)`` where each align is the
    share of applied weight that resolves ``BULL``/``BEAR`` (0..1).

    When ``drop_unresolved`` (default True), ``None`` labels are excluded from
    the weight denominator: a timeframe that hasn't resolved (e.g. higher-TF
    warmup) neither dilutes nor vetoes a confirmed signal on the resolved
    timeframes. Set False to treat unresolved as a neutral third vote.

    ``divergent`` is True when at least one resolved label is ``BULL`` and
    another resolved label is ``BEAR``.
    """
    pairs = [
        (lab, lower_tf_weight if is_lowest_tf(iv, ivs) else 1.0)
        for lab, iv in zip(labels, ivs)
    ]
    if drop_unresolved:
        pairs = [(lab, w) for lab, w in pairs if lab is not None]
    total = sum(w for _, w in pairs)
    if total <= 0:
        return (0.0, 0.0, False)
    long_w = sum(w for lab, w in pairs if lab == "BULL")
    short_w = sum(w for lab, w in pairs if lab == "BEAR")
    has_bull = any(lab == "BULL" for lab, _ in pairs)
    has_bear = any(lab == "BEAR" for lab, _ in pairs)
    return (round(long_w / total, 6), round(short_w / total, 6), has_bull and has_bear)
