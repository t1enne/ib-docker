"""Run all available screens over the nsdq universe and surface converging signals.

Two families of screens exist, and they are NOT equal in what they claim:

  * **Absolute screens** — `momentum`, `macd_divergence`, `mfi_divergence`,
    `obv_divergence`, `rsi_divergence`. Each independently scores whether a
    fresh condition (compression-breakout, a divergence print) fired on a
    symbol. A nonzero score means the condition actually exists, regardless of
    the rest of the tape. These are the only votes that can genuinely
    *converge*.

  * **Relative screen** — `rs` (relative strength). It cross-sectionally ranks
    every symbol against one benchmark and mechanically tags the top/bottom
    ~10% (top_pct) long/short. A high `rs` score means "cheapest vs benchmark",
    NOT "fresh signal". It fires by construction, so it is shown as a
    discretionary overlay/context — never counted as corroboration.

Convergence therefore = symbols where two or more *absolute* screens fired the
same direction. Volume is the only truth; a convergence that the divergence
screens contradict is flagged, not celebrated.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import pandas as pd

from src.bt.screen.screens import _discover, init_screen, resolve_screen_params
from src.bt.screen.adapter import state_per_interval

UNIVERSE = "universes/nsdq.json"
BENCHMARKS = ["QQQ", "SPY"]
INTERVALS = ["1d", "4h"]
START = "2020-01-01"
END = "2026-12-31"

# `rs` is a cross-sectional ranking, not an absolute condition — it must never
# "corroborate" a fresh signal.
RELATIVE_SCREENS = {"rs"}
ABSOLUTE_SCREENS = {
    "momentum",
    "macd_divergence",
    "mfi_divergence",
    "obv_divergence",
    "rsi_divergence",
}


@dataclass(frozen=True)
class _Vote:
    screen: str
    action: str
    score: float
    signal: str


def load_symbols(path: str) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    syms = [s.upper() for s in data["symbols"]]
    for b in BENCHMARKS:
        if b not in syms:
            syms.append(b)
    return syms


def run_all(symbols: list[str]) -> dict[str, dict[str, _Vote]]:
    """votes[symbol][screen] = _Vote for the latest daily bar."""
    states = state_per_interval(
        symbols, _ts(START), _ts(END), INTERVALS, benchmarks=BENCHMARKS
    )
    daily = states["1d"]
    votes: dict[str, dict[str, _Vote]] = {}
    for name in sorted(_discover().keys()):
        mod = init_screen(name)
        resolved = resolve_screen_params(name, {})
        results = mod.on_state(daily, resolved)
        for r in results:
            if r.score <= 0 or r.action == "flat":
                continue
            votes.setdefault(r.symbol, {})[name] = _Vote(
                name, r.action, r.score, r.signals[0] if r.signals else ""
            )
    return votes


def _ts(v: str) -> pd.Timestamp:
    ts = pd.Timestamp(v)
    assert isinstance(ts, pd.Timestamp)
    return ts


def parse_args(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(
        description="Run all available screens over a universe JSON file."
    )
    parser.add_argument(
        "universe",
        nargs="?",
        default=UNIVERSE,
        help="path to universe JSON (default: %(default)s)",
    )
    return parser.parse_args(argv).universe


def main(argv: list[str] | None = None) -> None:
    symbols = load_symbols(parse_args(argv))
    print(f"Universe: {len(symbols)} symbols (incl bench {BENCHMARKS})")
    screens = sorted(_discover().keys())
    abs_screens = sorted(ABSOLUTE_SCREENS)
    print(f"Absolute screens: {abs_screens}")
    print(f"Relative screen : {sorted(RELATIVE_SCREENS)}")

    votes = run_all(symbols)

    # --- absolute-screen convergence --------------------------------------
    print("\n=== CONVERGING SIGNALS — independent absolute screens ===")
    conv: list[tuple[str, int, float, str, list[tuple[str, str, float]]]] = []
    for sym, by_screen in votes.items():
        abs_votes = {k: v for k, v in by_screen.items() if k in ABSOLUTE_SCREENS}
        if not abs_votes:
            continue
        calls = len(abs_votes)
        strength = sum(v.score for v in abs_votes.values())
        dirs = {v.action for v in abs_votes.values()}
        clean = calls >= 2 and len(dirs) == 1
        blows = sorted(
            ((k, v.action, v.score) for k, v in abs_votes.items()),
            key=lambda x: -x[2],
        )
        conv.append((sym, calls, strength, "CONVERGE" if clean else "MIXED", blows))
    conv.sort(key=lambda r: (-r[1], -r[2]))
    if not conv:
        print("  none")
    for sym, calls, strength, tag, blows in conv:
        print(f"\n{sym:<6} {tag}  votes={calls} sum={strength:.2f}")
        for sc, a, s in blows:
            print(f"    {sc:<16} {a:<5} {s:.2f}")

    # --- relative-strength overlay (context only, never corroboration) ----
    print("\n=== RELATIVE-STRENGTH overlay (context, not a fresh signal) ===")
    rs_rows: list[tuple[str, str, float, list[tuple[str, str, float]]]] = []
    for sym, by_screen in votes.items():
        abs_votes = {k: v for k, v in by_screen.items() if k in ABSOLUTE_SCREENS}
        rel = by_screen.get("rs")
        if rel is None:
            continue
        blows = sorted(
            ((k, v.action, v.score) for k, v in abs_votes.items()),
            key=lambda x: -x[2],
        )
        rs_rows.append((sym, rel.action, rel.score, blows))
    rs_rows.sort(key=lambda r: -r[2])
    for sym, ra, rs, blows in rs_rows:
        abs_note = " | ABS:" + ", ".join(f"{k}={v:.2f}{a}" for k, a, v in blows) if blows else " | no absolute screens"
        print(f"{sym:<6} rs[{ra}]={rs:.2f}{abs_note}")


if __name__ == "__main__":
    main()
