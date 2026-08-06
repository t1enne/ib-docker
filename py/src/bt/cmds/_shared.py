"""Shared CLI helpers used across the bt command modules."""

from __future__ import annotations

import json
import re

import click
import pandas as pd


def _json_default(o):
    """JSON fallback encoder for non-native objects at the output edge.

    Converts pandas Timestamps (equity/z-score) and Enum members to plain
    values. Used only when serializing CLI output — never inside engine logic.
    """
    if isinstance(o, pd.Timestamp):
        return str(o)
    value = getattr(o, "value", None)
    if value is not None:
        return value
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def parse_param_grid(raw: str) -> dict:
    """Parse a JSON-like param grid, tolerating unquoted bare keys.

    Accepts strict JSON and shorthand like ``{ma_slow: [9, 14, 21]}``.
    Column values (keys) with no quotes are wrapping in double quotes before
    parsing. Single-quoted text stays literal.
    """
    try:
        return json.loads(raw, parse_int=int, parse_float=float)
    except json.JSONDecodeError:
        pass

    # Tolerate unquoted bare keys anywhere in the JSON: wrap each bare key
    # (identifier not already preceded by a quote) in double quotes.
    quoted = re.sub(
        r"(?P<notsquote>^|[^\"'])([A-Za-z_][A-Za-z0-9_]*)\s*:",
        r'\g<notsquote>"\2" :',
        raw,
    )
    try:
        parsed = json.loads(quoted, parse_int=int, parse_float=float)
    except json.JSONDecodeError as exc:
        raise ValueError(f"param_grid is not valid JSON: {exc}")
    if not isinstance(parsed, dict):
        raise ValueError("param_grid must be a JSON object.")
    return parsed


def cli_ts(dt) -> pd.Timestamp:
    """Convert a click DateTime value into a non-NaT Timestamp."""
    ts = pd.Timestamp(dt)
    if pd.isna(ts):
        raise click.UsageError(f"Invalid datetime: {dt}")
    assert isinstance(ts, pd.Timestamp)
    return ts
