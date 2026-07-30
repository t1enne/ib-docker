"""Minimal declarative table formatting. No dependencies. Pure functions."""

from dataclasses import dataclass, field
from typing import Sequence

type Align = str  # "<" | ">" | "^"


@dataclass(frozen=True)
class Col:
    """Single column definition. Label doubles as header text."""

    label: str
    align: Align = "<"
    fmt: str = ""


@dataclass(frozen=True)
class Table:
    """Declarative table: columns + rows of strings, render later."""

    columns: tuple[Col, ...]
    rows: tuple[tuple[str, ...], ...] = field(default_factory=tuple)


def _pad(value: str, width: int, align: Align) -> str:
    fill = width - len(value)
    if align == ">":
        return " " * fill + value
    if align == "^":
        left = fill // 2
        return " " * left + value + " " * (fill - left)
    return value + " " * fill  # "<"


def render(table: Table, *, sep: str = "  ") -> list[str]:
    """Render a Table into lines. Returns list of str (no trailing newlines).

    Column widths auto-expand to fit the widest cell (label or data).
    """
    cols = table.columns
    widths = [len(c.label) for c in cols]

    for row in table.rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    lines: list[str] = []

    # Header
    header = sep.join(_pad(c.label, widths[i], c.align) for i, c in enumerate(cols))
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    for row in table.rows:
        line = sep.join(
            _pad(cell, widths[i], cols[i].align) for i, cell in enumerate(row)
        )
        lines.append(line)

    return lines


def render_from_dicts(
    headers: Sequence[str],
    rows: Sequence[dict[str, str]],
    *,
    align: Align = "<",
    sep: str = "  ",
) -> list[str]:
    """Convenience: build and render a Table from list-of-dicts.

    All columns share the same alignment. For mixed alignment, use Table + Col directly.
    """
    if not rows:
        return []

    cols = tuple(Col(label=h, align=align) for h in headers)
    str_rows = tuple(tuple(str(row.get(h, "")) for h in headers) for row in rows)
    return render(Table(columns=cols, rows=str_rows), sep=sep)
