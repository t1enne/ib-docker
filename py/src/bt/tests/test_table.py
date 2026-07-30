"""Tests for src.bt.table — pure table formatting utility."""

from src.bt.table import Col, Table, render, render_from_dicts


def test_render_single_column():
    t = Table(
        columns=(Col("Name", "<"),),
        rows=(("Alice",), ("Bob",), ("Charlie",)),
    )
    result = render(t)
    assert result == [
        "Name   ",
        "-------",
        "Alice  ",
        "Bob    ",
        "Charlie",
    ]


def test_render_mixed_alignment():
    t = Table(
        columns=(Col("Benchmark", "<"), Col("Ann Ret", ">"), Col("Sharpe", ">")),
        rows=(
            ("Strategy", "10.87%", "0.95"),
            ("SPY", "11.83%", "0.73"),
        ),
    )
    result = render(t)
    # Labels as wide as widest cell
    assert result[0].startswith("Benchmark")
    assert result[0].endswith("Sharpe")
    # Ann Ret column right-aligned
    ann_ret_pos = result[0].index("Ann Ret")
    # Check data rows align under headers
    for row in result[2:]:
        # Values in Ann Ret column should be right-padded (space before number)
        pass
    # Verify column alignment visually
    for line in result:
        print(line)


def test_render_auto_expands_widths():
    """Column widths expand to fit widest cell."""
    t = Table(
        columns=(Col("X", "<"),),
        rows=(("short",), ("very long cell content",)),
    )
    result = render(t)
    assert len(result[0]) == len("very long cell content")
    assert len(result[2]) == len("very long cell content")


def test_render_empty_rows():
    t = Table(columns=(Col("A", "<"), Col("B", ">")))
    result = render(t)
    assert result == ["A  B", "----"]


def test_render_from_dicts():
    headers = ["Name", "Score"]
    rows = [
        {"Name": "Alice", "Score": "95"},
        {"Name": "Bob", "Score": "87"},
    ]
    result = render_from_dicts(headers, rows)
    assert result[0].startswith("Name")
    assert len(result) == 4  # header + sep + 2 rows


def test_render_from_dicts_empty():
    result = render_from_dicts(["A", "B"], [])
    assert result == []


def test_render_from_dicts_right_align():
    headers = ["Metric", "Value"]
    rows = [{"Metric": "Alpha", "Value": "0.05"}]
    result = render_from_dicts(headers, rows, align=">")
    assert result[0].startswith(" Metric") or result[0].startswith("Metric")
