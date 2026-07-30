"""Tests for src.bt.table — critical formatters."""

from src.bt.table import Col, Table, render, render_from_dicts


def test_render_single_column():
    t = Table(columns=(Col("Name", "<"),), rows=(("Alice",), ("Bob",), ("Charlie",)))
    assert render(t) == ["Name   ", "-------", "Alice  ", "Bob    ", "Charlie"]


def test_render_from_dicts():
    headers = ["Name", "Score"]
    rows = [{"Name": "Alice", "Score": "95"}, {"Name": "Bob", "Score": "87"}]
    result = render_from_dicts(headers, rows)
    assert result[0].startswith("Name")
    assert len(result) == 4  # header + sep + 2 rows
