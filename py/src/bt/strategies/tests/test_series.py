"""Tests for SeriesView — the cursor-truncated series type (DSL)."""

import numpy as np
import pytest

from src.bt.strategies.series import SeriesView


def _view(values, n: int) -> SeriesView:
    """SeriesView with a fixed visible length ``n`` (tests slice logic directly)."""
    return SeriesView(np.asarray(values, dtype=np.float64), lambda: n)


def test_len_matches_visible_window():
    v = _view([10.0, 20.0, 30.0, 40.0], n=3)
    assert len(v) == 3
    assert v.visible == 3


def test_negative_index_counts_from_cursor_tail():
    v = _view([10.0, 20.0, 30.0, 40.0], n=4)
    assert v[-1] == 40.0
    assert v[-2] == 30.0
    assert v[-4] == 10.0
    assert v.last() == 40.0


def test_absolute_index_clamped_to_visible():
    # Engine semantics: the visible window always starts at element 0 (bars
    # append from 0 and the cursor truncates the tail). n=2 => bars [10,20].
    v = _view([10.0, 20.0, 30.0, 40.0], n=2)
    assert v[0] == 10.0
    assert v[1] == 20.0
    # A huge absolute index clamps to the last VISIBLE bar (never a future one).
    assert v[99] == 20.0


def test_empty_view_raises():
    v = _view([], n=0)
    with pytest.raises(IndexError):
        v[-1]
    assert len(v) == 0


def test_negative_index_beyond_visible_raises():
    v = _view([10.0, 20.0], n=1)  # only 10.0 visible
    with pytest.raises(IndexError):
        v[-2]


def test_cursor_narrows_update_visible_length_lazily():
    # The crucible: a view over a growing series must never expose a bar beyond
    # the current cursor. n=3 then n=2 shrinks visibility (e.g. new run/window).
    v = _view([10.0, 20.0, 30.0], n=3)
    assert v[-1] == 30.0
    # Same view object now sees a shorter cursor -> -1 is the *new* last bar.
    v._lengther = lambda: 2  # type: ignore[attr-defined]
    assert v[-1] == 20.0


def test_iteration_is_cursor_truncated():
    v = _view([10.0, 20.0, 30.0, 40.0], n=3)
    assert list(v) == [10.0, 20.0, 30.0]


def test_nz_replaces_nan():
    v = _view([np.nan], n=1)
    assert v.nz() == 0.0
    assert v.nz(5.0) == 5.0
    v2 = _view([7.0], n=1)
    assert v2.nz() == 7.0


def test_change_difference():
    v = _view([10.0, 12.0, 15.0], n=3)
    assert v.change() == 3.0  # 15 - 12
    assert v.change(bars=2) == 5.0  # 15 - 10


def test_change_nan_guarded():
    v = _view([10.0, 12.0, np.nan], n=3)
    assert v.change() == 0.0
