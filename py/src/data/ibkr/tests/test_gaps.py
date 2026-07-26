"""Test gap calculation logic — pure functions in candles.py.

No API calls. Only tests for calculate_gaps, find_internal_gaps,
and _merge_and_sort_gaps.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.data.ibkr.candles import (
    calculate_gaps,
    _merge_and_sort_gaps,
)


# ── calculate_gaps ──────────────────────────────────────────────

# Correct ms timestamps for UTC dates:
#   2026-01-01 00:00:00 UTC = 1767225600000
#   2026-06-01 00:00:00 UTC = 1780272000000
#   2026-02-01 00:00:00 UTC = 1769904000000

_JAN1 = 1767225600000
_JUN1 = 1780272000000


class TestCalculateGaps:
    """Pure function tests — no DB, no API."""

    def test_no_existing_data(self):
        """No data at all → full range is one gap."""
        gaps = calculate_gaps(
            datetime(2026, 1, 1),
            datetime(2026, 6, 1),
            None,
            None,
        )
        assert len(gaps) == 1
        assert gaps[0] == (datetime(2026, 1, 1), datetime(2026, 6, 1))

    def test_full_coverage(self):
        """Existing data covers the whole range → no gaps."""
        gaps = calculate_gaps(
            datetime(2026, 1, 1),
            datetime(2026, 6, 1),
            _JAN1,
            _JUN1,
        )
        assert gaps == []

    def test_backward_gap_only(self):
        """Request starts before oldest data → gap at start."""
        gap = calculate_gaps(
            datetime(2025, 12, 1),
            datetime(2026, 6, 1),
            _JAN1,
            _JUN1,
        )
        assert len(gap) == 1
        assert gap[0][0] == datetime(2025, 12, 1)
        assert gap[0][1] == datetime(2026, 1, 1)

    def test_forward_gap_only(self):
        """Request ends after newest data → gap at end."""
        gap = calculate_gaps(
            datetime(2026, 1, 1),
            datetime(2026, 7, 1),
            _JAN1,
            _JUN1,
        )
        # Jun 1 → Jul 1 spans trading days → gap preserved
        assert len(gap) == 1
        assert gap[0][0] == datetime(2026, 6, 1)
        assert gap[0][1] == datetime(2026, 7, 1)

    def test_forward_gap_weekend_filtered(self):
        """Forward gap that is purely weekend → filtered out.

        Jun 5 2026 is a Friday. Jun 7 is Sunday.
        Gap from Fri evening to Sun morning = only non-trading days.
        """
        jun5_fri_19 = datetime(2026, 6, 5, 19, 0)
        jun7_sun_10 = datetime(2026, 6, 7, 10, 0)
        # Newest existing = Jun 5 19:00 (Fri after market)
        newest_ts = int(jun5_fri_19.replace(tzinfo=timezone.utc).timestamp() * 1000)
        gaps = calculate_gaps(
            datetime(2026, 1, 1),
            jun7_sun_10,
            _JAN1,
            newest_ts,
        )
        # The forward gap is Fri 19:00 → Sun 10:23 — all non-trading
        assert len(gaps) == 0

    def test_forward_gap_during_trading_day(self):
        """Forward gap starting during market hours → preserved."""
        jun5_fri_10 = datetime(2026, 6, 5, 10, 0)
        newest_ts = int(jun5_fri_10.replace(tzinfo=timezone.utc).timestamp() * 1000)
        gaps = calculate_gaps(
            datetime(2026, 1, 1),
            datetime(2026, 6, 7, 10, 0),
            _JAN1,
            newest_ts,
        )
        # Gap starts Fri 10:00 during market → real gap
        assert len(gaps) == 1

    def test_both_gaps(self):
        """Range extends both before oldest and after newest."""
        gaps = calculate_gaps(
            datetime(2025, 11, 1),
            datetime(2026, 7, 1),
            _JAN1,
            _JUN1,
        )
        assert len(gaps) == 2
        assert gaps[0][0] == datetime(2025, 11, 1)
        assert gaps[0][1] == datetime(2026, 1, 1)
        assert gaps[1][0] == datetime(2026, 6, 1)
        assert gaps[1][1] == datetime(2026, 7, 1)

    def test_exact_boundary_no_gap(self):
        """Request exactly = existing range → no gap."""
        gaps = calculate_gaps(
            datetime(2026, 1, 1),
            datetime(2026, 6, 1),
            _JAN1,
            _JUN1,
        )
        assert gaps == []

    def test_existing_is_subset(self):
        """Existing data is in the middle → gaps before and after.

        1768089600000 = 2026-01-10 00:00:00 UTC
        1771747200000 = 2026-02-22 00:00:00 UTC  (example mid-range)
        """
        oldest_ts = int(
            datetime(2026, 1, 10, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000
        )
        newest_ts = int(
            datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000
        )
        gaps = calculate_gaps(
            datetime(2026, 1, 1),
            datetime(2026, 12, 31),
            oldest_ts,
            newest_ts,
        )
        assert len(gaps) == 2
        assert gaps[0][0] == datetime(2026, 1, 1)
        assert gaps[0][1] == datetime(2026, 1, 10)
        assert gaps[1][0] == datetime(2026, 4, 20)
        assert gaps[1][1] == datetime(2026, 12, 31)

    def test_only_oldest_provided(self):
        """Only oldest, no newest → forward gap."""
        gaps = calculate_gaps(
            datetime(2026, 1, 1),
            datetime(2026, 6, 1),
            _JAN1,
            None,
        )
        assert len(gaps) == 1
        assert gaps[0][0] == datetime(2026, 1, 1)
        assert gaps[0][1] == datetime(2026, 6, 1)

    def test_only_newest_provided(self):
        """Only newest, no oldest → backward gap."""
        gaps = calculate_gaps(
            datetime(2026, 1, 1),
            datetime(2026, 6, 1),
            None,
            _JUN1,
        )
        assert len(gaps) == 1
        assert gaps[0][0] == datetime(2026, 1, 1)
        assert gaps[0][1] == datetime(2026, 6, 1)


# ── _merge_and_sort_gaps ────────────────────────────────────────


class TestMergeAndSortGaps:
    def test_empty(self):
        assert _merge_and_sort_gaps([]) == []

    def test_single(self):
        g = [(datetime(2026, 1, 1), datetime(2026, 1, 5))]
        assert _merge_and_sort_gaps(g) == g

    def test_merge_overlapping(self):
        gaps = [
            (datetime(2026, 1, 1), datetime(2026, 1, 10)),
            (datetime(2026, 1, 5), datetime(2026, 1, 15)),
        ]
        merged = _merge_and_sort_gaps(gaps)
        assert len(merged) == 1
        assert merged[0] == (datetime(2026, 1, 1), datetime(2026, 1, 15))

    def test_merge_adjacent(self):
        """End of first = start of second → merged."""
        gaps = [
            (datetime(2026, 1, 1), datetime(2026, 1, 5)),
            (datetime(2026, 1, 5), datetime(2026, 1, 10)),
        ]
        merged = _merge_and_sort_gaps(gaps)
        assert len(merged) == 1
        assert merged[0] == (datetime(2026, 1, 1), datetime(2026, 1, 10))

    def test_disjoint_kept_separate(self):
        g1 = (datetime(2026, 1, 1), datetime(2026, 1, 5))
        g2 = (datetime(2026, 1, 10), datetime(2026, 1, 15))
        merged = _merge_and_sort_gaps([g2, g1])  # unsorted input
        assert len(merged) == 2
        assert merged[0] == g1
        assert merged[1] == g2

    def test_unsorted_input(self):
        gaps = [
            (datetime(2026, 2, 1), datetime(2026, 2, 10)),
            (datetime(2026, 1, 1), datetime(2026, 1, 10)),
        ]
        merged = _merge_and_sort_gaps(gaps)
        assert merged[0][0] == datetime(2026, 1, 1)
        assert merged[1][0] == datetime(2026, 2, 1)
