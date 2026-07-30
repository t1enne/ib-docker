"""Test gap calculation logic — pure functions in candles.py."""

from __future__ import annotations

from datetime import datetime, timezone

from src.data.ibkr.candles import calculate_gaps, _merge_and_sort_gaps

_JAN1 = 1767225600000
_JUN1 = 1780272000000


class TestCalculateGaps:
    def test_no_existing_data(self):
        gaps = calculate_gaps(datetime(2026, 1, 1), datetime(2026, 6, 1), None, None)
        assert len(gaps) == 1
        assert gaps[0] == (datetime(2026, 1, 1), datetime(2026, 6, 1))

    def test_full_coverage(self):
        assert (
            calculate_gaps(datetime(2026, 1, 1), datetime(2026, 6, 1), _JAN1, _JUN1)
            == []
        )

    def test_backward_gap(self):
        gaps = calculate_gaps(datetime(2025, 12, 1), datetime(2026, 6, 1), _JAN1, _JUN1)
        assert len(gaps) == 1
        assert gaps[0][0] == datetime(2025, 12, 1)

    def test_forward_gap(self):
        gaps = calculate_gaps(datetime(2026, 1, 1), datetime(2026, 7, 1), _JAN1, _JUN1)
        assert len(gaps) == 1
        assert gaps[0][1] == datetime(2026, 7, 1)

    def test_forward_gap_weekend_filtered(self):
        jun5_fri_19 = datetime(2026, 6, 5, 19, 0)
        jun7_sun_10 = datetime(2026, 6, 7, 10, 0)
        newest_ts = int(jun5_fri_19.replace(tzinfo=timezone.utc).timestamp() * 1000)
        gaps = calculate_gaps(datetime(2026, 1, 1), jun7_sun_10, _JAN1, newest_ts)
        assert len(gaps) == 0


class TestMergeAndSortGaps:
    def test_merge_overlapping(self):
        gaps = [
            (datetime(2026, 1, 1), datetime(2026, 1, 10)),
            (datetime(2026, 1, 5), datetime(2026, 1, 15)),
        ]
        merged = _merge_and_sort_gaps(gaps)
        assert len(merged) == 1
        assert merged[0] == (datetime(2026, 1, 1), datetime(2026, 1, 15))

    def test_disjoint_kept_separate(self):
        g1 = (datetime(2026, 1, 1), datetime(2026, 1, 5))
        g2 = (datetime(2026, 1, 10), datetime(2026, 1, 15))
        merged = _merge_and_sort_gaps([g2, g1])  # unsorted
        assert len(merged) == 2
        assert merged[0] == g1
        assert merged[1] == g2
