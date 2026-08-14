"""Tests for the process-pool runner (`src.bt.parallel`).

Validates that `run_in_processes`:
- broadcasts `init_data` via `pool_init` (readable from the per-worker
  `WORKER_STATE` cache),
- runs `fn(task)` for each task and returns results in input order,
- invokes `on_complete(index, result)` in input order,
- handles the empty-tasks, single-task and sequential (`workers=1`) paths.

Real-process (workers > 1) equivalence against the backtest engine is covered
by ``test_sweep_pooled_matches_sequential_engine`` in the sweep suite — we
avoid a redundant forkserver spawn here to keep `make check` in budget
(spawn is ~2s per pool). Worker functions must be module-level to pickle by
qualified name across the process boundary.
"""

from __future__ import annotations

import time

import pytest

from src.bt.parallel import WORKER_STATE, run_in_processes


def _echo_worker(task: tuple[int, str]) -> tuple[str, int]:
    """Task = (payload, key); returns (key, payload + shared byte)."""
    byte = WORKER_STATE["byte"]
    return (task[1], task[0] + byte)


def _slow_independent(task: tuple[float, int]) -> int:
    """Sleep by task's first field so completion order != input order."""
    delay, ident = task
    time.sleep(delay)
    return ident


def _tasks() -> list[tuple[int, str]]:
    return [(1, "a"), (2, "b"), (3, "c"), (4, "d")]


def _expected(tasks: list[tuple[int, str]], byte: int) -> list[tuple[str, int]]:
    return [(s, payload + byte) for payload, s in tasks]


def test_sequential_streams_results_in_input_order():
    """The sequential fallback runs the worker and streams in input order.

    Real-process equivalence (workers > 1) is covered by
    ``test_sweep_pooled_matches_sequential_engine`` in the sweep suite — no
    redundant forkserver spawn here, keeping `make check` in budget.
    """
    tasks = _tasks()
    order: list[int] = []

    results = run_in_processes(
        _echo_worker,
        tasks,
        workers=1,
        init_data={"byte": 100},
        on_complete=lambda i, r: order.append(i),
    )

    assert results == _expected(tasks, 100)
    assert order == [0, 1, 2, 3]


def test_results_in_input_order_after_out_of_order_completion():
    """Slower tasks land in their original slot; on_complete stays in order."""
    tasks = [(0.05, 0), (0.0, 1), (0.03, 2), (0.0, 3)]
    order: list[int] = []

    results = run_in_processes(
        _slow_independent,
        tasks,
        workers=1,
        on_complete=lambda i, r: order.append(i),
    )

    assert results == [0, 1, 2, 3]
    assert order == [0, 1, 2, 3]


def test_worker_state_broadcast_to_clean_worker():
    """A fresh worker (no init_data seen before) reads the broadcast cache."""
    res = run_in_processes(_echo_worker, _tasks(), workers=1, init_data={"byte": 5})
    assert res == _expected(_tasks(), 5)


def test_empty_tasks_return_empty():
    assert run_in_processes(_echo_worker, [], workers=2, init_data={"byte": 1}) == []


def test_single_task_runs_without_pool():
    res = run_in_processes(_echo_worker, [(7, "x")], workers=4, init_data={"byte": 3})
    assert res == [("x", 10)]


@pytest.mark.slow
# process-pool-spawning (capitalized to 2 workers) — skipped via `-m not slow`.
def test_workers_capped_at_task_count():
    """Requesting more workers than tasks must not error or change results."""
    tasks = [(1, "a"), (2, "b")]  # 2 tasks
    # workers=8 far exceeds the task count -> effectively 2 workers.
    res = run_in_processes(_echo_worker, tasks, workers=8, init_data={"byte": 0})
    assert res == [("a", 1), ("b", 2)]

    # And equals the sequential result.
    seq = run_in_processes(_echo_worker, tasks, workers=1, init_data={"byte": 0})
    assert res == seq


def test_on_complete_optional_is_harmless():
    res = run_in_processes(_echo_worker, _tasks(), workers=1, init_data={"byte": 0})
    assert res == _expected(_tasks(), 0)
