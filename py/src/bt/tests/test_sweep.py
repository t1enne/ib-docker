"""Tests for the param sweep module (grid math, config building, ranking)."""

from src.bt.sweep import (
    _build_config,
    product_grid,
)
from src.bt.types import StrategyConfig


def _cfg(**strategy_params) -> StrategyConfig:
    return StrategyConfig(
        name="test",
        strategy_type="dummy",
        symbols=["A", "B"],
        stop_loss=0.5,
        take_profit=0.8,
        initial_capital=100000.0,
        position_size=0.95,
        commission=0.05,
        training_start="2020-01-01",
        training_end="2020-01-02",
        trading_start="2020-01-02",
        trading_end="2021-01-01",
        bars=["1d"],
        strategy_params={"base": 1, **strategy_params},
        benchmark_symbols=["A"],
    )


def test_product_grid_empty():
    assert product_grid({}) == [{}]


def test_product_grid_single():
    assert product_grid({"a": [1, 2]}) == [{"a": 1}, {"a": 2}]


def test_product_grid_cartesian():
    grid = product_grid({"a": [1], "b": [10, 20]})
    assert grid == [{"a": 1, "b": 10}, {"a": 1, "b": 20}]


def test_product_grid_order_stable():
    grid = product_grid({"x": [1, 2], "y": [3, 4]})
    assert grid[0] == {"x": 1, "y": 3}
    assert grid[-1] == {"x": 2, "y": 4}


def test_build_config_overrides_strategy_params():
    cfg = _cfg()
    out = _build_config(cfg, {"base": 99, "extra": "new"})
    assert out.strategy_params["base"] == 99
    assert out.strategy_params["extra"] == "new"
    assert out.strategy_params["base"] != cfg.strategy_params["base"]


def test_build_config_overrides_config_fields():
    cfg = _cfg()
    out = _build_config(cfg, {"stop_loss": 0, "take_profit": 0})
    assert out.stop_loss == 0
    assert out.take_profit == 0
    assert out is not cfg  # returns a fresh copy, no mutation


def test_build_config_mixed():
    cfg = _cfg()
    out = _build_config(cfg, {"position_size": 0.5, "drift_tolerance": 0.1})
    assert out.position_size == 0.5
    assert out.strategy_params["drift_tolerance"] == 0.1
    # untouched fields preserved
    assert out.stop_loss == cfg.stop_loss
    assert out.strategy_params["base"] == 1
