# Algos module for trading algorithms
import src.bt.algos.pairs_trading_functional
import src.bt.algos.ema_cross
import src.bt.algos.vol_extension_pullback
import src.bt.algos.yesterday_high_breakout


def init_strat(strat_name: str):
    match strat_name:
        case "pnd":
            return src.bt.algos.pairs_trading_functional
        case "ema_cross":
            return src.bt.algos.ema_cross
        case "volatility_expansion_pullback_continuation":
            return src.bt.algos.vol_extension_pullback
        case "yesterday_high_breakout":
            return src.bt.algos.yesterday_high_breakout


__all__ = ["init_strat"]
