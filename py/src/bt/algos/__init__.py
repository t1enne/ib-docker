# Algos module for trading algorithms
import src.bt.algos.trend_following
import src.bt.algos.breakout_ema
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
        case "breakout_ema":
            return src.bt.algos.breakout_ema
        case "trend_following":
            return src.bt.algos.trend_following


__all__ = ["init_strat"]
