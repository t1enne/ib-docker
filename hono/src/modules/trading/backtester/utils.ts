import { invariant } from "es-toolkit";
import { TradeDir, type Context, type Position } from "./types";
import { isNumber } from "es-toolkit/compat";

export function getExitPrice(ctx: Context, openposition: Position) {
  const { direction, sl, tp } = openposition;
  const long = direction == TradeDir.long;
  const candle = ctx.ohlcv;
  invariant(isNumber(sl) && isNumber(tp), "should be defined");
  if (long) {
    if (candle.low <= sl) {
      return sl;
    }
    if (candle.high >= tp) {
      return tp;
    }
    return;
  }
  // short
  if (candle.high >= sl) {
    return sl;
  }
  if (candle.low <= tp) {
    return tp;
  }
}
