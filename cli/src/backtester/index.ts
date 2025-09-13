import { invariant, isNotNil } from "es-toolkit";
import {
  type OHLCV,
  type Context,
  type Strategy,
  type Position,
  type BacktestResult,
  TradeDir,
} from "./types";
import { getExitPrice } from "./utils";

export function displayResult(result: BacktestResult) {
  const out = `Backtest Stats:
Total Return: ${(result.stats.totalReturn * 100).toFixed(2)}%
Start Date: ${result.start}
End Date: ${result.end}
Starting Equity: ${result.equity.at(0)?.toFixed(2)}
End Equity: ${result.equity.at(-1)?.toFixed(2)}
Total Trades: ${result.stats.totalTrades}
Sharpe Ratio: ${result.stats.sharpeRatio.toFixed(2)}
Max Drawdown: ${(result.stats.maxDrawdown * 100).toFixed(2)}%
`;

  console.log(out);
}

export class BacktestEngine {
  run(
    data: OHLCV[],
    strategy: Strategy,
    initialCapital: number = 10000,
  ): BacktestResult {
    console.log(`running strategy: ${strategy.name}`);
    let equity = initialCapital;
    const positions: Record<"open" | "closed", Position[]> = {
      open: [],
      closed: [],
    };
    const equityHistory: number[] = [initialCapital];

    for (let i = 0; i < data.length; i++) {
      const candle = data[i];
      const isLast = i == data.length - 1;
      invariant(candle, "exists");
      var context: Context = {
        ohlcv: candle,
        sentiment: Math.random() * 2 - 1, // Placeholder: random sentiment
        marketCycle: "bull", // Placeholder
        positions,
        equity,
      };
      for (let i = positions.open.length - 1; i >= 0; i--) {
        const openpos = positions.open[i]!;
        const { direction, size, entryPrice } = openpos;
        // update trailing stop
        let exitPrice: number | undefined;
        if (openpos.trailAmount) {
          const trail = openpos.trailAmount;
          if (direction == TradeDir.long) {
            if (!openpos.trailingSL) {
              openpos.trailingSL = entryPrice - trail;
            }
            openpos.trailingSL = Math.max(
              openpos.trailingSL,
              candle.high - trail,
            );
            if (candle.low <= openpos.trailingSL) {
              exitPrice = openpos.trailingSL;
            }
          } else {
            if (!openpos.trailingSL) {
              openpos.trailingSL = entryPrice + trail;
            }
            openpos.trailingSL = Math.min(
              openpos.trailingSL,
              candle.low + trail,
            );
            if (candle.high >= openpos.trailingSL) {
              exitPrice = openpos.trailingSL;
            }
          }
        }
        // check SL/TP if not trailing exit
        if (!exitPrice) {
          exitPrice = getExitPrice(context, openpos);
        }
        const skip = !(isNotNil(exitPrice) || isLast);
        if (skip) {
          continue;
        }
        const dirMultiplier = direction == TradeDir.long ? 1 : -1;
        const exit = exitPrice ?? candle.close;
        const pnl = (exit - entryPrice) * size * dirMultiplier;
        equity += pnl;
        console.log(pnl);
        positions.open.splice(i, 1);
        positions.closed.push({
          ...openpos,
          exitPrice,
          exitTimestamp: candle.timestamp,
          pnl,
        });
      }
      // calculate unrealized P&L
      let unrealized = 0;
      for (const pos of positions.open) {
        const dirMultiplier = pos.direction == TradeDir.long ? 1 : -1;
        unrealized +=
          (candle.close - pos.entryPrice) * pos.size * dirMultiplier;
      }
      const totalEquity = equity + unrealized;
      equityHistory.push(totalEquity);

      const signal = strategy.getSignal(context);
      const trade = strategy.handleSignal(context, signal);
      if (trade) {
        positions.open.push(trade);
      }
    }

    // Calculate stats
    const finalEquity =
      equityHistory[equityHistory.length - 1] || initialCapital;
    const totalReturn = (finalEquity - initialCapital) / initialCapital;

    const returns = equityHistory
      .slice(1)
      .map((e, i) => (e - equityHistory[i]!) / equityHistory[i]!);

    const avgReturn =
      returns.length > 0
        ? returns.reduce((a, b) => a + b, 0) / returns.length
        : 0;
    const stdReturn =
      returns.length > 0
        ? Math.sqrt(
            returns.reduce((a, b) => a + (b - avgReturn) ** 2, 0) /
              returns.length,
          )
        : 1;
    const sharpeRatio = stdReturn !== 0 ? avgReturn / stdReturn : 0;
    let peak = initialCapital;
    let maxDrawdown = 0;
    for (const e of equityHistory) {
      if (e > peak) peak = e;
      const drawdown = (peak - e) / peak;
      if (drawdown > maxDrawdown) maxDrawdown = drawdown;
    }

    // console.log("Starting price", data.at(0)!.close);
    // console.log("End price", data.at(-1)!.close);

    return {
      equity: equityHistory,
      start: new Date(data.at(0)!.timestamp).toISOString(),
      end: new Date(data.at(-1)!.timestamp).toISOString(),
      stats: {
        totalReturn,
        sharpeRatio,
        maxDrawdown,
        totalTrades: positions.closed.length,
      },
    };
  }
}
