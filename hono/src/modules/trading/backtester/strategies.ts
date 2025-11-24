import { TradeDir, type Context, type Position, type Strategy } from "./types";

export class BuyAndHold implements Strategy {
  name = "buyandhold";
  getSignal(context: Context): number {
    if (context.positions.open.length > 0) {
      return 0;
    }
    return 1;
  }

  handleSignal(context: Context, signal: number) {
    if (signal === 0) {
      return undefined;
    }

    const candle = context.ohlcv;
    const equity = context.equity;
    const allocation = equity;
    const size = allocation / candle.close;
    return {
      direction: signal > 0 ? TradeDir.long : TradeDir.short,
      tp: 1e10,
      sl: -1,
      size,
      entryPrice: candle.close,
      // trailAmount: candle.close * 0.05, // 5% trailing stop
      timestamp: candle.timestamp,
    } satisfies Position;
  }
}

export class EMACrossStrategy implements Strategy {
  name = "emacross";
  private fastPrices: number[] = [];
  private slowPrices: number[] = [];
  private fastEMA: number | null = null;
  private slowEMA: number | null = null;
  private prevSignal = 0;

  constructor(
    private fast: number,
    private slow: number,
  ) {}

  getSignal(context: Context): number {
    const price = context.ohlcv.close;
    this.fastPrices.push(price);
    this.slowPrices.push(price);

    // Keep last 50 for fast EMA, 200 for slow EMA
    if (this.fastPrices.length > this.fast) this.fastPrices.shift();
    if (this.slowPrices.length > this.slow) this.slowPrices.shift();

    const hasEnoughCandles = this.slowPrices.length >= this.slow;
    if (!hasEnoughCandles) {
      return 0;
    }

    // Calculate EMA
    const fastAlpha = 2 / (50 + 1);
    const slowAlpha = 2 / (200 + 1);

    this.fastEMA = this.fastEMA
      ? fastAlpha * price + (1 - fastAlpha) * this.fastEMA
      : this.fastPrices.reduce((a, b) => a + b, 0) / this.fastPrices.length;
    this.slowEMA = this.slowEMA
      ? slowAlpha * price + (1 - slowAlpha) * this.slowEMA
      : this.slowPrices.reduce((a, b) => a + b, 0) / this.slowPrices.length;

    if (!this.fastEMA || !this.slowEMA) return 0;

    const signal = this.fastEMA > this.slowEMA ? 1 : -1;

    // Signal on crossover
    if (signal !== this.prevSignal) {
      this.prevSignal = signal;
      return signal;
    }

    this.prevSignal = signal;
    return 0;
  }

  handleSignal(context: Context, signal: number) {
    if (signal === 0) {
      return undefined;
    }

    const candle = context.ohlcv;
    const equity = context.equity;
    const allocation = equity * 0.1; // 10% of equity
    const size = allocation / candle.close;
    const tp = signal > 0 ? candle.close * 1.1 : candle.close * 0.9;
    const sl = signal > 0 ? candle.close * 0.9 : candle.close * 1.1;
    return {
      direction: signal > 0 ? TradeDir.long : TradeDir.short,
      tp,
      sl,
      size,
      entryPrice: candle.close,
      trailAmount: candle.close * 0.1, // 20% trailing stop
      timestamp: candle.timestamp,
    } satisfies Position;
  }
}

// Add more strategies here
