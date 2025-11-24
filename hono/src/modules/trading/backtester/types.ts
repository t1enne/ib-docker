export enum TradeDir {
  long,
  short,
}
export interface OHLCV {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Context {
  equity: number;
  ohlcv: OHLCV;
  sentiment: number; // -1 to 1
  marketCycle: "bull" | "bear" | "sideways";
  positions: { open: Position[]; closed: Position[] };
  // Add more meta indicators as needed
}

export interface Strategy {
  name: string;
  getSignal(context: Context): number; // Signal: -1 to 1
  handleSignal(context: Context, signal: number): Position | undefined;
}

export interface Position {
  direction: TradeDir;
  tp: number;
  sl: number;
  timestamp: number;
  size: number;
  entryPrice: number;
  trailAmount?: number;
  trailingSL?: number;
  exitPrice?: number;
  exitTimestamp?: number;
  pnl?: number;
}

export interface BacktestResult {
  equity: number[];
  start: string;
  end: string;
  stats: {
    totalReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
    totalTrades: number;
  };
}
