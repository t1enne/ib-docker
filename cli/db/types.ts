export interface DatabaseSchema {
  symbol: SymbolSchema;
  candle: CandleSchema;
}

export interface CandleSchema {
  ticker: string;
  conid: number;
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface SymbolSchema {
  conid: number;
  ticker: string;
  name: string | null;
  market: string;
  currency: string;
}
