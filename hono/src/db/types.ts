export interface DatabaseSchema {
  symbols: ISymbol;
  watchlist: Watchlist;
  watchlist_symbols: WatchlistSymbolsTable;
  ohlcv_1min: OhlcvTable;
  ohlcv_5min: OhlcvTable;
  ohlcv_15min: OhlcvTable;
  ohlcv_30min: OhlcvTable;
  ohlcv_1h: OhlcvTable;
  ohlcv_4h: OhlcvTable;
  ohlcv_1d: OhlcvTable;
  ohlcv_1w: OhlcvTable;
}

export interface OhlcvTable {
  id?: number;
  symbol_id: number;
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Watchlist {
  id?: number;
  name: string;
  notes: string | null;
  strategy: string | null;
}

export interface WatchlistSymbolsTable {
  id?: number;
  watchlist_id: number;
  symbol_id: number;
}

export interface ISymbol {
  id: number;
  symbol: string;
  name: string | null;
  market: string;
  currency: string;
}
