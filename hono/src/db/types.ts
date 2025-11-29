export interface DatabaseSchema {
  symbol: ISymbol;
  watchlist: IWatchlist;
  watchlist_symbol: IWatchlistSymbol;
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
  timestamp: number; //date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface IWatchlist {
  id?: number;
  name: string;
  notes: string | null;
  strategy: string | null;
}

export interface IWatchlistSymbol {
  id?: number;
  watchlist_id: number;
  symbol_id: number;
}

export interface ISymbol {
  id: number;
  ticker: string;
  name: string | null;
  market: string;
  currency: string;
}
