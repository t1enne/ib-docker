import { Kysely, SqliteDialect } from "kysely";
import Database from "better-sqlite3";
import path from "path";

export interface DatabaseSchema {
  symbols: {
    id: number;
    symbol: string;
    name: string | null;
  };
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
  id: number;
  symbol_id: number;
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const dbPath = path.join(process.cwd(), "db.sqlite");
const dialect = new SqliteDialect({
  database: new Database(dbPath) as any,
});
