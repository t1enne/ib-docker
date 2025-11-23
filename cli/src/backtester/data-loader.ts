import { db } from "../../db/db";
import type { OHLCV } from "./types";

export const loadOHLCV = (table: string, symbolId: number): Promise<OHLCV[]> =>
  db
    .selectFrom(table as "ohlcv_1d")
    .where("symbol_id", "=", symbolId)
    .orderBy("timestamp", "asc")
    .selectAll()
    .execute();
