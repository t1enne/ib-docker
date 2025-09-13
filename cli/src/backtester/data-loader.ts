import { db } from "../../db/db";
import type { OHLCV } from "./types";

export async function loadOHLCV(
  table: string,
  symbolId: number,
): Promise<OHLCV[]> {
  return await db
    .selectFrom(table as "ohlcv_1d")
    .where("symbol_id", "=", symbolId)
    .orderBy("timestamp", "asc")
    // .limit(limit)
    .selectAll()
    .execute();
}
