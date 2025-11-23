import { invariant } from "es-toolkit";
import { db } from "../db/db";
import type { Watchlist } from "../db/types";

export interface WatchlistWithSymbols extends Watchlist {
  symbols: {
    id: number;
    symbol: string;
    name?: string | null;
    market: string;
    currency: string;
  }[];
}

export async function createWatchlist(
  name: string,
  notes?: string,
  strategy?: string,
): Promise<number> {
  const result = await db
    .insertInto("watchlist")
    .values({ name, notes, strategy })
    .returning("id")
    .executeTakeFirst();
  invariant(result?.id, "error");
  return result.id;
}

export async function getWatchlist(id: number): Promise<Watchlist | undefined> {
  const result = await db
    .selectFrom("watchlist")
    .where("id", "=", id)
    .selectAll()
    .executeTakeFirst();
  return result;
}

export async function getAllWatchlists(): Promise<Watchlist[]> {
  const result = await db.selectFrom("watchlist").selectAll().execute();
  return result.map((r) => ({
    id: r.id!,
    name: r.name,
    notes: r.notes,
    strategy: r.strategy,
  }));
}

export async function updateWatchlist(
  id: number,
  updates: Partial<Pick<Watchlist, "name" | "notes" | "strategy">>,
): Promise<void> {
  console.log(id, updates);
  await db.updateTable("watchlist").set(updates).where("id", "=", id).execute();
}

export async function deleteWatchlist(id: number): Promise<void> {
  await db
    .deleteFrom("watchlist_symbols")
    .where("watchlist_id", "=", id)
    .execute();
  await db.deleteFrom("watchlist").where("id", "=", id).execute();
}

export async function addSymbolsToWatchlist(
  watchlistId: number,
  symbolIds: number[],
): Promise<void> {
  const values = symbolIds.map((symbolId) => ({
    watchlist_id: watchlistId,
    symbol_id: symbolId,
  }));
  await db.insertInto("watchlist_symbols").values(values).execute();
}

export async function removeSymbolsFromWatchlist(
  watchlistId: number,
  symbolIds: number[],
): Promise<void> {
  await db
    .deleteFrom("watchlist_symbols")
    .where("watchlist_id", "=", watchlistId)
    .where("symbol_id", "in", symbolIds)
    .execute();
}

export async function getSymbolsInWatchlist(watchlistId: number): Promise<
  {
    id: number;
    symbol: string;
    name?: string | null;
    market: string;
    currency: string;
  }[]
> {
  return (await db
    .selectFrom("watchlist_symbols")
    .innerJoin("symbols", "symbols.id", "watchlist_symbols.symbol_id")
    .where("watchlist_symbols.watchlist_id", "=", watchlistId)
    .select([
      "symbols.id",
      "symbols.symbol",
      "symbols.name",
      "symbols.market",
      "symbols.currency",
    ])
    .execute()) as {
    id: number;
    symbol: string;
    name?: string | null;
    market: string;
    currency: string;
  }[];
}

export async function getWatchlistWithSymbols(
  id: number,
): Promise<WatchlistWithSymbols | undefined> {
  const watchlist = await getWatchlist(id);
  if (!watchlist) return undefined;
  const symbols = await getSymbolsInWatchlist(id);
  return { ...watchlist, symbols };
}

export async function getSymbolId(conid: string): Promise<number | undefined> {
  console.log({ conid });
  const result = await db
    .selectFrom("symbols")
    .where("id", "=", +conid)
    .select("id")
    .executeTakeFirst();
  return result?.id;
}
