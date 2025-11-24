import { attemptAsync, invariant } from "es-toolkit";
import type { SymbolInfo } from "../../types/ibkr";
import { db } from "../../db/db";
import { client } from "../shared/client";

export async function getContractInfo(conid: number) {
  const savedSymbol = await db
    .selectFrom("symbols")
    .selectAll()
    .where("symbols.id", "=", conid)
    .executeTakeFirst();
  return savedSymbol ?? fetchContractInfo(conid);
}

export async function fetchContractInfo(conid: number) {
  const ep = `iserver/contract/${conid}/info`;
  const [err, r] = await attemptAsync(() => client.get<SymbolInfo>(ep));
  invariant(r, `failed call to ${ep}: ${err}`);
  const values = await db
    .insertInto("symbols")
    .values({
      id: r.data.con_id,
      name: r.data.company_name,
      currency: r.data.currency,
      symbol: r.data.symbol,
      market: r.data.exchange,
    })
    .returningAll()
    .executeTakeFirst();
  return values;
}
