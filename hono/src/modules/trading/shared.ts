import { attemptAsync, invariant } from "es-toolkit";
import type { SymbolInfo } from "../../types/ibkr";
import { db } from "../../db/db";
import { client } from "../shared/client";

export async function getContractInfo(conid: number) {
  const savedSymbol = await (db as any)
    .selectFrom("symbol")
    .selectAll()
    .where("id", "=", conid)
    .executeTakeFirst();
  return savedSymbol ?? fetchContractInfo(conid);
}

export async function fetchContractInfo(conid: number) {
  const ep = `iserver/contract/${conid}/info`;
  const [err, r] = await attemptAsync(() => client.get<SymbolInfo>(ep));
  invariant(r, `failed call to ${ep}: ${err}`);
  console.log({ r });
  const values = await db
    .insertInto("symbol")
    .values({
      id: r.data.con_id,
      name: r.data.company_name,
      currency: r.data.currency,
      ticker: r.data.symbol,
      market: r.data.exchange,
    })
    .returningAll()
    .executeTakeFirst();
  return values;
}
