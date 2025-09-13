import { db } from "../db/db";
import type { ISymbol } from "../db/types";
import { attemptAsync, invariant } from "es-toolkit";
import type { SymbolInfo } from "../types/ibkr";
import { client } from "./shared";

async function fetchContractInfo(conid: number) {
  const ep = `iserver/contract/${conid}/info`;
  const [err, r] = await attemptAsync(() => client.get<SymbolInfo>(ep));
  invariant(r, `failed call to ${ep}: ${err}`);
  return r.data;
}

export async function getContractInfo(conid: number): Promise<ISymbol> {
  const contract = await db
    .selectFrom("symbols")
    .selectAll()
    .where("id", "=", conid)
    .executeTakeFirst();

  if (contract) {
    return contract;
  }
  const cinfo = await fetchContractInfo(conid);
  return await db
    .insertInto("symbols")
    .values({
      id: conid,
      market: cinfo.exchange,
      currency: cinfo.currency,
      name: cinfo.company_name,
      symbol: cinfo.symbol,
    })
    .returningAll()
    .executeTakeFirstOrThrow();
}
