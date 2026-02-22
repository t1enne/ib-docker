import { db } from "../db/db";
import { attemptAsync, invariant } from "es-toolkit";
import { client } from "./shared";
import type { SymbolSchema } from "../db/types";
import type { SymbolInfo } from "../types/ibkr";

async function fetchContractInfo(conid: number) {
  const ep = `iserver/contract/${conid}/info`;
  const [err, r] = await attemptAsync(() => client.get<SymbolInfo>(ep));
  invariant(r, `failed call to ${ep}: ${err}`);
  return r.data;
}

export async function getContractInfo(conid: number): Promise<SymbolSchema> {
  const contract = await db
    .selectFrom("symbol")
    .selectAll()
    .where("conid", "=", conid)
    .executeTakeFirst();

  if (contract) {
    return contract;
  }
  const cinfo = await fetchContractInfo(conid);
  return await db
    .insertInto("symbol")
    .values({
      conid: conid,
      market: cinfo.exchange,
      currency: cinfo.currency,
      name: cinfo.company_name,
      ticker: cinfo.symbol,
    })
    .returningAll()
    .executeTakeFirstOrThrow();
}
