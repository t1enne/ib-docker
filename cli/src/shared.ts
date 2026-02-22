import axios from "axios";
import { attemptAsync, invariant } from "es-toolkit";
import https from "https";
import type { SymbolInfo } from "../types/ibkr";
import { db } from "../db/db";

export const client = axios.create({
  baseURL: "https://localhost:5000/v1/api/",
  httpsAgent: new https.Agent({ rejectUnauthorized: false }),
  timeout: 10000,
});

export async function getContractInfo(conid: number) {
  const savedSymbol = await db
    .selectFrom("symbol")
    .selectAll()
    .where("symbol.conid", "=", conid)
    .executeTakeFirst();
  return savedSymbol ?? fetchContractInfo(conid);
}

export async function fetchContractInfo(conid: number) {
  const ep = `iserver/contract/${conid}/info`;
  const [err, r] = await attemptAsync(() => client.get<SymbolInfo>(ep));
  invariant(r, `failed call to ${ep}: ${err}`);
  const values = await db
    .insertInto("symbol")
    .values({
      conid: r.data.con_id,
      name: r.data.company_name,
      currency: r.data.currency,
      ticker: r.data.symbol,
      market: r.data.exchange,
    })
    .returningAll()
    .executeTakeFirst();
  return values;
}
