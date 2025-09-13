import axios from "axios";
import { attemptAsync, invariant } from "es-toolkit";
import https from "https";
import type { SymbolInfo } from "../types/ibkr";

export const client = axios.create({
  baseURL: "https://localhost:5000/v1/api/",
  httpsAgent: new https.Agent({ rejectUnauthorized: false }),
});

export async function fetchContractInfo(conid: number) {
  const ep = `iserver/contract/${conid}/info`;
  const [err, r] = await attemptAsync(() => client.get<SymbolInfo>(ep));
  invariant(r, `failed call to ${ep}: ${err}`);
  return r.data;
}
