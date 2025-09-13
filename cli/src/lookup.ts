import { attemptAsync, invariant } from "es-toolkit";
import { client } from "./shared";
import type { ISecurity } from "../types/ibkr";

export async function lookup(symbol: string) {
  const payload = { symbol };
  const [err, r] = await attemptAsync(() =>
    client.post("iserver/secdef/search", payload),
  );
  console.log(r);
  invariant(r, `${err}`);
  return r.data as ISecurity[];
}
