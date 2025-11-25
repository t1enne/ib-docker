import { attemptAsync, invariant } from "es-toolkit";
import { client } from "../shared/client";
import type { ISecurity } from "../../types/ibkr";

export async function lookup(symbol: string) {
  const [err, r] = await attemptAsync(() =>
    client.post<ISecurity[]>("iserver/secdef/search", {
      symbol: symbol.toUpperCase(),
    }),
  );
  invariant(r, `${err}`);
  return r.data;
}
