import { attemptAsync, invariant } from "es-toolkit";
import { client } from "../shared/client";
import type { ISecurity } from "../../types/ibkr";

export async function lookup(symbol: string) {
  const payload = { symbol };
  const [err, r] = await attemptAsync(() =>
    client.post<{ data: ISecurity[] }>("iserver/secdef/search", payload),
  );
  invariant(r, `${err}`);
  return r.data.data;
}
