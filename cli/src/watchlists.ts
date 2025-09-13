/**
 * More info:
 * https://www.interactivebrokers.com/campus/ibkr-api-page/webapi-ref/#tag/Trading-Watchlists/paths/~1iserver~1watchlists/get
 */
import { attemptAsync, delay, invariant } from "es-toolkit";
import { client } from "./shared";
import type { IWatchedSecurity, IWatchlist } from "../types/ibkr";
import * as v from "valibot";

export async function getWatchlistInstruments(id: string) {
  // id must be a numeric string
  v.parse(v.pipe(v.string(), v.regex(/\d+/)), id);

  const [err, r] = await attemptAsync(() =>
    client.get("iserver/watchlist", { params: { id } }),
  );
  if (err) {
    console.error(`${err}`);
    process.exit(1);
  }
  const instruments = r!.data.instruments as IWatchedSecurity[];
  return instruments;
}

export async function getWatchlists() {
  const params = { SC: "USER_WATCHLIST" };
  const [err, r] = await attemptAsync(() =>
    client.get("iserver/watchlists", { params }),
  );
  invariant(r, `${err}`);
  const watchlists = r.data.data.user_lists as IWatchlist[];
  const securitiesMatrix = await Promise.all(
    watchlists.map((wl) =>
      delay(500).then(() => getWatchlistInstruments(wl.id)),
    ),
  );
  return securitiesMatrix.flat();
}
