import { getWatchlists } from "./watchlists";
import { candles } from "./candles";
import { delay } from "es-toolkit";

export async function getWatchlistCandles(bar = "1d", period = "720d") {
  const securities = await getWatchlists();
  console.log(`Got ${securities.length} securities`);
  return Promise.all(
    securities
      .flat()
      .map((sec) =>
        delay(500).then(() => candles({ conid: sec.conid, bar, period })),
      ),
  );
}
