import { Hono } from "hono";
import { Main } from "../components/main";
import { lookup } from "./trading/lookup";
import { candles } from "./trading/candles";
import { getContractInfo } from "./trading/shared";
import {
  createWatchlist,
  addSymbolsToWatchlist,
  getAllWatchlists,
  getSymbolsInWatchlist,
} from "./trading/watchlists";
import { client } from "./shared/client";
import { db } from "../db/db";
import { BAR_INTERVAL } from "../consts/interval";
import { AddableSymbol } from "../components/addable-symbol";
import symbols from "./symbols/";

const main = new Hono();

main.route("/symbols", symbols);

main.get("/", (c) => c.render(<Main />));

main.get("/tickle", async (c) => {
  const r = await client.get("/tickle");
  return c.json(r.data);
});

main.post("/lookup", async (c) => {
  const formData = await c.req.formData();
  const symbol = formData.get("symbol")! as string;
  const results = await lookup(symbol);
  if (!results) {
    return c.text("nothing");
  }
  if (results.length == 0) {
    return c.render(<p>nothing found</p>);
  }
  const html = results.map((r) => <AddableSymbol symbol={r} />);
  return c.render(<>{html}</>);
});

main.post("/download-candles", async (c) => {
  const fd = await c.req.formData();
  const { conid, period, bar } = Object.fromEntries(fd.entries());
  await candles({ conid: +conid!, period: period!, bar: bar! });
  return c.html(
    "<p class='text-green-600'>Candles downloaded successfully</p>",
  );
});

main.get("/candles/:conid/:bar?", async (c) => {
  const conid = +c.req.param("conid");
  const bar = c.req.param("bar") || "1d";
  if (!BAR_INTERVAL.includes(bar as any))
    return c.render(<p>Invalid bar interval</p>);
  const tableName = `ohlcv_${bar}`;

  const symbolInfo = await getContractInfo(conid);
  if (!symbolInfo) return c.render(<p>Symbol not found</p>);

  const candlesData: any[] = await (db as any)
    .selectFrom(tableName)
    .where("symbol_id", "=", symbolInfo.id)
    .orderBy("timestamp", "desc")
    .limit(100)
    .selectAll()
    .execute();

  return c.render(
    <div>
      <h2 className="text-xl font-bold mb-4">
        Candles for {symbolInfo.symbol} ({bar})
      </h2>
      <table className="table-auto border-collapse border border-gray-300">
        <thead>
          <tr>
            <th className="border border-gray-300 px-4 py-2">Timestamp</th>
            <th className="border border-gray-300 px-4 py-2">Open</th>
            <th className="border border-gray-300 px-4 py-2">High</th>
            <th className="border border-gray-300 px-4 py-2">Low</th>
            <th className="border border-gray-300 px-4 py-2">Close</th>
            <th className="border border-gray-300 px-4 py-2">Volume</th>
          </tr>
        </thead>
        <tbody>
          {candlesData.map((candle) => (
            <tr key={candle.id}>
              <td className="border border-gray-300 px-4 py-2">
                {candle.timestamp.toISOString()}
              </td>
              <td className="border border-gray-300 px-4 py-2">
                {candle.open}
              </td>
              <td className="border border-gray-300 px-4 py-2">
                {candle.high}
              </td>
              <td className="border border-gray-300 px-4 py-2">{candle.low}</td>
              <td className="border border-gray-300 px-4 py-2">
                {candle.close}
              </td>
              <td className="border border-gray-300 px-4 py-2">
                {candle.volume}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>,
  );
});

main.get("/watchlists", async (c) => {
  const watchlists = await getAllWatchlists();
  const watchlistsWithCounts = await Promise.all(
    watchlists.map(async (w) => {
      const symbols = await getSymbolsInWatchlist(w.id!);
      return { ...w, symbolCount: symbols.length };
    }),
  );

  return c.render(
    <div>
      <h2 className="text-xl font-bold mb-4">Watchlists</h2>
      <ul className="space-y-2">
        {watchlistsWithCounts.map((w) => (
          <li key={w.id} className="border p-4 rounded">
            <a
              href={`/watchlist/${w.id}`}
              className="text-blue-600 hover:underline"
            >
              {w.name}
            </a>
            <p>Symbols: {w.symbolCount}</p>
            {w.notes && <p>Notes: {w.notes}</p>}
          </li>
        ))}
      </ul>
    </div>,
  );
});

main.get("/watchlist/:id", async (c) => {
  const id = +c.req.param("id");
  const symbols = await getSymbolsInWatchlist(id);

  return c.render(
    <div>
      <h2 className="text-xl font-bold mb-4">Watchlist Symbols</h2>
      <table className="table-auto border-collapse border border-gray-300">
        <thead>
          <tr>
            <th className="border border-gray-300 px-4 py-2">Symbol</th>
            <th className="border border-gray-300 px-4 py-2">Name</th>
            <th className="border border-gray-300 px-4 py-2">Market</th>
            <th className="border border-gray-300 px-4 py-2">Currency</th>
            <th className="border border-gray-300 px-4 py-2">Actions</th>
          </tr>
        </thead>
        <tbody>
          {symbols.map((s) => (
            <tr key={s.id}>
              <td className="border border-gray-300 px-4 py-2">{s.symbol}</td>
              <td className="border border-gray-300 px-4 py-2">
                {s.name || ""}
              </td>
              <td className="border border-gray-300 px-4 py-2">{s.market}</td>
              <td className="border border-gray-300 px-4 py-2">{s.currency}</td>
              <td className="border border-gray-300 px-4 py-2">
                <a
                  href={`/candles/${s.id}`}
                  className="text-blue-600 hover:underline"
                >
                  View Candles
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>,
  );
});

export default main;
