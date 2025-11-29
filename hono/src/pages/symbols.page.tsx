import { sql } from "kysely";
import { Layout } from "../components/layout";
import { db } from "../db/db";

export default async function Page() {
  const symbols = await db
    .selectFrom("symbol")
    .select((eb) => [
      "symbol.id",
      "symbol.market",
      "symbol.currency",
      "symbol.name",
      "symbol.ticker",
      eb
        .selectFrom("ohlcv_1d")
        .select(eb.fn.countAll().as("candleCount"))
        .whereRef("ohlcv_1d.symbol_id", "=", "symbol.id")
        .as("candle_count"),
    ])
    .execute();
  const watchlists = await db.selectFrom("watchlist").selectAll().execute();

  return (
    <Layout>
      <div>
        <div className="overflow-auto h-[600px] border">
          <table className="table-auto border-collapse w-full">
            <thead>
              <tr>
                <th className="border px-4 py-2">Ticker</th>
                <th className="border px-4 py-2">Market</th>
                <th className="border px-4 py-2">Currency</th>
                <th className="border px-4 py-2">Candles</th>
                <th className="border px-4 py-2">Watchlists</th>
              </tr>
            </thead>
            <tbody>
              {symbols.map((s) => {
                return (
                  <tr key={s.id} id={`symbol-${s.id}`}>
                    <td className="border px-4 py-2">
                      <a href={`symbols/${s.ticker}`}>{s.ticker}</a>
                    </td>
                    <td className="border px-4 py-2">{s.market}</td>
                    <td className="border px-4 py-2">{s.currency}</td>
                    <td className="border px-4 py-2">{s.candleCount}</td>
                    <td className="border px-4 py-2">{}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </Layout>
  );
}
