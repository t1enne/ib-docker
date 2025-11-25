import { db } from "../db/db";
import { Layout } from "./layout";

export async function Main() {
  const symbols = await (db as any).selectFrom("symbol").selectAll().execute();
  return (
    <Layout>
      <div className="space-y-6">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <nav className="flex gap-4 mb-6">
          <a href="/watchlists" className="text-blue-600 hover:underline">
            Watchlists
          </a>
        </nav>

        {/* Download Candles */}
        <div>
          <h2 className="text-lg font-semibold">Download Candles</h2>
          <form
            hx-post="/download-candles"
            hx-target="#download-result"
            hx-swap="innerHTML"
          >
            <input
              list="conid-options"
              id="conid-choice"
              className="border p-2 mr-2"
              name="conid"
              placeholder="symbol"
            />

            <datalist id="conid-options">
              {symbols.map((s: any) => (
                <option value={s.id}>{s.symbol}</option>
              ))}
            </datalist>
            <input
              type="text"
              name="period"
              placeholder="Period (e.g. 90d)"
              className="border p-2 mr-2"
            />
            <select name="bar" className="border p-2 mr-2">
              <option value="1min">1min</option>
              <option value="5min">5min</option>
              <option value="15min">15min</option>
              <option value="30min">30min</option>
              <option value="1h">1h</option>
              <option value="4h">4h</option>
              <option value="1d">1d</option>
              <option value="1w">1w</option>
            </select>
            <button type="submit" className="bg-green-500 text-white px-4 py-2">
              Download
            </button>
          </form>
          <div id="download-result" className="mt-4"></div>
        </div>
      </div>
    </Layout>
  );
}
