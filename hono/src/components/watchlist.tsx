import { invariant } from "es-toolkit";
import { db } from "../db/db";
import { getWatchlistWithSymbols } from "../modules/trading/watchlists";

interface Props {
  watchlist: Awaited<ReturnType<typeof getWatchlistWithSymbols>>;
}

export async function Watchlist({ watchlist }: Props) {
  invariant(watchlist, "Exists");
  const allSymbols = await db.selectFrom("symbol").selectAll().execute();

  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <div>
          <h1>
            Name: <b>{watchlist.name}</b>
          </h1>
          <p>Notes: {watchlist.notes}</p>
          <p>Strategy: {watchlist.strategy}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h2>Symbols in Watchlist ({watchlist.symbols.length})</h2>
          <div className="max-h-96 overflow-y-auto">
            <div>
              {watchlist.symbols.map((symbol) => (
                <div
                  key={symbol.id}
                  className="flex justify-between items-center border-b"
                >
                  <div>
                    <a href={`/symbols/${symbol.ticker}`}>{symbol.ticker}</a>
                    <span>
                      {" "}
                      {symbol.market} • {symbol.currency}
                    </span>
                    {symbol.name && <span>({symbol.name})</span>}
                  </div>
                  <button
                    class="link"
                    hx-delete={`/watchlists/${watchlist.id}/symbols/${symbol.id}`}
                    hx-confirm={`Remove ${symbol.ticker} from watchlist?`}
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Add Symbols */}
        <div>
          <h2>Add Symbols</h2>
          <div>
            <form
              hx-post={`/watchlists/${watchlist.id}/symbols`}
              hx-target="#add-symbols-result"
              hx-swap="innerHTML"
            >
              <div>
                <div className="max-h-48 overflow-y-auto border">
                  {allSymbols
                    .filter(
                      (symbol) =>
                        !watchlist.symbols.some((ws) => ws.id === symbol.id),
                    )
                    .map((symbol) => (
                      <label
                        key={symbol.id}
                        className="flex items-center space-x-2 p-1"
                      >
                        <input
                          type="checkbox"
                          name="symbolIds"
                          value={symbol.id}
                        />
                        <span className="text-sm">
                          {symbol.ticker} - {symbol.market} ({symbol.currency})
                          {symbol.name && ` - ${symbol.name}`}
                        </span>
                      </label>
                    ))}
                </div>
              </div>
              <button type="submit" className="px-4 py-2 ">
                Add Selected Symbols
              </button>
            </form>
            <div id="add-symbols-result" className="mt-2"></div>
          </div>
        </div>
      </div>
    </div>
  );
}
