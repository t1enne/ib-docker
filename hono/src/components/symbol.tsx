import { ISymbol, OhlcvTable } from "../db/types";

interface Props {
  symbol: ISymbol;
  candles?: OhlcvTable[];
  showDownload?: boolean;
}

function CandleDownloader({ symbolId }: { symbolId: number }) {
  return (
    <div>
      <h2 className="text-lg font-semibold">Download Candles</h2>
      <form
        hx-post="/download-candles"
        hx-target="#download-result"
        hx-swap="innerHTML"
      >
        <div className="flex gap-2">
          <input type="hidden" name="conid" value={symbolId} />
          <input type="date" name="startTime" placeholder="Start Date" />
          <input type="text" name="period" placeholder="Period (e.g. 90d)" />
          <select name="bar" className="border p-2">
            <option value="1d">1d</option>
            <option value="1w">1w</option>
            <option value="4h">4h</option>
            <option value="1h">1h</option>
            <option value="30min">30min</option>
            <option value="15min">15min</option>
            <option value="5min">5min</option>
            <option value="1min">1min</option>
          </select>
        </div>
      </form>
      <div id="download-result" className="mt-4"></div>
    </div>
  );
}

export function Symbol({ symbol, candles, showDownload }: Props) {
  return (
    <div className="space-y-6">
      <div className="">
        <div className="hidden">{symbol.id}</div>
        <div>
          <a href={`/symbols/${symbol.ticker}`}>
            <b>{symbol.name}</b>
          </a>
        </div>
        <p>MARKET: {symbol.market}</p>
        <p>CURRENCY: {symbol.currency}</p>
        <p>ID: {symbol.id}</p>
      </div>
      {showDownload && <CandleDownloader symbolId={symbol.id} />}
    </div>
  );
}
