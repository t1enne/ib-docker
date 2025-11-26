import dayjs from "dayjs";
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
          <input type="date" name="startDate" placeholder="Start Date" />
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
      {candles && (
        <>
          {/* Market Data Table */}
          <h2 className="text-lg font-semibold">Market Data (1d)</h2>
          <div className="overflow-auto h-[600px] border">
            <table className="table-auto border-collapse w-full">
              <thead>
                <tr>
                  <th className="border px-4 py-2">Timestamp</th>
                  <th className="border px-4 py-2">Open</th>
                  <th className="border px-4 py-2">High</th>
                  <th className="border px-4 py-2">Low</th>
                  <th className="border px-4 py-2">Close</th>
                  <th className="border px-4 py-2">Volume</th>
                </tr>
              </thead>
              <tbody>
                {candles.map((candle) => (
                  <tr key={candle.id}>
                    <td className="border px-4 py-2">
                      {dayjs(candle.timestamp).format("YY/MM/DD-HH:mm")}
                    </td>
                    <td className="border px-4 py-2">{candle.open}</td>
                    <td className="border px-4 py-2">{candle.high}</td>
                    <td className="border px-4 py-2">{candle.low}</td>
                    <td className="border px-4 py-2">{candle.close}</td>
                    <td className="border px-4 py-2">{candle.volume}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
