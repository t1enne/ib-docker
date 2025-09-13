import { BacktestEngine, displayResult } from "./index";
import { BuyAndHold, EMACrossStrategy } from "./strategies";
import { loadOHLCV } from "./data-loader";

async function main() {
  // Load OHLCV data for symbol_id 1 from 1d table
  const data = await loadOHLCV("ohlcv_1d", 265598); // Limit to 500 for faster testing

  if (data.length === 0) {
    console.log("No data found");
    return;
  }

  const engine = new BacktestEngine();
  const bandh = engine.run(data, new BuyAndHold());
  displayResult(bandh);
  const emacross = engine.run(data, new EMACrossStrategy(9, 14));
  displayResult(emacross);
}

main().catch(console.error);
