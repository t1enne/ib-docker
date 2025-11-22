import { attemptAsync, invariant } from "es-toolkit";
import { candles } from "../src/candles";
import { parseArgs } from "util";
const { positionals } = parseArgs({ allowPositionals: true });

const [conid, period, bar] = positionals;
invariant(conid, "Missing contract id");
invariant(period, "Missing period. Ex: 90d");
invariant(bar, "Missing bar. Ex: 1h");

const [e] = await attemptAsync(() => candles({ conid: +conid, bar, period }));
if (e) {
  console.error(e);
}
