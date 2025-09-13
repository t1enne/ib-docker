import { parseArgs } from "util";
import { lookup } from "../src/lookup";
import { invariant } from "es-toolkit";

const { positionals } = parseArgs({ allowPositionals: true });
const symbol = positionals[0];
invariant(symbol, "need to pass a symbol");
const r = await lookup(symbol.toUpperCase());
const out = r
  .map(
    (sec) =>
      `${sec.conid};${sec.companyHeader};${sec.companyName};${sec.symbol}`,
  )
  .join("\n");
console.log(out);
