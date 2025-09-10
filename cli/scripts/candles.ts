import { attemptAsync } from "es-toolkit";
import { client } from "./shared";
import { parseArgs } from "util";

const { positionals } = parseArgs({
  args: Bun.argv.slice(-3),
  tokens: true,
  allowPositionals: true,
});

const [conid, period, bar] = positionals;

const params = { conid, period, bar };
// Dict("conid" => string(conid), "period" => period, "bar" => interval)
const [err, r] = await attemptAsync(() =>
  client.get("iserver/marketdata/history", { params }),
);
if (!!err) {
  console.error(`${err}`);
  process.exit(1);
}
const data = r!.data as Array<{
  o: number;
  c: number;
  h: number;
  l: number;
  v: number;
  t: number;
}>;
console.log(data);
