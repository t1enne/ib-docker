import { attemptAsync } from "es-toolkit";
import { client } from "./shared";
import { parseArgs } from "util";

const { positionals } = parseArgs({
  args: Bun.argv.slice(-1),
  tokens: true,
  allowPositionals: true,
});

const payload = { symbol: positionals[0]?.toUpperCase() };
const [err, r] = await attemptAsync(() =>
  client.post("iserver/secdef/search", payload),
);
if (!!err) {
  console.error(`${err}`);
  process.exit(1);
}

const options = r!.data.map(
  (sec, i) =>
    `${i + 1}) ${sec.companyHeader} [${sec.conid}]. ${sec.description}`,
);

console.log(options.join("\n"));
