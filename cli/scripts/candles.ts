import * as v from "valibot";
import { BAR_INTERVAL } from "../consts/interval";
import { parseArgs } from "util";
import { candles } from "../src/candles";

const ArgsSchema = v.object({
  conid: v.number(),
  bar: v.picklist(BAR_INTERVAL),
  period: v.pipe(v.string(), v.endsWith("d")),
});

const { positionals } = parseArgs({ allowPositionals: true });
const [conid, bar, period] = positionals;
const _args = { conid: +conid!, bar, period };
const args = v.parse(ArgsSchema, _args);
await candles(args);
