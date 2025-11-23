import { attemptAsync, invariant } from "es-toolkit";
import { getContractInfo } from "../src/shared";
import { parseArgs } from "util";

const { positionals } = parseArgs({ allowPositionals: true });
const ids = positionals;
invariant(ids, "need to pass conid");
invariant(Array.isArray(ids), "need to pass a conid");
ids.forEach(symbol);

export async function symbol(id: string) {
  const [err, r] = await attemptAsync(() => getContractInfo(+id));
  invariant(r, `${err}`);
  console.log(r);
}
