import { parseArgs } from "util";
import { getContractInfo } from "../src/get-contract-info";
import { invariant } from "es-toolkit";

const { positionals } = parseArgs({ allowPositionals: true });
const conid = positionals[0];
invariant(conid, "need to pass a conid");

const info = await getContractInfo(+conid);
console.log(info);
