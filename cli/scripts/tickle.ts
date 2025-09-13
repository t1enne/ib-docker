import { attemptAsync, invariant } from "es-toolkit";
import { client } from "../src/shared";

const [err, r] = await attemptAsync(() => client.post("tickle"));
invariant(r, `${err}`);
console.log(r.data);
