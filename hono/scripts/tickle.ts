import { attemptAsync, invariant } from "es-toolkit";
import { client } from "../src/modules/shared/client";

const [err, r] = await attemptAsync(() => client.post("tickle"));
invariant(r, `${err}`);
console.log(r.data);
