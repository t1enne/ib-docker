import { attemptAsync } from "es-toolkit";
import { client } from "./shared";

const [err, r] = await attemptAsync(() => client.get("iserver/auth/status"));
if (!!err) {
  console.error(`${err}`);
  process.exit(1);
}

console.log(r!.data);
