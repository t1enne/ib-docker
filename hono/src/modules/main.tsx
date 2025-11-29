import { Hono } from "hono";
import { Main } from "../components/main";
import { lookup } from "./trading/lookup";
import { candles } from "./trading/candles";
import { client } from "./shared/client";
import { AddableSymbol } from "../components/addable-symbol";
import symbols from "./symbols/";
import watchlists from "./watchlists/";

const main = new Hono();

main.route("/symbols", symbols);
main.route("/watchlists", watchlists);

main.get("/", (c) => c.render(<Main />));

main.get("/tickle", async (c) => {
  const r = await client.get("/tickle");
  return c.json(r.data);
});

main.post("/lookup", async (c) => {
  const formData = await c.req.formData();
  const symbol = formData.get("symbol")! as string;
  const results = await lookup(symbol);
  if (!results) {
    return c.text("nothing");
  }
  if (results.length == 0) {
    return c.render(<p>nothing found</p>);
  }
  const html = results.map((r) => <AddableSymbol symbol={r} />);
  return c.render(<>{html}</>);
});

main.post("/download-candles", async (c) => {
  const fd = await c.req.formData();
  const { conid, period, bar, startDate } = Object.fromEntries(fd.entries());
  console.log(Object.fromEntries(fd.entries()));
  await candles({
    conid: +conid!,
    period: period?.length ? period : undefined,
    startDate: startDate?.length ? startDate : undefined,
    bar: bar!,
  });
  return c.html(
    "<p class='text-green-600'>Candles downloaded successfully</p>",
  );
});

export default main;
