import { Hono } from "hono";
import SymbolsPage from "../../pages/symbols.page";
import SymbolPage from "../../pages/symbol.page";
import SymbolAddPage from "../../pages/symbol-add.page";
import { getContractInfo } from "../trading/shared";

const controller = new Hono();

controller.get("/", async (c) => c.html(<SymbolsPage />));
controller.get("/add", async (c) => c.html(<SymbolAddPage />));
controller.get("/:ticker", async (c) =>
  c.html(<SymbolPage ticker={c.req.param().ticker} />),
);
controller.post("/:conid", async (c) => {
  const conid = c.req.param("conid");
  const symbolInfo = await getContractInfo(+conid);
  if (!symbolInfo)
    return c.html("<p class='text-red-600'>Symbol not found</p>");

  return c.html("<p class='text-green-600'>Symbol added</p>");
});

export default controller;
