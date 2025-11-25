import { Hono } from "hono";
import SymbolsPage from "../../pages/symbols.page";
import SymbolPage from "../../pages/symbol.page";
import SymbolAddPage from "../../pages/symbol-add.page";

const controller = new Hono();

controller.get("/", async (c) => c.html(<SymbolsPage />));
controller.get("/add", async (c) => c.html(<SymbolAddPage />));
controller.get("/:ticker", async (c) =>
  c.html(<SymbolPage ticker={c.req.param().ticker} />),
);

export default controller;
