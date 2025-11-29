import { Hono } from "hono";
import WatchlistsPage from "../../pages/watchlists.page";
import WatchlistPage from "../../pages/watchlist.page";
import WatchlistAddPage from "../../pages/watchlist-add.page";
import {
  createWatchlist,
  deleteWatchlist,
  addSymbolsToWatchlist,
  removeSymbolsFromWatchlist,
  getWatchlistWithSymbols,
} from "../trading/watchlists";

const controller = new Hono();

controller.get("/", async (c) => c.html(<WatchlistsPage />));
controller.get("/add", async (c) => c.html(<WatchlistAddPage />));
controller.get("/:id", async (c) => {
  const id = parseInt(c.req.param("id"));
  const watchlist = await getWatchlistWithSymbols(id);
  if (!watchlist) {
    return c.html("<p class='text-red-600'>Watchlist not found</p>");
  }
  return c.html(<WatchlistPage watchlist={watchlist} />);
});

controller.post("/", async (c) => {
  const formData = await c.req.formData();
  const name = formData.get("name") as string;
  const notes = formData.get("notes") as string;
  const strategy = formData.get("strategy") as string;

  if (!name) {
    return c.html("<p class='text-red-600'>Name is required</p>");
  }

  const id = await createWatchlist(
    name,
    notes || undefined,
    strategy || undefined,
  );
  return c.redirect(`/watchlists/${id}`);
});

controller.delete("/:id", async (c) => {
  const id = parseInt(c.req.param("id"));
  await deleteWatchlist(id);
  return c.html("<p class='text-green-600'>Watchlist deleted</p>");
});

controller.post("/:id/symbols", async (c) => {
  const id = parseInt(c.req.param("id"));
  const formData = await c.req.formData();
  const symbolIds = formData
    .getAll("symbolIds")
    .map((s) => parseInt(s as string));

  if (symbolIds.length === 0) {
    return c.html("<p class='text-red-600'>No symbols selected</p>");
  }

  await addSymbolsToWatchlist(id, symbolIds);
  return c.html("<p class='text-green-600'>Symbols added to watchlist</p>");
});

controller.delete("/:id/symbols/:symbolId", async (c) => {
  const id = parseInt(c.req.param("id"));
  const symbolId = parseInt(c.req.param("symbolId"));
  // const symbolIds = formData
  //   .getAll("symbolIds")
  //   .map((s) => parseInt(s as string));

  // if (symbolIds.length === 0) {
  //   return c.html("<p class='text-red-600'>No symbols selected</p>");
  // }

  await removeSymbolsFromWatchlist(id, [symbolId]);
  return c.html("<p class='text-green-600'>Symbols removed from watchlist</p>");
});

export default controller;

