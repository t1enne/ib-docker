import { parseArgs } from "util";
import {
  createWatchlist,
  updateWatchlist,
  deleteWatchlist,
  addSymbolsToWatchlist,
  removeSymbolsFromWatchlist,
  getSymbolId,
  getWatchlistWithSymbols,
} from "../src/watchlists";
import { debounce, isNil, isNotNil } from "es-toolkit";
import type { Watchlist } from "../db/types";
import { candles } from "../src/candles";

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    symbols: { type: "string", short: "s" },
    name: { type: "string" },
    notes: { type: "string" },
    strategy: { type: "string" },
  },
});

const [subcommand, ...args] = positionals;

async function handleCreate(args: string[], values: any) {
  const name = args[0] || values.name;
  if (!name) {
    console.error("Need name for watchlist");
    process.exit(1);
  }
  const id = await createWatchlist(name, values.notes, values.strategy);
  console.log(`Created watchlist ${id}`);
}

async function handleUpdate(args: string[], values: any) {
  const id = args[0];
  if (!id) {
    console.error("Need watchlist id");
    process.exit(1);
  }
  const updates: Partial<Pick<Watchlist, "name" | "notes" | "strategy">> = {};
  if ("name" in values) updates.name = values.name;
  if ("notes" in values) updates.notes = values.notes;
  if ("strategy" in values) updates.strategy = values.strategy;
  await updateWatchlist(+id, updates);
  console.log(`Updated watchlist ${id}`);
}

async function handleDelete(args: string[]) {
  const id = args[0];
  if (!id) {
    console.error("Need watchlist id");
    process.exit(1);
  }
  await deleteWatchlist(+id);
  console.log(`Deleted watchlist ${id}`);
}

async function handleAddSymbols(args: string[], values: any) {
  const id = args[0];
  if (!id || !values.symbols) {
    console.error("Need watchlist id and --symbols");
    process.exit(1);
  }
  const symbolStrings = values.symbols.split(",").map((s: string) => s.trim());
  const symbolIds = await Promise.all(symbolStrings.map(getSymbolId));
  const validIds = symbolIds.filter(isNotNil) as number[];
  if (validIds.length > 0) {
    await addSymbolsToWatchlist(+id, validIds);
  }
  const invalid = symbolStrings.filter(
    (_: string, i: number) => symbolIds[i] === undefined,
  );
  if (invalid.length > 0) {
    console.warn(`Symbols not found: ${invalid.join(", ")}`);
  }
  console.log(`Added symbols to watchlist ${id}`);
}

async function handleRemoveSymbols(args: string[], values: any) {
  const id = args[0];
  if (!id || !values.symbols) {
    console.error("Need watchlist id and --symbols");
    process.exit(1);
  }
  const symbolStrings = values.symbols.split(",").map((s: string) => s.trim());
  const symbolIds = await Promise.all(symbolStrings.map(getSymbolId));
  const validIds = symbolIds.filter(isNotNil) as number[];
  if (validIds.length > 0) {
    await removeSymbolsFromWatchlist(+id, validIds);
  }
  const invalid = symbolStrings.filter(
    (_: string, i: number) => symbolIds[i] === undefined,
  );
  if (invalid.length > 0) {
    console.warn(`Symbols not found: ${invalid.join(", ")}`);
  }
  console.log(`Removed symbols from watchlist ${id}`);
}

async function handleCandles() {
  const id = args[0];
  const bar = args[1] ?? "1d";
  const period = args[2] ?? "90d";
  if (isNil(id)) {
    console.error("Need watchlist id and --symbols");
    process.exit(1);
  }
  const wl = await getWatchlistWithSymbols(+id);
  console.log(wl);
  const dl = debounce((s) => candles({ conid: s.id, bar, period }), 200);
  const results = wl?.symbols.map(dl);
}

function showUsage() {
  console.log(
    'Usage: bun run scripts/watchlists.ts <create|update|delete|add-symbols|remove-symbols> [args] [--symbols "AAPL,GOOGL"] [--name "name"] [--notes "notes"] [--strategy "strategy"]',
  );
}

async function main() {
  const handlers: Record<
    string,
    (args: string[], values: any) => Promise<void>
  > = {
    create: handleCreate,
    update: handleUpdate,
    delete: handleDelete,
    "add-symbols": handleAddSymbols,
    "remove-symbols": handleRemoveSymbols,
    candles: handleCandles,
  };

  if (!subcommand) {
    showUsage();
    return;
  }

  const handler = handlers[subcommand];
  if (handler) {
    await handler(args, values);
  } else {
    showUsage();
  }
}

main().catch(console.error);
