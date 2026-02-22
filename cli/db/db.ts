import Database from "bun:sqlite";
import { CamelCasePlugin, CompiledQuery, Kysely } from "kysely";
import path from "path";
import type { DatabaseSchema } from "./types";
import { BunSqliteDialect } from "kysely-bun-sqlite";

const dbPath = path.join(process.cwd(), "..", "data", "db.sqlite");

const _db = new Database(dbPath);

const dialect = new BunSqliteDialect({
  database: _db,
  onCreateConnection: async (c) => {
    await c.executeQuery(CompiledQuery.raw("PRAGMA journal_mode=WAL"));
  },
});

export const db = new Kysely<DatabaseSchema>({
  dialect,
  log: ["query", "error"],
  plugins: [new CamelCasePlugin()],
});

process.on("SIGTERM", () => {
  console.log("Received SIGTERM, closing database...");
  _db.close();
  process.exit(0);
});

process.on("SIGINT", () => {
  console.log("Received SIGINT, closing database...");
  _db.close();
  process.exit(0);
});
