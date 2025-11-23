import type { Kysely } from "kysely";

export async function up(db: Kysely<any>) {
  await db.schema
    .createTable("watchlist")
    .addColumn("id", "integer", (col) => col.primaryKey().autoIncrement())
    .addColumn("name", "text", (col) => col.notNull())
    .addColumn("notes", "text")
    .addColumn("strategy", "text")
    .execute();

  await db.schema
    .createTable("watchlist_symbols")
    .addColumn("id", "integer", (col) => col.primaryKey().autoIncrement())
    .addColumn("watchlist_id", "integer", (col) => col.notNull().references("watchlist.id"))
    .addColumn("symbol_id", "integer", (col) => col.notNull().references("symbols.id"))
    .addUniqueConstraint("unique_watchlist_symbol", ["watchlist_id", "symbol_id"])
    .execute();
}

export async function down(db: Kysely<any>) {
  await db.schema.dropTable("watchlist_symbols").execute();
  await db.schema.dropTable("watchlist").execute();
}
