import type { Kysely } from "kysely";

export async function up(db: Kysely<any>) {
  await db.schema
    .createTable("watchlist")
    .addColumn("id", "integer", (col) => col.primaryKey().autoIncrement())
    .addColumn("symbol", "text", (col) => col.notNull())
    .addColumn("target_price", "real")
    .addColumn("exit_price", "real")
    .addColumn("notes", "text")
    .execute();
}

export async function down(db: Kysely<any>) {
  await db.schema.dropTable("watchlist").execute();
}
