import type { Kysely } from "kysely";
import { BAR_INTERVAL } from "../../consts/interval";

export async function up(db: Kysely<any>) {
  await db.schema
    .createTable("symbol")
    .addColumn("id", "integer", (col) => col.primaryKey().autoIncrement())
    .addColumn("ticker", "text", (col) => col.notNull())
    .addColumn("market", "text", (c) => c.notNull())
    .addColumn("currency", "text", (c) => c.notNull())
    .addColumn("name", "text")
    .execute();

  for (const interval of BAR_INTERVAL) {
    const tableName = `ohlcv_${interval}`;
    await db.schema
      .createTable(tableName)
      .addColumn("id", "integer", (col) => col.primaryKey().autoIncrement())
      .addColumn("symbol_id", "integer", (col) =>
        col.notNull().references("symbol.id"),
      )
      .addColumn("timestamp", "datetime", (col) => col.notNull())
      .addColumn("open", "real", (col) => col.notNull())
      .addColumn("high", "real", (col) => col.notNull())
      .addColumn("low", "real", (col) => col.notNull())
      .addColumn("close", "real", (col) => col.notNull())
      .addColumn("volume", "real", (col) => col.notNull())
      .execute();

    // Create indexes
    await db.schema
      .createIndex(`${tableName}_symbol_id_idx`)
      .on(tableName)
      .column("symbol_id")
      .execute();

    await db.schema
      .createIndex(`${tableName}_timestamp_idx`)
      .on(tableName)
      .column("timestamp")
      .execute();

    await db.schema
      .createIndex(`${tableName}_symbol_timestamp_idx`)
      .on(tableName)
      .columns(["symbol_id", "timestamp"])
      .unique()
      .execute();
  }

  // Index for symbol
  await db.schema
    .createIndex("symbol_symbol_idx")
    .on("symbol")
    .column("ticker")
    .execute();
}

export async function down(db: Kysely<any>) {
  const tables = ["symbol", ...BAR_INTERVAL.map((b) => `ohlcv_${b}`)];
  for (const name of tables) {
    console.log(`Dropping ${name}`);
    await db.schema.dropTable(name).execute();
  }
}
