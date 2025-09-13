import type { Kysely } from "kysely";
import { BAR_INTERVAL } from "../../consts/interval";

export async function up(db: Kysely<any>) {
  await db.schema
    .createTable("symbols")
    .addColumn("id", "integer", (col) => col.primaryKey().autoIncrement())
    .addColumn("symbol", "text", (col) => col.notNull())
    .addColumn("market", "text", (c) => c.notNull())
    .addColumn("currency", "text", (c) => c.notNull())
    .addColumn("name", "text")
    .execute();

  // Create OHLCV tables for each interval

  for (const interval of BAR_INTERVAL) {
    const tableName = `ohlcv_${interval}`;
    await db.schema
      .createTable(tableName)
      .addColumn("id", "integer", (col) => col.primaryKey().autoIncrement())
      .addColumn("symbol_id", "integer", (col) =>
        col.notNull().references("symbols.id"),
      )
      .addColumn("timestamp", "integer", (col) => col.notNull())
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
      .execute();
  }

  // Index for symbols
  await db.schema
    .createIndex("symbols_symbol_idx")
    .on("symbols")
    .column("symbol")
    .execute();
}

export async function down(db: Kysely<any>) {
  const tables = ["symbols", ...BAR_INTERVAL];
  for (const name of tables) {
    console.log(`Dropping ${name}`);
    db.schema.dropTable(`ohlcv_${name}`).execute();
  }
}
