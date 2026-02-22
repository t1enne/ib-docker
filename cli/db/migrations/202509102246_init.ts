import type { Kysely } from "kysely";

const candleTableName = `candle`;

export async function up(db: Kysely<any>) {
  await db.schema
    .createTable("symbol")
    .addColumn("conid", "integer", (col) => col.primaryKey())
    .addColumn("ticker", "text", (col) => col.notNull())
    .addColumn("market", "text", (c) => c.notNull())
    .addColumn("currency", "text", (c) => c.notNull())
    .addColumn("name", "text")
    .execute();

  await db.schema
    .createTable(candleTableName)
    .addColumn("id", "integer", (col) => col.primaryKey().autoIncrement())
    .addColumn("ticker", "text", (col) => col.notNull())
    .addColumn("conid", "integer", (col) =>
      col.notNull().references("symbol.conid"),
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
    .createIndex(`${candleTableName}_symbol_id_idx`)
    .on(candleTableName)
    .column("conid")
    .execute();

  await db.schema
    .createIndex(`${candleTableName}_timestamp_idx`)
    .on(candleTableName)
    .column("timestamp")
    .execute();

  await db.schema
    .createIndex(`${candleTableName}_symbol_timestamp_idx`)
    .on(candleTableName)
    .columns(["conid", "timestamp"])
    .unique()
    .execute();

  // Index for symbol
  await db.schema
    .createIndex("symbol_symbol_idx")
    .on("symbol")
    .column("symbol")
    .execute();
}

export async function down(db: Kysely<any>) {
  const tables = ["symbol", candleTableName];
  for (const name of tables) {
    console.log(`Dropping ${name}`);
    await db.schema.dropTable(name).execute();
  }
}
