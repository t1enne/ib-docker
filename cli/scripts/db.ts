import { Kysely, SqliteDialect } from 'kysely'
import Database from 'better-sqlite3'
import path from 'path'

export interface DatabaseSchema {
  symbols: {
    id: number
    symbol: string
    name: string | null
  }
  ohlcv_1min: OhlcvTable
  ohlcv_5min: OhlcvTable
  ohlcv_15min: OhlcvTable
  ohlcv_30min: OhlcvTable
  ohlcv_1h: OhlcvTable
  ohlcv_4h: OhlcvTable
  ohlcv_1d: OhlcvTable
  ohlcv_1w: OhlcvTable
}

export interface OhlcvTable {
  id: number
  symbol_id: number
  timestamp: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

const dbPath = path.join(process.cwd(), 'db.sqlite')
const dialect = new SqliteDialect({
  database: new Database(dbPath) as any
})

export const db = new Kysely<DatabaseSchema>({
  dialect
})

export async function setupDatabase() {
  // Create symbols table
  await db.schema
    .createTable('symbols')
    .addColumn('id', 'integer', (col) => col.primaryKey().autoIncrement())
    .addColumn('symbol', 'text', (col) => col.notNull().unique())
    .addColumn('name', 'text')
    .execute()

  // Create OHLCV tables for each interval
  const intervals = ['1min', '5min', '15min', '30min', '1h', '4h', '1d', '1w']

  for (const interval of intervals) {
    const tableName = `ohlcv_${interval}`
    await db.schema
      .createTable(tableName)
      .addColumn('id', 'integer', (col) => col.primaryKey().autoIncrement())
      .addColumn('symbol_id', 'integer', (col) => col.notNull().references('symbols.id'))
      .addColumn('timestamp', 'integer', (col) => col.notNull())
      .addColumn('open', 'real', (col) => col.notNull())
      .addColumn('high', 'real', (col) => col.notNull())
      .addColumn('low', 'real', (col) => col.notNull())
      .addColumn('close', 'real', (col) => col.notNull())
      .addColumn('volume', 'real', (col) => col.notNull())
      .execute()

    // Create indexes
    await db.schema
      .createIndex(`${tableName}_symbol_id_idx`)
      .on(tableName)
      .column('symbol_id')
      .execute()

    await db.schema
      .createIndex(`${tableName}_timestamp_idx`)
      .on(tableName)
      .column('timestamp')
      .execute()

    await db.schema
      .createIndex(`${tableName}_symbol_timestamp_idx`)
      .on(tableName)
      .columns(['symbol_id', 'timestamp'])
      .execute()
  }

  // Index for symbols
  await db.schema
    .createIndex('symbols_symbol_idx')
    .on('symbols')
    .column('symbol')
    .execute()
}

export async function closeDatabase() {
  await db.destroy()
}

// Run setup if called directly
if (import.meta.main && process.argv[2] === 'setup') {
  await setupDatabase()
  console.log('Database setup complete')
  process.exit(0)
}