import { Layout } from "../components/layout";
import { Symbol } from "../components/symbol";
import { db } from "../db/db";

interface Props {
  ticker: string;
}
export default async function Page({ ticker }: Props) {
  const symbol = await db
    .selectFrom("symbol")
    .selectAll()
    .where("ticker", "=", ticker)
    .executeTakeFirstOrThrow();

  const candlesData = await db
    .selectFrom("ohlcv_1d")
    .where("symbol_id", "=", symbol.id)
    .orderBy("timestamp", "desc")
    .limit(50)
    .selectAll()
    .execute();

  return (
    <Layout>
      <Symbol symbol={symbol} candles={candlesData} showDownload />
    </Layout>
  );
}
