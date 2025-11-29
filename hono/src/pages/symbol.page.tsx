import { Layout } from "../components/layout";
import { Symbol } from "../components/symbol";
import { db } from "../db/db";
import dayjs from "dayjs";

interface Props {
  ticker: string;
}
export default async function Page({ ticker }: Props) {
  const symbol = await db
    .selectFrom("symbol")
    .selectAll()
    .where("ticker", "=", ticker)
    .executeTakeFirstOrThrow();

  // Then get OHLCV stats
  const ohlcvStats = await db
    .selectFrom("ohlcv_1d")
    .select(({ fn }) => [
      fn.countAll().as("total"),
      fn.min("timestamp").as("min"),
      fn.max("timestamp").as("max"),
    ])
    .where("symbol_id", "=", symbol.id)
    .executeTakeFirstOrThrow();

  console.log(ohlcvStats);

  return (
    <Layout>
      <Symbol symbol={symbol} showDownload />
      <h6 className="font-bold">Stats</h6>
      <div>
        <b>Days:</b> {ohlcvStats.total}
        <br />
        <b>Min:</b> {dayjs(ohlcvStats.min).format("YY/MM/DD-HH:mm:ss")}
        <br />
        <b>Max:</b> {dayjs(ohlcvStats.max).format("YY/MM/DD-HH:mm:ss")}
        <br />
      </div>
    </Layout>
  );
}
