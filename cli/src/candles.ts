import { attemptAsync, invariant } from "es-toolkit";
import { client, fetchContractInfo } from "./shared";
import { db } from "../db/db";
import * as v from "valibot";
import { BAR_INTERVAL } from "../consts/interval";

const ArgsSchema = v.object({
  conid: v.number(),
  bar: v.picklist(BAR_INTERVAL),
  period: v.pipe(v.string(), v.endsWith("d")),
});

export async function candles({
  conid,
  period,
  bar,
}: v.InferInput<typeof ArgsSchema>) {
  v.parse(ArgsSchema, { conid, period, bar });

  const symbolInfo = await fetchContractInfo(+conid);
  invariant(symbolInfo, "Failed retrieving symbol");

  const params = {
    conid,
    period, // 90d
    bar, // 5mins
  };

  const [err, r] = await attemptAsync(() =>
    client.get("iserver/marketdata/history", { params }),
  );
  if (!!err) {
    console.error(`${err}`);
    process.exit(1);
  }
  invariant(!!r?.data.data.length, "Missing data");
  const data = r.data.data as Array<{
    o: number;
    c: number;
    h: number;
    l: number;
    v: number;
    t: number;
  }>;

  const tableName = `ohlcv_${bar as "1h"}` as const;
  const maxTimestampResult = await db
    .selectFrom(tableName)
    .select((eb) => eb.fn.max("timestamp").as("maxTs"))
    .where("symbol_id", "=", symbolInfo.con_id)
    .executeTakeFirst();

  const maxTs = maxTimestampResult?.maxTs as number | null;
  const filteredData = data.filter((item) => maxTs === null || item.t > maxTs);

  if (filteredData.length == 0) {
    process.exit(0);
  }

  console.log(`Inserting ${filteredData.length} rows for ${symbolInfo.symbol}`);

  await db
    .insertInto(tableName)
    .values(
      filteredData.map((item) => ({
        symbol_id: symbolInfo.con_id,
        timestamp: item.t,
        open: item.o,
        high: item.h,
        low: item.l,
        close: item.c,
        volume: item.v,
      })),
    )
    .execute();
}
