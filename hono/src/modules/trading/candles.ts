import { attemptAsync, invariant } from "es-toolkit";
import { getContractInfo } from "./shared";
import { db } from "../../db/db";
import * as v from "valibot";
import { BAR_INTERVAL } from "../../consts/interval";
import { client } from "../shared/client";
import dayjs from "dayjs";

const ArgsSchema = v.object({
  conid: v.number(),
  bar: v.picklist(BAR_INTERVAL),
  period: v.optional(v.pipe(v.string(), v.endsWith("d"))),
  startTime: v.optional(v.string()),
});

export async function candles({
  conid,
  period,
  bar,
  startTime,
}: v.InferInput<typeof ArgsSchema>) {
  v.parse(ArgsSchema, { conid, period, bar, startTime });

  const symbolInfo = await getContractInfo(+conid);
  invariant(symbolInfo, "Failed retrieving symbol");
  console.debug(`Getting candles for ${symbolInfo.ticker}`, symbolInfo);

  const tableName = `ohlcv_${bar}` as "ohlcv_1d";

  const params = {
    conid,
    bar,
    period,
    startTime: startTime
      ? dayjs(startTime).format("YYYYMMDD-HH:mm:ss")
      : undefined,
  };
  console.debug(`Getting candles with p: ${JSON.stringify(params, null, 2)}`);

  const [err, r] = await attemptAsync(() =>
    client.get("iserver/marketdata/history", { params }),
  );
  if (!!err) {
    console.error(`Failed getting marketdata: ${err}`);
    return;
  }
  if (!r?.data?.data || r.data.data.length === 0) {
    console.debug(`No candles found`);
    return;
  }
  const data = r.data.data as Array<{
    o: number;
    c: number;
    h: number;
    l: number;
    v: number;
    t: number;
  }>;

  console.debug(`Inserting ${data.length} candles`);
  await db
    .insertInto(tableName)
    .values(
      data.map((item) => ({
        symbol_id: symbolInfo.id,
        timestamp: item.t,
        open: item.o,
        high: item.h,
        low: item.l,
        close: item.c,
        volume: item.v,
      })),
    )
    .onConflict((oc) =>
      oc.columns(["timestamp", "symbol_id"]).doUpdateSet({
        open: (eb) => eb.ref("excluded.open"),
        high: (eb) => eb.ref("excluded.high"),
        low: (eb) => eb.ref("excluded.low"),
        close: (eb) => eb.ref("excluded.close"),
        volume: (eb) => eb.ref("excluded.volume"),
      }),
    )
    .execute();
}
