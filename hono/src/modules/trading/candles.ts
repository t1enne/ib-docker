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
  period: v.pipe(v.string(), v.endsWith("d")),
});

export async function candles({
  conid,
  period,
  bar,
}: v.InferInput<typeof ArgsSchema>) {
  v.parse(ArgsSchema, { conid, period, bar });

  const symbolInfo = await getContractInfo(+conid);
  invariant(symbolInfo, "Failed retrieving symbol");
  console.debug(`Getting candles for ${symbolInfo.ticker}`, symbolInfo);

  const tableName = `ohlcv_${bar}` as "ohlcv_1d";
  await db
    .deleteFrom(tableName)
    .where("symbol_id", "=", symbolInfo.id)
    .execute();

  const params = { conid, bar, period };

  const [err, r] = await attemptAsync(() =>
    client.get("iserver/marketdata/history", { params }),
  );
  if (!!err) {
    console.error(`Failed getting marketdata: ${err}`, err);
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
  console.log(data.slice(0, 10));
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
    .execute();
}
