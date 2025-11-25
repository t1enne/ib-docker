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

  return (
    <Layout>
      <Symbol symbol={symbol} />
    </Layout>
  );
}
