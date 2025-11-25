import { Layout } from "../components/layout";
import { Symbol } from "../components/symbol";
import { db } from "../db/db";

export default async function Page() {
  const symbols = await db.selectFrom("symbol").selectAll().execute();
  return (
    <Layout>
      <div>
        <ul>
          {symbols.map((symbol) => (
            <Symbol symbol={symbol} />
          ))}
        </ul>
      </div>
    </Layout>
  );
}
