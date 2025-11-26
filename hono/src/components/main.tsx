import { db } from "../db/db";
import { Layout } from "./layout";

export async function Main() {
  const symbols = await (db as any).selectFrom("symbol").selectAll().execute();
  return (
    <Layout>
      <div className="space-y-6">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <nav className="flex gap-4 mb-6">
          <a href="/watchlists" className="text-blue-600 hover:underline">
            Watchlists
          </a>
        </nav>


      </div>
    </Layout>
  );
}
