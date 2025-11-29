import { Layout } from "../components/layout";
import { db } from "../db/db";

export default async function Page() {
  const watchlists = await db.selectFrom("watchlist").selectAll().execute();

  return (
    <Layout>
      <div>
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-2xl font-bold">Watchlists</h1>
        </div>
        <div className="overflow-auto h-[600px] border">
          <table className="table-auto border-collapse w-full">
            <thead>
              <tr>
                <th className="border px-4 py-2">Name</th>
                <th className="border px-4 py-2">Notes</th>
                <th className="border px-4 py-2">Strategy</th>
                <th className="border px-4 py-2">Actions</th>
              </tr>
            </thead>
            <tbody>
              {watchlists.map((w) => (
                <tr key={w.id} id={`watchlist-${w.id}`}>
                  <td className="border px-4 py-2">
                    <a
                      href={`/watchlists/${w.id}`}
                      className="text-blue-600 hover:underline"
                    >
                      {w.name}
                    </a>
                  </td>
                  <td className="border px-4 py-2">{w.notes || ""}</td>
                  <td className="border px-4 py-2">{w.strategy || ""}</td>
                  <td className="border px-4 py-2">
                    <button
                      hx-delete={`/watchlists/${w.id}`}
                      hx-confirm="Are you sure you want to delete this watchlist?"
                      className="text-red-600 hover:text-red-800"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Layout>
  );
}

