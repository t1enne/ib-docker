import { Layout } from "../components/layout";

export default async function Page() {
  return (
    <Layout>
      <div>
        <h2 className="text-lg font-semibold mb-4">Create New Watchlist</h2>
        <form hx-post="/watchlists" hx-target="body" hx-swap="innerHTML">
          <div className="mb-4">
            <input
              type="text"
              id="name"
              name="name"
              required
              placeholder="Enter watchlist name"
            />
          </div>
          <div className="mb-4">
            <textarea
              id="notes"
              name="notes"
              rows={3}
              placeholder="Optional notes about this watchlist"
            />
          </div>
          <div className="mb-4">
            <input
              type="text"
              id="strategy"
              name="strategy"
              placeholder="Optional trading strategy"
            />
          </div>
          <button type="submit">Create Watchlist</button>
        </form>
      </div>
    </Layout>
  );
}

