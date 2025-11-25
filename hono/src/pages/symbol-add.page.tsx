import { Layout } from "../components/layout";

export default async function Page({}) {
  return (
    <Layout>
      <div>
        <h2 className="text-lg font-semibold">Lookup Symbol</h2>
        <form hx-post="/lookup" hx-target="#lookup-results" hx-swap="innerHTML">
          <input
            type="text"
            name="symbol"
            placeholder="Enter symbol"
            className="border p-2 mr-2"
          />
          <button
            hidden
            type="submit"
            className="bg-blue-500 text-white px-4 py-2"
          >
            Lookup
          </button>
        </form>
        <div id="lookup-results" className="mt-4"></div>
      </div>
    </Layout>
  );
}
