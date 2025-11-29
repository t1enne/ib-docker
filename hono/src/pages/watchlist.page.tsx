import { Layout } from "../components/layout";
import { Watchlist } from "../components/watchlist";
import type { IWatchlist } from "../db/types";
import { getWatchlistWithSymbols } from "../modules/trading/watchlists";

interface Props {
  watchlist: Awaited<ReturnType<typeof getWatchlistWithSymbols>>;
}

export default async function Page({ watchlist }: Props) {
  return (
    <Layout>
      <Watchlist watchlist={watchlist} />
    </Layout>
  );
}

