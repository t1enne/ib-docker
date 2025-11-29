import { db } from "../db/db";

function getSymbols() {
  return db.selectFrom("symbol").select("ticker").execute();
}

function getWatchlists() {
  return db.selectFrom("watchlist").select("id").execute();
}

export async function Navigation() {
  const data = await Promise.all([getSymbols(), getWatchlists()]);
  const [symbols, watchlists] = data;
  const symbolRoutes = symbols.map((s) => `/symbols/${s.ticker}`);
  const watchlistRoutes = watchlists.map((w) => `/watchlists/${w.id}`);
  const routes = ["/", "/symbols", "/symbols/add", "/watchlists", "/watchlists/add", ...symbolRoutes, ...watchlistRoutes];

  return (
    <div
      x-data={`{
        show: false,
        query: '',
        filteredRoutes: [],
        selectedIndex: 0,
        routes: [ ${routes.map((r) => `'${r}'`).join(", ")} ]
      }`}
      x-init="
        filteredRoutes = routes;
        $watch('query', (value) => {
          filteredRoutes = routes.filter(route =>
            route.toLowerCase().includes(value.toLowerCase())
          );
          selectedIndex = 0;
        });
        document.addEventListener('keydown', (e) => {
          if (e.key === '/' && !show && e.target.tagName !== 'INPUT') {
            e.preventDefault();
						query = window.location.pathname;
            show = true;
            $nextTick(() => $refs.input.focus());
          } else if (e.key === 'Escape' && show) {
            show = false;
            query = '';
          }
        });
      "
      x-show="show"
      className="fixed inset-0 bg-black z-50"
    >
      <div className="p-4 w-full">
        <input
          x-ref="input"
          x-model="query"
          type="text"
          placeholder="Type to navigate..."
          className="w-full p-2 border mb-2"
          x-on:keydown="if ($event.key === 'Enter' && filteredRoutes.length > 0) { window.location.href = filteredRoutes[selectedIndex]; } else if ($event.key === 'ArrowDown') { $event.preventDefault(); selectedIndex = Math.min(selectedIndex + 1, filteredRoutes.length - 1); } else if ($event.key === 'ArrowUp') { $event.preventDefault(); selectedIndex = Math.max(selectedIndex - 1, 0); }"
        />
        <ul className="max-h-60 overflow-y-auto">
          <template x-for="(route, index) in filteredRoutes">
            <li
              x-bind:class="index === selectedIndex ? 'text-white' : ''"
              className="p-2 hover:text-white"
              x-on:click="window.location.href = route"
            >
              <a hx-boost x-bind:href={`route`}>
                <span x-text="route" className=""></span>
              </a>
            </li>
          </template>
        </ul>
      </div>
    </div>
  );
}
