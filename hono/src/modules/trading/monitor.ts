// @ts-ignore
import blessed from "blessed";
// @ts-ignore
import WebSocket from "ws";
import { client } from "../shared/client";
import { attemptAsync } from "es-toolkit";

interface Watchlist {
  id: string;
  name: string;
}

interface Instrument {
  conid: number;
  symbol: string;
  secType: string;
  exchange: string;
  companyName: string;
}

class WatchlistMonitor {
  private screen: blessed.Widgets.Screen;
  private watchlistList: blessed.Widgets.ListElement;
  private instrumentList: blessed.Widgets.ListElement;
  private statusBox: blessed.Widgets.BoxElement;
  private ws: WebSocket | null = null;
  private watchlists: Watchlist[] = [];
  private currentInstruments: Instrument[] = [];
  private prices: Map<number, any> = new Map();

  constructor() {
    this.screen = blessed.screen({
      smartCSR: true,
      title: "IBKR Watchlist Monitor",
    });

    this.setupUI();
    this.setupKeys();
    this.loadWatchlists();
  }

  private setupUI() {
    // Watchlist list on the left
    this.watchlistList = blessed.list({
      parent: this.screen,
      top: 0,
      left: 0,
      width: "30%",
      height: "90%",
      border: "line",
      label: " Watchlists ",
      style: {
        selected: { bg: "blue" },
      },
      keys: true,
      vi: true,
    });

    // Instrument list on the right
    this.instrumentList = blessed.list({
      parent: this.screen,
      top: 0,
      left: "30%",
      width: "70%",
      height: "90%",
      border: "line",
      label: " Instruments ",
      style: {
        selected: { bg: "green" },
      },
    });

    // Status box at the bottom
    this.statusBox = blessed.box({
      parent: this.screen,
      bottom: 0,
      left: 0,
      width: "100%",
      height: "10%",
      border: "line",
      label: " Status ",
      content: "Loading watchlists...",
    });

    this.screen.render();
  }

  private setupKeys() {
    this.screen.key(["escape", "q", "C-c"], () => {
      this.cleanup();
      process.exit(0);
    });

    this.watchlistList.on("select", (_item: any, index: number) => {
      const wl = this.watchlists[index];
      if (wl) {
        this.loadWatchlistInstruments(wl.id);
      }
    });
  }

  private async loadWatchlists() {
    try {
      const [err, r] = await attemptAsync(() =>
        client.get<{ data: { user_lists: Watchlist[] } }>("iserver/watchlists", { params: { SC: "USER_WATCHLIST" } }),
      );

      if (err) {
        this.statusBox.setContent(
          `Error loading watchlists: ${(err as any)?.message || err}`,
        );
        this.screen.render();
        return;
      }

      this.watchlists = r!.data.data.user_lists;
      const names = this.watchlists.map((wl) => `[${wl.id}] ${wl.name}`);
      this.watchlistList.setItems(names);
      this.statusBox.setContent(`Loaded ${this.watchlists.length} watchlists`);
      this.screen.render();
    } catch (error) {
      this.statusBox.setContent(`Error: ${error}`);
      this.screen.render();
    }
  }

  private async loadWatchlistInstruments(watchlistId: string) {
    try {
      const [err, r] = await attemptAsync(() =>
        client.get<{ instruments: Instrument[] }>("iserver/watchlist", { params: { id: watchlistId } }),
      );

      if (err) {
        this.statusBox.setContent(
          `Error loading instruments: ${(err as any)?.message || err}`,
        );
        this.screen.render();
        return;
      }

      this.currentInstruments = r!.data.instruments;
      this.updateInstrumentDisplay();
      this.subscribeToPrices();
      this.statusBox.setContent(
        `Loaded ${this.currentInstruments.length} instruments`,
      );
      this.screen.render();
    } catch (error) {
      this.statusBox.setContent(`Error: ${error}`);
      this.screen.render();
    }
  }

  private updateInstrumentDisplay() {
    const items = this.currentInstruments.map((inst) => {
      const price = this.prices.get(inst.conid);
      const priceStr = price ? `${price["31"] || "N/A"}` : "Loading...";
      return `${inst.symbol} (${inst.exchange}) - ${priceStr}`;
    });
    this.instrumentList.setItems(items);
    this.screen.render();
  }

  private subscribeToPrices() {
    if (this.ws) {
      this.ws.close();
    }

    this.ws = new WebSocket("wss://localhost:5000/v1/api/ws");

    this.ws.on("open", () => {
      this.statusBox.setContent("WebSocket connected");
      this.screen.render();

      // Subscribe to prices for all instruments
      this.currentInstruments.forEach((inst) => {
        const message = `smd+${inst.conid}+{"fields":["31","84","85","86","88"]}`;
        this.ws!.send(message);
      });
    });

    this.ws.on("message", (data: any) => {
      try {
        const message = JSON.parse(data.toString());
        if (message.conid && message["31"]) {
          this.prices.set(message.conid, message);
          this.updateInstrumentDisplay();
        }
      } catch (e) {
        // Ignore non-JSON messages
      }
    });

    this.ws.on("error", (error: any) => {
      this.statusBox.setContent(`WebSocket error: ${error.message}`);
      this.screen.render();
    });

    this.ws.on("close", () => {
      this.statusBox.setContent("WebSocket disconnected");
      this.screen.render();
    });
  }

  private cleanup() {
    if (this.ws) {
      this.ws.close();
    }
  }

  public run() {
    this.screen.render();
  }
}

// Run the monitor
const monitor = new WatchlistMonitor();
monitor.run();

