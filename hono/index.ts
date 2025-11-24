import { Hono } from "hono";
import { logger } from "hono/logger";
import { jsxRenderer } from "hono/jsx-renderer";
import { serveStatic, upgradeWebSocket, websocket } from "hono/bun";
import main from "./src/modules/main";

const app = new Hono();

app.use("*", logger(), jsxRenderer(), serveStatic({ root: "./static" }));

app.route("/", main);
app.get(
  "/ws",
  upgradeWebSocket((_) => {
    console.log("Upgrading websockets");
    return {
      onOpen: () => console.log("Opened"),
      onClose: () => {
        console.log(`WS connection closed`);
      },
      onError: () => {
        console.log(`WS connection errored`);
      },
    };
  }),
);

export default { fetch: app.fetch, websocket };
