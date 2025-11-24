import { Hono } from "hono";
import { Main } from "../components/main";

const main = new Hono();

main.get("/", (c) => c.render(<Main />));

export default main;
