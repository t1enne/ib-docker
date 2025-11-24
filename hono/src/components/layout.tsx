import { FC } from "hono/jsx";
import { raw } from "hono/html";

export const Layout: FC = (props) => {
  return (
    <>
      {raw("<!doctype html>")}
      <html>
        <head>
          <title>IBKR</title>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <meta name="color-scheme" content="light dark" />
          <link rel="icon" href="/favicon.ico" />
          <link
            rel="stylesheet"
            type="text/css"
            href="https://cdn.jsdelivr.net/npm/@phosphor-icons/web@2.1.1/src/regular/style.css"
          />
          <link
            rel="stylesheet"
            type="text/css"
            href="https://cdn.jsdelivr.net/npm/@phosphor-icons/web@2.1.1/src/fill/style.css"
          />
          <script type="module" src="/index.js" defer />
          <script
            src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"
            defer
          />
          <link rel="stylesheet" href="/main.css" />
          <script
            src="https://cdn.jsdelivr.net/npm/htmx.org@2.0.6/dist/htmx.min.js"
            crossorigin="anonymous"
          />
          <script
            src="https://cdn.jsdelivr.net/npm/htmx-ext-ws@2.0.2"
            crossorigin="anonymous"
          />
          <script
            type=""
            dangerouslySetInnerHTML={{
              __html: `
				htmx.config.globalViewTransitions = true;
				htmx.config.useTemplateFragments = true;
				htmx.config.defaultSwapStyle = "outerHTML";
				htmx.config.allowEval = false;
				`,
            }}
          />
        </head>
        <body hx-ext="ws" ws-connect="/ws">
          <aside
            id="toaster"
            x-data
            className="toast fixed z-10 top-0 left-0 pl-12 pt-24 pointer-events-none"
          />
          <main id="root" className="p-4">
            {props.children}
          </main>
        </body>
      </html>
    </>
  );
};
