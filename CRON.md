# Scheduling run_pipeline.sh with cron

`run_pipeline.sh` is designed to be cron-safe (no TTY, detached compose,
stderr log capture). Add a crontab entry for user `nasrt`:

```bash
crontab -e -u nasrt
```

## Required environment in cron

Cron does NOT source your interactive shell, so export the secrets and tool
paths the pipeline needs. Put these at the top of your crontab:

```
# --- PATH (uv, docker inherited via /usr/bin; pi/node live under nvm) ---
PATH=/home/nasrt/.local/bin:/usr/bin:/bin:/home/nasrt/.nvm/versions/node/v24.16.0/bin

# --- pi provider/model (same values your interactive shell uses) ---
PI_PROVIDER=deepseek
PI_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=REPLACE_ME

# --- Telegram ---
TELEGRAM_TOKEN=REPLACE_ME
TELEGRAM_CHAT=REPLACE_ME

# --- IBKR creds (also in py/.env, but explicit beats ambient) ---
IBKR_USERNAME=REPLACE_ME
IBKR_PASSWORD=REPLACE_ME
TRADING_MODE=paper
```

> **Security note:** these lines contain secrets in plaintext inside your
> crontab (`/var/spool/cron/crontabs/nasrt`, readable only by root and you,
> mode 0600). If that is unacceptable, source a `0600` env file instead:
> `0 8 * * 1-5  . /home/nasrt/ibkr_cron.env && /home/nasrt/Documents/code/dev/ibkr/run_pipeline.sh`

## Example schedules

Run every weekday at 08:00, gateway downtime tolerated, log to a file:

```bash
# Minute Hour Day  Mon  * *  command
0 8 * * 1-5  PATH=... env >> /home/nasrt/Documents/code/dev/ibkr/data/pipeline.out 2>&1 \
             /home/nasrt/Documents/code/dev/ibkr/run_pipeline.sh
```

Reuse an already-running gateway instead of bouncing it each time:

```bash
0 8 * * 1-5  SKIP_GATEWAY=1 /home/nasrt/Documents/code/dev/ibkr/run_pipeline.sh
```

> If you use `SKIP_GATEWAY=1`, make sure the gateway is running first
> (e.g. start it with a separate `docker compose up -d` at boot).

## Tuning knobs (env vars)

| Var              | Default        | Meaning                                   |
|------------------|----------------|-------------------------------------------|
| `UNIVERSES`      | nsdq + sector + biotech | array of universe paths, edit in script   |
| `BENCHMARKS`     | QQQ, SPY, XBI | RS benchmark downloaded alongside each universe (index-aligned with `UNIVERSES`) |
| `DL_FROM`        | `2026-01-01`   | candle history start date                 |
| `TRADER_PROMPT`  | `~/.pi/agent/prompts/trader.md` | pi system prompt for analysis |
| `PI_MODEL`       | (ambient)      | model for the pi analysis step            |
| `PI_TIMEOUT`     | `600` s        | max pi analysis time                      |
| `GATEWAY_TIMEOUT`| `180` s        | max wait for gateway healthcheck          |
| `SKIP_GATEWAY`   | unset          | `1` = reuse running gateway (still waits) |
| `KEEP_RECENT`    | `20`           | number of screen/analysis artifacts to retain |
| `DRY`            | unset          | `1` = print commands, don't execute       |

## Failure handling

- `set -euo pipefail`: any step failing with nonzero exit aborts the run.
- A `trap ... EXIT` always runs cleanup (artifact pruning + orphaned-browser
  kill) even when a step fails.
- **No valid session check:** before login the script curls `/v1/api/tickle`. If
  it returns HTTP 200 with `iserver.authStatus.authenticated == true` it skips
  login entirely; otherwise it logs in and then *re-checks* tickle, failing fast
  if the login did not actually authenticate. This avoids unnecessary logins and
  re-authentication prompts on every cron tick.
- Nothing is sent to Telegram unless the pi analysis produced non-empty output
  (prevents a telegram claiming a signal list from a failed run).
- The telegram message body is exactly the pi signals report (piped on stdin,
  via `-T` bold title + `-` stdin text).
- Every `log()` line is appended to `data/pipeline.log` AND stderr, so you
  still get the record even when the redirected cron stdout is empty.
- Screen reports (`data/screen_reports_*.txt`) and analysis
  (`data/analysis_*.md`) are kept with timestamped filenames; the cleanup step
  prunes them beyond `KEEP_RECENT` (default 20).

## Verification

```bash
DRY=1 ./run_pipeline.sh        # preview every command without running
./run_pipeline.sh              # full run (requires gateway up / session)
```

## Per-universe RS benchmark

Each universe's `rs` screen is scored against a benchmark relevant to its own
symbols, and that benchmark's candles are downloaded as part of Step 3:

| Universe            | RS benchmark |
|---------------------|-------------|
| `universes/nsdq.json`    | QQQ     |
| `universes/sector.json`  | SPY     |
| `universes/biotech.json` | XBI     |

The mapping lives in `scripts/run_screens.py` (`UNIVERSE_BENCHMARKS`) and is
mirrored by the `BENCHMARKS` array in `run_pipeline.sh`. Keep the two lists in
sync when you add universes.

The cleanup step also kills orphaned headless Chromium/Playwright processes
left by a previously interrupted login (crash recovery), scoped to the current
user.
