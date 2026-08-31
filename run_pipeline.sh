#!/usr/bin/env bash
#
# run_pipeline.sh — end-to-end IBKR trading signal pipeline.
#
#  1. Start the IBKR Client Portal Gateway (docker compose up -d) and wait for
#     its healthcheck /v1/api/tickle to pass.
#  2. If /tickle already reports an authenticated session (200 + authenticated:true)
#     skip login; otherwise login via Playwright (scripts/login_ibkr.py).
#  3. Download candles for each configured universe (data dl -U ...).
#  4. Run all screens (scripts/run_screens.py) over each universe.
#  5. Pipe the combined screen report to a headless `pi` run using the Trader
#     prompt (~/.pi/agent/prompts/trader.md) to produce a TRADE/SCALE/PASS
#     signal list.
#  6. Send the pi signals report to Telegram (the report IS the message body).
#  Cleanup (trap on EXIT) prunes old artifacts and orphaned headless browsers.
#
# Designed to be safe to run from cron. Any step failing with a nonzero exit
# aborts the pipeline. Missing data is fetched; existing candles are NOT
# refetched, so repeat runs are cheap once the DB is warm.
#
# Usage:
#   ./run_pipeline.sh                 # run everything
#   DRY=1 ./run_pipeline.sh           # print commands, don't execute
#   SKIP_GATEWAY=1 ./run_pipeline.sh  # reuse an already-running gateway
#
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$PROJECT_DIR/py"

# Universes to screen. Paths are relative to PY_DIR.
UNIVERSES=(
  "universes/nsdq.json"
  "universes/sector.json"
  "universes/biotech.json"
)

# RS benchmark downloaded alongside each universe (index-aligned with UNIVERSES)
# and used by the rs screen: nsdq->QQQ, sector->SPY, biotech->XBI.
BENCHMARKS=(
  "QQQ"
  "SPY"
  "XBI"
)

# Start-of-history date for candle downloads (fixed; candles are not refetched).
DL_FROM="${DL_FROM:-2026-01-01}"

# Trader prompt used as the system prompt for the pi analysis step.
TRADER_PROMPT="${TRADER_PROMPT:-$HOME/.pi/agent/prompts/trader.md}"

# Model for the pi analysis step (override freely; default keeps it cheap/fast).
PI_MODEL="${PI_MODEL:-}"

# Log file for the whole run. All log lines are appended here AND echoed to
# stderr, so a cron invocation can redirect nothing and still keep a record.
LOG_FILE="${LOG_FILE:-$PROJECT_DIR/data/pipeline.log}"
if [[ "${DRY:-0}" != "1" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")" && : >> "$LOG_FILE"
fi

# Logging helpers — wall-clock time per line, written to both the log file and
# stderr so stdout stays clean for piped data flow.
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
log() {
  local line="[$(stamp)] $*"
  if [[ "${DRY:-0}" != "1" ]]; then printf '%s\n' "$line" >> "$LOG_FILE"; fi
  printf '%s\n' "$line" >&2
}

# ── Helpers ─────────────────────────────────────────────────────────────────
die() { log "FATAL: $*" >&2; exit 1; }

run_or_dry() {
  # run_or_dry "label" cmd args...
  local label="$1"; shift
  if [[ "${DRY:-0}" == "1" ]]; then
    log "[DRY] $label: $*"
    return
  fi
  log "==> $label: $*"
  "$@"
}

# Check whether an authenticated gateway session already exists.
# A valid session: /tickle returns HTTP 200 with iserver.authStatus.authenticated
# == true. Anything else (401, or authenticated:false) means we must log in.
session_is_valid() {
  local code body
  code="$(curl -ks -H 'Accept: application/json' --max-time 15 -o /tmp/ibkr_tickle.json \
    -w '%{http_code}' "https://localhost:5000/v1/api/tickle" 2>/dev/null || true)"
  rm -f /tmp/ibkr_tickle.json
  if [[ "$code" != "200" ]]; then
    log "Gateway session check: http $code -> not valid"
    return 1
  fi
  # Re-tickle into a file so jq can parse the authenticated flag.
  curl -ks -H 'Accept: application/json' --max-time 15 -o /tmp/ibkr_tickle.json \
    "https://localhost:5000/v1/api/tickle" 2>/dev/null || { rm -f /tmp/ibkr_tickle.json; return 1; }
  local auth
  auth="$(jq -r '.iserver.authStatus.authenticated // false' /tmp/ibkr_tickle.json 2>/dev/null)"
  rm -f /tmp/ibkr_tickle.json
  log "Gateway session authenticated: $auth"
  [[ "$auth" == "true" ]]
}


# Wait for the gateway to be reachable; timeout in seconds.
# Treat 200 and 4xx as "server is up" — an unauthenticated gateway returns
# 401 on /tickle, and the login step runs afterwards to establish the session.
wait_for_gateway() {
  local url="${1:-https://localhost:5000/v1/api/tickle}"
  local timeout="${GATEWAY_TIMEOUT:-180}"
  local waited=0
  local code
  log "Waiting for gateway at $url (timeout ${timeout}s)..."
  while [[ "$waited" -lt "$timeout" ]]; do
    code="$(curl -ks --max-time 5 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || true)"
    # 200..499 = server answered (500 is a live-byte error, still server up).
    if [[ "$code" =~ ^[0-9]{3}$ ]] && [[ "$code" -lt 500 ]]; then
      log "Gateway is up (http $code)."
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  die "Gateway did not become reachable within ${timeout}s."
}

# ── Step 1: Start & wait for gateway ────────────────────────────────────────
step_gateway() {
  if [[ "${DRY:-0}" == "1" ]]; then
    log "[DRY] gateway up: docker compose --ansi never --project-directory $PROJECT_DIR up -d"
    return
  fi
  if [[ "${SKIP_GATEWAY:-0}" == "1" ]]; then
    log "SKIP_GATEWAY=1 — assuming gateway already running."
    wait_for_gateway
    return
  fi
  log "=== Step 1: docker compose up -d ==="
  # Feed /dev/null so compose can never block waiting on a TTY (cron-safe).
  # --ansi never avoids ANSI control chars in logs; -d detaches the gateway.
  run_or_dry "gateway up" \
    docker compose --ansi never --project-directory "$PROJECT_DIR" up -d
  wait_for_gateway
}

# ── Step 2: Login ───────────────────────────────────────────────────────────
step_login() {
  log "=== Step 2: check session / login to gateway ==="
  if session_is_valid; then
    log "Session already valid — skipping login."
    return
  fi
  log "No valid session — logging in."
  # login_ibkr.py loads IBKR_USERNAME/PASSWORD from py/.env via its load_env().
  run_or_dry "login" uv --directory "$PY_DIR" run scripts/login_ibkr.py
  # Fail fast if login did not actually establish an authenticated session.
  if [[ "${DRY:-0}" != "1" ]] && ! session_is_valid; then
    die "Login completed but /tickle still reports unauthenticated."
  fi
  log "Session authenticated after login."
}

# ── Step 3: Download candles per universe ───────────────────────────────────
step_download() {
  log "=== Step 3: download candles (from $DL_FROM) ==="
  local i
  for i in "${!UNIVERSES[@]}"; do
    local uni="${UNIVERSES[$i]}"
    local bench="${BENCHMARKS[$i]:-}"
    log "  downloading: $uni"
    run_or_dry "download $uni" \
      uv --directory "$PY_DIR" run ibkr data dl -U "$uni" -f "$DL_FROM"
    if [[ -n "$bench" ]]; then
      log "  downloading benchmark: $bench"
      run_or_dry "download benchmark $bench" \
        uv --directory "$PY_DIR" run ibkr data dl "$bench" -f "$DL_FROM"
    fi
  done
}

# ── Step 4: Run screens per universe ────────────────────────────────────────
# Output is captured to stdout (one report per universe) so it can be piped to
# the pi analysis step.
step_screens() {
  log "=== Step 4: run screens ==="
  for uni in "${UNIVERSES[@]}"; do
    log "  screening: $uni"
    run_or_dry "screen $uni" \
      uv --directory "$PY_DIR" run scripts/run_screens.py "$uni"
  done
}

# ── Step 5: Pi analysis (Trader prompt) ─────────────────────────────────────
step_analysis() {
  log "=== Step 5: pi analysis (trader prompt) ==="
  [[ -f "$TRADER_PROMPT" ]] || die "trader prompt not found: $TRADER_PROMPT"

  local sysprompt
  sysprompt="$(cat "$TRADER_PROMPT")"

  local sysprompt_flag=()
  if [[ -n "$sysprompt" ]]; then
    sysprompt_flag=(--system-prompt "$sysprompt")
  fi

  local model_flag=()
  if [[ -n "$PI_MODEL" ]]; then
    model_flag=(--model "$PI_MODEL")
  fi

  if [[ "${DRY:-0}" == "1" ]]; then
    log "[DRY] would pipe screen output to: pi --no-tools --no-session ${model_flag[*]} ${sysprompt_flag[*]} -p <trader analysis prompt>"
    return
  fi

  # Read screen reports: if a filename is passed, use it; else read stdin.
  local reports="$1"
  local body
  body="$(cat "$reports")"

  printf '%s\n' "Analyze the following screen results from the IBKR pipeline as a senior discretionary trader. Apply the Trader rules (volume confirmation, ATR-based risk, the six-point plan) and, for every symbol with a convergent or notable signal, give a one-word verdict TRADE / SCALE / PASS with a one-line reason. Group verdicts per universe. End with a single prioritized signal list (symbol, direction, verdict, key reason). Be terse.

SCREEN RESULTS:
${body}" | \
    timeout "${PI_TIMEOUT:-600}" pi --no-tools --no-session \
      ${model_flag[@]+"${model_flag[@]}"} \
      ${sysprompt_flag[@]+"${sysprompt_flag[@]}"} \
      -p
}

# ── Step 6: Telegram ────────────────────────────────────────────────────────
step_telegram() {
  log "=== Step 6: telegram ==="
  local analysis="$1"
  if [[ "${DRY:-0}" == "1" ]]; then
    log "[DRY] telegram would send contents of $analysis"
    return 0
  fi
  [[ -s "$analysis" ]] || die "empty analysis output from pi."
  # Send the pi signals report AS the message body: pipe it on stdin and pass
  # '-' as the positional text arg (telegram.sh only reads stdin when TEXT=='-').
  # -M renders markdown; -T adds a bold title line above the report.
  cat "$analysis" | timeout 60 \
    /home/nasrt/.agents/skills/telegram/telegram.sh -M -T "IBKR signal run — $(stamp)" - \
    || { log "WARN: telegram send returned nonzero."; return; }
  log "Telegram sent ($(wc -l <"$analysis") lines)."
}

# ── Cleanup ────────────────────────────────────────────────────────────────
# Runs after a completed pipeline (best-effort; failures never abort the run).
#  - Prune old screen/analysis artifacts beyond KEEP_RECENT (default 20).
#  - Kill orphaned headless chromium/playwright child processes left by a
#    previously interrupted login (crashed cron runs). Only targets chromium
#    owned by the current user to avoid nuking unrelated browsers.
step_cleanup() {
  log "=== Cleanup ==="
  local keep="${KEEP_RECENT:-20}"
  if [[ "${DRY:-0}" != "1" ]]; then
    ls -1t "$PROJECT_DIR"/data/screen_reports_*.txt 2>/dev/null | tail -n +$((keep + 1)) | xargs -r rm -f
    ls -1t "$PROJECT_DIR"/data/analysis_*.md 2>/dev/null | tail -n +$((keep + 1)) | xargs -r rm -f
    # Orphaned headless browsers from interrupted Playwright logins.
    pgrep -u "$(id -u)" -f "chromium .*headless.*playwright|chrome-headless" 2>/dev/null \
      | xargs -r kill -9
  else
    log "[DRY] cleanup: prune old reports (keep $keep), kill orphaned headless chromium"
  fi
  log "Cleanup done."
}


# ── Main ────────────────────────────────────────────────────────────────────
main() {
  log "==================================================="
  log "IBKR pipeline start (universes: ${UNIVERSES[*]})"
  log "==================================================="

  # Always clean up on exit (success or failure).
  trap step_cleanup EXIT

  step_gateway
  step_login
  step_download

  # In DRY mode nothing is written, so skip tee-ing empty artifacts.
  local reports
  local analysis
  if [[ "${DRY:-0}" == "1" ]]; then
    reports="/dev/null"
    analysis="/dev/null"
    step_screens
    step_analysis "$reports"
    step_telegram "$analysis"
  else
    # Capture both screen reports AND tee to a file so the analysis has a stable
    # record and telegram can log what was sent. Logs go to stderr (kept on the
    # terminal); only the screen data flows into the report file.
    reports="$PROJECT_DIR/data/screen_reports_$(date +%Y%m%d_%H%M%S).txt"
    step_screens | tee "$reports"

    analysis="$PROJECT_DIR/data/analysis_$(date +%Y%m%d_%H%M%S).md"
    step_analysis "$reports" | tee "$analysis"

    step_telegram "$analysis"
  fi

  log "Pipeline complete."
  log "==================================================="
}

main
