#!/usr/bin/env bash
# Isolated, local-only nexus sandbox for the HERB benchmark (PG-engine era).
#
# One virgin HOME under .herb-sandbox/home holds EVERY nexus surface: config dir,
# credentials, the bundled Postgres cluster, the engine binary and its logs. The
# service and the cluster bind loopback ports the OS allocates, so the sandbox cannot
# collide with a live managed service, a live local service or their config:
#
#   - env -i allowlist: HOME and NEXUS_CONFIG_DIR point into the sandbox, NX_LOCAL=1,
#     NX_NO_TELEMETRY=1. No NX_SERVICE_URL / NX_SERVICE_TOKEN, no VOYAGE_API_KEY, no
#     ANTHROPIC_API_KEY ever reaches a child (an ambient key would flip the engine
#     off the local bge-768 embedder or spend money).
#   - `nx init -y --no-autostart`: no launchd/systemd unit is registered. The unit
#     label is a constant keyed on the uid, not on HOME, so an autostart unit would
#     land in the operator's real session domain.
#   - after `up`, the sandbox service URL is read back and the script refuses
#     (non-zero) unless it is loopback.
#
# The claude CLI (needed only by nx_answer) cannot log in under a virgin HOME. Set
# HERB_SANDBOX_LLM=1 and supply CLAUDE_CODE_OAUTH_TOKEN in the caller's environment to
# let that one variable through, e.g. with the nexus repo's helper (keychain item
# nexus-automation-oauth-token; it needs the REAL HOME, so it wraps this script, not
# the other way round):
#   HERB_SANDBOX_LLM=1 python3 <nexus>/tests/e2e/lib/claude_credentials.py run -- \
#       ./benchmark/herb_sandbox.sh run python3 -m benchmark.herb_ladder --nx-answer
# Without HERB_SANDBOX_LLM=1 the token is dropped, so indexing (whose post-store hooks
# can queue aspect extraction, an LLM call) can never spend.
#
# Usage:
#   ./benchmark/herb_sandbox.sh up                    # provision (idempotent) and start
#   ./benchmark/herb_sandbox.sh run <command...>      # run a command inside the sandbox env
#   ./benchmark/herb_sandbox.sh <command...>          # same as run
#   ./benchmark/herb_sandbox.sh status                # service URL + loopback check
#   ./benchmark/herb_sandbox.sh down                  # stop the service and its Postgres
#   ./benchmark/herb_sandbox.sh destroy               # down, then remove the sandbox tree
#   ./benchmark/herb_sandbox.sh --self-test           # the env-allowlist checks, no nexus needed
#
# Overrides: HERB_SANDBOX_DIR (default <repo>/.herb-sandbox), NX_BIN (default: nx on PATH).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SB="${HERB_SANDBOX_DIR:-$REPO/.herb-sandbox}"
SB_HOME="$SB/home"
SB_CFG="$SB_HOME/.config/nexus"
REAL_HOME="$HOME"

_die() { echo "[herb_sandbox] $*" >&2; exit 1; }

# The sandbox must not be (or contain) the operator's real home or config.
_assert_sandbox_path() {
  case "$SB" in
    "$REAL_HOME"|"$REAL_HOME/"|"$REAL_HOME/.config"|"$REAL_HOME/.config/"*|"/"|"")
      _die "refusing sandbox path '$SB': it is or contains the real HOME / nexus config" ;;
  esac
  case "$SB" in
    */.herb-sandbox|"$REPO"/*|/tmp/*|/private/tmp/*|/var/folders/*) : ;;
    *) _die "refusing sandbox path '$SB': it must end in .herb-sandbox or live under the repo or a temp dir" ;;
  esac
}

NX_BIN="${NX_BIN:-$(command -v nx || true)}"

# The interpreter of the nx install (nx is a shim that exec's <tools>/current/bin/nx; the
# sibling python3 has nexus importable). Resolved from the REAL HOME, before the swap.
_nx_python() {
  local gen
  gen="$(readlink "$REAL_HOME/.local/share/nexus/tools/current" 2>/dev/null || true)"
  if [ -n "$gen" ] && [ -x "$gen/bin/python3" ]; then echo "$gen/bin/python3"; fi
}

# The allowlisted environment, NUL-separated.
_sandbox_env() {
  local nx_dir py_dir
  nx_dir="$(dirname "$NX_BIN")"
  # The caller's python3 (it carries the harness deps, rich and pytest), not whichever the bare
  # sandbox PATH finds first.
  py_dir="$(dirname "$(command -v python3)")"
  local e=(
    "HOME=$SB_HOME"
    "NEXUS_CONFIG_DIR=$SB_CFG"
    "PATH=$py_dir:$nx_dir:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    "TERM=${TERM:-dumb}"
    "NX_LOCAL=1"
    "NX_NO_TELEMETRY=1"
    "NX_PYTHON=${NX_PYTHON:-$(_nx_python)}"
    "HERB_SANDBOX=1"
    "HERB_DIR=${HERB_DIR:-$REAL_HOME/git/HERB}"
  )
  if [ -n "${HTTPS_PROXY:-}" ]; then e+=("HTTPS_PROXY=$HTTPS_PROXY"); fi
  if [ -n "${HTTP_PROXY:-}" ]; then e+=("HTTP_PROXY=$HTTP_PROXY"); fi
  if [ "${HERB_SANDBOX_LLM:-0}" = 1 ] && [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    e+=("CLAUDE_CODE_OAUTH_TOKEN=$CLAUDE_CODE_OAUTH_TOKEN")
  fi
  printf '%s\0' "${e[@]}"
}

_in_sandbox() {
  local -a e=()
  local x
  while IFS= read -r -d '' x; do e+=("$x"); done < <(_sandbox_env)
  env -i "${e[@]}" "$@"
}

# `nx daemon service status` prints "host: <h>" and "port: <p>" lines (pg_port is a different key).
_service_url() {
  _in_sandbox "$NX_BIN" daemon service status 2>&1 \
    | awk '$1=="host:"{h=$2} $1=="port:"{p=$2} END{if(h!=""&&p!="") printf "http://%s:%s\n", h, p}'
}

_assert_loopback() {
  case "$1" in
    http://127.0.0.1:*|http://localhost:*|http://\[::1\]:*) return 0 ;;
    *) return 1 ;;
  esac
}

cmd_up() {
  _assert_sandbox_path
  [ -x "$NX_BIN" ] || _die "nx not found (set NX_BIN)"
  mkdir -p "$SB_HOME/.cache/nexus" "$SB_CFG"
  # Seed the ONNX model cache (a 400 MB download otherwise) from the operator's cache.
  # Cache only, never config.
  local d
  for d in onnx_models ort_native djl; do
    if [ ! -d "$SB_HOME/.cache/nexus/$d" ] && [ -d "$REAL_HOME/.cache/nexus/$d" ]; then
      cp -Rc "$REAL_HOME/.cache/nexus/$d" "$SB_HOME/.cache/nexus/$d" 2>/dev/null \
        || cp -R "$REAL_HOME/.cache/nexus/$d" "$SB_HOME/.cache/nexus/$d"
    fi
  done
  echo "[herb_sandbox] nx init (local mode, virgin HOME $SB_HOME)" >&2
  _in_sandbox "$NX_BIN" init -y --no-autostart
  local url
  url="$(_service_url)"
  if [ -z "$url" ] || ! _assert_loopback "$url"; then
    echo "[herb_sandbox] service URL '$url' is not loopback; stopping the sandbox" >&2
    cmd_down || true
    exit 3
  fi
  echo "[herb_sandbox] up: $url" >&2
}

cmd_status() {
  _assert_sandbox_path
  local url
  url="$(_service_url)"
  echo "sandbox: $SB"
  echo "service: ${url:-<none>}"
  if [ -n "$url" ] && _assert_loopback "$url"; then echo "loopback: yes"; else echo "loopback: NO"; exit 3; fi
}

cmd_down() {
  _assert_sandbox_path
  if [ ! -d "$SB_CFG" ]; then echo "[herb_sandbox] nothing to stop" >&2; return 0; fi
  _in_sandbox "$NX_BIN" daemon service stop --with-pg || true
  # Indexing starts a per-tenant aspect-worker daemon (a foreground process with no stop verb).
  # End exactly the pids whose argv names THIS sandbox's config dir, nothing else.
  local pid
  # shellcheck disable=SC2009 # needs the argv, which pgrep -f would also match across other trees
  for pid in $(ps -axo pid=,command= | grep -F -- "--config-dir $SB_CFG" | grep -vF grep | awk '{print $1}'); do
    kill "$pid" 2>/dev/null || true
  done
}

cmd_destroy() {
  _assert_sandbox_path
  cmd_down
  # A postmaster that survived the stop would keep a SysV segment over a deleted data dir.
  # shellcheck disable=SC2009 # pgrep -f would also match a peer's argv; this matches this tree's path only
  if ps -axo pid=,command= | grep -F "postgres -D $SB_CFG" | grep -vF grep >/dev/null; then
    _die "a postgres for $SB_CFG is still running; not removing the tree (stop it by pid, never by pattern)"
  fi
  rm -rf "$SB"
  echo "[herb_sandbox] removed $SB" >&2
}

cmd_self_test() {
  # No nexus needed: the env allowlist, the loopback test and the path guards.
  local fail=0 out bad ok
  export CLAUDE_CODE_OAUTH_TOKEN=dummy VOYAGE_API_KEY=v ANTHROPIC_API_KEY=a NX_SERVICE_URL=https://x NX_SERVICE_TOKEN=t
  NX_BIN="${NX_BIN:-/usr/bin/true}"
  out="$(_sandbox_env | tr '\0' '\n')"
  for bad in VOYAGE_API_KEY ANTHROPIC_API_KEY NX_SERVICE_URL NX_SERVICE_TOKEN CLAUDE_CODE_OAUTH_TOKEN; do
    if grep -q "^$bad=" <<<"$out"; then echo "FAIL: $bad leaked"; fail=1; fi
  done
  grep -qx "HOME=$SB_HOME" <<<"$out" || { echo "FAIL: HOME is not the sandbox"; fail=1; }
  grep -qx "NEXUS_CONFIG_DIR=$SB_CFG" <<<"$out" || { echo "FAIL: NEXUS_CONFIG_DIR is not the sandbox"; fail=1; }
  grep -qx "NX_LOCAL=1" <<<"$out" || { echo "FAIL: NX_LOCAL"; fail=1; }
  grep -qx "NX_NO_TELEMETRY=1" <<<"$out" || { echo "FAIL: NX_NO_TELEMETRY"; fail=1; }
  out="$(HERB_SANDBOX_LLM=1 _sandbox_env | tr '\0' '\n')"
  grep -q "^CLAUDE_CODE_OAUTH_TOKEN=" <<<"$out" || { echo "FAIL: token not passed with HERB_SANDBOX_LLM=1"; fail=1; }
  if grep -q "^VOYAGE_API_KEY=" <<<"$out"; then echo "FAIL: voyage key leaked"; fail=1; fi
  for ok in 127.0.0.1 localhost; do _assert_loopback "http://$ok:1234" || { echo "FAIL: loopback $ok"; fail=1; }; done
  if _assert_loopback "https://api.conexus-nexus.com"; then echo "FAIL: managed URL passed loopback"; fail=1; fi
  if (SB="$REAL_HOME"; _assert_sandbox_path) 2>/dev/null; then echo "FAIL: real HOME accepted as sandbox"; fail=1; fi
  if (SB="$REAL_HOME/.config/nexus"; _assert_sandbox_path) 2>/dev/null; then echo "FAIL: live config accepted as sandbox"; fail=1; fi
  if [ "$fail" = 0 ]; then echo "herb_sandbox self-test: OK"; fi
  exit "$fail"
}

case "${1:-}" in
  up) cmd_up ;;
  down) cmd_down ;;
  status) cmd_status ;;
  destroy) cmd_destroy ;;
  --self-test) cmd_self_test ;;
  run) shift; _assert_sandbox_path; _in_sandbox "$@" ;;
  "") _die "usage: herb_sandbox.sh {up|down|status|destroy|run <cmd...>|<cmd...>}" ;;
  *) _assert_sandbox_path; _in_sandbox "$@" ;;
esac
