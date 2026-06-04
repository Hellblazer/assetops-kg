#!/usr/bin/env bash
# Isolated, local-only nexus sandbox for the HERB benchmark.
#
# Every nexus state surface — T2 (SQLite), T3 (local ChromaDB), catalog, logs,
# daemon sockets — lives under .herb-sandbox/ and uses the local bge-768
# embedder. It NEVER touches the real knowledgebase, Chroma Cloud, or Voyage:
# cloud credentials are scrubbed from the environment and NX_LOCAL=1 forces the
# local backend.
#
# Usage:
#   ./benchmark/herb_sandbox.sh nx collection list
#   ./benchmark/herb_sandbox.sh .venv/bin/python -m benchmark.run_herb --mode index
#   ./benchmark/herb_sandbox.sh .venv/bin/python -m benchmark.run_herb --mode eval
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SB="$REPO/.herb-sandbox"
mkdir -p "$SB/config" "$SB/chroma" "$SB/fastembed"

export NEXUS_CONFIG_DIR="$SB/config"
export NX_LOCAL_CHROMA_PATH="$SB/chroma"
export NX_LOCAL=1
# Scrub cloud creds so a stray default can't leak the benchmark corpus to
# Chroma Cloud / Voyage. Local mode needs none of these.
unset CHROMA_API_KEY VOYAGE_API_KEY CHROMA_TENANT CHROMA_DATABASE 2>/dev/null || true

# One-time: record the 768-dim local embedder for this config dir.
if [ ! -f "$SB/config/config.yml" ]; then
  echo "[herb_sandbox] first run — provisioning bge-768 local embedder" >&2
  nx init --embedder bge-768 -y
fi

# Ensure the isolated daemons are up (idempotent). T2 has ensure-running;
# T3 start is a no-op spawn when already alive.
nx daemon t2 ensure-running >/dev/null 2>&1 || true
nx daemon t3 status >/dev/null 2>&1 || nx daemon t3 start >/dev/null 2>&1 || true

exec "$@"
