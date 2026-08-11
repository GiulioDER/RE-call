#!/usr/bin/env bash
set -euo pipefail

BRANCH="${RECALL_BRANCH:-codex/enterprise-rag-bench}"
ROOT="${RECALL_ROOT:-/workspace/RE-call}"
DATA_DIR="$ROOT/.benchdata/enterprise-rag-v1.0.0"
PG_PORT="${RECALL_PG_PORT:-55432}"
PGDATA="${RECALL_PGDATA:-/workspace/recall-pgdata}"
PGSOCK="${RECALL_PGSOCK:-/workspace/recall-pgsock}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

[[ -n "${VOYAGE_API_KEY:-}" ]] || fail "set VOYAGE_API_KEY before setup"
[[ -n "${OPENROUTER_API_KEY:-}" ]] || fail "set OPENROUTER_API_KEY before setup"

if [[ ! -d "$ROOT/.git" ]]; then
  mkdir -p "$(dirname "$ROOT")"
  git clone --branch "$BRANCH" --single-branch https://github.com/GiulioDER/RE-call.git "$ROOT"
else
  git -C "$ROOT" fetch origin "$BRANCH"
  git -C "$ROOT" checkout "$BRANCH"
  git -C "$ROOT" reset --hard "origin/$BRANCH"
fi

cd "$ROOT"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[voyage,sparse,bench,pool]"

mkdir -p "$DATA_DIR"
python - <<'PY'
import json
import urllib.request
from pathlib import Path

base = Path(".benchdata/enterprise-rag-v1.0.0")
with urllib.request.urlopen(
    "https://api.github.com/repos/onyx-dot-app/EnterpriseRAG-Bench/releases/latest"
) as response:
    release = json.load(response)
assets = {asset["name"]: asset["browser_download_url"] for asset in release["assets"]}
for name in ("questions.jsonl", "all_documents.zip"):
    out = base / name
    if out.exists() and out.stat().st_size > 0:
        print(f"{name} exists {out.stat().st_size}")
        continue
    print(f"downloading {name}")
    urllib.request.urlretrieve(assets[name], out)
    print(f"{name} {out.stat().st_size}")
PY

has_native_pgvector() {
  compgen -G "/usr/share/postgresql/*/extension/vector.control" >/dev/null
}

if [[ -z "${RECALL_DSN:-}" ]]; then
  if have docker; then
    docker rm -f recall-enterprise-pg >/dev/null 2>&1 || true
    docker run -d \
      --name recall-enterprise-pg \
      -e POSTGRES_USER=recall \
      -e POSTGRES_PASSWORD=recall \
      -e POSTGRES_DB=recall_bench \
      -p "$PG_PORT:5432" \
      pgvector/pgvector:pg16 >/dev/null
    export RECALL_DSN="postgresql://recall:recall@127.0.0.1:${PG_PORT}/recall_bench"
    sleep 5
  elif have initdb && have pg_ctl && has_native_pgvector; then
    mkdir -p "$PGSOCK"
    if [[ ! -s "$PGDATA/PG_VERSION" ]]; then
      initdb -D "$PGDATA" --auth=trust --encoding=UTF8 --locale=C.UTF-8
    fi
    if ! pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then
      pg_ctl -D "$PGDATA" -l /workspace/recall-postgres.log -o "-p $PG_PORT -k $PGSOCK" start
    fi
    createdb -h "$PGSOCK" -p "$PG_PORT" recall_bench 2>/dev/null || true
    export RECALL_DSN="postgresql:///?host=${PGSOCK}&port=${PG_PORT}&dbname=recall_bench"
  else
    fail "no RECALL_DSN, no docker, and no native Postgres plus pgvector"
  fi
fi

cat > .env <<EOF
RECALL_DSN="${RECALL_DSN}"
VOYAGE_API_KEY="${VOYAGE_API_KEY}"
OPENROUTER_API_KEY="${OPENROUTER_API_KEY}"
EOF
chmod 600 .env

set -a
source ./.env
set +a

if [[ "$RECALL_DSN" == postgresql://recall:*@127.0.0.1:* ]]; then
  python - <<'PY'
import os
import psycopg

conn = psycopg.connect(os.environ["RECALL_DSN"], autocommit=True)
conn.execute("create extension if not exists vector")
print("pgvector ok")
PY
elif [[ "$RECALL_DSN" == postgresql:///\?host=* ]]; then
  python - <<'PY'
import os
import psycopg

conn = psycopg.connect(os.environ["RECALL_DSN"], autocommit=True)
conn.execute("create extension if not exists vector")
print("pgvector ok")
PY
fi

./scripts/enterprise_rag_vast_preflight.sh

cat <<EOF
ready
root=$ROOT
smoke:
  cd $ROOT
  ./scripts/enterprise_rag_vast_smoke.sh
run:
  cd $ROOT
  nohup ./scripts/enterprise_rag_vast_run.sh > logs/enterprise_rag_vast_top_splade.log 2>&1 & echo \\$! > enterprise_rag_vast_top_splade.pid
EOF
