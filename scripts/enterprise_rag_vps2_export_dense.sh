#!/usr/bin/env bash
set -euo pipefail

ROOT="${RECALL_ROOT:-/home/sentiment/enterprise-rag-run/RE-call}"
TABLE="${ENTERPRISE_RAG_DENSE_TABLE:-ber_voy_lex_12k_full}"
OUT="${ENTERPRISE_RAG_DENSE_DUMP:-/home/sentiment/enterprise-rag-run/enterprise_rag_dense_12k.pgcustom}"

cd "$ROOT"
source .venv/bin/activate
set -a
[[ ! -f .env ]] || source ./.env
set +a

[[ -n "${RECALL_DSN:-}" ]] || {
  echo "ERROR: RECALL_DSN is required" >&2
  exit 1
}

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT" "$OUT.sha256"

pg_dump "$RECALL_DSN" \
  --format=custom \
  --compress=9 \
  --no-owner \
  --no-acl \
  --table "$TABLE" \
  --file "$OUT"

sha256sum "$OUT" > "$OUT.sha256"
ls -lh "$OUT" "$OUT.sha256"
