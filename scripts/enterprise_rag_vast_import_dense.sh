#!/usr/bin/env bash
set -euo pipefail

ROOT="${RECALL_ROOT:-/workspace/RE-call}"
DUMP="${1:-${ENTERPRISE_RAG_DENSE_DUMP:-/workspace/enterprise_rag_dense_12k.pgcustom}}"
TABLE="${ENTERPRISE_RAG_DENSE_TABLE:-ber_voy_lex_12k_full}"
TENANT="${ENTERPRISE_RAG_TENANT:-enterprise-rag-voyage-lexical-chunk12k-full}"
SPARSE_TABLE="${RECALL_SPARSE_TABLE:-recall_sparse_v1}"
REQUIRE_SPLADE="${ENTERPRISE_RAG_REQUIRE_SPLADE:-0}"

cd "$ROOT"
source .venv/bin/activate
set -a
[[ ! -f .env ]] || source ./.env
set +a

[[ -n "${RECALL_DSN:-}" ]] || {
  echo "ERROR: RECALL_DSN is required" >&2
  exit 1
}
[[ -s "$DUMP" ]] || {
  echo "ERROR: missing dense dump: $DUMP" >&2
  exit 1
}

python - <<'PY'
import os
import psycopg

conn = psycopg.connect(os.environ["RECALL_DSN"], autocommit=True)
conn.execute("create extension if not exists vector")
PY

pg_restore \
  --clean \
  --if-exists \
  --no-owner \
  --no-acl \
  --dbname "$RECALL_DSN" \
  "$DUMP"

python - <<PY
import os
import psycopg

table = "$TABLE"
tenant = "$TENANT"
sparse_table = "$SPARSE_TABLE"
require_splade = "$REQUIRE_SPLADE"
conn = psycopg.connect(os.environ["RECALL_DSN"])
with conn.cursor() as cur:
    cur.execute(
        f"select count(*), count(distinct source) from {table} where tenant_id = %s",
        (tenant,),
    )
    chunks, sources = cur.fetchone()
    cur.execute("select to_regclass(%s) is not null", (sparse_table,))
    sparse_exists = cur.fetchone()[0]
    sparse_ids = 0
    if sparse_exists:
        cur.execute(
            f"""
            select count(distinct id)
            from {sparse_table}
            where tenant_id = %s and chunk_table = %s
            """,
            (tenant, table),
        )
        sparse_ids = cur.fetchone()[0]
print(
    f"imported_table={table} tenant={tenant} chunks={chunks} sources={sources} "
    f"sparse_ids={sparse_ids} require_splade={require_splade}"
)
if chunks < 500000:
    raise SystemExit(f"imported dense table is too small: {chunks}")
if require_splade == "1" and sparse_ids < chunks:
    raise SystemExit(
        f"imported SPLADE sidecar is incomplete: {sparse_ids} sparse ids for {chunks} chunks"
    )
PY

mkdir -p results/enterprise_rag
date -Is > results/enterprise_rag/vast_dense_import.done
