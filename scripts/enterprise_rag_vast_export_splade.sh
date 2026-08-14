#!/usr/bin/env bash
set -euo pipefail

ROOT="${RECALL_ROOT:-/workspace/RE-call}"
TABLE="${ENTERPRISE_RAG_DENSE_TABLE:-ber_voy_lex_12k_full}"
TENANT="${ENTERPRISE_RAG_TENANT:-enterprise-rag-voyage-lexical-chunk12k-full}"
SPARSE_TABLE="${RECALL_SPARSE_TABLE:-recall_sparse_v1}"
OUT="${ENTERPRISE_RAG_SPLADE_DUMP:-/workspace/enterprise_rag_splade_full.pgcustom}"

cd "$ROOT"
source .venv/bin/activate
set -a
[[ ! -f .env ]] || source ./.env
set +a

[[ -n "${RECALL_DSN:-}" ]] || {
  echo "ERROR: RECALL_DSN is required" >&2
  exit 1
}

python - <<PY
import os
import psycopg

table = "$TABLE"
tenant = "$TENANT"
sparse_table = "$SPARSE_TABLE"

conn = psycopg.connect(os.environ["RECALL_DSN"])
with conn.cursor() as cur:
    cur.execute(
        f"select count(*), count(distinct source) from {table} where tenant_id = %s",
        (tenant,),
    )
    chunks, sources = cur.fetchone()
    cur.execute("select to_regclass(%s) is not null", (sparse_table,))
    sparse_exists = cur.fetchone()[0]
    if not sparse_exists:
        raise SystemExit(f"{sparse_table} does not exist")
    cur.execute(
        f"""
        select count(*), count(distinct id), count(distinct profile_id)
        from {sparse_table}
        where tenant_id = %s and chunk_table = %s
        """,
        (tenant, table),
    )
    sparse_rows, sparse_ids, sparse_profiles = cur.fetchone()
    cur.execute(
        f"""
        select profile_id, count(*)
        from {sparse_table}
        where tenant_id = %s and chunk_table = %s
        group by 1
        order by 2 desc, 1
        """,
        (tenant, table),
    )
    profiles = cur.fetchall()

print(
    f"source_table={table} tenant={tenant} chunks={chunks} sources={sources} "
    f"sparse_rows={sparse_rows} sparse_ids={sparse_ids} sparse_profiles={sparse_profiles}"
)
for profile_id, count in profiles:
    print(f"sparse_profile={profile_id} rows={count}")
if chunks < 500000:
    raise SystemExit(f"chunk table is too small: {chunks}")
if sparse_ids < chunks:
    raise SystemExit(
        f"learned sparse sidecar is incomplete: {sparse_ids} sparse ids for {chunks} chunks"
    )
PY

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT" "$OUT.sha256" "$OUT.sizes.txt" "$OUT.status.env"

pg_dump "$RECALL_DSN" \
  --format=custom \
  --compress=9 \
  --no-owner \
  --no-acl \
  --table "$TABLE" \
  --table "$SPARSE_TABLE" \
  --file "$OUT"

pg_restore -l "$OUT" | grep -q "TABLE DATA public $TABLE"
pg_restore -l "$OUT" | grep -q "TABLE DATA public $SPARSE_TABLE"

sha256sum "$OUT" > "$OUT.sha256"
{
  ls -lh "$OUT" "$OUT.sha256"
  printf "dump_manifest_tables=\\n"
  pg_restore -l "$OUT" | grep -E "TABLE DATA public ($TABLE|$SPARSE_TABLE)"
} > "$OUT.sizes.txt"
{
  printf "table=%q\\n" "$TABLE"
  printf "tenant=%q\\n" "$TENANT"
  printf "sparse_table=%q\\n" "$SPARSE_TABLE"
  printf "dump=%q\\n" "$OUT"
  printf "created_at=%q\\n" "$(date -Is)"
} > "$OUT.status.env"

cat "$OUT.sizes.txt"
