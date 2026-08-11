#!/usr/bin/env bash
set -euo pipefail

ROOT="${RECALL_ROOT:-$PWD}"
cd "$ROOT"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

need_file() {
  local path="$1"
  [[ -s "$path" ]] || fail "missing or empty file: $path"
}

need_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "missing environment variable: $name"
}

need_env VOYAGE_API_KEY
need_env OPENROUTER_API_KEY
need_env RECALL_DSN

need_file ".benchdata/enterprise-rag-v1.0.0/questions.jsonl"
need_file ".benchdata/enterprise-rag-v1.0.0/all_documents.zip"

python - <<'PY'
from recall._env import load_dotenv
from recall.embeddings import resolve_embedder
from recall.sparse import inspect_sparse_device, resolve_sparse_device
from recall.store import PgVectorStore
import os

load_dotenv()
embedder = resolve_embedder("voyage:voyage-4-large")
report = inspect_sparse_device("cuda")
device = resolve_sparse_device("cuda", report=report)
print(f"embedder_dim={embedder.dim}")
print(f"sparse_device={device}")
print(f"gpu_name={report.device_name}")
print(f"cuda_build={report.torch_cuda_build}")
print(f"free_vram_mb={report.free_vram_mb}")

with PgVectorStore(
    os.environ["RECALL_DSN"],
    dim=embedder.dim,
    pool_size=4,
) as store:
    store.ensure_schema()
with PgVectorStore(
    os.environ["RECALL_DSN"],
    dim=embedder.dim,
    table="bench_enterprise_rag_top_full",
    tenant="enterprise-rag-top-full",
    pool_size=4,
) as store:
    store.ensure_schema()
    print(f"bench_existing_chunks={store.count()}")
PY

python - <<'PY'
import json
import zipfile
from pathlib import Path

questions = Path(".benchdata/enterprise-rag-v1.0.0/questions.jsonl")
docs = Path(".benchdata/enterprise-rag-v1.0.0/all_documents.zip")
with questions.open(encoding="utf-8") as handle:
    first = json.loads(next(handle))
print(f"first_question_id={first['question_id']}")
with zipfile.ZipFile(docs) as zf:
    txt = sum(1 for name in zf.namelist() if name.endswith(".txt"))
print(f"document_text_files={txt}")
PY

echo "preflight ok"
