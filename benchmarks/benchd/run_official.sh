#!/usr/bin/env bash
# Official Bench'd run for RE-call: pins, hashes, meter snapshots, signed manifest, verify.
#
# Usage:
#   VOYAGE_API_KEY=... OPENROUTER_API_KEY=... RECALL_BENCHD_DSN=postgresql://... \
#     bash benchmarks/benchd/run_official.sh <harness-dir> <benchmark-slug> [workers]
#
# The script refuses to run unless:
#   - the RE-call working tree is clean (a dirty tree cannot be pinned to a commit),
#   - the adapter inside the harness is byte-identical to the committed one here,
#   - both API keys and the DSN are present.
#
# It never wipes a database: point RECALL_BENCHD_DSN at a fresh, dedicated instance.
set -euo pipefail

HARNESS_DIR=${1:?harness dir}
BENCH=${2:?benchmark slug (longmemeval-v1 | locomo-v1)}
WORKERS=${3:-4}  # measured 2026-08-23: 2.4x at 4 workers, score stable 12/13/12 across 1/4/8
REPO_DIR=$(cd "$(dirname "$0")/../.." && pwd)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$HARNESS_DIR/runs-official/$BENCH-$STAMP"

: "${VOYAGE_API_KEY:?VOYAGE_API_KEY must be set}"
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set}"
: "${RECALL_BENCHD_DSN:?RECALL_BENCHD_DSN must be set}"

# ---- pins ------------------------------------------------------------------
if [ -n "$(git -C "$REPO_DIR" status --porcelain)" ]; then
  echo "REFUSED: RE-call working tree is dirty; commit first so the run pins to a SHA" >&2
  exit 1
fi
RECALL_SHA=$(git -C "$REPO_DIR" rev-parse HEAD)
HARNESS_SHA=$(git -C "$HARNESS_DIR" rev-parse HEAD)
ADAPTER_REPO_SHA=$(python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$REPO_DIR/benchmarks/benchd/recall_adapter.py")
ADAPTER_LIVE_SHA=$(python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$HARNESS_DIR/benchd_harness/adapters/recall_adapter.py")
if [ "$ADAPTER_REPO_SHA" != "$ADAPTER_LIVE_SHA" ]; then
  echo "REFUSED: adapter in harness differs from the committed adapter" >&2
  exit 1
fi

# ---- champion configuration (tuned 2026-08-23, preregistered) --------------
export RECALL_BENCHD_EMBEDDER=voyage:voyage-4
export RECALL_BENCHD_RERANKER=voyage
export RECALL_BENCHD_SPARSE=lexical
export RECALL_BENCHD_TOP_K=10
export RECALL_BENCHD_GRANULARITY=session
export RECALL_BENCHD_SYNTH=deepseek/deepseek-v4-pro-0813
export RECALL_BENCHD_SYNTH_REASONING=off
export RECALL_BENCHD_SYNTH_MAX_TOKENS=2000
export RECALL_BENCHD_THRESHOLD=0.0
export RECALL_BENCHD_ABSTAIN=suppress
export RECALL_BENCHD_INGEST_CACHE=1
export RECALL_BENCHD_VERSION_TAG="recall-rag+${RECALL_SHA:0:12}+adapter-${ADAPTER_REPO_SHA:0:12}"

mkdir -p "$OUT"
python "$REPO_DIR/benchmarks/benchd/count_tokens.py" openrouter > "$OUT/meter-before.json"

# ---- dataset hash (download happens on first load; hash whatever is used) --
python - "$BENCH" > "$OUT/dataset.json" <<'EOF'
import hashlib, json, sys
from pathlib import Path
slug = sys.argv[1]
name = {"longmemeval-v1": ("longmemeval", "longmemeval_oracle.json"),
        "locomo-v1": ("locomo", "locomo10.json")}[slug]
path = Path.home() / ".cache" / "benchd" / name[0] / name[1]
if not path.exists():
    from benchd_harness.benchmarks import get_benchmark
    get_benchmark(slug).load_items()
print(json.dumps({"path": str(path),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "bytes": path.stat().st_size}))
EOF

if [ "${RECALL_BENCHD_DRYRUN:-0}" = "1" ]; then
  echo "DRY RUN OK: pins verified, meter snapshotted, dataset hashed. Stopping before spend."
  echo "  recall:  $RECALL_SHA"
  echo "  harness: $HARNESS_SHA"
  echo "  adapter: $ADAPTER_REPO_SHA"
  cat "$OUT/dataset.json"
  exit 0
fi

# ---- the run ---------------------------------------------------------------
cd "$HARNESS_DIR"
benchd run -a re-call -b "$BENCH" --judge --key ./keys/private.key \
  --workers "$WORKERS" --out "$OUT" | tee "$OUT/console.log"

python "$REPO_DIR/benchmarks/benchd/count_tokens.py" openrouter > "$OUT/meter-after.json"

MANIFEST=$(ls "$OUT"/run_*/manifest.signed.json | head -1)
benchd verify "$MANIFEST" | tee "$OUT/verify.log"
python "$REPO_DIR/benchmarks/benchd/count_tokens.py" manifest "$MANIFEST" > "$OUT/token-recount.txt"

# ---- run record ------------------------------------------------------------
python - "$OUT" "$BENCH" "$WORKERS" "$RECALL_SHA" "$HARNESS_SHA" "$ADAPTER_REPO_SHA" "$MANIFEST" <<'EOF'
import json, os, sys
out, bench, workers, recall_sha, harness_sha, adapter_sha, manifest = sys.argv[1:8]
record = {
    "benchmark": bench,
    "workers": int(workers),
    "recall_sha": recall_sha,
    "harness_sha": harness_sha,
    "harness_upstream": "https://github.com/benchdai/harness",
    "harness_patch": "benchmarks/benchd/harness-workers.patch (adds --workers; per-item pipeline unchanged)",
    "adapter_sha256": adapter_sha,
    "manifest": manifest,
    "config": {k: v for k, v in os.environ.items() if k.startswith("RECALL_BENCHD_")},
}
with open(os.path.join(out, "run-record.json"), "w") as fh:
    json.dump(record, fh, indent=2, sort_keys=True)
print("run record written")
EOF

echo "OFFICIAL RUN COMPLETE: $OUT"
