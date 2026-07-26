#!/usr/bin/env bash
# Build + score the merged-S LongMemEval arm, crash-tolerantly.
#
# Two facts about this host shape the script, both measured rather than assumed:
#
#   1. Postgres died once mid-index (backend exit 2, 2026-07-26 10:52 UTC) with the host at
#      ~47 MB free of 12.3 GB. Free memory, not disk, is the constraint.
#   2. `Indexer` skips an already-indexed file on a stored content-hash comparison BEFORE
#      embedding, and writes each file's rows in one transaction. So a crash costs the current
#      file, not the run: re-invoking resumes.
#
# Therefore: wait for headroom, run, and on a non-zero exit go back and wait again. Every restart
# is a resume. The gate is a floor on FREE memory, so this job yields to whatever else is running
# instead of competing with it — it must not be the process that kills someone else's benchmark.

set -uo pipefail
cd "$(dirname "$0")" || exit 1

# Measured, then corrected. The first value here was 1800, taken from an early 1.4 GB reading of
# the Oracle indexer. Sampling the same process later showed its working set oscillating between
# ~1.5 GB and ~3.25 GB (ONNX arena growth plus Windows working-set trimming — it is not a leak;
# the count came back down while the chunk count held still). A gate below the PEAK would admit
# the job and then page, which is what preceded the Postgres reset this script exists to survive.
MIN_FREE_MB=${MIN_FREE_MB:-3500}
POLL_S=${POLL_S:-120}
TABLE=${TABLE:-lme_s}
OUT=s_out/labelled_s.json
ERR=s_out/labelled_s.err
LOG=s_out/runner.log

free_mb() {
  powershell -NoProfile -Command \
    '[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1KB)' 2>/dev/null | tr -d ' \r\n'
}

rows() {
  docker exec recall-db-1 psql -U recall -d recall -tAc \
    "SELECT count(*) FROM ${TABLE}" 2>/dev/null | tr -d ' \r\n'
}

say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

say "runner start: table=${TABLE} min_free=${MIN_FREE_MB}MB"

attempt=0
while true; do
  # Yield to other tenants of this machine until there is real headroom.
  while true; do
    f=$(free_mb)
    [ -z "$f" ] && f=0
    if [ "$f" -ge "$MIN_FREE_MB" ]; then break; fi
    say "waiting for memory: free=${f}MB < ${MIN_FREE_MB}MB"
    sleep "$POLL_S"
  done

  attempt=$((attempt + 1))
  started=$(date -u +%s)
  say "attempt ${attempt}: free=$(free_mb)MB rows=$(rows)"

  .venv/Scripts/python.exe -m recall.eval.labelled \
    --corpus ./s_out/corpus --questions ./s_out/questions.json --table "$TABLE" \
    > "$OUT" 2> "$ERR"
  rc=$?

  # Diagnostic only — the authoritative completeness check is the sources_expected /
  # sources_indexed pair the harness now records, which catches a short table whether or not a
  # reset was logged. This just names the cause when one did happen.
  if docker logs --since "$((($(date -u +%s) - started) + 30))s" recall-db-1 2>&1 \
       | grep -qiE "exited with exit code|database system was interrupted"; then
    say "NOTE: postgres reset inside this build window — verify sources_indexed in ${OUT}"
  fi

  # Success is a parseable report, not merely exit 0 — a truncated or empty stdout with a zero
  # status would otherwise read as a finished benchmark.
  if [ $rc -eq 0 ] && .venv/Scripts/python.exe -c "import json,sys; json.load(open('$OUT'))" 2>/dev/null; then
    say "DONE after ${attempt} attempt(s): rows=$(rows)"
    exit 0
  fi

  say "attempt ${attempt} failed (rc=${rc}) rows=$(rows) — resuming after backoff"
  tail -3 "$ERR" 2>/dev/null | sed 's/^/    /' | tee -a "$LOG"
  sleep "$POLL_S"
done
