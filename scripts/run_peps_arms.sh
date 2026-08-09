#!/usr/bin/env bash
# The two PEPs invocations behind the rerank x pool-width 2x2, serially, in ONE process.
#
# Prior work: [[closed-hypothesis-recall-rerank-pool-interaction-2026-08-05]] is what this
# produced; the prediction it was run against is benchmarks/PREREGISTRATION-peps-rerank-pool.md and
# the scorer is benchmarks/score_peps_rerank_pool.py. Related, and NOT this: benchmarks/
# rerank_pool_arms.py measures LOCOMO at a fixed pool, which is the arm this experiment contrasts.
#
# Each `--rerank` invocation reports BOTH a `hybrid` and a `hybrid+rerank` arm, so two invocations
# give the full 2x2: A1/A2 at candidate-k 20, A3/A4 at candidate-k 250. `k=5` throughout, so the
# ONLY thing that varies is the pool.
#
# WRITTEN AFTER TWO REAL FAILURES, and each defence below is one of them:
#
#   1. A duplicate copy of the driver ran concurrently. Both wrote the same output path, both
#      contended for CPU, one died on ConcurrentMigrator, and the surviving ck250 arm was killed
#      after 94 minutes leaving a 0-byte file and NO traceback. => the LOCK below. Same defence,
#      same reason, as scripts/run_locomo_arms.sh.
#   2. The original driver reported `rc=0 bytes=NNNNN` and that was treated as success. An exit
#      code is not a measurement. => verify_arm() below reads the ARTIFACT and checks the
#      apparatus invariant, so a run that completes but scored the wrong population fails HERE
#      rather than silently becoming a published number.
#
# ⚠️ MIGRATION HAZARD. Run this from a checkout whose recall/migrations/sql/0008_*.sql matches what
# the target database already has. `0008` was edited IN PLACE (OIDC, #202): against a pre-existing
# DB a newer checkout raises MigrationChecksumMismatch, and the whole point of reusing --table is
# that re-indexing costs hours. The published arms were produced at de2a712 for exactly this
# reason. Pinning also keeps all four arms on ONE commit, which is what makes the deltas paired.
#
# Usage:
#   scripts/run_peps_arms.sh                       # defaults below
#   PEPS_CORPUS=/path/to/peps scripts/run_peps_arms.sh
#   RECALL_DSN=... PEPS_PY=/path/to/python OUT=results/peps_rerank_pool/raw \
#       PEPS_TABLE=peps_bge scripts/run_peps_arms.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY=${PEPS_PY:-python}
DSN=${RECALL_DSN:-postgresql://recall:recall@localhost:5434/recall}
CORPUS=${PEPS_CORPUS:-peps-corpus/peps}
TABLE=${PEPS_TABLE:-peps_bge}
QUESTIONS=recall/eval/peps_questions.json
OUT=${OUT:-results/peps_rerank_pool/raw}
POOLS=${PEPS_POOLS:-"20 250"}

# The apparatus invariant, asserted per arm. On the SAMPLE SIZES, never the rates: `false_abstain`
# .rate legitimately moves with candidate_k (0.0227 at ck20, 0.0909 at ck250) because
# research_search shares the pool (EVAL-002), so asserting the rate would false-void a valid arm.
EXPECT_SCORED=88
EXPECT_FALSE_ABSTAIN_N=44
EXPECT_ABSTENTION_N=11

# ---- preflight: fail in seconds, not after an hour of indexing --------------------------------
for p in "$CORPUS" "$QUESTIONS"; do
  [ -e "$p" ] || { echo "FATAL: missing $p (set PEPS_CORPUS / run from the repo root)"; exit 1; }
done
command -v "$PY" >/dev/null 2>&1 || [ -x "$PY" ] || { echo "FATAL: no python at '$PY' (set PEPS_PY)"; exit 1; }

mkdir -p "$OUT"

# ---- one lock: a second copy exits instead of racing the first --------------------------------
LOCK=$OUT/.arms.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FATAL: $LOCK exists; another run is in progress (or died; rmdir it to clear)."
  exit 1
fi

LOG=$OUT/run.log
SAMPLES=$OUT/resource_samples.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---- resource sampler -------------------------------------------------------------------------
# The killed arm left 91 minutes of stderr silence and no traceback, which makes memory exhaustion
# and an external kill indistinguishable afterwards. Sampling discriminates them: a ramp to the
# ceiling is memory, a flat line that simply stops is something else killing it. Windows-only
# (powershell); skipped silently elsewhere rather than failing the run.
SAMPLER_PID=""
if command -v powershell.exe >/dev/null 2>&1; then
  ( while true; do
      powershell.exe -NoProfile -Command "
        \$p = Get-Process python -ErrorAction SilentlyContinue | Sort-Object WS -Desc | Select-Object -First 1;
        \$os = Get-CimInstance Win32_OperatingSystem;
        '{0} rss_mb={1} free_mb={2}' -f (Get-Date -Format HH:mm:ss), [int](\$p.WS/1MB), [int](\$os.FreePhysicalMemory/1KB)
      " 2>/dev/null | tr -d '\r' >> "$SAMPLES"
      sleep 60
    done ) & SAMPLER_PID=$!
fi
cleanup() { [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null; rmdir "$LOCK" 2>/dev/null; }
trap cleanup EXIT

# ---- verify the ARTIFACT, not the exit code ---------------------------------------------------
# Paths and expectations reach Python through argv, never pasted into the `python -c` source
# string (scripts/run_locomo_arms.sh documents why that matters for values that can contain quotes).
# The source goes in on stdin via a QUOTED heredoc (<<'PYEOF') rather than `python -c '...'`.
# `-c` with a single-quoted shell string cannot contain a single quote: `arms[a].get('hit_at_5')`
# would silently close the shell string mid-source, and escaping quotes inside an f-string
# expression is a Python syntax error in its own right. A quoted heredoc is literal, so both
# classes of bug are structurally impossible. Values still arrive as argv, never interpolated.
verify_arm() {  # $1 = report path, $2 = requested candidate_k
  "$PY" - "$1" "$2" "$EXPECT_SCORED" "$EXPECT_FALSE_ABSTAIN_N" "$EXPECT_ABSTENTION_N" <<'PYEOF'
import json, sys

path = sys.argv[1]
want_ck, want_scored, want_fa, want_aa = map(int, sys.argv[2:6])
try:
    d = json.load(open(path, encoding="utf-8"))
except Exception as exc:
    sys.exit(f"  ARTIFACT INVALID: {path} does not parse ({exc})")

bad = []
if d.get("candidate_k") != want_ck:
    bad.append(f"candidate_k {d.get('candidate_k')} != requested {want_ck} (I4)")
scored = d.get("questions", {}).get("retrieval_scored_on")
if scored != want_scored:
    bad.append(f"retrieval_scored_on {scored} != {want_scored}")
got_fa = d.get("false_abstain", {}).get("n")
if got_fa != want_fa:
    bad.append(f"false_abstain.n {got_fa} != {want_fa}. The abstention split leaked; "
               "no retrieval number from this run is usable")
got_aa = d.get("abstention_accuracy", {}).get("n")
if got_aa != want_aa:
    bad.append(f"abstention_accuracy.n {got_aa} != {want_aa}")

arms = d.get("arms", {})
for a in ("hybrid", "hybrid+rerank"):
    if a not in arms:
        bad.append(f"arm {a!r} missing")
    else:
        got_n = arms[a].get("hit_at_5", {}).get("n")
        if got_n != want_scored:
            bad.append(f"arm {a!r} scored n={got_n} != {want_scored}")

# I2: an identical miss SET means the reranker did not run, or was inert, and the arm is not a
# measurement. Identical RATES would not catch it: two different miss sets can share a rate.
if all(a in arms for a in ("hybrid", "hybrid+rerank")):
    base = {m["id"] for m in arms["hybrid"]["misses"]}
    rer = {m["id"] for m in arms["hybrid+rerank"]["misses"]}
    if base == rer:
        bad.append("I2: hybrid+rerank has an IDENTICAL miss set to hybrid, so the reranker is inert")

if bad:
    sys.exit("  ARTIFACT INVALID:\n    " + "\n    ".join(bad))

ck = d["candidate_k"]
n = arms["hybrid"]["hit_at_5"]["n"]
hyb = arms["hybrid"]["hit_at_5"]["rate"]
rer_rate = arms["hybrid+rerank"]["hit_at_5"]["rate"]
fa_rate = d["false_abstain"]["rate"]
print(f"  verified: candidate_k={ck} n={n} hybrid={hyb} hybrid+rerank={rer_rate} "
      f"(false_abstain.rate={fa_rate}, moves with the pool, by design)")
PYEOF
}

# ---- the arms ---------------------------------------------------------------------------------
say "=== START. HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo '?') table=$TABLE ==="
rc_all=0
for ck in $POOLS; do
  report=$OUT/peps_ck${ck}.json
  # PEPS_VERIFY_ONLY re-checks reports that already exist without re-running a ~3h job. It exists
  # so the verifier below is TESTABLE: a guard nobody has watched fail is a hypothesis, not a guard.
  if [ -n "${PEPS_VERIFY_ONLY:-}" ]; then
    say "arm candidate-k=$ck : verify-only"
    if ! verify_arm "$report" "$ck" | tee -a "$LOG"; then rc_all=1; fi
    continue
  fi
  say "arm candidate-k=$ck : start"
  "$PY" -m recall.eval.labelled \
      --corpus "$CORPUS" --questions "$QUESTIONS" --glob '*.rst' \
      --embedder fastembed -k 5 --candidate-k "$ck" --rerank --score-retrieval-on all \
      --table "$TABLE" --dsn "$DSN" \
      > "$report" 2> "$OUT/peps_ck${ck}.err"
  rc=$?
  say "arm candidate-k=$ck : rc=$rc bytes=$(wc -c < "$report" 2>/dev/null || echo 0)"
  if [ "$rc" -ne 0 ]; then
    say "  FAILED. last stderr:"; tail -3 "$OUT/peps_ck${ck}.err" | tee -a "$LOG"
    [ -s "$SAMPLES" ] && { say "  last resource samples:"; tail -3 "$SAMPLES" | tee -a "$LOG"; }
    rc_all=1
    continue
  fi
  if ! verify_arm "$report" "$ck" | tee -a "$LOG"; then rc_all=1; fi
done

say "=== DONE. rc=$rc_all ==="
[ "$rc_all" -eq 0 ] && say "Score with: $PY -m benchmarks.score_peps_rerank_pool"
exit "$rc_all"
