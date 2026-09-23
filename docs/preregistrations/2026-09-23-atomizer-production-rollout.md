# Micro atomizer production rollout: placement, production path, live gates

Status: frozen when committed. Nothing below the "Result" heading may be written before the
measurement, and nothing above it may be edited afterwards; corrections are appended.

## Question

The `micro` atomizer passed on C8/CAMBench (`2026-09-23-c8-atomizer-round3.md`) and on the memory
tenant's dense-only proxy (`2026-09-23-memory-atomizer-check.md`). The operator authorized a
memory-only production rollout and a latency budget of about 40 ms p95. Three things were never
measured and gate the rollout:

- **M1.** Does placing the rescue after fusion (`RECALL_ATOMIC_RESCUE_PLACEMENT=fused`) keep C8's
  rank-8 gains while removing its rank-1 losses?
- **M2.** On the memory tenant's **real serving path** (`recall_mcp.service.search_memory`, fusion
  with the lexical leg, reranker, trust policy and calibration autoload), with the artifact built
  by the production builder (per-chunk context, content-addressed store), which placement, if
  either, beats no rescue, and is the atomic stage inside the budget?
- **M3.** After deployment, does the live memory MCP server serve the atomic stage without errors
  or refusals, and does rollback work?

## Frozen apparatus

- Code: PR #706 (`claude/atomizer-production`) plus the study harnesses, on branch
  `claude/atomizer-measure`. VPS2 for M2 and M3 (memory tenant, user-authorized; no AML service,
  port or checkout touched), VPS3 for question writing and M1.
- **M2.** Generation pinned at the start of the run and recorded in the appendix. Artifact built by
  `scripts/build_atomic_micro_artifact.py` into a private registry under a fresh view store; the
  build's embedded and reused counts are recorded, and a second build of the same generation must
  embed nothing. Harness `scripts/memory_atomizer_production_check.py`, arms `off`, `dense`,
  `fused`, `k=10`. Fresh questions: seed 20260925, one span per source from 240 sources, excluding
  every source named as **gold** in `~/.codex/evals/atomic-fact-*` records and every source of the
  134 questions of the memory check; writer Llama 3.3 70B with the memory-note prompt; split
  dev/confirm by source hash. Hashes appended before evaluation.
- **M1.** C8 offline replay (`scripts/c8_atomizer_round3.py`) with a `micro_fused` arm; fresh
  questions from all CAMBench sessions, seed 20260926, disjoint from every span of the round-1 and
  round-3 probe files. Arms `off`, `micro` (dense placement), `micro_fused`.
- **M3.** Serving checkout fast-forwarded by `scripts/session-serving.sh sync`; `.env` gains
  `RECALL_ATOMIC_RESCUE_MODE=active`, the chosen placement and the production registry; the refresh
  pipeline's artifact step switches to the micro builder. Backups of every edited VPS2 file.

## Predictions

My recorded bias is to over-predict benefits two to four times and under-predict costs.

1. **Builder.** The first build embeds the views of every chunk (about 76,000 views, a one-time
   Voyage cost under USD 0.50); an immediate second build embeds 0 chunks and writes a
   byte-identical matrix.
2. **Questions (M2).** 180 to 225 kept.
3. **Headroom (M2).** `off` exact@6 on dev between 50% and 75%.
4. **`fused` (M2 dev).** Net exact@6 +2 to +6 against `off`, at most 2 losses, top five unchanged
   on every query, no errors.
5. **`dense` (M2 dev).** Net exact@6 +1 to +6, with at least one query whose top five changes.
6. **Latency (M2).** Atomic stage p95 between 20 and 45 ms, p99 below 90 ms.
7. **Trust (M2).** Trust state differs from `off` on at most 5% of queries in either arm.
8. **M1.** `micro_fused` net exact@8 +1 to +4 against `off`, 0 rank-1 losses on probes and tasks;
   `micro` (dense) keeps at least one rank-1 loss on the task sentinel.

## Decision rules

- **Placement (M2 dev).** `fused` is eligible with net exact@6 at least +2, at most 2 losses, top
  five unchanged on every query and no errors. `dense` is eligible with net exact@6 at least +2, at
  most 2 losses, at most 1 exact@1 loss and no errors. Choose `fused` if eligible, else `dense` if
  eligible, else no rollout.
- **Confirm (M2), once, chosen placement only against `off`.** Passes with net exact@6 at least
  +1, at most 2 losses, no errors (and top five unchanged, for `fused`).
- **Latency.** Atomic stage p95 at most 40 ms and p99 at most 80 ms over dev and confirm together.
  A breach is reported to the operator before any rollout.
- **Rollout (M3)** only after M2 passes. Live gates: a real MCP handshake lists the tools; 20 live
  `recall_search` calls through the memory MCP server return no error and carry the
  `atomic_rescue` stage; a rollback rehearsal (mode `off`, restart, stage absent; mode `active`,
  restart, stage present) passes. Any failure restores `mode=off` and the backed-up files.
- M1 decides only C8's recommended placement; it does not gate the memory rollout.

## Result

Not yet run.
