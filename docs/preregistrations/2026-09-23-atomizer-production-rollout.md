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

## Apparatus appendix, appended 2026-09-23 before any evaluation

Apparatus only; no arm had been evaluated when it was written.

- **M2 generation pinned:** `gen_b6aefc110e0d42588f3a57c98aa29129` (the active memory generation at
  the start of the run).
- **M2 questions:** spans file SHA-256 `78520bfa288551b9…` (240 spans; 305 sources excluded: 171
  former gold, 134 used by the memory check). Question file SHA-256 `49e00f9363e8508b…`: **206
  kept** (30 copied the span, 3 not answerable), 103 dev and 103 confirm; 136 `sentiment-agent`, 56
  `recall`, 14 `agent-memory-bench` sources. Llama 3.3 70B, USD 0.017118. Prediction 2 (180 to 225)
  is already visible and **confirmed**.
- **Builder, first run** (production builder, bounded, `embed.lock` held, view store
  `~/.codex/atomic-view-store/memory.sqlite`): 11,735 chunks, 11,700 parents, **76,572 views**,
  11,696 distinct chunk texts embedded (76,557 views), 313.6 MB matrix, 285.3 s.
- **Builder, second run** into a separate registry: **11,700 chunks reused, 0 embedded**, 21.0 s;
  matrix SHA-256 `d8d9d1de774953cc…` identical to the first, `views.json` byte-identical.
  Prediction 1 is already visible and **confirmed**.

## Result: M2 dev, appended 2026-09-23 after the run

The "Not yet run." line under **Result** is left as written; this section is the result. Report
`~/atomizer-prod/report-dev.json` and rows `~/atomizer-prod/rows-dev.jsonl` on VPS2 (private),
103 dev questions, generation `gen_b6aefc110e0d42588f3a57c98aa29129` unchanged across the run, 0
errors in every arm. The harness ran under `systemd-run --scope -p CPUQuota=150% nice -n 15`.

| arm | exact@1 | exact@5 | exact@6 | exact@10 | abstained |
|---|---:|---:|---:|---:|---:|
| off | 57 | 79 | 83 | 91 | 30 |
| dense | 60 | 85 | 89 | 92 | 22 |
| fused | 53 | 76 | 83 | 94 | 22 |

Paired against `off` (gains/losses): `dense` exact@1 +7/−4, exact@6 +6/−0, top five changed on 49
queries; `fused` exact@1 +0/−4, exact@6 +1/−1, top five changed on 28. Trust state changed on 0
queries in either arm. Atomic stage: `dense` p50 78.2, p95 212.4, p99 334.9 ms; `fused` p50 59.7,
p95 176.9, p99 287.4 ms.

**Scoring.**

- Prediction 3 (off exact@6 50% to 75%): **falsified**, 83 of 103 (80.6%). Less headroom than
  predicted.
- Prediction 4 (`fused` net +2 to +6, top five unchanged): **falsified** on both counts. Net 0, top
  five changed on 28 queries. No errors held.
- Prediction 5 (`dense` net +1 to +6, some top five change): **confirmed**, net +6.
- Prediction 6 (atomic p95 20 to 45 ms, p99 below 90): **falsified**, by four to five times.
- Prediction 7 (trust state differs on at most 5%): **confirmed** as worded, 0 of 103. Abstention,
  which the prediction did not name, flipped from abstain to answer on 8 queries in each arm and
  never the other way.

**Decisions under the frozen rules.** `fused` is not eligible (net 0, top five changed). `dense` is
not eligible (4 exact@1 losses against a limit of 1). **No rollout.** The confirm split was **not**
run and stays unread. The latency breach is reported to the operator.

### Diagnosis, appended the same day, post hoc and labelled as such

Neither mechanism below was predicted; both were found by reading the rows and the code after the
result, so they are explanations, not tests.

**1. The rescued hit carries the micro-view cosine into the trust layer.** `scored_loader` is
called with `selection.score`, the cosine of a 24-word view, and `evaluate` returns `ok + rest`: a
hit whose score clears the certified threshold is placed ahead of every hit that does not. The
threshold was certified on whole-chunk cosines, and a short view that matches the query scores
higher than its chunk. So the rescued parent was promoted above top-five hits that sat below the
threshold, and on queries where `off` abstained it became the one `ok` hit and turned the
abstention into an answer.

- `fused`: all 28 top-five changes are a new id entering the top five (at positions 1 to 5: 8, 5,
  5, 3, 7). All 8 abstention flips are among them, and all 4 exact@1 losses are queries where `off`
  abstained and `fused` answered with the rescue in first place.
- `dense`: 7 of its 8 abstention flips are among its 49 top-five changes, and 3 of its 4 exact@1
  losses are abstain-to-answer flips.

This violates the design's own premise that the rescue changes rank only, not the trust decision.
It is a defect in the rescue, not in the trust layer.

**2. The latency is CFS throttling of OpenBLAS under the harness's CPU quota.** Measured on VPS2
the same day, `matrix @ query` on the 76,572 × 1024 float32 matrix, 40 samples each, host load
about 4 to 5 on 12 cores during the official AML run:

| setting | p50 ms | p95 ms |
|---|---:|---:|
| quota 150%, 12 BLAS threads (the harness) | 119.3 | 201.8 |
| quota 150%, 4 threads | 17.3 | 109.5 |
| quota 150%, 2 threads | 18.4 | 54.9 |
| quota 150%, 1 thread | 27.8 | 46.8 |
| no quota, 12 threads | 18.8 | 45.9 |
| no quota, 2 threads | 16.6 | 38.6 |

Twelve threads under a 150% quota spend the period's CPU budget in a few milliseconds and then
wait for the next period. Memory mapping is ruled out: the same selection from an in-memory copy
measured p50 100.2 against 100.9 mapped. The quota was a property of the measurement, but the
effect is not only an artefact: any production host that bounds the server's CPU would see it, and
even unbounded the p95 sits at the edge of the 40 ms budget on a loaded host.

**What follows, as a proposal and not a result.** (a) The rescued parent must carry its own chunk
cosine against the query, never the view's. (b) The selection must bound its own BLAS threads.
Both are code changes to the measured system, so any re-measurement is a **new** preregistration on
dev, and the untouched confirm split then decides. The numbers above stay as they are.

## M1 apparatus appendix, appended 2026-09-23 before any M1 question was written

Apparatus only. M1's question, prediction 8 and its decision rule are frozen above and unchanged.

- **Code:** `scripts/c8_atomizer_round3.py` on `claude/atomizer-measure` at the commit that adds
  this appendix. New: arm `micro_fused` (`fused_replay`: fuse the unmodified dense and lexical
  rankings exactly as the `off` arm does, then apply the production `insert_atomic_rescue_fused`,
  which places the dense-rank-selected winner at final rank six and leaves the fused top five
  unchanged); `fresh_spans(..., dev_only=False)`; `probes --m1` (seed 20260926, every session,
  split label `m1`, ids `m1-`); `--used-probes` accepts several files. Round 3's defaults are
  unchanged. Two new tests with recorded red proofs in `tests/test_c8_atomizer_round3.py`.
- **Questions:** `probes --m1 --used-probes private/probes.jsonl private/probes-dev3.jsonl`
  (round 1 and round 3 probe files, so every M1 span is disjoint from both), writer
  `meta-llama/llama-3.3-70b-instruct` with the round 3 prompt, schema and rejection rules, cap USD
  0.50 in code. The probe file's SHA-256 and counts are appended before evaluation.
- **Evaluation:** `evaluate --split m1 --arms off micro micro_fused`, the frozen CAMBench corpus
  and the existing Code4 vector cache on VPS3, plus the 34 task prompts, which every split carries.
- **Where:** VPS3 only, `/home/sentiment/atomizer-c8`, bounded and niced as in round 3.
