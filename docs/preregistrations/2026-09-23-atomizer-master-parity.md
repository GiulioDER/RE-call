# Micro atomizer on the memory tenant: master parity before moving the serving checkout

Status: frozen when committed. Nothing below the "Result" heading may be written before the
measurement, and nothing above it may be edited afterwards; corrections are appended.

## Question

The memory tenant serves the micro atomizer from `a42f035c`, the commit round 3 measured
(`2026-09-23-atomizer-production-rollout-r3-latency.md`). Master is now `986b70c9`, 14 commits
later, including #711 and #713, which restructured the trusted-search path and split the rescue
into `select_atomic_rescue` plus `place_atomic_selection_dense` and `place_atomic_selection_fused`.
None of those changes was meant to move a rank or a score. Does master serve the memory tenant
the same results as `a42f035c`, within the operator's 80 ms p95 budget?

## Frozen apparatus

- **Code:** `origin/master` at `986b70c9`, in a detached worktree of `serving-master` on VPS2. No
  migration and no dependency differs from `a42f035c` (checked with `git diff a42f035c 986b70c9 --
  recall/migrations pyproject.toml`, both empty).
- **Unchanged from round 3:** generation `gen_b6aefc110e0d42588f3a57c98aa29129` (r204, now retired
  but not collected; 11,735 chunks); artifact `~/atomizer-prod/registry`; questions file
  `49e00f9363e8508b…`, all 206 questions, dev then confirm; harness
  `scripts/memory_atomizer_production_check.py` as merged to master in #726 (its helpers are the
  verbatim copies in `scripts/atomizer_study_common.py`); arms `off`, `dense`, `fused`, `k=10`;
  scope `MemoryMax=4G`, `MemorySwapMax=0`, no CPU quota, nice 0. Output files carry an `-r4`
  suffix.
- **Baseline:** round 3's rows, `rows-{dev,confirm}-r3.jsonl`.
- **Parity** is computed per question for the `dense` arm (the placement in production): exact
  rank, top five and abstention, round 4 against round 3.

## Predictions

My recorded bias is to over-predict benefits and under-predict costs.

1. **Parity:** `dense` exact rank matches round 3 on at least 200 of 206, top five on at least
   195, abstention on at least 203. Round 3 matched round 2 on 206 of 206, so I expect the same or
   within one or two questions of it; any mismatch comes from query-time nondeterminism, not code.
2. **Latency:** `dense` atomic stage p95 between 30 and 90 ms, p99 below 160, pooled over 206.
   Host load decides most of this, and it is not controlled.
3. **Errors:** 0 in every arm.

## Decision rules

- **Parity passes** with exact rank matching on at least 200 of 206, top five on at least 195 and
  abstention on at least 203. Every mismatch is listed; a mismatch caused by the code change fails
  the round.
- **Latency passes** with p95 at most 80 ms (the operator's budget) and p99 at most 160 ms (twice
  p95, my derivation, as in round 3's appendix).
- **If both pass,** moving the serving checkout to master is eligible, and is reported to the
  operator before it is done. **If parity fails,** serving stays on `a42f035c`.

## Result

Not yet run.

## Result, appended 2026-09-23 after the run

The "Not yet run." line above is left as written. Reports `~/atomizer-prod/report-{dev,confirm}-r4.json`
and rows `~/atomizer-prod/rows-{dev,confirm}-r4.jsonl` (VPS2, private); 206 questions; 0 errors in
every arm. VPS2 load average about 3.5 to 4.3. (A first launch failed before evaluating anything:
`systemd-run --user` could not reach the user bus because the ssh session had no
`XDG_RUNTIME_DIR`; the relaunch set it.) The harness flags `active_generation_unchanged=False`
because the hourly refresh promoted a new production generation during the run; the run is pinned
to r204, so that does not touch what was measured.

| arm | atomic p50 | atomic p95 | atomic p99 | max |
|---|---:|---:|---:|---:|
| `dense` | 21.6 | **39.0** | 96.0 | 171.8 |
| `fused` | 20.9 | 55.1 | 102.1 | 215.8 |

**Parity, `dense` against round 3:** exact rank **205 of 206**, top five **201 of 206**,
abstention **206 of 206**. Paired quality unchanged: exact@6 +4/−0 on dev and on confirm. The `off`
arm, which runs no rescue at all, itself changed its top five on 5 questions and its exact rank on
1, so part of the drift is present with no atomizer code involved.

**Every `dense` mismatch explained, measured after the run and labelled as such.** For the five
mismatched questions and one control, each question's query vector was computed once and frozen,
then `off` and `dense` were replayed through `search_memory` on r204 under `a42f035c` and under
`986b70c9` (each run importing `recall` from its own checkout, checked from `recall.__file__`).
**All 12 replays returned identical top ten lists on both commits.** So none of the mismatches
comes from the code change; they come from Voyage query vectors differing between calls, visible
directly in `mem-0ea3c8…`, whose exact rank was 5 in round 3, 6 in round 4 and 4 on the frozen
vector.

**Scoring.**

1. Parity (at least 200, 195, 203): **confirmed** (205, 201, 206).
2. Latency (p95 30 to 90, p99 below 160): **confirmed** (39.0, 96.0).
3. Errors 0: **confirmed**.

**Decision under the frozen rules.** Parity passes, with no mismatch caused by the code; latency
passes (p95 39.0 against 80, p99 96.0 against 160). **Moving the serving checkout to master is
eligible**, and is reported to the operator before it is done.

## Serving moved to master, appended 2026-09-23 after the operator approved it

- Waited for `embed.lock`: the hourly project refresh (a `re-call-code-gen` generation build) held
  it, and swapping modules under it could break it partway. Repointed only once it was free,
  re-checking the lock in the same command.
- `serving` now points to `~/recall-repos/atomic-micro-prod-986b70c9`, a clean detached worktree of
  `serving-master` at `986b70c9`, the commit this record measured. Previous target, kept for
  rollback: `atomic-micro-prod-a42f035c` (backed up with `.env` under suffix
  `before-master-986b70c9-<timestamp>`). Schema 0025 compatible; `session-serving.sh verify`:
  handshake 22 tools.
- **Live gate, two samples of 20 `recall_search` calls through fresh memory MCP servers:** 0
  errors, 40 of 40 carrying the `atomic_rescue` stage, all `trusted`. Stage times: sample 1 p50
  23.0 ms with two adjacent slow calls (87.2 and 164.7 ms) in an otherwise 15.7 to 29.9 ms run;
  sample 2 p50 23.4 ms, max 47.6 ms. Host load about 4.5. The budget is judged on the 206-question
  run above (p95 39.0 ms); these samples gate errors and stage presence. **Passes.**
- **Rollback rehearsal on the real `.env`:** mode off, fresh server, stage 0 of 3; mode active,
  fresh server, stage 3 of 3; all `trusted`, 0 errors; `.env` restored with no other key changed.
  **Passes.** Active generation r206 has its micro artifact ready.

## Serving moved again, to master `4cf7aa50`, appended 2026-09-24 at the operator's request

- Master had moved five commits past `986b70c9`: #694 (evidence proof obligations, off by
  default), #728 (planner statistics after a generation build), #731 (indexing performance) and two
  docs commits. No migration and no dependency changed. None touched `recall/atomic_rescue.py`,
  `recall/trust.py`, `recall/retriever.py`, `recall/generation_store.py` or
  `recall_mcp/service.py`; the only change on the search path's modules, in `recall/embeddings.py`,
  is inside `FastEmbedEmbedder`'s passage path, which the memory tenant (hosted Voyage Context 4)
  does not use. So no new parity run was needed for search; the live gates below were run instead.
  #728 and #731 do change the generation build the hourly refresh drives, which the next refresh
  that finds a changed corpus exercises.
- `embed.lock` free; `serving -> ~/recall-repos/atomic-micro-prod-4cf7aa50` (clean detached worktree
  of `serving-master`). Previous target, kept for rollback: `atomic-micro-prod-986b70c9` (backed up
  with `.env` under suffix `before-master-4cf7aa50-<timestamp>`). Schema 0025 compatible;
  `session-serving.sh verify`: handshake 22 tools.
- **Live gate, 20 `recall_search` calls through a fresh memory MCP server:** 0 errors, 20 of 20
  carrying the `atomic_rescue` stage, all `trusted`, 1 abstained. Stage p50 26.7 ms, max 57.8 ms,
  load about 3.8. **Passes.**
- **Rollback rehearsal on the real `.env`:** off 0 of 3, active 3 of 3, all `trusted`, 0 errors;
  `.env` restored with no other key changed. Active generation r208 has its micro artifact ready.
  **Passes.**
