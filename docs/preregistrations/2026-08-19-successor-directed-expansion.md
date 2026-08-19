# Pre registration: successor directed retrieval expansion

**Date:** 2026-08-19   **Status:** predicted, not yet measured

## The question

When a retrieved memory is superseded and its declared successor is **not** in the candidate pool,
does fetching that successor by name turn the resulting abstention into a correct successor answer,
without raising the superseded trust rate above 0.00?

Answerable by a rate over a stated denominator, not by judgement.

## Why this is worth measuring

`resolve_successor` returns a successor's **filename** (`recall/trust.py:148`) and nothing more. The
file is then promoted to verdict `ok` only if it appears in `promoted_files`, which is built purely
from chunks already present in `result.hits` (`recall/trust.py:561`). Nothing fetches it. So validity is enforced as a **post filter** over a
pool selected without regard to validity: the superseded chunk is demoted, no hit earns `ok`, and
`trusted_search` abstains with a reason that names a successor sitting in the index, unretrieved.

This is already recorded as a known limit in `results/FINDINGS.md:185`, in exactly these words:
"the redirect requires the successor to be *retrieved* (it is not re-queried)". This record is the
attempt to close it, and to bound what closing it is worth.

## Baseline and treatment

**Baseline:** current `trusted_search`. Candidate pool is `DEFAULT_CANDIDATE_K = 20` per leg
(`recall/retriever.py:40`), fused, optionally reranked, truncated to `k`, then verdicted.

**Treatment:** after the first trust evaluation, for every hit with verdict `superseded` whose
resolved `superseded_by` file contributed no chunk to the pool, rerun the existing source scoped
retrieval `search(query, chunks_per_source, source=<successor file>)`, merge unique chunks, and
re evaluate. This reuses the mechanism `expand_retrieval_by_source` already implements
(`recall/retriever.py:96`) with a different trigger: the supersession edge rather than a relational
query pattern. Bounded by a maximum number of distinct successor files per query. Off by default;
the baseline path must be unchanged when disabled.

## What I already know

| Fact | Where |
|---|---|
| Successor is the top trusted answer in 3 of 4 cases on the 22 document eval corpus | `results/FINDINGS.md:165` |
| Successor accuracy 0.14, 95% Wilson [0.09, 0.20], n=150, on the generated scale corpus | `results/FINDINGS.md:233` |
| That generated column is **disclaimed**: it measures token discrimination, not the trust layer | `results/FINDINGS.md:376` and `results/FINDINGS.md:383` |
| Superseded trust rate is 0.00 at coverage 1.00, n=250 | `results/FINDINGS.md:231` |
| Timestamp recency is not a substitute, steelmanned: 0.83 to 1.00 STR | `docs/ENTAILMENT_SUPERSESSION_STUDY.md:115` |

**Neither existing fixture can exhibit the condition.** The eval corpus is 22 documents against a
fused pool of up to 40 distinct chunks, so the successor is essentially always in the pool and the
failure never arises. The generated corpus is large enough but its successor column is already
withdrawn. A new fixture is therefore part of the apparatus, not an optimisation of it.

## The design decision that makes this measurable: stratify, do not tune

Rather than build a fixture tuned to make the successor fall outside the pool at a chosen rate,
run the baseline first and **partition the supersession queries by pool membership**:

- **Stratum A**, successor present in the baseline fused pool.
- **Stratum B**, successor absent from the baseline fused pool.

The primary metric is reported over **stratum B only**. This fixes the denominator to the queries
the change can possibly affect, and means fixture difficulty sets the stratum's *size*, which is
reported, rather than silently setting the headline *rate*.

## Prediction

Denominators stated. All rates are over supersession queries unless said otherwise.

| Metric | Denominator | Prediction |
|---|---|---|
| Successor recovery rate, treatment | stratum B | **0.70 to 0.90** |
| Successor recovery rate, baseline | stratum B | **0.00**, by construction |
| Successor accuracy, treatment | stratum A | **unchanged**, within +/- 0.05 |
| Superseded trust rate (`str_trust`) | all trust queries | **0.00**, exactly unchanged |
| Trust coverage | all trust queries | **rises** by the share of stratum B recovered |
| Abstention accuracy on unanswerable controls | expected abstain queries | **no decrease greater than 0.05** |
| p50 latency, triggering queries | queries that fire the fetch | **no more than 2x baseline** |
| p50 latency, non triggering queries | queries that do not fire | **within noise**, no more than 1.1x |

**Not 1.00 on stratum B, and the reason is specific.** Building `promoted_files` (`recall/trust.py:561`) requires
the *stale* hit to have scored at or above the calibrated threshold. A superseded hit **below**
threshold leaves its fetched successor at `low_confidence`, unpromoted, and the query still
abstains. So there is a ceiling here that is set by calibration and not by retrieval, and I expect
it to account for most of the shortfall from 1.00.

**Mechanism prediction, separately falsifiable:** the gain comes from adding a chunk that was
absent, not from reordering chunks that were already present. A gain visible on stratum A would
falsify the mechanism even if the aggregate improved.

## Invariants

- [ ] Every fetched successor chunk carries its own cosine and its own verdict; none is served
      without one.
- [ ] `str_trust` cannot rise. A fetch that causes a stale memory to be served as `ok` is a defect,
      not a trade.
- [ ] The fetch never crosses the tenant, generation, or corpus binding. The scoped search goes
      through the same store, so `tenant_id` is enforced in SQL, but this is asserted rather than
      assumed.
- [ ] Baseline output is byte stable when the feature is disabled.
- [ ] At most one scoped search per distinct successor file, and at most `max_sources` of them per
      query. A supersession cycle cannot cause unbounded fetching.
- [ ] Duplicate chunk ids are emitted at most once.
- [ ] The re evaluation does not re enter expansion. One round only.

## Apparatus verification, before any outcome is read

Predicting the outcome does not reveal a broken harness. Checked first, in this order:

1. **Stratum B is non empty**, and its size is reported with the result. An empty stratum means the
   fixture is inert and the quality result must not be interpreted at all. This is precisely what
   the 22 document eval corpus would produce, and it would otherwise read as a clean null.
2. **Baseline successor recovery on stratum B is 0.00.** It cannot be anything else: a chunk absent
   from the pool cannot be the top trusted answer. A non zero baseline means the stratification is
   reading the wrong pool, most likely the post truncation `hits` rather than the pre truncation
   fused pool.
3. **A known answer case passes both arms.** One supersession pair with the successor deliberately
   inside the pool, which both baseline and treatment must answer correctly.
4. **The fixture does not repeat the recorded corpus defect.** Documents must be real prose that
   differs in meaning, not one sentence with a swapped opaque token; that defect is what withdrew
   the successor column at `results/FINDINGS.md:376`.

## What would falsify this

- Successor recovery on stratum B below **0.40**. The mechanism would then be doing something other
  than what it claims.
- Any increase in `str_trust` above 0.00, at any n. This falsifies the change outright regardless
  of recovery, because it is the one number the trust layer exists to hold.
- Abstention accuracy on unanswerable controls falling by more than 0.05.
- A gain on stratum A comparable to the gain on stratum B: the mechanism is then not "fetch what was
  missing" and the explanation is wrong even if the number is good.
- p50 latency on triggering queries above 2x baseline.
- Stratum B empty, which falsifies nothing and measures nothing. Reported as an apparatus failure,
  never as a null result.

## Confounds I can name now

- **Fixture difficulty is mine to choose.** Stratum B's *size* is a design parameter, so the
  aggregate successor accuracy gain is not a corpus independent number. Only the stratum B rate is
  quotable outside this fixture, and even that is conditioned on this corpus's supersession shape.
- **The two arms use different retrieval paths.** The scoped fetch runs the filtered HNSW arm, which
  sets `iterative_scan=relaxed_order` and thereby trades truncation for approximation
  (`recall/store.py:288`). On a single source filter over a handful of chunks this should not bite,
  but the arms are not identical retrieval and a difference could come from there.
- **Reranking degrades as the pool widens**, which is why `FUSED_RERANK_POOL_CAP` exists
  (`recall/retriever.py:228`): a 547 candidate pool lost 0.0513 R@100 where 200 gained 0.0226. Adding successor chunks widens the pool, so with a
  reranker enabled a real recovery gain could be partly cancelled by a ranking loss. Run both with
  and without the reranker rather than picking one.
- **Calibration sets the ceiling, not retrieval.** Per the promotion rule above, a mis calibrated
  threshold caps stratum B recovery independently of whether the fetch worked. If recovery comes in
  low, separate "the successor was never fetched" from "the successor was fetched and not promoted"
  before concluding anything, and report the two counts separately.
- **Coverage is the exact complement of abstention** (`results/FINDINGS.md:241`), so the coverage row
  and the abstention row are one fact reported twice. They are not independent corroboration.
- **A recovered answer is not a correct answer.** Successor recovery counts whether the successor
  became the top trusted hit, not whether it answers the question. It is named by that denominator
  deliberately.

## Decision rule, fixed in advance

| Outcome | Action |
|---|---|
| Stratum B recovery at least 0.40, `str_trust` stays 0.00, all invariants hold, latency within 2x | Keep the feature, opt in, and open a separate default promotion decision |
| Any invariant fails, or `str_trust` rises at all | Reject the implementation regardless of recovery |
| Stratum B recovery below 0.40 with the fetch confirmed to have run | Record the null, keep the code out of the default path, and re examine the promotion rule rather than the fetch |
| Stratum B empty, or apparatus check 1 to 4 fails | Do not interpret the quality result. Fix the fixture and re run under this same record |

## How it will be measured

A probe modelled on `benchmarks/document_bundle_probe.py`, reporting both arms over one fixture in
one process, against a session owned database:

```bash
eval "$(scripts/session-db.sh up)"
python benchmarks/successor_expansion_probe.py
```

Metrics by their existing names where they exist: `str_trust` (`superseded_trust_rate`),
`successor_acc` (`successor_accuracy`), `trust_coverage`, `abstain_acc`, all in
`recall/eval/metrics.py`. "Successor recovery rate over stratum B" is new and must be defined in the
probe, not derived by hand afterwards. Every rate is reported with its n and a 95% Wilson interval,
matching `results/FINDINGS.md`.

Written against `origin/master` at `5c621cab`, in a worktree created off it, so every `path:line`
citation above was verified against the tree the measurement will run on. Both citation gates were
run before this record was committed.

## Implementation note (2026-08-19, written BEFORE any measurement)

The prediction above is unchanged and stays unchanged. This records one deviation from the
**treatment as described**, written down now rather than alongside the number, so that it cannot
read as a story fitted to a result.

**The record says "after the first trust evaluation ... and re evaluate". The implementation does
it in one pass, before `evaluate`.** The trigger needs only the supersession map, and
`trusted_search` already reads that map before it verdicts anything, so the expansion sits between
those two steps. A second evaluation was never necessary.

Consequences, all in the safe direction:

- The invariant "the re evaluation does not re enter expansion, one round only" is now trivially
  true rather than enforced by a flag: there is no second evaluation to re enter.
- Every hit, fetched or original, is verdicted exactly once by the same `evaluate` call. That is a
  stronger form of the invariant that each fetched chunk carries its own verdict.
- `resolve` is injected into the expander rather than imported, because `recall.trust` already
  imports `recall.retriever` and the reverse would be a cycle. The closure `trusted_search` passes
  mirrors `_verdict`: a file in `unresolved` resolves to nothing, so an ambiguous edge fetches
  nothing.
- It deliberately does **not** mirror `not_yet_known`, which outranks `superseded` per hit. A hit
  replayed before its own write time can therefore cost one scoped search whose chunks cannot
  change any verdict. Wasted work, never a wrong answer, and the alternative needs the hit dates at
  a point that does not otherwise have them.

Shipped off by default, as `successor_expansion=` on `trusted_search`. Invariant tests are in
`tests/test_successor_expansion.py`: 14 of them, covering the falsifiers that need no corpus. The
superseded hit is never promoted, an already present successor is never refetched, the disabled
path returns the identical object, the fetch is bounded, and a cycle terminates. They pin the
mechanism only, and say nothing about the rate, which is what the record above is for.

## Result

Not yet measured.
