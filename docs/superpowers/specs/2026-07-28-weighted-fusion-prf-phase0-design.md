# Weighted fusion + leg-triggered PRF — Phase 0 diagnostic design

**Date:** 2026-07-28 · **Status:** design written, not yet implemented
**Scope:** Phase 0 **only** — a diagnostic that decides whether Phases 1 and 2 get built at all.
No fusion change and no second retrieval pass are in scope here.

## Why

Two levers were proposed together, each aimed at a fact this repo has already measured:

- **Weighted fusion.** §9a measured that a 5× deeper candidate pool *lowers* hit@5 — **0.671 →
  0.596** — and diagnosed the mechanism: `_rrf` scores `dense[r]` and `sparse[r]` identically, so a
  deeper pool interleaves five times as many low-rank candidates into every prefix.
- **A second retrieval pass (PRF).** §7 found the bottleneck is candidate recall, not ranking: the
  cross-encoder converted 3 of 31 misses, so for ~28 the right document was never in the window.
  A reranker cannot fix that; a second *retrieval* can.

The proposed trigger — fire the second pass only when the lexical leg was more decisive than the
dense leg — rests on an assumption that has never been measured: **that leg disagreement selects
for retrieval failures.** If it selects for successes instead (sparse is decisive precisely because
it nailed a rare-term query), the whole design inverts.

Phase 0 measures that assumption, and two others, before any feature code exists. It is designed to
be able to **kill the idea cheaply**. Everything it needs is local: `bge-small`, one CPU pass, $0.

## The blocker that has to be fixed first

`conf(L)` (defined below) needs each leg's **own** ranking scores. The dense leg returns cosines.
The sparse leg does not return `ts_rank`.

`PgVectorStore.query_sparse` computes `ts_rank(c.tsv, q.tsq) AS rank` in a subquery, orders by it,
and then the **outer SELECT overwrites `score` with the dense cosine** whenever `vec` is passed —
which `HybridRetriever.search` always does. The lexical score is computed and discarded before it
reaches any caller.

That is deliberate and correct for its current purpose (it lets a lexical-only hit report a cosine
comparable with dense hits downstream, and the trust layer's thresholds depend on `score` being a
cosine). It is also fatal to this measurement: computing `conf(sparse)` from what is returned today
would compare *dense-cosine decisiveness over the sparse-retrieved set* against *dense-cosine
decisiveness over the dense-retrieved set*. That is not leg disagreement. It would run without
error and produce a plausible firing rate for a trigger that measures nothing.

**Prerequisite change:** surface `ts_rank` alongside the cosine, without changing `score`. Add an
opt-in `with_rank: bool = False` parameter to `query_sparse` that, when set, returns
`tuple[list[ScoredChunk], list[float]]` — the hits unchanged plus their `ts_rank` values in the
same order. `ScoredChunk` is a frozen dataclass shared by every retrieval path and the whole trust
layer, so a new field there would ripple through code this measurement has no business touching; a
parallel return keeps `score`'s cosine contract exactly as it is and leaves the default call site
byte-identical. This is the only production code change in Phase 0, and it is additive.

## Definitions, fixed before measurement

For leg `L` with its own native candidate scores `s₁ ≥ s₂ ≥ … ≥ s_n` (n = `candidate_k` = 20;
cosine for dense, `ts_rank` for sparse):

```
conf(L) = (s₁ − μ_L) / σ_L        # z-score of the leg's top hit within its own candidates
conf(L) = 0                        when σ_L = 0 or the leg returned nothing
```

**Affine-invariant by construction** — scaling or shifting a leg's scores leaves `conf` unchanged.
That is the property that makes cosine and `ts_rank` comparable without normalizing incompatible
scales, and it is asserted as a property test, not assumed.

```
m = min(len(sparse_scores), len(dense_scores))          # common candidate depth
trigger(query) ≡ conf(sparse[:m]) > conf(dense[:m])     # both legs scored at depth m
trigger(query) ≡ False                                   when m < 2
```

Zero constants: a comparison between two scale-free quantities, not a threshold. This is
deliberate — §2 established that a fitted cosine threshold does not transfer across embedders, and
a trigger built on one would inherit that failure.

> **Amendment, 2026-07-28, before any run.** The definition first committed here was
> `conf(sparse) > conf(dense)` at each leg's natural depth. That is biased, and the bias was
> caught by the review loop while implementing, with **no data in existence** — which is the only
> point at which a preregistered definition may be changed.
>
> `conf` is the z-score of a sample *maximum*, and the expected maximum grows with sample size.
> Measured on pure iid Gaussian noise, 4,000 trials, no signal at all: **E[conf] = 1.389 (n=5),
> 1.669 (n=10), 1.945 (n=20)**. The dense leg always returns exactly `candidate_k` candidates — it
> is a top-k vector scan — while the sparse leg returns only chunks whose `tsv` matched the
> tsquery, often far fewer. At natural depths the comparison would therefore have measured **how
> many chunks matched the query text**, not which leg was more decisive: a query with 5 sparse hits
> starts ~0.55 z-units behind before any evidence is weighed. Q1 and Q2 — the two answers this
> phase exists to produce — would both have been confounded by tsquery match count.
>
> Scoring both legs at the common depth `m` removes it by construction, at the cost of ignoring
> the deeper leg's tail. Nothing else changed: not the predictions, not the decision rules, not the
> kill gates. Implemented as `recall.eval.legconf.more_decisive`, with
> `test_more_decisive_is_not_fooled_by_leg_length` as the regression guard.
>
> **The fix is partial, and the residual is measured, not hoped away.** Simulated on iid noise
> (200k trials/point): an equal-length 5-vs-5 comparison fires **50.0%** of the time, but a
> 5-candidate sparse leg against a 20-candidate dense leg fires **35.1%**, and against a
> 40-candidate dense leg **33.6%**. Truncating a *larger* pool to its top m yields order statistics
> clustered more tightly near the maximum than a fresh m-sized draw, so matching the sample sizes
> does not fully match the distributions. (For scale: the unfixed definition fired 11.9% on the
> same comparison.) The trigger therefore still carries some correlation with tsquery match count,
> and match count plausibly correlates with question difficulty.
>
> **Consequence for Q1, binding:** the diagnostic records `n_dense` and `n_sparse` per question and
> reports Q1 **stratified by sparse-leg depth** alongside the pooled figure. A firing/not-firing
> gap that exists in the pooled number but vanishes inside every depth bin is the confound
> talking, and **does not clear the Q1 gate**. The pooled Q1 alone does not settle it.

## What Phase 0 measures

Instrument `HybridRetriever.search` to dump, per query: each leg's native candidate scores and ids,
`conf` per leg, the fused ranking, and whether the gold chunk appeared in each leg's pool. Run on
**LOCOMO** (n=1,536 answerable, 10 conversations, pool 20, bge-small, no rerank — the §9a
configuration exactly) and on the **private 46** (`recall.eval.labelled`).

**Sequencing:** the LOCOMO arm runs first and is the arm that decides, because it is public,
n=1,536, and has a published baseline to check the apparatus against. The private-46 arm needs a
second probe threading and the corpus owner's data; it follows once LOCOMO has cleared its gates,
and its role is replication in the opposite regime (§7's low-cosine, jargon-heavy corpus), not
adjudication.

### Q1 — does leg disagreement select for failures?

Split queries on `trigger`, compare hit@5 in each group. Wilson CIs per group; two-proportion test
across them.

### Q2 — what is the firing rate?

Rate plus CI, overall and per LOCOMO category.

### Q3 — where is the gold chunk on firing queries that missed at k=5?

Three disjoint buckets, which map one-to-one onto the two proposed phases:

| bucket | meaning | whose job |
|---|---|---|
| **(a)** in the fused candidate union, ranked below 5 | retrieved but mis-ranked | **Phase 1** (weighted fusion) |
| **(b)** in neither leg's candidate pool | never retrieved at all | **Phase 2** (PRF — a new query vector is the only thing that can reach it) |
| **(c)** no gold chunk in the index | labelling defect | neither — reported, excluded |

Bucket (a) is sub-split by which leg held it. **(b) is the ceiling on what PRF can possibly buy.**

## The apparatus check — asserted in code, not eyeballed

The instrumented run **must reproduce hit@5 = 0.671 and hit@20 = 0.855** on pool 20, matching
`results/locomo/postfix_pool20.json`, and must score **exactly 1,536 answerable questions**.
Asserted in the harness; a mismatch fails the run.

The two rate asserts carry a **±0.01 tolerance**, the answerable count **none**. HNSW index builds
are nondeterministic (§5b, §6), so demanding equality on a rate would fail an honest rerun — while
±0.01 is far too tight to absorb a structural defect, since a doubled corpus moves a headline rate
by very much more than that. The count is the doubled-corpus check and is exact.

This is not ceremony. A corrupted apparatus does not raise — it returns plausible numbers and a
manufactured finding, which is exactly how a doubled corpus (11,764 vs 5,882 rows) produced a
believable result on 2026-07-27. **Exit code 0 is not a measurement.** Instrumentation that changes
the retrieved set has broken the thing it was measuring, and this assert is what catches it.

Two further invariants, as tests:

- `conf()` is affine-invariant: scaling/shifting a leg's scores does not move the weights.
- `trigger` cannot fire when `use_sparse=False` or the sparse leg returns nothing (`conf(sparse)=0`
  and the comparison is strict), so ablation arms degrade to today's behaviour rather than
  silently entering a different code path.

## Preregistered before running

Written and committed **before** the dump is generated, per `docs/RESEARCH_PROTOCOL.md`. Each
carries its reasoning, so a hit can be scored as *right for the right reason*.

- **Q2 firing rate: LOCOMO 15–35%, private 40–60%.** LOCOMO is ordinary prose and §8 established
  the local embedder is not the bottleneck there, so dense should usually be the decisive leg. The
  private corpus is the opposite regime — §7 measured sparse (0.348) *above* dense (0.326).
- **Q1: firing-group hit@5 lower by ≥0.10 absolute** (e.g. 0.60 against 0.70, not a 10% relative
  change). If the legs disagree about what is relevant, at least one of them is wrong, and the
  fused prefix is being built from conflicting evidence.
- **Q1 falsification path, and it is live.** Sparse may be decisive precisely *because* it matched a
  rare term — a name, a number, a codename — in which case the firing group has *higher* hit@5, the
  trigger selects for successes, and the design is dead. This is the single load-bearing assumption
  of the whole proposal and the cheapest thing in it to check. It is stated here so that outcome
  cannot be reinterpreted as a partial win afterwards.
- **Q3: LOCOMO splits roughly 30–50% into bucket (a).** The private corpus skews hard into (b) —
  §7 found ~28 of 31 misses never entered the window.
- **A null result ships.** Phase 0's product is a decision, not a gain. "The trigger does not select
  for failures" is a publishable finding and closes a lane, which is worth more than an unmeasured
  feature.

## Decision rules, fixed in advance

| gate | rule | consequence |
|---|---|---|
| **Q1** | firing-group hit@5 ≥ non-firing group | **Stop.** No PRF. The trigger premise is false. |
| **Q2** | firing rate outside 5–50% | Redesign the trigger. Near 0% ⇒ unmeasurable on public data, the exact failure this trigger was chosen to avoid. Near 100% ⇒ not a trigger, and the latency argument dies with it. |
| **Q3** | bucket (b) share ≈ 0 | **No Phase 2.** PRF has nothing to fetch; the work reduces to weighted fusion alone. |
| **Q3** | bucket (a) share ≈ 0 | **No Phase 1 gain available on this corpus** — report it rather than shipping a fusion change with no measurable effect. |

Any gate that fires is written up and the lane closes or narrows. A gate that cannot go green
tells you only about itself.

## Out of scope

- **The fusion change itself (Phase 1)** and **the second pass (Phase 2).** Phase 0 adds no
  weighting and issues no extra query. One variable at a time.
- **The privileged-vs-unprivileged expansion-leg question.** It cannot be decided by argument and
  needs PRF to exist first; it is a Phase 2 A/B on one index pass, not a Phase 0 question.
- **Any embedder, chunking, pool-depth or rerank change.** §9a's pool-100 control already exists
  and mixing it in would confound fusion weighting with pool depth.
- **LongMemEval.** Its configuration is pre-fix (§10's configuration note) and is not comparable.

## What gets published

- The three answers with CIs, per corpus and per LOCOMO category.
- The firing rate, which is the number that decides whether the latency story in the pitch survives.
- **The predictions above, scored** — including the ones that were wrong.
- The retained per-query dump as the artifact behind every figure. Per
  `reference-recall-docs-evidence-tier-convention`, no figure ships without a retained artifact or
  a reproduce command.
- If a kill gate fires: a closure note, and no Phase 1 or 2.
