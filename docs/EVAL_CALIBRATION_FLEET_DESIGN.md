# Eval calibration fleet: certifying that the harness reports the number it claims

> Design, 2026-08-15. Status: implemented, shipped on branch `claude/eval-calibration-fleet`.
> Scope: `tests/fleet/`, covering `recall/eval/harness._score_config` and
> `recall/eval/promotion/aggregate.py`. Test only, nothing ships in the wheel.
>
> **Extended 2026-08-15.** `run_trust_eval` and `run_nearmiss_eval` took an optional injected
> `store_factory` (`recall/eval/harness.py`, test seam only, default path unchanged for every
> existing caller), and the fleet grew Surface C and Surface D to reach them as far as a scripted
> store can honestly drive their published rates. See those two sections below and the updated
> "Declared non-coverage".

## The problem this exists to solve

On 2026-08-09 I published MTRAG numbers that compared our **raw** metric against IBM's
**conditioned** one. The cause is recorded in `results/mtrag_generation/fix_idk_conditioning.py`:
the official scorer reads the answerability label as lower case `answerability`, the release files
spell it `Answerability`, so the key was never found, every label branch was skipped, and the
function fell through to a branch that **penalises** an abstention instead of rewarding it.

Two properties of that failure decide this design.

1. **Every individual piece was correct.** `idk_eval`, `underspecified_eval` and the raw metrics
   were all computed properly. Only the final combination step was broken. Unit tests on the
   pieces could not have caught it, and `recall/eval/metrics.py` is already unit tested function
   by function.
2. **The tell was an invariant violation, not a wrong-looking number.** The conditioned values came
   out *below* the raw ones, when a correct abstention can only raise them. Nobody had written
   that invariant down, so nothing compared against it.

`recall/eval/arm_check.py` is the closest existing guard and it does not close this. It asserts a
**direction** (`DIFFERS` / `SET_IDENTICAL` / `IDENTICAL`) and never a **magnitude**. On 08-09 the
numbers did differ. They differed by the wrong amount.

The fleet is the upgrade from direction to magnitude: systems whose behaviour is known by
construction, driven through the real scoring path, asserted against closed-form values derived by
hand.

## What was considered and rejected

**Seeded PRF injection, as in `egnaro9/reference-fleet`.** That project makes defective responders
deterministic with a SHA256 PRF over `(member_id, seed, request_index)`, because its members are
stochastic LLM wrappers and the PRF is what buys exactness. Our members are scripted stores, which
are already deterministic by construction, so the rate machinery would add a layer without adding
any certainty. Rejected as copying the mechanism while missing the reason for it.

**Plain pytest functions, one per member.** Idiomatic for this repo, which has roughly 150 focused
test files and no test framework of its own. Rejected because coverage becomes invisible: there is
no single place answering "which failure classes does this certify, and what does each one miss",
and a legible answer key is the entire point.

**End to end against real Postgres**, building a corpus with `recall/eval/synthetic.py`. Highest
fidelity and it would cover indexing, but it needs a live database and cannot be a normal CI test.

**A public board, as `reference-fleet` publishes.** Out of scope by decision: this is an internal
regression guard, so it fails CI and produces no artifact.

## Architecture

One test-only package, `tests/fleet/`, in three files.

### `members.py`, the answer key

A frozen dataclass per member:

| field | meaning |
|---|---|
| `name` | stable identifier, used in the parametrised test id |
| `defect` | what this member embodies, in one sentence |
| `build` | callable producing the scripted inputs for its surface |
| `expected` | the closed-form result, derived by hand, with the derivation in a comment |
| `does_not_catch` | **required** non-empty string naming this member's blind spot |

`does_not_catch` is required rather than optional. The same discipline is already used by
mem-bench's verifier checks and by the money-path mutation registry's declared blind spots, and an
optional field would be empty on every member within a month.

### `scripted.py`, the controlled systems

A `QueryKeyedStore` extending the `tests/fakes.py` pattern: a mapping from query text to the exact
dense and sparse rows returned, with per-row scores. It must control two things at once.

- **The retrieved id list**, which drives `precision_at_k`, `recall_at_k`, `mrr` and `ndcg_at_k`.
- **`gap_warning`**, which drives `false_confident_rate`. `HybridRetriever` computes it at
  `recall/retriever.py:317` as `gap_warning(list(dense_score.values()), self._gap_threshold)`,
  from **dense cosine scores only** and against `recall.guards.DEFAULT_GAP_THRESHOLD`. Choosing
  scores either side of that threshold drives it deterministically.

`tests/fakes.py:FakeStore` returns the same rows for every query and `VectorKeyedFakeStore` keys on
the embedding vector, so neither can express "question 3 retrieves differently from question 4"
without a contrived embedder. Keying on query text is the smallest addition that makes per-question
scripting readable.

### `test_fleet.py`, the runner

One parametrised test walking the table, plus the meta-tests in "Proving the fleet can fire" below.

## Surface A: `harness._score_config`

`_score_config(store, embedder, queries, fusion, reranker)` at `recall/eval/harness.py:129` takes
`store` and `embedder` directly, so a scripted store reaches the whole scoring path with no
database.

Every query in the fleet corpus has exactly **one** relevant document and `k=10`, which gives a
closed form for every aggregate. With the gold document at 1-based rank `r`:

```
R@5     = 1.0 if r <= 5 else 0.0
P@5     = 0.2 if r <= 5 else 0.0      (one relevant, so P@5 caps at 1/5)
MRR     = 1/r
nDCG@10 = 1/log2(r+1)                 (idcg = 1/log2(2) = 1 for a single relevant doc)
```

| member | defect | expected | why it earns its place |
|---|---|---|---|
| `perfect-rank-1` | none, the clean twin | `R@5 1.0`, `MRR 1.0`, `nDCG 1.0`, `P@5 0.2` | the baseline every other member must differ from |
| `gold-at-rank-3` | ranking degraded, retrieval intact | `R@5 1.0`, `MRR 1/3`, `nDCG 0.5`, `P@5 0.2` | separates set metrics from ranked metrics. A scorer that conflated them passes `R@5` and fails here |
| `boundary-rank-5` | gold exactly at the k window edge | `R@5 1.0`, `MRR 0.2`, `nDCG 1/log2(6)` | pins the inclusive edge |
| `boundary-rank-6` | gold one past the edge | `R@5 0.0`, `P@5 0.0`, `MRR 1/6`, `nDCG 1/log2(7)` | pins the exclusive edge. This repo has a documented history of 1-based and 0-based rank confusion, recorded in `metrics.latency_report`'s docstring |
| `gold-dropper-half` | gold absent entirely on half the questions | `R@5 0.5`, `MRR 0.5`, `nDCG 0.5`, `P@5 0.1` | pins the mean over questions, not just the per-question value |
| `guard-never-fires` | dense scores above threshold on every unanswerable query | `fcr_with_guard 1.0` | with the next row, pins the polarity of the `not g` negation inside `false_confident_rate` |
| `guard-fires-on-half` | dense scores below threshold on half the unanswerable queries, above on the rest | `fcr_with_guard 0.5` | the interior point, not the 0.0 extreme, is what defeats a scorer that computes `any()`/`all()` instead of a mean over the class; the 0.0 extreme is already covered by every other member (all of which have `fcr_with_guard 0.0`) |
| `no-answerable-queries` | declared blind spot, see below | `p_at_5 0.0` (current behaviour) | pins today's value and names the inconsistency |

`gold-dropper-half` is built as 10 answerable queries, 5 with the gold at rank 1 and 5 with the gold
absent from the top 10 entirely.

### The declared blind spot

`recall/eval/metrics.py` states its convention plainly: a rate with no data is NaN, because 0.0
"would read as a PERFECT superseded-trust rate and a CATASTROPHIC accuracy at the same time" and
NaN "forces publishers to render 'n/a' instead of a fake number".

`_score_config` does not honour it. At `recall/eval/harness.py:168` four headline metrics use
`mean(ps) if ps else 0.0`, so `p_at_5`, `r_at_5`, `mrr` and `ndcg_at_10` publish a fake `0.0` when a
configuration has no answerable queries, while `fcr_with_guard` on the same return **is**
NaN-on-empty via `false_confident_rate`. One return object, two conventions.

It is latent rather than active: the shipped eval corpus has answerable queries, so the branch is
not reached today.

**Decision: declare it, do not fix it here.** The `no-answerable-queries` member pins the current
`0.0` and names the inconsistency in its docstring. Changing the empty-case semantics of four
published metrics is a real change to a published artifact's shape and deserves its own reviewed
diff, not a quiet ride inside a test-only PR. The member is written so that whoever fixes it will
see exactly which assertion to update and why.

## Surface B: `promotion/aggregate.py`

Pure functions over `QuestionRecord` sequences, so no fakes are needed at all. This is the highest
stakes surface in scope, because it decides whether something ships.

`aggregate.py`'s module docstring **documents** four refusals: unpaired arms, NaN safety metrics,
unanswerable questions in the quality set, and latency measured on a loaded host. Documented is not
proven. These members execute them.

| member | construction | expected |
|---|---|---|
| `identical-arms` | candidate records identical to baseline | paired delta exactly 0, gate refuses |
| `treatment-strictly-better` | baseline gold at rank 6, candidate at rank 1, on every question | `hit@5` delta exactly 1.0, gate **promotes** |
| `safety-regressed` | quality improved, `false_abstention` worse than tolerance | gate refuses. Proves the safety axis can veto rather than merely being reported |
| `nan-safety-class` | an arm with no unanswerable questions | `build_safety` raises `ValueError`, naming the empty class |
| `unpaired-arms` | one frozen question missing from the candidate arm | refusal from `_index_arm` / `_refuse_if_vacuous` |
| `latency-pending` | otherwise promotable, `certified_latency_p95_ms` left at its `None` default | gate refuses, because PENDING fails |

`latency-pending` was added during design review rather than in the original member list. Reading
`decide` showed that `treatment-strictly-better` cannot reach PROMOTE unless it explicitly supplies
`certified_latency_p95_ms` and `security_green=True`, which makes the default worth pinning in its
own right: it is the single explicit door named in the module docstring, and a door that silently
stopped blocking would look exactly like one that was never opened.

`nan-safety-class` matters most of the six. `build_safety` raises because, in its own words, "the
gate compares them with `>`, and every comparison against NaN is False, so an empty class would
read exactly like a passed check". That is the 08-09 failure shape exactly: a check that cannot
fail reads identically to a check that passed.

Members assert on `PromotionDecision`'s outcome and on the specific refusal raised, both of which
are deterministic. `evaluate_retrieval_promotion` draws a bootstrap, so members lower
`bootstrap_samples` for speed and no member asserts on a bootstrap-derived interval bound.

## Surface C: `run_trust_eval`

`run_trust_eval(dsn, embedders, corpus_dir=None, queries_path=None, touch_stale=True,
store_factory=None)` at `recall/eval/harness.py` built its store internally from a `dsn` until
`store_factory` gave it an injection point. `queries_path` is still read from disk regardless —
the seam only replaces how the STORE is built — so `run_surface_c` in `test_fleet.py` writes each
member's queries to `tmp_path` and passes `store_factory=lambda emb: nullcontext(store)`.

`tests/fleet/scripted.QueryKeyedTrustStore` extends `QueryKeyedStore` with the two things
`recall.trust.trusted_search` needs beyond a plain `HybridRetriever.search`: a `supersession()`
method (`trusted_search` calls it unconditionally whenever a search returns any hit, so a bare
`QueryKeyedStore` raises `AttributeError` the moment the trust layer runs at all), scriptable via
a `supersession_edges` constructor argument, and a documented no-op `touch_files` (this store's
chunks never carry an `indexed_at`, so a touch has nothing to move).

**The finding the brief asked this change to establish, stated plainly: `research_search`'s
`TrustPolicy.development()` does NOT force every verdict to `unverified`.** Reading
`recall/trust.py:trusted_search` closely: the forced-`unverified` branch is gated on
`calibration is None`, and both `run_trust_eval` and `run_nearmiss_eval` always pass an EXPLICIT
`calibration=cal` into `research_search`. So `evaluate()`'s real verdicts (`ok`, `superseded`,
`low_confidence`, ...) stand; only `trust_state` is stamped `degraded`. `_research_trust.py`'s
module docstring ("every hit comes back `unverified`") describes the no-calibration case, which
these two callers never take — the docstring is not wrong, it is just not the branch either
runner reaches. This means `superseded_trust_rate` is genuinely drivable, not structurally 0.0,
and the two members below prove it two different ways: by score alone, and by real supersession
detection.

| member | construction | expected | why it earns its place |
|---|---|---|---|
| `trust-catches-scripted-supersession` | a stale hit scores above threshold WITH a scripted `supersedes:` edge | `str_trust 0.0`, `trust_coverage 0.5`, `successor_acc 1.0`, `abstain_acc 1.0` | proves the trust layer's REAL supersession resolution (`recall.trust.resolve_successor`), not just a score comparison, is what keeps the stale hit out of `str_trust` — its successor is promoted from `low_confidence` to `ok` by the same steelman logic `evaluate()` documents |
| `trust-misses-unscripted-supersession` | the SAME high score, no edge scripted | `str_trust 1.0`, `trust_coverage 1.0`, `successor_acc 0.0`, `abstain_acc NaN` | the honest counterexample: without the metadata edge the trust layer has no signal beyond the score, so a genuinely stale memory (per the eval's own label) is served `ok`. `str_trust` flips from 0.0 to 1.0 driven by nothing but what `supersession()` returns — the two-value proof the brief required before adding either member |

`str_baseline` / `str_recency` are 1.0 in both members (both read the raw scripted hit, not a
verdict) — reused, not separately proven, since neither routes through the trust layer at all.

`test_the_fleet_detects_a_broken_supersession_resolver` monkeypatches
`recall.trust.resolve_successor` to always return `None` and requires
`trust-catches-scripted-supersession`'s `str_trust` to move off 0.0 — proof the member is reading
the real resolution, not a coincidence of the scripted scores.

## Surface D: `run_nearmiss_eval`

`run_nearmiss_eval(dsn, embedders, judge, ..., store_factory=None)` gained the same seam.
Unlike Surface C, `judge: EntailmentJudge` is a REQUIRED positional argument used to build all
three arms every call, even for a member that only cares about `ARM_THRESHOLD`'s row — so
`tests/fleet/scripted.AlwaysEntailJudge` (`judge(query, texts) -> [True] * len(texts)`) exists
purely to satisfy that requirement without crashing or ever demoting a hit. It mirrors `AcceptAll`
in `tests/test_eval_nearmiss.py`, the real (DB-backed) suite's own fixture for the identical
purpose, and inherits that suite's proven degeneracy: with a judge that never disagrees,
`ARM_STACKED` cannot differ from `ARM_THRESHOLD` on any published rate
(`recall/entailment.py`'s `apply_entailment` only ever demotes, never promotes).

| member | construction | expected (`threshold` / `threshold+entail` / `entail-only`) | why it earns its place |
|---|---|---|---|
| `nearmiss-above-threshold-fools-the-cosine-guard` | a near-miss distractor scores ABOVE the gap threshold | `nearmiss_fcr 1.0 / 1.0 / 1.0` | a cosine threshold cannot tell a confident distractor from a confident answer — the exact failure class near-miss testing exists to name |
| `nearmiss-below-threshold-reads-as-an-ordinary-gap` | the SAME distractor, scored below threshold | `nearmiss_fcr 0.0 / 0.0 / 1.0` | `nearmiss_fcr` flips from 1.0 to 0.0 under `threshold`/`threshold+entail`, driven by nothing but the store's returned score — the two-value proof for this rate. `entail-only`'s 1.0 is unchanged in both members: its permissive threshold (-1.0) passes any real cosine regardless, a genuine derived consequence, not vacuity |

`gap_fcr` / `false_abstain` / `mrr_answerable` come from a shared plain/gap query pair and are
identical across both members (0.0 / 0.0 / 1.0 under `threshold`/`threshold+entail`; `entail-only`
reads `gap_fcr 1.0` — its permissive threshold cannot tell the far-gap query from an answer
either, the mirror of what `nearmiss_fcr` shows). `test_the_fleet_detects_a_broken_near_miss_metric`
monkeypatches `recall.eval.harness.near_miss_false_confident_rate` to a constant `0.0` and
requires `nearmiss-above-threshold-fools-the-cosine-guard`'s `ARM_THRESHOLD` row to move.

## Assertions

**Expected values are derived, never captured.** Every `expected` is written by hand from the
formulas above and carries its derivation in a comment. This is the line between a fleet and a
snapshot test. A snapshot blesses whatever the code does today, so it can only detect *change*; a
fleet asserts what the code *must* do, so it detects change and pre-existing wrongness alike. If a
member's expected value ever has to be edited to make a test pass, that is a finding to
investigate, not a chore. The module docstring says so, because re-recording is how this kind of
suite rots.

Float comparison is `pytest.approx(expected, abs=1e-9)`, an explicit absolute tolerance either
way. `1/log2(6)` is irrational and an equality assert would be a lie about precision.

## Failure modes the runner distinguishes

Following the precedent `arm_check.EmptySampleError` sets in this repo, where an empty comparison
raises rather than returning a vacuous `("IDENTICAL", 0)`:

- **A member producing zero scored questions raises.** A fixture that measured nothing must not
  read like a defect that was absent.
- **A member whose expected values match the clean twin on every metric fails the table**, as
  vacuous. `gold-at-rank-3` matches `perfect-rank-1` on `R@5` and earns its place only through
  `MRR` and `nDCG`. This enforces mechanically what
  `feedback-a-mutation-sweep-cannot-see-a-fixture-built-from-the-passing-shape-2026-08-05` records
  as a lesson.

## Proving the fleet can fire

Green tests are evidence of nothing until they have been shown to go red, and a test written after
a fix and never shown to fail is a hypothesis rather than a guard. Four meta-tests:

- **Surface A**: monkeypatch `recall_at_k` to return `1.0` unconditionally, assert the fleet goes
  red.
- **Surface B**: monkeypatch `false_abstain_rate` to return `0.0` unconditionally, which makes the
  safety axis unable to register a regression, and assert `safety-regressed` goes red (its
  refusal, `"false abstention regresses"`, disappears and the candidate is wrongly promoted).
  `nan-safety-class` raises on `gap_false_confident_rate`, a different function untouched by this
  stub, so it is not part of this meta-test.
- **Surface C**: monkeypatch `recall.trust.resolve_successor` to always return `None`, assert
  `trust-catches-scripted-supersession`'s `str_trust` moves off 0.0 — proof the member reads real
  supersession resolution, not a coincidence of the scripted scores.
- **Surface D**: monkeypatch `recall.eval.harness.near_miss_false_confident_rate` to return `0.0`
  unconditionally, assert `nearmiss-above-threshold-fools-the-cosine-guard`'s `ARM_THRESHOLD` row
  moves off 1.0.

If any mutation leaves the suite green, the fleet is decorative and that is the finding.

## Declared non-coverage

Rolled up from every member's `does_not_catch` into one test that prints the list, so blind spots
appear in the output rather than staying buried per member.

- **Indexing and embedding are stubbed.** Nothing here says anything about real retrieval quality.
- **`run_trust_eval` and `run_nearmiss_eval` are now REACHED** (Surface C, Surface D, added
  2026-08-15 via an injected `store_factory`), but only as far as a scripted store can honestly
  drive their published rates:
  - **`str_recency`'s real "prefer the newest timestamp" tie-break is unexercised.** Every
    scripted hit carries `indexed_at=None`, so the recency arm's `max(pool, key=...)` degenerates
    to "first hit in the confident pool" rather than a real timestamp comparison; `touch_stale` is
    also passed `False` in every Surface C member, so `run_trust_eval`'s default re-sync
    simulation is never taken.
  - **The entailment DEMOTION mechanism (`ok` -> `not_entailed`) is unexercised.** Surface D's
    judge (`AlwaysEntailJudge`) never disagrees, so `ARM_STACKED` is only shown to be inert when
    the judge agrees — never shown to demote a confident near-miss, which is the one thing
    entailment exists to do that a cosine threshold cannot. A judge that can disagree per
    candidate is required to close this, and none is scripted.
  - **`TrustEvalResult`'s Wilson-CI fields (`*_ci`) and `n_*` sample counts are not
    independently asserted.** They are pure functions of flags Surface C already drives to
    differing values (`recall/eval/metrics.py`'s `wilson_ci`), so covering them separately would
    re-test that 12-line pure function rather than close a new class of blind spot — a scoping
    choice, not an incapability.
  - **`entail_latency_ms_mean` and `query_latency_ms_mean` are undrivable to a closed-form
    value.** They are wall-clock timings; nothing in `tests/fleet/members.py`'s "derived, never
    captured" discipline can write a formula for a measured duration, so no member asserts on
    them.
- **LOCOMO, BEAM, MTRAG and the ladder are out of scope for v1.** `locomo.run_conversation` has the
  same `store` and `embedder` seam, so the `QueryKeyedStore` reaches it whenever someone wants it.
- **Generation and judging are untouched.** The 08-09 conditioning bug lived in an upstream IBM
  scorer, and this fleet would not have caught that bug at its own site. It closes the *class*, an
  aggregation step that silently computes something other than what its name claims, not that
  specific instance. Stating this plainly is better than implying we have vaccinated against the
  incident.
- **`p_at_5`'s `0.0`-on-empty is pinned as current behaviour**, not asserted as correct.

## Success criteria

1. `pytest tests/fleet/` passes with all eighteen members: eight on Surface A, six on Surface B,
   two on Surface C, two on Surface D.
2. All four mutation meta-tests turn the suite red when applied, verified by running them.
3. Every member carries a non-empty `does_not_catch`, enforced by a test rather than by review.
4. No file outside `tests/` is modified, except `recall/eval/harness.py`'s `store_factory` seam
   (Part A, additive-only: the default path is byte-for-byte unchanged for every existing caller,
   verified by the six known call sites still passing untouched).
