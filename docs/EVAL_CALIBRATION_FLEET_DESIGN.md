# Eval calibration fleet: certifying that the harness reports the number it claims

> Design, 2026-08-15. Status: approved, not yet implemented.
> Scope: `tests/fleet/`, covering `recall/eval/harness._score_config` and
> `recall/eval/promotion/aggregate.py`. Test only, nothing ships in the wheel.

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
| `guard-always-fires` | dense scores below threshold | `fcr_with_guard 0.0` | a polarity flip is invisible to any fixture that is symmetric in the two classes |
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

## Assertions

**Expected values are derived, never captured.** Every `expected` is written by hand from the
formulas above and carries its derivation in a comment. This is the line between a fleet and a
snapshot test. A snapshot blesses whatever the code does today, so it can only detect *change*; a
fleet asserts what the code *must* do, so it detects change and pre-existing wrongness alike. If a
member's expected value ever has to be edited to make a test pass, that is a finding to
investigate, not a chore. The module docstring says so, because re-recording is how this kind of
suite rots.

Float comparison is `math.isclose` with an explicit absolute tolerance. `1/log2(6)` is irrational
and an equality assert would be a lie about precision.

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
a fix and never shown to fail is a hypothesis rather than a guard. Two meta-tests:

- **Surface A**: monkeypatch `recall_at_k` to return `1.0` unconditionally, assert the fleet goes
  red.
- **Surface B**: monkeypatch `false_abstain_rate` to return `0.0` unconditionally, which makes the
  safety axis unable to register a regression, and assert `safety-regressed` and `nan-safety-class`
  go red.

If either mutation leaves the suite green, the fleet is decorative and that is the finding.

## Declared non-coverage

Rolled up from every member's `does_not_catch` into one test that prints the list, so blind spots
appear in the output rather than staying buried per member.

- **Indexing and embedding are stubbed.** Nothing here says anything about real retrieval quality.
- **`run_trust_eval` and `run_nearmiss_eval` are unreached.** Both take a `dsn` and build a real
  store internally via `_throwaway_store`, and additionally need `store.touch_files`, a calibration
  fitted from real cosines, and an entailment judge. Reaching them requires refactoring them to
  accept an injected store, which turns a test-only change into a harness refactor. **Deferred by
  decision**, to be its own change.
- **LOCOMO, BEAM, MTRAG and the ladder are out of scope for v1.** `locomo.run_conversation` has the
  same `store` and `embedder` seam, so the `QueryKeyedStore` reaches it whenever someone wants it.
- **Generation and judging are untouched.** The 08-09 conditioning bug lived in an upstream IBM
  scorer, and this fleet would not have caught that bug at its own site. It closes the *class*, an
  aggregation step that silently computes something other than what its name claims, not that
  specific instance. Stating this plainly is better than implying we have vaccinated against the
  incident.
- **`p_at_5`'s `0.0`-on-empty is pinned as current behaviour**, not asserted as correct.

## Success criteria

1. `pytest tests/fleet/` passes on `origin/master` with all fourteen members: eight on surface A,
   six on surface B.
2. Both mutation meta-tests turn the suite red when applied, verified by running them.
3. Every member carries a non-empty `does_not_catch`, enforced by a test rather than by review.
4. No file outside `tests/` is modified.
