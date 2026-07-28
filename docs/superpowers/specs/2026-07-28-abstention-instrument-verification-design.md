# Abstention instrument verification — design

> Status: **approved design, pre-implementation.** 2026-07-28.
> Written against `origin/master` at `9eb3bc1` (the #103 rerank merge).

## 1. Goal

Establish which of this repo's published abstention claims still describe the shipped pipeline,
and make every future result artifact carry its own proof of what it was measured against.

This cycle deliberately **adds no capability**. It ends in a written verdict that gates the next
one. The reason is on the record: on 2026-07-27 two concurrent runs indexed the LOCOMO corpus
twice (11,764 rows against a correct 5,882), nothing errored, every depth came in ~0.05 low, and
the run produced a *believable but manufactured* finding. Exit code 0 is not a measurement.

## 2. Why this and not a new signal

The temptation is to attack over-abstention directly — a combined signal, an entity feature, a
softer policy. Every one of those is fit against a baseline. If the baseline cannot be
reproduced, the improvement cannot be attributed.

Three defects found while scoping this, each verified rather than assumed:

- **No result artifact in this repo records the corpus row count it was measured against.**
  `grep` over `results/**/*.json` returns nothing. This includes `postfix_abstention.json` and
  `postfix_pool20.json`, the current headline artifacts. The 2026-07-27 failure is therefore
  **undetectable retroactively** on every published number.
- **The §9c re-run left no retained evidence at all.** This repo's own `.gitignore` calls
  `results/locomo/*.log` transient and says the JSON beside it is the artifact — and for §9c, no
  JSON was ever written. The only trace, a `postfix_entailment_sweep.log` that stops after 9
  conversations with no summary, is untracked and gitignored by design, so it isn't part of this
  repository. `FINDINGS.md` records §9c as "not re-measured", which is true and undersells it: it
  *was* attempted, it died, and left nothing this repo can check.
- **#103 verified its run and the verification evaporated.** `scripts/run_locomo_arms.sh` holds
  `EXPECTED_ROWS=5882`, takes a lock, drops each table and checks the count after every arm. That
  discipline was real. It reached the runner's stdout and stopped there — `baseline_verified.json`,
  `rerank_modern.json` and `rerank_shipped.json` record embedder, `k`, `candidate_k`, `reranker`,
  `conversations` and `elapsed_s`, and no row count. In six months those files are
  indistinguishable from a doubled-corpus run.

The third is the load-bearing one: the argument for Phase 0 comes from the **best-executed**
experiment in the repo, not the worst. A process that only survives while its author is watching
is not a process.

## 3. What is already true (verified 2026-07-28)

Recorded so this cycle does not redo work that landed while it was being scoped.

| thing | state |
|---|---|
| tenant-refusal guard | **on master**, `recall/eval/locomo.py:274-276` — refuses to index over an existing tenant |
| `scripts/run_locomo_arms.sh` | **on master** — lock, table drop, row-count assertion per arm |
| `docs/RESEARCH_PROTOCOL.md` | **on master** — predict before measuring, assert invariants in code |
| #103 rerank result | **merged** (`9eb3bc1`), hit@5 0.671 → 0.777, verified at 5,882 rows |
| #101 calibration auto-load | **merged** (`8748df0`) |

**#101 does not invalidate the §9b tables.** `recall/eval/locomo_abstention.py:168` passes
`calibration=cal` **explicitly** to `trusted_search`, so the auto-load bug never touched that
harness. #101's blast radius was on library users, not on published results.

**#103 is not re-run by this cycle.** Its numbers are sound and the `rerank_modern` arm alone cost
3.8 hours. What it lacked is addressed by Phase 0 going forward, not by re-measuring backwards.

## 4. Phase 0 — artifacts carry their own row count

The only code change in this cycle.

Every eval runner that writes a result JSON also writes the corpus row count it measured, the
tenant/table it read, and the git SHA of the tree that produced it. The count is read from the
store and **asserted before any metric is computed**, not after.

**It moves from the shell script into the harness.** A wrapper script is bypassed by anyone
invoking the module directly — which is how the 2026-07-27 run was launched. A guard that only
fires on the blessed path does not guard the path that failed.

Runners in scope: `recall/eval/locomo.py`, `recall/eval/locomo_abstention.py`,
`recall/eval/locomo_entailment_sweep.py`, `recall/eval/labelled.py`,
`recall/eval/longmemeval_perq.py`.

Pinned by a test that writes a result with a deliberately wrong expected count and asserts the
runner refuses. Verified on the **detection path**: the test must go red when the assertion is
reverted.

Existing artifacts are **not** back-filled. A row count reconstructed after the fact is a claim,
not a measurement, and writing one into an old file would be the same class of error this phase
exists to remove.

## 5. Phase 1 — inventory

One table in `results/INSTRUMENT_STATUS.md`: for every abstention claim, whether it is current,
stale, or unfalsifiable, with the artifact that settles it.

Starting state, verified while scoping:

| claim | status | basis |
|---|---|---|
| §9b LOCOMO abstention, 4 modes | **current** | `postfix_abstention.json`, post-#81/#84, calibration passed explicitly |
| §9b abstention **with rerank on** | **unmeasured** for the calibrated and judge modes | #103 measured only the default mode (0.00, unchanged) |
| §9c entailment ROC sweep | **stale, no retained artifact** — re-run died at 9/10, no JSON ever written | §2 |
| §10 LongMemEval, all rows | **stale and unfalsifiable** — pre-#81/#84, no artifact retained | `FINDINGS.md` §10 note |
| §7 private-corpus abstention | current, but not independently checkable (private corpus) | — |
| §8 PEP abstention | current, cheap and public to re-establish | — |
| *every row above* | **none records its row count** | §2 |

The `with rerank on` row is new and follows from #103 landing: reranking moves hit@1 from 0.398 to
0.553, so every top-1-based abstention signal now has a second input distribution that has never
been characterised. Rerank remains off by default, so §9b still describes the shipped default —
but "the shipped default" and "the best configuration we ship" are no longer the same thing.

## 6. Phase 2 — re-measure, cheapest first

Each run gets a written prediction committed **before** it starts, per `RESEARCH_PROTOCOL.md`, and
is scored afterwards for whether it was right *for the right reason*.

1. **§9c entailment sweep** (~1h, one model pass, analytic threshold sweep). First question is not
   the ROC — it is *why the previous run died at 9/10*. A harness that can stop nine tenths of the
   way through and leave a plausible-looking log is a defect independent of its output.
2. **§8 PEP abstention arm** — public corpus, public questions, cheap. Re-establishes a working
   abstention measurement on the current pipeline, which the inventory otherwise lacks.
3. **§10 LongMemEval** — an explicit **decision, not an automatic re-run.** 6h39m to rebuild the
   merged index alone. Its conclusion rests on signal *separability* (AUC ≤ 0.753 against a ~0.90
   bar), and `FINDINGS.md` already argues a better candidate pool does not turn a relevance signal
   into an answerability signal. Two defensible outcomes, and the cycle picks one on the record:
   - **re-run** — restores a checkable artifact for the strongest negative result in the document; or
   - **demote** — mark §10 "measured on a superseded configuration, artifact not retained", and
     stop citing it as load-bearing evidence anywhere.

   Demoting is not a retreat. An unfalsifiable claim that is cited as evidence is worse than one
   that is labelled.

## 7. Phase 3 — verdict, and what it gates

`results/INSTRUMENT_STATUS.md` closes with a statement of which abstention claims are checkable on
the current pipeline.

That statement is the **gate for the next cycle**. No combined signal, no entity-mismatch feature
and no abstention-policy change is fit against a baseline this document marks unfalsifiable.

## 8. Out of scope, deliberately

No new signal. No threshold change. No abstention-policy change. No change to `trust.py`,
`calibration.py` or `guards.py`. No re-measurement of #103. No retrieval work — the mem0-teardown
plans (fusion weighting, lemmatisation, entity channel) are a separate track and are untouched
here.

Named because each is a live temptation with a plausible case, and because a verification cycle
that quietly grows a feature stops being one.

## 9. Risks

- **The inventory could come back mostly green**, making Phase 2 cheap and this cycle look like
  overhead. That is a good outcome, not a wasted one — "checkable" is the deliverable, and it is
  currently unknown rather than known-good.
- **Concurrent sessions share this clone.** Another session merged #103 while this spec was being
  written. Re-read `origin/master` before acting on any line number here; "nothing to commit" or
  "already merged" is a race, not an error.
- **Phase 0 touches five runners.** If the row-count assertion is wrong in one of them it blocks a
  measurement rather than corrupting it, which is the safe direction — but it must not be added to
  a runner without a detection-path test, or it becomes decoration.
