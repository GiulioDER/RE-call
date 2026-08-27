# Pre-registration: the same three arms, with the description actually in the index

**Date:** 2026-08-27   **Status:** predicted, not yet measured
**Apparatus:** the corrected `scripts/agent_ab_prepare_discoverability_corpus.py` +
`scripts/agent_ab_probe_discoverability.py` at commit `247cf67b`, pinned by
`scripts/agent_ab_discoverability_tests.py`.

## The question

The 2026-08-27 probe (`2026-08-27-memo-discoverability-authoring.md`) reported rescue 0 of 14 in
three arms. A DEEP audit then found that **the generated `description:` reached no index in any
arm**: it was written into YAML frontmatter, and the corpus builder chunks
`parse_document(text).human_body`, which is the document with frontmatter removed. Measured on the
built corpora: the description appeared in indexed chunk text for **0 of 40** memos, the title for
**40 of 40**.

So: with the identical generated surfaces now emitted as indexed body text, does any arm rescue any
of the 14 recorded misses?

## Why this is a new registration and not a re-run

The first record's arms are frozen and its result stands for what it measured. This changes the
treatment (where the description lands), so it is a different experiment and gets its own
predictions. What it deliberately does NOT change is the generated content.

**The 190 triples from the first run are reused verbatim**, seeded into the resume cache with the
current terms stamp. This is legitimate and is the stronger design: `rewrite_model` and
`rewrite_prompt` in `rewrites.json` are byte-identical to the committed constants (verified before
writing this record), and temperature, max_tokens, reasoning and input truncation were hoisted to
constants without changing their values. Reusing them makes run 1 and run 2 differ in **exactly one
variable, the placement of the description**, which is the question. Generating fresh triples at
temperature 0 would be expected to reproduce them and would add a confound if it did not.

## Design

Identical to the first record except where stated:

- Same evidence corpus (`recall-agentab-corpus`, generation `gen_f01fc522`), same recovery, same
  four arms, same builder, chunker, embedder (`fastembed`) and calibration query set.
- Same replay: the recorded `recall_search` queries of `agent-ab-skill-001`'s admitted on-arm
  `memory_only` sessions, same stdio transport, same top-5.
- Same registered exclusion, `ts-raise-on-missing`, for the same reason (its governing memo is the
  one reconstruction-approximate governing source). Scored population: **14 misses, 26 hits**, now
  asserted by the instrument rather than assumed.
- **What changed in the arms:** the generated description is emitted as body text under the title
  in `retitle` and `restructured`, and in the pointer document in `pointer`; a frontmatter block is
  synthesized when a source has none, so all 190 memos receive the whole treatment rather than 163.

**Reconstruction is disclosed up front, not discovered afterwards.** The chunk store holds
`human_body`, so a drifted source can never be rebuilt byte-exactly and the learned joiner scores
`0/167`. The corrected script now REFUSES this state; this run passes
`--allow-lossy-reconstruction` deliberately, and the refusal's own condition is checked first: no
governing memo of any SCORED session may be among the reconstructed set. That check is recorded in
the result below, and if it fails the run is abandoned rather than flagged.

## What I predict

Calibration applied explicitly, per `[[i-over-predict-effect-magnitudes]]`: this is a
DESCRIPTION-side intervention, wording-side interventions in this project have now measured at or
below the bottom of their band **three consecutive times** (appended aliases 0, query expansion
3/15, authored surfaces 0/14), and that memo's revised rule says to predict the bottom of the
discounted band and treat anything above it as the surprise. The ceiling is 14.

| # | Prediction | Falsified if |
|---|---|---|
| 1 | **Best-arm rescue: 0 to 2 of 14.** One extra sentence of the same generated vocabulary, in arms whose five task phrasings were already indexed and rescued nothing | 3 or more |
| 2 | **MECHANISM, and the one that makes this run interpretable at all: `retitle`'s top-5 differs from RUN 1's `retitle` on at least 50% of the 78 queries.** Run 1's retitle was title-only; if adding an indexed sentence to all 190 memos moves fewer than half the result lists, the treatment is barely reaching the ranking and a null says little | under 50% |
| 3 | **Retention: at least 25 of 26 in every arm.** The description is one more sentence competing corpus-wide, which the first run's zero-loss result suggests is affordable | any arm below 25 |
| 4 | **`ts-lf-rewrite`: 0 of 6.** Its memo cannot say "version"; the description is derived from the same text that never mentions it, so the causal gap is untouched by placement | 1 or more (which would be the most informative single outcome in this run) |
| 5 | **Cost: under 40 minutes wall clock and under 0.10 USD.** Four builds at ~6 minutes each measured on 2026-08-27, one replay at ~2 minutes, zero generation calls because the triples are reused | either bound exceeded |

## Decision rule, full partition

Read from the best arm with retention >= 24 of 26:

- **Rescue >= 3**: placement was the binding constraint, not vocabulary. The first record's
  conclusion is materially wrong and must be superseded rather than corrected; preregister a live
  agent A/B before any claim about agent-level effect.
- **Rescue 1 or 2**: a real but marginal effect that the first run could not have seen. The
  generation-side lane reopens at low priority; one further registered iteration is licensed and
  the authoring guideline is worth writing for the description alone.
- **Rescue 0, with prediction 2 confirmed** (the treatment demonstrably moved the rankings): the
  first record's conclusion stands and is now strengthened rather than merely corrected, because
  the fully-applied treatment was measured. The generation-side lane closes for good.
- **Rescue 0, with prediction 2 falsified** (rankings barely moved): the run is INCONCLUSIVE, not a
  null. A treatment that does not perturb the ranking has not been tested, and saying so is the
  whole point of registering the mechanism metric beside the outcome.
- **Apparatus gate fails, or any arm fails the divergence floor**: VOID, no verdict, exit code 3
  and a `*.VOID.json` artifact.

## How it will be measured

```bash
eval "$(scripts/session-db.sh up)"
python -u scripts/agent_ab_prepare_discoverability_corpus.py \
  --evidence-dsn postgresql://recall:recall@127.0.0.1:5407/agent_ab \
  --out benchmarks/artifacts/agent_ab/discoverability-rerun --allow-lossy-reconstruction
# four sequential builds into probe2_{control,retitle,restructured,pointer}
python -u scripts/agent_ab_probe_discoverability.py --archive ~/.claude/archive/agent-ab-skill-001 \
  --control-dsn <session>/probe2_control --arm retitle=<session>/probe2_retitle \
  --arm restructured=<session>/probe2_restructured --arm pointer=<session>/probe2_pointer \
  --exclude-base ts-raise-on-missing
```

## What I already know

- Run 1's per-arm retrieval for all 78 queries is in
  `~/.claude/archive/discoverability-probe-2026-08-27/discoverability-probe.json`, which is what
  prediction 2 is measured against.
- Three prior generation-side results, all at or under the bottom of their bands:
  `[[query-side-expansion-reproduces-the-blind-spot]]`,
  `[[authored-discoverability-surfaces-cannot-close-the-formulation-gap]]`.
- The audit's own lesson, which is why prediction 2 exists at all:
  `[[a-null-is-the-cheapest-result-to-fabricate]]`. Four paths in the old apparatus printed the
  published null exactly; a second null from a treatment that never moved a ranking would be the
  fifth.

## Confounds I can name now

1. **I know run 1's answer.** The predictions above are anchored on it, and prediction 1's band is
   centred on zero partly because zero is what I have seen three times. Prediction 2 is the guard:
   it can falsify the run without reference to the outcome.
2. **Reused triples.** If the first run's generation was itself poor, this run inherits that. It is
   the correct trade for isolating placement, and it means a null here is a null about placement,
   not about generated surfaces in general.
3. **25 lossy reconstructions**, identical across all four arms and disclosed above. Bounded by the
   check that no scored governing memo is among them.
4. **New voiding criteria added after run 1** (denominator assertion, arm divergence floor). Run 1
   satisfied both post hoc, so they are not fitted to produce a particular answer here.

<!-- frozen_above -->

## Result (2026-08-27)

**Status: measured, valid, and all five predictions confirmed.** The apparatus gate passed on the
registered population: **14 of 14** misses and **26 of 26** hits reproduced, `population_matches_
registration=true`, `top_k=5`, `void=false`, exit 0.

**Measured: rescue 0 of 14 in every arm, with the treatment demonstrably live.**

| arm | rescue | direct | retention | direct | divergence from control | vs RUN 1's same arm |
|---|---|---|---|---|---:|---:|
| retitle | 0/14 | 0 | **25/26** | 25 | 0.923 | **98.7%** |
| restructured | 0/14 | 0 | 26/26 | 26 | 0.987 | 93.6% |
| pointer | 0/14 | 0 | 26/26 | 26 | 0.962 | 84.6% |

Artifact: `benchmarks/artifacts/agent_ab/discoverability-rerun/rerun-probe.json`.

**The apparatus validated itself in a way the first run could not.** Control was rebuilt
independently and answered **identically to run 1 on 78 of 78 queries**, at the same 1,019 chunks.
Two rebuilds, five days of corpus drift apart in their inputs, bit-stable baseline: that is the
evidence a single run's passing gate cannot supply.

**Predictions against measurements:**

| # | predicted | measured | verdict |
|---|---|---|---|
| 1 | best-arm rescue 0 to 2 of 14 | **0** | confirmed, at the bottom of the band |
| 2 | retitle differs from run 1's retitle on >= 50% of 78 queries | **98.7%** (77 of 78) | confirmed, far above |
| 3 | retention >= 25 of 26 in every arm | 25, 26, 26 | confirmed, retitle exactly at the bound |
| 4 | `ts-lf-rewrite` 0 of 6 | **0 of 6** | confirmed |
| 5 | under 40 minutes and under 0.10 USD | ~28 minutes, **0.00 USD** | confirmed (zero generation calls: triples reused) |

Five for five is not a sign of insight; four of them are the classes
`[[i-over-predict-effect-magnitudes]]` already says I estimate well (mechanism rates, costs, and a
benefit predicted at the bottom of its band per that memo's revised rule). Recorded because a
predicted null is information, and because the rule that produced prediction 1 has now earned a
fourth consecutive confirmation.

**Decision, per the registered partition: rescue 0 with prediction 2 CONFIRMED, so the first
record's conclusion stands and is strengthened rather than corrected. The generation-side lane
closes for good.** The fully-applied treatment has now been measured: every one of the 190 memos
carried a searcher-oriented title, a searcher-oriented description and (in two arms) five task
phrasings into the indexed text; the rankings moved on 98.7%, 93.6% and 84.6% of queries; and not
one of the 14 recorded misses was rescued in any arm.

🔑 **The retention loss is the most informative single number in this run.** `retitle` lost
`ts-sample-covers-tail#r4`, and it lost it by **displacement, not substitution**: for the query
`chunk_id file text sampling`, `sorted-sample-plus-early-stop-is-head-bias.md` sat at rank 1 in
control and fell out of the top 5 entirely, replaced by neighbours that gained the same treatment.
Nothing about that memo got worse. Everything around it got a searcher-oriented sentence, and the
ranking is a competition. That is the mechanism this whole lane has been circling stated in one
case: **uniform authoring improvements do not accumulate, they cancel, and past the cancellation
they can cost you the hit you already had.** A memo cannot be written into first place when every
competitor is written the same way, and the one query where the treatment changed a scored outcome,
it changed it for the worse.

**What this closes and what it leaves.** Four registered attempts, four directions, all measured:
query-side expansion 3/15, bottom-appended aliases 0 effect, authored surfaces with the treatment
partly unindexed 0/14, and now authored surfaces fully indexed 0/14 with a live instrument. No
fifth generation-side variant is licensed. The formulation gap is a RELEVANCE problem: the
governing memo is frequently not the semantically closest document to a goal query, and cannot be
made so by writing, because writing is available to every document at once. What remains is
retrieval-side (a scorer that can follow a task-to-hazard causal link, or lexical/hybrid matching
where an exact trigger term exists) or interactive search behaviour, and either needs its own
registration.

See [[a-null-is-the-cheapest-result-to-fabricate]] for why prediction 2 was registered at all: it
is the check whose absence made the first run's null unprovable, and here it is what turns a second
0 of 14 from a repeat into a verdict.
