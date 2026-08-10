# Preregistration: is LOCOMO retrieval coverage-bound or ranking-bound?

**Written 2026-08-06, before any number from this run was seen.**

## Prior work searched, and what it already settles

`docs_search(source_type="memory", ...)` run twice before writing this, on the SPLADE/MT-RAG lever
and on LOCOMO depth curves. Prior work found, and it is directly on point:

- **[[project-recall-nearmiss-signal-exhaustion-2026-07-29]] §7, the k-sweep.** Already measured a
  LOCOMO gold `hit@k` depth curve at `candidate_k=250`:

  | k | 1 | 3 | 5 | 10 | 20 | 50 |
  |---|---|---|---|---|---|---|
  | gold hit@k | 0.375 | 0.555 | 0.640 | 0.730 | 0.805 | **0.910** |

- **§11 of the same memo**: pool-level `voyage:rerank-2.5` over a top-50 pool lifts hit@5 from
  0.640 to 0.870, and its own conclusion is that *"the residual is pool recall (hit@50 0.910), not
  ranking."*
- **[[closed-hypothesis-recall-rerank-pool-interaction-2026-08-05]]**: the local MiniLM
  cross-encoder gets **worse** as the pool widens (PEPs, n=88), so "widen the pool and rerank with
  what we ship" is already falsified as a combination.
- **[[project-recall-mtrag-retrieval-coverage-bottleneck-2026-08-06]]**: the ~0.95 saturation
  threshold this gate applies.

⚠️ **This prior work revised prediction P2 below, before the run.** Recorded explicitly so the
change is auditable and cannot be mistaken for fitting a prediction to a result.

### What is still genuinely unmeasured, and why this run is not redundant

The k-sweep is **pooled across categories**, stops at **depth 50**, ran on the **ladder arm** of 200
answerable questions at **`candidate_k=250`**. The mem0 head-to-head is a different configuration on
a different population. Three cells are new here:

1. **Per-category**, so cat1 can be read on its own. cat1 is the category the head-to-head loss was
   attributed to, and no depth curve has ever been broken out by category.
2. **Depth 100**, past where the k-sweep stopped.
3. **The published head-to-head configuration**: `candidate_k=20`, all 10 conversations, the full
   question set, which is the arm the Mem0 comparison actually uses.

## Why this exists

SPLADE was built and measured on MT-RAG, where it bought **coverage** and only coverage: reranked
R@100 +0.0303, CI [+0.0173, +0.0437], Holm-significant, while reranked nDCG@5 was −0.0043 with a CI
spanning zero. SPLADE is a coverage instrument. Wiring it into the Mem0 head-to-head is only worth
the work if LOCOMO's failure is a **coverage** failure.

## The apparatus

`recall.eval.locomo` scores `hit@k` by depth from ONE retrieval per question. Judge-free: LOCOMO
`qa` rows carry `evidence` as dialog-turn ids, one turn is one indexed document, a hit is string
equality on the turn id. No generator, no judge variance, no LLM cost.

Two facts about the published configuration that shape the design:

- `DEFAULT_CANDIDATE_K = 20`. The published head-to-head arm retrieves with a candidate pool of
  **20 per leg** and delivers `k=20`. Depths past 20 are unreachable without widening the pool.
- The module docstring records a prior pooled result: **hit@5 0.671 against hit@20 0.855**.

⚠️ Raising `--candidate-k` **changes the fusion**, so arm B is a different configuration, not a
deeper look at arm A. Reported as such, never merged into one curve.

## Arms

| arm | candidate_k | depths scored |
|---|---|---|
| A (published config) | 20 | 1, 3, 5, 10, 20 |
| B (widened pool) | 100 | 1, 3, 5, 10, 20, 50, 100 |

Same embedder as the published head-to-head (`fastembed` = `BAAI/bge-small-en-v1.5`, 384-dim), same
10 conversations, reranker OFF (its default, and what the published number used).

## Predictions, recorded before the run

- **P1.** In arm A, cat1 `hit@k` rises materially from k=5 to k=20, and cat1's `hit@20` is the
  **lowest** of the four answerable categories. *(Untouched by prior work: no depth curve has been
  broken out by category.)*
- **P2 (the deciding cell), REVISED by the prior work above.** cat1 `hit@100` in arm B lands
  **between 0.90 and 0.95**, i.e. high but short of saturation. Original prediction before I found
  the k-sweep was "above 0.90"; the measured pooled hit@50 = 0.910 at a wider pool makes the
  interesting question specifically whether cat1 **clears 0.95**, so that is the number the
  decision rule is written against.
- **P3.** The archived "on cat1 failures the gold was never retrieved 71% of the time" figure is
  about the **delivered k=20 context under a pool of 20**, and will NOT reproduce as absence from a
  widened pool. It is mostly a pool-width and ranking artefact, not evidence the retriever cannot
  find the turn.
- **P4.** Arm A's ceiling is set by its pool. Arm B beats arm A at every shared depth, and the gap
  at depth 20 is material, meaning the published head-to-head arm is leaving already-available
  recall on the table through `candidate_k` alone, before SPLADE is considered.

## Decision rule, fixed in advance

Applying the MT-RAG saturation threshold of ~0.95:

- **cat1 hit@100 (arm B) ≥ 0.95** ⇒ coverage is saturated on LOCOMO. SPLADE is the wrong lever and
  should NOT be wired for the Mem0 rematch. The payoff is pool width and ranking.
- **cat1 hit@100 < 0.90** ⇒ coverage has real room ⇒ wire SPLADE and re-run the head-to-head.
- **0.90 to 0.95** ⇒ ambiguous. Report both, decide on the absolute count of cat1 evidence turns
  still missing at depth 100, not on the rate.

⚠️ Whatever the verdict on SPLADE, P4 is decided separately: if arm B beats arm A at depth 20, the
`candidate_k` lever is free and is the first thing to change in the rematch. ⛔ It does **not**
license turning the shipped reranker on over a wide pool, which
[[closed-hypothesis-recall-rerank-pool-interaction-2026-08-05]] already falsified.

## Invariants asserted before believing any number

1. The k=5 row of the depth curve must reproduce a standalone k=5 run exactly. `run()` checks this
   coherence internally and raises on mismatch, so a run that completes has passed it.
2. Every category cell must have non-zero `n`. A rate over an empty cell is not a measurement.
3. The corpus must be indexed exactly once. `index_conversation` raises rather than double-index,
   which would depress every `hit@k` without erroring.
4. Arm A and arm B must agree at the depths where the pool does not bind, or the pool is not doing
   what it is documented to do.
