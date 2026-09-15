# Source gold retrieval leg reachability preregistration

Date registered: 2026-09-13.

## Question

On the current certified Voyage 4 memory generation, does source level gold disappear during
candidate generation, RRF fusion, final selection, or trust admission?

This audit is independent of the Voyage Context 4 production work. It diagnoses the current
baseline so the next intervention targets the stage that actually loses the answer.

## Why the old metric is insufficient

The 2026-08-17 query set deliberately labels every chunk from an answering memo as relevant. That
choice avoided letting an embedder select its own labels, but it makes complete gold infeasible
when the source has more chunks than the final context budget. It also left one current query
pointing at `recall-full-suite-takes-12-minutes.md`, which is now explicitly superseded by
`full-suite-takes-31-minutes-not-12.md`.

The new immutable input is
`docs/preregistrations/2026-09-13-memory-queries-source-gold.json`, SHA256
`06E5CFB2A345D3108EE5AE9E2D0BC2CD74FBA455D46F56658BF496B2447E088F`. It preserves every legacy
`relevant_ids` value, adds one source level gold label per answerable query, and redirects only the
known superseded source. It contains the same 50 queries, 22 answerable and 28 unanswerable.

Source gold answers a diagnostic question: did retrieval find the correct memo? It is not a claim
that every chunk from that memo answers the query. Essential fact coverage remains a separate
follow up after this audit identifies the loss boundary.

## Frozen protocol

1. Pin one certified generation and one committed source checkout.
2. Run all 50 queries once through the production MCP reasoning entry point with graph expansion
   off and the fast retrieval profile.
3. Reuse the exact served query vector to fetch the top 100 dense candidates and top 100 lexical
   candidates from the same generation.
4. Reconstruct unweighted RRF with constant 60 at per leg candidate pools 20 and 100. The pool 20
   arm reproduces the production candidate boundary. The pool 100 arm measures deeper fusion
   without pretending it is the current serving order.
5. Report source hit rates at 5, 10, 20, 50, and 100 for dense, lexical, their oracle union, and
   reconstructed RRF for both candidate pools. Also report trusted served source hit at 5 and
   unanswerable abstention.
6. Classify every answerable query into exactly one boundary:
   `reachable_fused_top10`, `selection_loss_fused_11_to_20`,
   `fusion_loss_from_leg_top20`, `deep_candidate_only_21_to_100`, or
   `candidate_generation_miss_at_100`.

Candidate identity is exposed only when both `RECALL_BENCHMARK_PIN=1` and
`RECALL_BENCHMARK_RETRIEVAL_LEG_AUDIT=1` are set. The production default remains off.

## Predictions made before measurement

1. Source level RRF hit at 10 will be 18 to 21 of 22. The prior complete chunk metric made 18
   queries look incomplete, but most should already retrieve at least one chunk from the correct
   source.
2. The oracle union at 100 will contain the gold source for at least 20 of 22 queries.
3. At least three queries will be recoverable from a retrieval leg but lost before RRF top 10.
4. Lexical top 20 will uniquely recover at least two queries that dense top 20 misses. These
   technical memory questions often preserve exact identifiers and operation vocabulary.
5. Trusted served source hit at 5 will be lower than raw RRF source hit at 5, because trust can
   abstain or remove a current low confidence chunk even when the correct source was retrieved.

## Decision rules

1. If RRF source hit at 10 is at least 18 of 22, retire complete chunk set recall as evidence that
   base candidate generation is the dominant live failure. Proceed to source scoped document
   expansion and essential fact coverage.
2. If at least five queries are in `fusion_loss_from_leg_top20` or
   `selection_loss_fused_11_to_20`, preregister a ranking intervention. A reranker is licensed only
   for candidates already inside the union.
3. If at least three queries are `candidate_generation_miss_at_100`, prioritize hierarchical
   source retrieval or late interaction. Do not widen the existing flat pool and call that a new
   mechanism.
4. If the live successor for the suite duration query is absent from the union at 100, treat corpus
   freshness or source identity as a blocker before interpreting aggregate retrieval quality.
5. Graph relation authoring remains closed regardless of this result. It reopens only under its
   separately registered candidate headroom gate.

## Reproduction command

The result record will replace `<committed-checkout>` and `<source-commit>` with the immutable
values created by the preregistration commit, and `<generation-id>` with the generation verified
immediately before the run.

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/<committed-checkout>'
$env:RECALL_SOURCE_COMMIT='<source-commit>'
python -m scripts.run_live_retrieval_leg_audit --query-set docs/preregistrations/2026-09-13-memory-queries-source-gold.json --output docs/results/2026-09-13-live-retrieval-leg-source-gold.json --generation-id <generation-id>
```
