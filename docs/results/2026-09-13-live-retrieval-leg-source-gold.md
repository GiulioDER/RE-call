# Live source gold retrieval leg audit

Date measured: 2026-09-13.

## Verdict

Flat candidate generation is not the main retrieval quality bottleneck on this set. Dense retrieval
found the correct source for all 22 answerable queries by rank 20. Production style RRF found 20 of
22 by rank 10. The trusted serving result contained the correct source for only 17 of 22 queries.

The highest ROI next experiment is therefore source scoped document expansion followed by
essential fact coverage measurement. A general reranker, a wider flat candidate pool, lexical
weighting, hierarchical retrieval, late interaction, and further graph relation authoring are not
licensed by this result.

Every number in this report is reproduced by the command below and recorded in
`docs/results/2026-09-13-live-retrieval-leg-source-gold.json`, SHA256
`BD54D97765713CA21E6E29FF9CF06071DE4B6A936A2F28E9763764F8F4C34591`.

## Immutable lineage

* Source commit: `0c23a291ff6e45aa39c761d39b0c55badf80a5c0`.
* Remote checkout: `/home/sentiment/recall-repos/source-gold-leg-audit-0c23a291`.
* Generation: `gen_56dce932a4444411b488bc860fea4fb7`.
* Calibration: `cal_b458bf825c6d47469c5511b3a5038dff`.
* Pipeline fingerprint: `77c918cd93f9200b36e505ae874d49a1949f04a8a85ce5b8b72a8c135b472db7`.
* Query set: 50 queries, including 22 answerable and 28 unanswerable.
* Query set SHA256: `06E5CFB2A345D3108EE5AE9E2D0BC2CD74FBA455D46F56658BF496B2447E088F`.
* Retrieval profile: `fast`, graph expansion off, RRF constant 60.

## Results

| Arm | Hit at 5 | Hit at 10 | Hit at 20 | Hit at 50 | Hit at 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 20/22 | 20/22 | 22/22 | 22/22 | 22/22 |
| Lexical | 15/22 | 17/22 | 18/22 | 22/22 | 22/22 |
| Oracle union | 20/22 | 20/22 | 22/22 | 22/22 | 22/22 |
| RRF from each leg's top 20 | 19/22 | 20/22 | 20/22 | 22/22 | 22/22 |
| RRF from each leg's top 100 | 18/22 | 20/22 | 21/22 | 22/22 | 22/22 |

The trusted serving result contained the correct source for 17 of 22 answerable queries. Retrieval
abstained on 24 of 28 unanswerable queries. Four unanswerable queries received trusted evidence.
Answer generation was disabled, so this run measures retrieval and trust admission, not answer
wording or answer correctness.

## Post hoc evaluator correction

The first report incorrectly stated that all 28 unanswerable queries abstained. The scorer used the
top level reasoning outcome, which is always `abstained` when the answer provider is disabled. The
retained trusted evidence bundles show four false retrieval answers, at query indices 22, 30, 42,
and 46. The corrected scorer treats an empty trusted evidence bundle as abstention and a nonempty
bundle as an answer. Raw query rows are unchanged. Red proof node
`retrieval-leg-abstention-01` reproduces the old false count.

## Loss boundaries

* `reachable_fused_top10`: 20 queries, indices 0 through 10 except 11, then 12 through 20.
* `selection_loss_fused_11_to_20`: zero queries.
* `fusion_loss_from_leg_top20`: two queries, indices 11 and 21.
* `deep_candidate_only_21_to_100`: zero queries.
* `candidate_generation_miss_at_100`: zero queries.

The two fusion losses are:

1. Index 11, `what is wrong with a guard I wrote that always passes`. The gold source
   `recall/guards-that-cannot-fail.md` was dense rank 13 and lexical rank 23.
2. Index 21, `why can gate 1774 never turn green`. The gold source
   `sentiment-agent/redacted-94559ce9ad31` was dense rank 17
   and lexical rank 41.

No query was lexical only at rank 20. Four were dense only at rank 20, indices 10, 11, 16, and 21.
The other 18 were present in both legs by rank 20. Increasing the fusion pool from 20 to 100 made
rank 5 worse, from 19 to 18 correct sources, and did not improve rank 10.

## Trusted serving misses

Five answerable queries did not expose the gold source in trusted evidence:

| Index | Short description | Dense rank | Lexical rank | Serving result |
| ---: | --- | ---: | ---: | --- |
| 9 | branch looked regressed | 3 | 1 | `corpus_gap`, no evidence |
| 11 | guard always passes | 13 | 23 | three trusted chunks from adjacent sources |
| 14 | Redis backend decision | 1 | 1 | `corpus_gap`, no evidence |
| 15 | maker rebate materiality | 2 | 16 | one trusted chunk from another source |
| 21 | gate 1774 cannot settle | 17 | 41 | `corpus_gap`, no evidence |

This split matters. Three misses are admission failures even though the right source was already
retrieved. Two retain adjacent evidence while omitting the gold source. Only indices 11 and 21 are
also fusion losses at rank 10.

The freshness control passed. The live successor
`recall/full-suite-takes-31-minutes-not-12.md` was dense rank 1, lexical rank 7, and present in
trusted evidence.

## Prediction and decision rule outcomes

1. RRF hit at 10 was predicted at 18 to 21 of 22. Observed 20 of 22. Confirmed.
2. Union hit at 100 was predicted at least 20 of 22. Observed 22 of 22. Confirmed.
3. At least three losses before RRF rank 10 were predicted. Observed two. Refuted.
4. At least two lexical only recoveries at rank 20 were predicted. Observed zero. Refuted.
5. Trusted serving hit was predicted below raw RRF hit at 5. Observed 17 versus 19. Confirmed.

Decision rule 1 fires because RRF hit at 10 is at least 18. Complete chunk set recall should no
longer be used as evidence that base candidate generation is the dominant problem. The next test is
source scoped document expansion with essential fact labels.

Decision rule 2 does not fire because only two, not five, queries were lost in selection or fusion.
Decision rule 3 does not fire because no gold source was absent from the union at 100. Decision rule
4 passes because the live suite duration successor is present. The graph lane remains closed.

## Next experiment

For each answerable query, select the best source already present in the top retrieval results, then
assemble neighboring chunks only within that source under the existing evidence token budget. Score
essential fact coverage, unsupported context, trusted source hit, and abstention. Use the five
trusted serving misses as the diagnostic slice, but keep all 50 queries as the decision set.

The experiment should compare:

1. Current trusted chunks.
2. Source scoped ordinal expansion around the highest ranked chunk from each selected source.
3. A small whole document summary or fact capsule only if ordinal expansion exceeds the token
   budget.

The primary gate should be essential fact coverage, not complete chunk recall. Safety constraints
remain unchanged: every expanded chunk receives independent trust evaluation, and expansion must
not cross source boundaries or follow authored graph relations.

## Reproduction

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/source-gold-leg-audit-0c23a291'
$env:RECALL_SOURCE_COMMIT='0c23a291ff6e45aa39c761d39b0c55badf80a5c0'
python -m scripts.run_live_retrieval_leg_audit --query-set docs/preregistrations/2026-09-13-memory-queries-source-gold.json --output docs/results/2026-09-13-live-retrieval-leg-source-gold.json --generation-id gen_56dce932a4444411b488bc860fea4fb7
```

Focused verification before measurement:

```powershell
python -m pytest tests/test_live_retrieval_leg_audit.py tests/test_live_graph_performance_attribution.py tests/test_reasoning_api.py tests/test_reasoning_embedding_reuse.py tests/test_reasoning_deterministic_caches.py -q
python -m ruff check recall_mcp/service.py scripts/run_live_tty_graph_precision.py scripts/run_live_retrieval_leg_audit.py tests/test_live_retrieval_leg_audit.py
python -m mypy --explicit-package-bases --ignore-missing-imports recall_mcp/service.py scripts/run_live_tty_graph_precision.py scripts/run_live_retrieval_leg_audit.py
```

The focused pytest result was 60 passed on 2026-09-13. Ruff and mypy passed on the same date.
