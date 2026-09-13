# Live document expansion essential fact result

Date measured: 2026-09-13.

## Verdict

The existing source scoped document expansion configuration is a no go on the certified Voyage 4
memory corpus. It added no essential fact coverage to the trusted candidate pool. Document bundle
ordering then reduced complete queries from 14 to 13 and covered facts from 16 to 15.

This closes the configuration of two selected sources, eight chunks per source, and a five item
bundle. It does not support making document expansion or structural expansion a default. The next
highest ROI target is calibrated source conditioned trust admission, not another expansion or
ranking mechanism.

Every measured value below is recorded in
`docs/results/2026-09-13-live-document-expansion-essential-facts.json`, SHA256
`287529F99B47114559B9745A486CDDFAE3006ACE2B2258C1117FD137C0DD89A9`, and reproduced by the
command in this report.

## Immutable lineage

* Source commit: `7e6a0d5ee9872feb40c0c9545f54792191b8357a`.
* Remote checkout: `/home/sentiment/recall-repos/document-expansion-7e6a0d5e`.
* Generation: `gen_56dce932a4444411b488bc860fea4fb7`.
* Calibration: `cal_b458bf825c6d47469c5511b3a5038dff`.
* Query set SHA256: `06E5CFB2A345D3108EE5AE9E2D0BC2CD74FBA455D46F56658BF496B2447E088F`.
* Essential fact label SHA256: `45C38731E138F6AED425635B78EE79B692E142F5EF040E0EFFFEB042AD47186B`.
* Evaluation set: 50 queries, including 22 answerable and 28 unanswerable.
* Gold: 25 essential facts across 22 answerable queries.
* Serving constraints: fast profile, graph off, candidate pool 20 per leg, five evidence items.

The active memory generation changed before this run to
`gen_ca914376ed4547b1b9d3ee64ae8168ec` and strict retrieval refused it with `LINEAGE_MISMATCH`.
The experiment therefore pinned the earlier certified generation used by the retrieval leg audit.
A preflight request confirmed its calibration remained certified and trusted.

## Results

| Arm | Complete queries | Facts covered | Gold source hit | Context precision | Unanswerable answers | Mean retrieval ms | P95 retrieval ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 14/22 | 16/25 | 17/22 | 0.4677 | 4/28 | 500.900 | 699.629 |
| Document expansion, retrieval prefix | 14/22 | 16/25 | 17/22 | 0.5217 | 4/28 | 486.304 | 830.319 |
| Document expansion, document bundle | 13/22 | 15/25 | 16/22 | 0.6102 | 4/28 | 486.304 | 830.319 |
| Structural expansion, document bundle | 13/22 | 15/25 | 16/22 | 0.6271 | 4/28 | 485.204 | 813.840 |

The full trusted pool after document expansion remained 14 of 22 complete and covered 16 of 25
facts. The full trusted structural pool was identical. Expansion therefore found no additional
trusted essential fact that a different five item selector could recover.

Document bundle ordering lost query index 10, `does master allow merge commits`. Baseline included
the gold source `recall/commit-signing-squash-merge.md`. The two document limit retained two
adjacent sources and removed the third, correct source. No query gained complete coverage.

## Remaining fact misses

Eight answerable queries remained incomplete in the baseline and both trusted expansion pools:

| Index | Query topic | Gold source trusted | Diagnostic reading |
| ---: | --- | --- | --- |
| 2 | current full suite duration | yes | the source was present, but the current duration fact chunk was absent |
| 9 | stale branch looked regressed | no | the result was an empty trusted bundle |
| 11 | guard always passes | no | adjacent guard memos displaced the gold source |
| 14 | Redis backend decision | no | the result was an empty trusted bundle |
| 15 | maker rebate materiality | no | one adjacent backlog chunk survived |
| 16 | paid API work | yes | the source was present, but its policy fact chunk was absent |
| 19 | clean Git status versus current HEAD | yes | the source was present, but its substrate fact chunk was absent |
| 21 | gate 1774 cannot settle | no | the result was an empty trusted bundle |

The preceding retrieval leg audit shows why this is primarily an admission problem. Dense top 20
already contained every gold source. Several essential chunks had raw cosines below the global
calibration threshold, including the current suite duration fact at 0.474, Redis decision at 0.484,
maker rebate fact at 0.416, and gate 1774 settlement fact at 0.316. Query index 11 differed: its gold
chunk scored 0.558 at dense rank 13, but the expansion policy considered only two sources selected
from the initial five item result.

The four unanswerable false retrieval answers were unchanged across every arm. They are query
indices 22, 30, 42, and 46. The earlier source gold report was corrected after its first scorer
mistakenly used the answer provider outcome instead of the evidence bundle decision.

## Prediction outcomes

1. Baseline complete coverage was predicted at 12 to 17 of 22. Observed 14. Confirmed.
2. Document bundle was predicted to gain at least two complete queries and three facts. It lost one
   query and one fact. Refuted.
3. Retrieval prefix was predicted not to outperform document bundle. It scored 14 versus 13.
   Refuted.
4. Structural bundle was predicted not to outperform document bundle. Both scored 13. Confirmed.
5. Treatments were predicted to add no more than one unanswerable answer. They added zero.
   Confirmed.
6. Document bundle context precision was predicted not to fall more than 0.10. It rose by 0.1425.
   Confirmed, but the precision gain accompanied lower fact coverage.
7. Document bundle p95 was predicted within three times baseline. The ratio was about 1.19.
   Confirmed.

The promotion rule does not fire because fact coverage did not improve. The retirement rule fires
because neither diagnostic trusted pool gained two complete queries. A fact aware selector is not
licensed because the fact was absent from the full trusted pool, not merely left outside the final
bundle.

## Next experiment

I recommend a source conditioned trust audit before implementation. Aggregate the existing top 20
chunks into source candidates using rank, cross leg agreement, number of supporting chunks, and
margin to the next source. Then measure whether a calibrated source confidence can safely contribute
to each chunk's own admission score.

The treatment must remain independently calibrated per chunk. It must not copy a trusted verdict
from one chunk to its neighbors. A reasonable candidate is an out of fold logistic gate over the
original chunk cosine and source evidence features. Evaluate essential fact coverage and the 28
unanswerable controls. This directly targets the observed pattern where the right source is known
but the right fact chunk falls below a corpus wide cosine threshold.

I would not spend the next cycle on a general reranker. Only two of 22 source gold queries were
fusion losses at rank 10, while eight of 22 lacked complete facts after trust. I would also keep
hierarchical retrieval and late interaction closed because no gold source was absent from the union
at rank 100.

## Reproduction

Gold coverability validation:

```powershell
python -m scripts.validate_memory_essential_facts --labels docs/preregistrations/2026-09-13-memory-essential-facts.json --recall-root C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-recall\memory --sentiment-root C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-progetto-sentimental\memory
```

Live treatment:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/document-expansion-7e6a0d5e'
$env:RECALL_SOURCE_COMMIT='7e6a0d5ee9872feb40c0c9545f54792191b8357a'
python -m scripts.run_live_document_expansion_audit --query-set docs/preregistrations/2026-09-13-memory-queries-source-gold.json --fact-labels docs/preregistrations/2026-09-13-memory-essential-facts.json --output docs/results/2026-09-13-live-document-expansion-essential-facts.json --generation-id gen_56dce932a4444411b488bc860fea4fb7
```

Verification before measurement:

```powershell
python -m pytest tests/test_live_retrieval_leg_audit.py tests/test_live_document_expansion_audit.py tests/test_document_bundles.py tests/test_reasoning_api.py tests/test_reasoning_embedding_reuse.py tests/test_reasoning_deterministic_caches.py tests/test_live_graph_performance_attribution.py -q
python -m ruff check recall_mcp/service.py scripts/run_live_tty_graph_precision.py scripts/run_live_retrieval_leg_audit.py scripts/run_live_document_expansion_audit.py scripts/validate_memory_essential_facts.py tests/test_live_retrieval_leg_audit.py tests/test_live_document_expansion_audit.py
python -m mypy --explicit-package-bases --ignore-missing-imports recall_mcp/service.py scripts/run_live_tty_graph_precision.py scripts/run_live_retrieval_leg_audit.py scripts/run_live_document_expansion_audit.py scripts/validate_memory_essential_facts.py
```

The focused regression result was 73 passed on 2026-09-13. Ruff and mypy passed on the same date.
