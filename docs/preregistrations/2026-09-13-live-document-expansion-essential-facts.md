# Live document expansion essential fact preregistration

Date registered: 2026-09-13.

## Question

Can the existing source scoped document expansion and document bundle features improve essential
answer fact coverage on the certified Voyage 4 memory corpus without materially weakening
abstention, context precision, or latency?

The 2026-09-13 retrieval leg audit found the gold source in dense top 20 for all 22 answerable
queries and in production style RRF top 10 for 20 of 22. Trusted serving exposed the gold source
for only 17 of 22. This makes the evidence assembly and trust boundary higher ROI than another
flat retrieval mechanism.

An earlier EnterpriseRAG result found that document grouping and structural expansion reduced
complete document coverage. Its own next valid experiment was a gold document conditioned selector
with independently repaired labels. This protocol performs that missing test on the live memory
corpus. It does not authorize a default from the earlier fixture result.

## Immutable inputs

The query set is
`docs/preregistrations/2026-09-13-memory-queries-source-gold.json`, SHA256
`06E5CFB2A345D3108EE5AE9E2D0BC2CD74FBA455D46F56658BF496B2447E088F`. It contains 50 queries,
including 22 answerable and 28 unanswerable.

The independent essential fact labels are
`docs/preregistrations/2026-09-13-memory-essential-facts.json`, SHA256
`45C38731E138F6AED425635B78EE79B692E142F5EF040E0EFFFEB042AD47186B`. They contain 25 facts for
the 22 answerable queries. Each fact is bound to the manually selected gold source and is satisfied
only when one chunk from that source contains at least its registered number of terms. The labels
are scoring inputs only. Retrieval, source selection, trust evaluation, and bundle selection cannot
read them.

All 25 labels are coverable in one chunk produced from their canonical source by the repository's
current `parse_document` and `chunk_text` functions. That property was checked before this protocol
was committed with:

```powershell
python -m scripts.validate_memory_essential_facts --labels docs/preregistrations/2026-09-13-memory-essential-facts.json --recall-root C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-recall\memory --sentiment-root C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-progetto-sentimental\memory
```

The validation returned 22 labels, 25 facts, and zero uncovered facts on 2026-09-13. The per source
hashes and supporting ordinals are emitted by the command for review.

## Frozen arms

All arms use the same committed source, generation, query set, Voyage 4 query vector, fast profile,
per leg candidate pool 20, graph off, item budget 5, and production trust policy.

1. `baseline`: current trusted retrieval order and current evidence prefix.
2. `document_retrieval`: `DocumentExpansionPolicy` with two selected sources and eight chunks per
   source, followed by the current retrieval ordered evidence prefix.
3. `document_bundle`: the same expanded trusted pool, followed by
   `EvidencePolicy(bundle_mode="document", max_documents=2)`.
4. `structural_bundle`: `StructuralExpansionPolicy` with two sources, eight chunks per source,
   radius two, and document bundle selection.

The experiment disables the policies' relational query trigger so every fixed query is measured.
This is a treatment assignment, not an oracle. Each policy selects sources only from its initial
retrieval. Every expanded chunk receives its own production trust verdict. The audit also records
the full trusted treatment pools so a miss can be classified as expansion failure or final bundle
selection failure. Pool coverage is diagnostic and is not a serving score because it exceeds the
five item context budget.

## Metrics

Primary metrics are complete essential fact queries and total essential facts covered. Secondary
metrics are gold source hit, gold source context precision, unanswerable answers, unanswerable
abstentions, mean retrieval latency, and p95 retrieval latency. Every query and every fact remains
visible in the artifact.

An unanswerable query counts as answered when its evidence bundle decision is `answer` or its item
list is nonempty. The top level reasoning outcome is forbidden as this metric because a disabled
answer provider makes that outcome `abstained` even when retrieval supplied trusted evidence.

## Predictions made before treatment measurement

1. Baseline complete essential fact coverage will be 12 to 17 of 22.
2. `document_bundle` will make at least two more answerable queries complete than baseline and will
   cover at least three additional facts.
3. `document_retrieval` will not outperform `document_bundle` on complete queries. Appending source
   chunks without changing final selection often leaves them outside the five item prefix.
4. `structural_bundle` will not outperform `document_bundle`. The current structural policy filters
   a relevance scoped source search to ordinal neighbors, so it can discard useful distant facts.
5. No treatment will add more than one unanswerable answer relative to baseline.
6. `document_bundle` context precision will not fall more than 0.10 absolute below baseline.
7. `document_bundle` p95 retrieval latency will remain at or below three times baseline.

## Decision rules

Promote `document_bundle` to a production candidate only if it gains at least two complete queries,
does not add more than one unanswerable answer, loses no more than 0.10 context precision, and stays
within three times baseline p95 latency.

If the diagnostic trusted pool gains at least two complete queries but its five item bundle does
not, work on fact aware selection rather than retrieval. If neither trusted pool gains at least two,
retire this document expansion configuration on this corpus. If fact coverage rises while source
hit does not, treat that as support for within source evidence assembly. If unanswerable answers
rise by more than one, reject default activation regardless of coverage.

This experiment cannot reopen graph relation authoring, a wider flat pool, lexical weighting,
general reranking, hierarchical retrieval, or late interaction. Their separate gates remain
unchanged.

## Reproduction command

The result report will replace `<committed-checkout>` and `<source-commit>` with this protocol's
commit and `<generation-id>` with the certified generation verified immediately before execution.

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/<committed-checkout>'
$env:RECALL_SOURCE_COMMIT='<source-commit>'
python -m scripts.run_live_document_expansion_audit --query-set docs/preregistrations/2026-09-13-memory-queries-source-gold.json --fact-labels docs/preregistrations/2026-09-13-memory-essential-facts.json --output docs/results/2026-09-13-live-document-expansion-essential-facts.json --generation-id <generation-id>
```
