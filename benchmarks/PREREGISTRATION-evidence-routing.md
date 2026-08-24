# Pre registration: evidence cost and query routing

Date: 2026 08 24

This record fixes the measurement before the first public evidence cost or routing result.

## Fixed configuration

1. Reader tokenizer: `cl100k_base` from `tiktoken==0.13.0`, with the encoding hash recorded in
   `tokenizer_metadata`.
2. Evidence budgets: 128, 256, 512, 1024, 2048, 4096, and 8192 tokens.
3. Query classifier: `query-class-v1`, using deterministic precedence.
4. Routing policy: `routing-v1`, with fast for lookup and list, quality for temporal and status,
   and quality with structural expansion for causal, comparative, and entity.
5. Unknown queries use fast and no expansion.
6. Operational measurements are reported separately and cannot enter retrieval quality aggregates.

## Fixed execution environment and analysis

1. The measurement host is VPS2. The run records the host fingerprint, CPU count, memory limit,
   Python version, package lock hash, model identifiers, retrieval configuration, and provider
   metadata before the first arm. A host or configuration mismatch invalidates the run.
2. Embedding and indexing use the existing single process and bounded batch safeguards. No second
   indexing or embedding process may run concurrently.
3. The primary routing comparison is paired active routing versus the unchanged baseline on the
   same question identities. The evidence curve is analyzed separately at each fixed budget.
4. Binary paired outcomes use exact McNemar tests. Confidence intervals use paired bootstrap
   resampling with 10,000 resamples. Query class comparisons use Benjamini Hochberg correction at
   q = 0.05 across the eight preregistered classes.

## Predictions

1. Exact rendered evidence tokens will never exceed the applied budget, and the median and p95
   token counts will be monotonic across the fixed budget ladder.
2. Increasing the evidence budget from 128 through 2,048 tokens will reduce false refusals and
   improve citation support for answerable questions, with diminishing gains at larger budgets.
3. The deterministic classifier will route lookup and list queries to fast, temporal and status
   queries to quality, and causal, comparative, and entity queries to quality with structural
   expansion. Unknown queries will use fast without expansion.
4. Active routing will satisfy the preregistered noninferiority gates overall, improve or remain
   neutral on temporal, status, causal, comparative, and entity questions, and keep the routing
   evidence cost within ten percent of baseline at matched quality.
5. Staged indexing and snapshot loading will improve readiness or startup measurements without
   changing retrieval quality aggregates, because they are operational measurements only.

## Measurement population and analysis

1. Corpora are the existing public benchmark corpora selected before the run. No question is
   removed after classification, and no gold answer, category label, or generated answer is
   available to the classifier.
2. Hardware, Python version, package lock, model identifiers, retrieval configuration, and
   provider metadata are recorded in each artifact. Startup and indexing measurements record the
   host and process configuration used for that run.
3. Quality comparisons use paired question identities. Accuracy, refusal, false refusal, and
   citation metrics are reported per budget and per query class. Confidence intervals and paired
   significance tests are computed by the existing benchmark analysis tooling, with multiplicity
   correction across query classes.
4. An observed within budget point is not treated as a budgeted quality result. A budgeted point
   requires the per question record to identify the applied evidence budget.

## Promotion gates

Routing can become the default only when paired evaluation shows overall noninferiority within one
percentage point, no class regression above three percentage points after correction, no false
refusal increase above one percentage point, no more than twice baseline p95 latency, and no more
than ten percent evidence token increase at matched quality.

Every published artifact must retain the per question records, tokenizer metadata, routing decision,
configuration, and operational claim family.
