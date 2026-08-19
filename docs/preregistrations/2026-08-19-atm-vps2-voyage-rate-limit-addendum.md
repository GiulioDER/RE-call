# ATM-Bench VPS2 Voyage rate-limit addendum

This addendum is committed after the first VPS2 attempt failed before writing
an artifact. The failure was an API rate-limit rejection, not a retrieval
measurement: `rerank-2.5` rejected the `candidate_k=200` traffic burst at the
project limit of 2,000,000 tokens per minute. No quality conclusion is drawn
from that attempt.

## Revised arm

The next VPS2 control keeps the same data, source snapshot, Voyage reranker,
reasoning environment, full question set, dense and lexical-hybrid arms, and
separate tenant tables. It changes only `candidate_k` from 200 to 100 so the
reranker request fits the observed project limit. The original `candidate_k=200`
prediction remains unchanged and is not retroactively replaced.

The revised run is expected to complete without a rate-limit failure. Before
measurement, I predict that the MiniLM hybrid arm will retain at least 95% of
its `candidate_k=200` complete-evidence Recall@10GT on the local full set, and
that Voyage reranking will improve answer hit@5 over the corresponding
unreranked arm by at least 2 percentage points. These are operational
predictions, not official ATM-Bench scores.

The runner will still set `RECALL_REASONING=1`,
`RECALL_REASONING_MODEL=openai/gpt-5-mini`, and the OpenRouter endpoint. The
current ATM retrieval runner records those settings but does not generate or
judge answers, so this artifact must not be described as an end-to-end
reasoning score.

## Execution record

* 2026-08-19: `candidate_k=200` attempt failed before artifact creation with
  Voyage `RateLimitError` at 2,000,000 TPM.
* 2026-08-19: revised `candidate_k=100` arm preregistered before rerun.
* 2026-08-19: `candidate_k=100` also failed before artifact creation with
  Voyage `RateLimitError` at 2,000,000 TPM.

## Second rate-limit correction

The `candidate_k=100` correction was insufficient because the ATM text items
are long enough to consume almost the complete two-million-token minute even
with 100 documents per request. I therefore preregister `candidate_k=25` for
the next attempt. This is the only change from the failed `candidate_k=100`
run. It is expected to complete without a rate-limit failure. I predict that
the MiniLM hybrid arm will achieve at least 0.70 complete-evidence Recall@10GT
and that the Voyage-reranked hybrid answer hit@5 will be at least 0.58 on the
full local set. These predictions are recorded before the next measurement.
