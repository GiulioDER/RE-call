# Calibrated semantic graph reranking quality result

This is the live result for the preregistration in
`docs/preregistrations/2026-09-10-calibrated-graph-rerank-quality.md`. The preregistration was
committed as `468720cc` before the first live request and was not edited afterward.

## Verdict

The quality gate failed to show the predicted gain. The calibrated reranker produced exactly the
same trusted evidence ordering as graph off on all 250 paired requests in the frozen run. The
candidate is not promoted from this result.

## Run identity

* Candidate implementation: `ab688757`
* VPS2 serving commit during measurement: `468720cc4e6b0db27ea9054eb8b1d2826f77376a`
* Active generation: `gen_b02a44a99917424ba3bd8011280e3712`
* Query set: `docs/preregistrations/2026-08-17-memory-queries.json`
* Query set SHA256: `63d290a61189758a88b47bfed981ae1331d87fc004b752d632c1f5b58f5aa192`
* Queries: 50, of which 22 were answerable and labelled
* Recorded requests: 250 per arm, after one warmup pass per arm
* Raw artifact: `C:\Users\gde00\AppData\Local\Temp\recall-live-calibrated-rerank-quality-20260910.json`
* Raw artifact SHA256: `7b25dcfe3815509cb87ec69385742b8267275e19cce8a5f22de65cbe1c4415e1`

The raw runner field is a hash `chunk_id`, while the frozen labels are `source:ordinal` values.
For the preregistered hit and MRR calculations, each trusted evidence item was therefore matched
using its returned source and ordinal, the only common identity format in the artifact and labels.

## Retrieval quality

| Metric | Graph off | Calibrated one hop | Delta |
| --- | ---: | ---: | ---: |
| Answerable hit at five | 0.500000 | 0.500000 | 0.000000 |
| Answerable MRR | 0.469697 | 0.469697 | 0.000000 |
| Empty trusted evidence | 3/22 | 3/22 | 0.000000 |

The paired hit deltas were 22 ties, with no wins or losses. The paired MRR deltas were also 22
ties. The deterministic paired bootstrap 95 percent intervals were `[0.000000, 0.000000]` for
both hit at five and MRR. The paired two sided sign flip p values were `1.0` for both metrics.

All 250 pairs had identical evidence sets, outcome, refusal reason, and trust state. Both arms
reported `trusted` for every recorded request. Both arms reported `abstained` because this
evidence assembly run had no answer provider. Unsupported claim rate and answer correctness are
therefore not measured.

## Graph activation

The one hop arm reported `ready` on all 250 requests. Graph candidates were discovered on 5 of
250 requests, with 40 candidates total. All 40 candidates were accepted by the graph relation
policy, but zero became new trusted evidence. The remaining graph requests were gated by
`no_trusted_seed` on 135 requests and `graph_gate_not_met` on 110 requests. The graph policy
fingerprint was
`335c500d17ad4901b887558a1c56342914cc51d8d7fcd0d7a2dacb2ab6a6acce`.

This means the live run exercised calibrated candidate scoring on only the five requests that
discovered candidates, and those candidates did not change the returned trusted evidence. It is
not evidence that the reranking formula improves activated graph cases with new trusted items.

## Latency context

| Metric | Graph off p50 / p95 ms | Calibrated one hop p50 / p95 ms |
| --- | ---: | ---: |
| Client observed | 1047.434 / 1375.782 | 1093.678 / 1328.545 |
| Total server | 477.986 / 598.258 | 480.807 / 593.761 |

Latency is descriptive only here. The preregistered quality decision failed because the primary
hit at five delta was zero, not because of a latency regression.

## Follow up

VPS2 was restored to the pre experiment serving commit `bccf4d80` after the measurement and the
serving verification handshake passed. The isolated branch remains available at
`codex/calibrated-graph-rerank` with the implementation and preregistration commits.
