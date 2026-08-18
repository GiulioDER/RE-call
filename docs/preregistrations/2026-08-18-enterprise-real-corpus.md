# Pre registration: EnterpriseRAG real corpus bundle selection

Date: 2026-08-18

## Scope

This is a real corpus follow up to the six case synthetic bundle benchmark. It is an offline
retrieval and evidence selection measurement, not a production launch gate. The corpus files are
local benchmark data with these SHA256 identities:

| file | SHA256 |
| --- | --- |
| `.benchdata/enterprise-rag-v1.0.0/questions.jsonl` | `F9524B9157CD43AAE36B99333A124738804306EA6D07F332D49FAA6D3D147905` |
| `.benchdata/enterprise-rag-v1.0.0/all_documents.zip` | `9D1174928696AD08BC15F3F104739519DE633C1605A4EC2034E0E3C0087BC5CD` |

The document archive is filtered to the 722 distinct document ids referenced by the 500 questions.
The filtered index contains 723 documents. The 500 question set contains 40
`intra_document_reasoning` questions, 93 questions with multiple expected documents, and 30
questions with no expected document ids.

## Fixed implementation and environment

* Embedder: local `hashing` embedder. No hosted embedding, generation, reranking, or judge call.
* Store: local Docker pgvector, table `bench_five_arm_enterprise`, tenant
  `five-arm-enterprise`.
* Retrieval: `k=8`, `candidate_k=20`, threshold `0.219`.
* Trust: explicit benchmark calibration with `TrustPolicy.development()` because the calibration
  is not certified and is not generation bound. Every result must therefore be reported as
  degraded and cannot be used to approve production serving.
* Document expansion: at most 2 sources and 8 chunks per source.
* Structural expansion: radius 2 with at most 2 sources and 8 chunks per source.
* Evidence bundles: at most 8 items and 2 documents. Answer slots use the preregistered JSONL
  labels in `benchmarks/enterprise_real_answer_slots.jsonl`.

## Arms

The five arms are the same as the synthetic preregistration:

1. `current_retrieval`
2. `document_grouping`
3. `structural_expansion`
4. `answer_slots`
5. `bundle_beam`

The answer slot and beam arms are measured only on labeled questions. Unlabeled questions are
recorded as not measured for those arms rather than silently treated as passing.

## Predictions

1. On the 40 intra document questions, structural expansion will not reduce complete document
   recall relative to current retrieval, and will improve at least one question where current
   retrieval misses an expected document.
2. On labeled questions, answer slots and bundle beam will reduce selected partial or forbidden
   evidence relative to structural expansion, while retaining complete slot coverage on at least
   80% of answerable labeled questions.
3. On the 10 labeled unanswerable questions, answer slots and bundle beam will abstain more often
   than current retrieval and document grouping. Any arm that answers more than 20% of these
   questions is not safe to promote.
4. Structural expansion and beam selection will add measurable selection latency. A latency win
   is not required, but p95 must be reported.
5. No arm will be promoted from this run because trust remains degraded. A live decision requires
   a certified, generation bound calibration and a separate serving latency measurement.

## Exploratory retrieval only result already measured

Before this protocol was written, an exploratory 500 question retrieval-only run reused the same
filtered index. It used no labels, so the slot and beam arms were not measured:

| arm | complete document sets | false positives | mean ms | p95 ms | trust |
| --- | ---: | ---: | ---: | ---: | --- |
| current retrieval | 129/500 | 30 | 29.7603 | 58.9199 | degraded 500 |
| document grouping | 104/500 | 30 | 29.6783 | 59.3465 | degraded 500 |
| structural expansion | 104/500 | 30 | 44.9327 | 86.3616 | degraded 500 |

This result is evidence against making the current two document grouping policy the default on
this corpus. It is not a certified answer quality result.

## Measurement result

Measured after the protocol and label set were committed. The fixed run used `candidate_k=20`:

| arm | questions | complete document sets | complete slots | false positives | mean ms | p95 ms | trust |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| current retrieval | 500 | 129 | not applicable | 20 | 27.4790 | 52.9516 | degraded 500 |
| document grouping | 500 | 104 | not applicable | 20 | 27.3449 | 52.7959 | degraded 500 |
| structural expansion | 500 | 104 | not applicable | 20 | 41.3551 | 70.3541 | degraded 500 |
| answer slots | 50 labeled | 10 | 0/40 answerable | 0/10 unanswerable | 44.4978 | 97.0845 | degraded 50 |
| bundle beam | 50 labeled | 10 | 0/40 answerable | 0/10 unanswerable | 167.5987 | 351.1791 | degraded 50 |

The 10 answer slot and beam document completions are the 10 unanswerable labels, where an empty
expected document set is treated as complete by the runner. They must not be read as answer quality.
Both selection arms abstained on all 40 answerable labels because the hashing retriever did not
surface their expected documents. An annotation coverability audit also found that only 65 of 109
v1 labeled slots had all of their terms in one indexed chunk. The labeled slot result is therefore
diagnostic only and does not validate the selection arms.

The candidate pool sensitivity addendum was then measured with `candidate_k=200`:

| arm | complete document sets | false positives | mean ms | p95 ms |
| --- | ---: | ---: | ---: | ---: |
| current retrieval | 109/500 | 20 | 45.9701 | 86.7251 |
| document grouping | 87/500 | 20 | 44.8189 | 83.7271 |
| structural expansion | 87/500 | 20 | 59.1977 | 107.9661 |
| answer slots | 0/40 answerable slots | 0/10 unanswerable | 57.5530 | 94.3944 |
| bundle beam | 0/40 answerable slots | 0/10 unanswerable | 170.4684 | 310.8782 |

The sensitivity check falsified its prediction: widening the candidate pool did not improve recall
and increased latency. The real corpus evidence says not to promote document grouping, structural
expansion, answer slots, or bundle beam as a serving default yet. The next valid experiment is a
gold-document-conditioned selector test with independently repaired labels, followed by a
certified calibration and a serving benchmark.

## Candidate pool sensitivity addendum

Added before the next measurement on 2026-08-18. The same indexed corpus, hashing embedder,
threshold, and five arms will be rerun with `candidate_k=200` instead of `candidate_k=20`.
This is an exploratory sensitivity check because the first labeled run showed a retrieval ceiling
before answer slot selection. The prediction is that current retrieval will recover more expected
document sets at the wider pool, and that structural expansion will only improve cases whose gold
document is present in the wider initial candidate set. This addendum does not change the launch
criteria or convert the development calibration into certified evidence.
