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

To be appended after the labeled run. Predictions above must not be edited.
