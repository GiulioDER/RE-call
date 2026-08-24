# EnterpriseRAG non reasoning retrieval and reranker experiment

**Date:** 2026-08-18  
**Status:** preregistered, execution pending  
**Benchmark:** EnterpriseRAG Bench release 1.0.0  
**Fixed split seed:** 366

## Question

Can a fixed retrieval configuration using Voyage rerank 2.5 improve document recall and exact
document set coverage on the frozen dev and confirmation slices without a material increase in
invalid extra documents, context latency, or measured provider cost?

## Runtime boundary

The retrieval and answer runners receive only the question text, question id, indexed documents,
and configuration. They do not read expected document ids, answer facts, gold answers, or
competitor answers. Those fields are read only by the official evaluator after an answer file has
been generated. The run uses no correction and skips citation stripping. Parallelism is one.

## Arms and grid

The primary comparison is no reranker against `voyage:rerank-2.5`. I will test candidate pools of
100, 200, and 400 and final values of 5, 8, and 12. I will test lexical sparse retrieval and both
lexical plus SPLADE only when the required SPLADE model and isolated index are available. The
answer model and prompt stay fixed for every answer side comparison.

The first reproduction is the five question project related dev slice at k 8, candidate pool 200,
Voyage embeddings, lexical sparse retrieval, and extractive output. I will repeat each selected
capture three times to check stability. A reranker answer run is allowed only for retrieval arms
that improve the paired retrieval result on the held out confirmation slice.

## Predictions

1. Voyage reranking will reproduce a positive project related dev result at k 8 and candidate pool
   200, with at least one recovered expected document and no retrieval losses on the five question
   screen.
2. On the confirmation slices, the Pareto candidate will be k 8 with candidate pool 200 or a
   smaller pool. Candidate pool 400 will add latency and invalid extras without a proportional
   recall gain. k 12 will improve recall in some categories but will add enough invalid extras or
   context length that it will not dominate k 8.
3. The largest paired retrieval gains will be in project related and completeness. Basic and
   constrained will be mixed. Conflicting information will require a separate answer policy even
   if retrieval recall improves.
4. Lexical plus SPLADE will be tested only if its isolated index has complete source coverage and
   stable query encoding. I predict that it will help some semantic and project related questions,
   but may add invalid extras when fused with lexical retrieval.
5. Retrieval improvement will not be treated as an answer score improvement. The answer side will
   be run only for retrieval winners, with the answer model and prompt held fixed, and may be
   rejected if correctness or completeness does not improve.

## Promotion gate

No arm is promoted from this experiment. A candidate can continue to confirmation only after a
positive paired result, stable repeated captures, no citation or document id mismatch, no
info not found regression, and recorded latency and cost. Full benchmark consideration requires a
homogeneous 500 question official evaluation and comparison with the 61.03 top five threshold.
