# AML clean reranker retrieval result

Date: 2026-09-19

## Decision

Select `B0_raw` over the tested Voyage `rerank-2.5` candidate. Keep graph OFF.
`B1_raw_rerank` failed the preregistered retrieval gate, so the 72-cell executable screen and the
1,020-cell confirmation were correctly not run. No scarce AML hosted run was spent.

This decision applies to the tested raw Hosted retrieval configuration and `rerank-2.5`. It does
not claim that another reranker cannot help. After the result was measured, the current Voyage
model table was checked and found to describe preview `rerank-3` as its highest-accuracy model and
the recommendation for most applications. The best-current-reranker question therefore remains
open and requires a separate preregistered comparison. The `rerank-2.5` result remains unchanged.

## Reproducible identities and population

* RE-call served commit: `7b54c04e84a4120ec385cbf4b87b1d1e9cde94f2`
* AMB harness commit: `fe45b4643d99ee41f2ef74ea1d0faad43f162db1`
* Embedding identity: `voyage-context-4-v1`
* Reranker identity: provider `voyage`, model `rerank-2.5`
* Conditions: present, absent, superseded, contradictory, adjacent
* Population: 34 queries per condition, three captures per query and arm
* Search requests: 1,020
* B1 reranker attempts and completions: 510 and 510
* Fallbacks and invalid permutations: 0 and 0
* Dense embedding passes: five, one per condition, with B1 reusing each B0 tenant

## Retrieval gate

| Present metric | B0 raw | B1 raw plus rerank | Delta or ratio |
| --- | ---: | ---: | ---: |
| Mean reciprocal rank | 0.322645 | 0.286091 | -0.036553 |
| Complete coverage at 100 | 0.941176 | 0.911765 | -0.029412 |
| Source-session recall | 1.000000 | 1.000000 | 0.000000 |
| Search p95 ms | 368.293 | 1107.133 | 3.0061x |

B1 failed present MRR improvement, coverage-at-100 nondecline, and the requirement that p95 stay
strictly below three times baseline. It passed source-session recall nondecline and the absolute
5,000 ms latency limit. B1 MRR was also lower in absent, superseded, contradictory, and adjacent,
so the loss was not confined to present.

The frozen reranker cost estimate is USD 1.544478325 for 510 requests. This is an estimate based
on candidate text volume, characters divided by four, and the preregistered USD 0.05 per million
tokens price. It is not a provider invoice. Exact billed cost and account allowance state were not
available. No DeepSeek or GPT 4o mini task cost was incurred.

## Graph and mechanism verification

The graph preflight found zero authored corpus relations, zero eligible corpus relations, and zero
served relation rows. Graph was therefore ineligible and was not silently added.

Every expected B1 Search invoked and completed Voyage `rerank-2.5`. Every full pretruncation output
was a duplicate-free permutation of its input. Provider and model identities, served commit,
candidate width 100, RRF constant 60, corpus hashes, and query hashes passed their gates.

Three of 510 B0 and B1 paired captures differed in pre-rerank candidate character volume while
candidate counts, query hashes, corpus hashes, and configuration identities matched. The affected
captures were one present `ts-glob-hidden` capture and two contradictory captures. The variation
is consistent with approximate retrieval tie variation across service restarts and remains an
explicit limitation. It does not rescue B1, which lost every condition-level MRR comparison.

## Immutable artifacts

* [Retrieval selector](2026-09-19-aml-clean-reranker-retrieval-selection.json), SHA-256
  `278ea73aa38947bdca2d1d6efe1b52754e633a3a446c0edadf4137f91a1df9af`
* [Graph preflight](2026-09-19-aml-clean-reranker-graph-preflight.json), SHA-256
  `324bc3a1ceacc56d5131510feecd39fab3205d2333a4262d1dcba24e788114ca`
* VPS2 raw results and logs:
  `/home/sentiment/agent-memory-bench-clean-reranker-2dc798d9/results/aml-clean-reranker-v1/7b54c04e-fe45b46-retrieval-r2`
* VPS2 graph artifact:
  `/home/sentiment/agent-memory-bench-clean-reranker-2dc798d9/results/aml-clean-reranker-v1/7b54c04e-fe45b46-graph-preflight-r2/preflight.json`

The failed pre-service wrapper attempt remains excluded and preserved under the paths recorded in
AMB preregistration amendment 1. It performed no service setup, Add, embedding, reranking, Search,
or task execution.

## Official readiness

The selected configuration for this completed comparison is `B0_raw`. The final local benchmark
candidate is not settled until the separately preregistered `rerank-3` check resolves. No official
AML run is authorized. Written organizer confirmation that Voyage embeddings are permitted in the
open-source track is also still required. If Voyage embeddings are disallowed, block both official
runs and redesign first.

## Post-result model discovery

Voyage's current [reranker documentation](https://docs.voyageai.com/docs/reranker) lists preview
`rerank-3` as highest accuracy and recommends it for most applications. The current
[pricing page](https://docs.voyageai.com/docs/pricing) lists the same USD 0.05 per million token
rate used by `rerank-2.5`, with a 200 million token allowance for `rerank-3`. This information was
recorded after the frozen `rerank-2.5` result and does not reinterpret its measurements.
