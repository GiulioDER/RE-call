# AML standard entailment screen result

Date: 2026-09-19

## Verdict

Select `B0_raw`. Keep the standard QNLI entailment judge OFF for the AML benchmark.
No continuation, executable task screen, AML Smoke, or AML Full run is authorized by this result.

The preregistered present stop fired after the complete 12 task paired present block. All mechanism
checks passed, but every decisive present gate failed.

| metric | B0 raw | B3 standard entailment | delta |
|---|---:|---:|---:|
| complete coverage at 100 | 0.9167 | 0.4167 | -0.5000 |
| mean reciprocal rank | 0.4996 | 0.3592 | -0.1404 |
| hit in returned budget | 1.0000 | 0.8333 | -0.1667 |
| Search p95, ms | 530.6 | 37,141.2 | +36,610.6 |
| Search p50, ms | 343.4 | 31,035.4 | +30,692.0 |

B3 attempted and completed the exact pinned judge on all 12 evaluated requests. It judged 2,091
candidates, accepted 1,430, rejected 661, and produced zero empty response abstentions. Exact
provider, model, revision, threshold, cardinality, corpus, query, generation, and served commit
checks passed. Fallbacks, invalid permutations, and sparse failures were zero.

The valid attempt issued 36 Search requests: 24 deciding present requests and 12 nondeciding absent
B0 requests that completed before the present stop was applied. No absent B3 Search was issued.
The absent B0 artifact is preserved as nondeciding provenance. No Add, Delete, embedding,
indexing, or sparse backfill operation was performed.

## Identities

* RE-call served commit: `0e616d03ed16d745f9d51f5d885b3a3d6117a911`
* AMB measurement harness commit: `b681a5796c7b5e5d5b3a8e54b53442386e372074`
* AMB stop selector commit: `8f4ce3959af3b2a6b54c072da61b92e411fa6cc9`
* model: `cross-encoder/qnli-distilroberta-base`
* model revision: `7dd04ee0a6040c06fb381ad7edcb8585f4d937fd`
* threshold: `0.5`
* model artifact digest: `8448f917fa256648b8fcfa78aa5dc1cecfa5d1a778a5d6812108d9ee53d47fef`
* dependencies: sentence-transformers 6.1.0, torch 2.14.0+cpu, transformers 5.17.0,
  tokenizers 0.23.2
* present namespace: `amb-clean-7b54c04e-fe45b46-retrieval-r2-present`
* present corpus SHA-256: `63de5f323603d5a32a5945bef89ec2b59e4866afa8b73905fd8d130cc350880e`

## Excluded attempt

The first attempt at `0e616d03-696169f-screen` called port 18006 while the dedicated service was
correctly configured on 18005. It failed before any Search request. Its sole service log remains
immutable on VPS2 with SHA-256
`4eeeb6cb32f916925d4373937cc86ee7115f33f2960faedef43ef38ec5e069c6`.

## Limitation

The stop rule deliberately prevented measuring B3 on the absent and adjacent targets. Therefore
this result does not claim that QNLI can never remove noise. It establishes the decision needed for
this benchmark: the shipped standard configuration is ineligible because it severely damages
present evidence and exceeds the latency ceiling by about 7.4 times.

