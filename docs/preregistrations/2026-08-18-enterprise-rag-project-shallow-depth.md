# EnterpriseRAG project shallow depth expansion

**Date:** 2026-08-18
**Benchmark commit:** `d36685e273713975ee20299bbf1ab64165575b3c`  
**Question release SHA256:** `f9524b9157cd43aae36b99333a124738804306ea6d07f332d49faa6d3d147905`  
**Development slice:** `results/enterprise_rag/project_slices/dev.ids`, 17 project-related questions  
**Confirmation slice:** `results/enterprise_rag/project_slices/confirmation.ids`, 23 project-related questions

## Hypothesis

The raw `k=12` arm recovered relevant documents that were just below the existing eight-document
cutoff, but its four additional documents created too many invalid extras. A fixed shallow
expansion to nine or ten submitted documents may retain part of the recall gain while satisfying
the invalid-extra guardrail. This is a retrieval-depth experiment only and does not add reasoning,
query expansion, reranking, answer decomposition, or answer-policy changes.

The raw k12 confirmation result is hypothesis-generation evidence only. The confirmation slice is
not used to choose between k9 and k10.

## Arms

All arms use the same official index and runtime configuration:

* table `ber_voy_lex_12k_full`
* tenant `enterprise-rag-voyage-lexical-chunk12k-full`
* Voyage 4 Large embeddings
* lexical sparse retrieval
* `candidate_k=200`
* no reranker
* no reasoning arm
* identical question order and document release
* three repeated retrieval captures per question

The baseline submits `k=8` documents. Candidate A submits `k=9`. Candidate B submits `k=10`.
Both candidates preserve the existing hybrid ranking and only change the document cutoff.

## Measurements

For each arm, record mean document recall, exact document-set coverage, invalid extra documents,
paired per-question gains and losses, retrieval latency, embedding and lexical provider calls,
capture stability, and the complete runtime manifest. Runtime must not read gold fields.

The primary development metric is mean document recall. The guardrails are:

1. mean invalid extra documents must increase by no more than 2.0 versus k8;
2. mean recall must not be lower than k8;
3. repeated captures must be stable at 1.0 for the selected arm;
4. no question may fail due to malformed output or missing document ids.

Select the higher-recall candidate among k9 and k10 only if it passes every guardrail. If neither
passes, reject shallow expansion. If both pass with equal recall, select k9 because it submits
fewer documents. Do not use confirmation results to break a tie or choose a candidate.

## Confirmation and promotion

Run only the selected development candidate on the frozen 23-question confirmation slice. The
candidate passes confirmation only if its mean recall is no lower than the paired k8 baseline,
the invalid-extra increase remains no greater than 2.0, and it has no material stability or
latency failure. Do not run an answer-quality comparison unless retrieval confirmation passes.

This experiment cannot claim a leaderboard improvement. Any promotion requires a new homogeneous
500-question official evaluation using the frozen evaluator flags and comparison with the official
leaderboard snapshot.
