# Atomic fact auxiliary retrieval exhaustive 22 source pilot

Status: predicted, no construction aggregate, retrieval result, or auxiliary embedding measured.

Registered 2026-09-16 after the preceding 250 source protocol stopped during deterministic pool
construction with exactly 22 eligible unique-query sources. The population size is known. No source
identity, query, fact, construction family, retrieval result, or embedding result has been inspected.

## Purpose and scope

Test whether production aligned atomic fact vectors can improve exact span and gold source retrieval
on the complete currently available source-disjoint population.

This is an exploratory pilot. Its maximum sample is 22 and was known before the prediction below.
It cannot authorize serving, establish a precise effect size, or replace a prospective confirmatory
set. Its purpose is to decide whether building a prospective set is worth the delay.

## Frozen pool construction

Use `scripts/run_production_atomic_fact_fresh_audit.py` at commit `8f0e2bf3`, with the same source
roots, four exact-hash exclusion inputs, seed `atomic-fact-fresh-audit-v1`, date exclusions, field
separation, fact eligibility, parent mapping, and production aligned rendering registered in
`2026-09-16-production-aligned-atomic-fact-fresh-audit.md`.

Set `--count 22`. This selects the complete available population reported by the earlier frozen
run. Write the row-level pool privately under
`C:\Users\gde00\.codex\evals\atomic-fact-pilot-22-2026-09-16`. Publish only its SHA256 and aggregate
metrics.

The construction gate requires:

1. exactly 22 source-disjoint rows and exact input hashes
2. exact source hashes at audit time
3. one complete gold view at the frozen parent ordinal for every row
4. at least two construction families, each with 100 percent gold coverage
5. no zero-view source and no rendered view above 800 characters
6. the same duplicate, collision, row growth, character growth, and parent crowding limits as the
   preceding protocol

If construction fails, return `STOP_ATOMIC_FACT_22_CONSTRUCTION` and do not embed.

## Frozen serving lineage and baseline

The current production baseline is:

1. generation `gen_cd269b86b7364e25843ed7d7dda7c419`
2. calibration `cal_1f12dcbfa2b64e10a1548051cd2a5fcd`
3. pipeline `57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86`
4. corpus `5fc44dffc89b0e4f2d769bb4b52c4f3671a1d3c9ee2b30462a1b4aa0f33d24fc`
5. embedding profile `voyage-context-4-v1`

Collect the exact current dense top 20 for every query with graph expansion, reranking, source
conditioning, and learned selection disabled. Abort on lineage mismatch or fewer than 20 unique
dense candidates.

## Frozen auxiliary corpus and embeddings

Build auxiliary views from the seven production memory roots named by the deployed refresh
manifest: `sentiment-agent`, `recall`, `ai-boost-av-safety`, `ai-boost-cad`, `steel`, `cca-demos`,
and `agent-memory-bench`. Use the exact production aligned extraction and rendering above. Each
source is one ordered Voyage document group. Each view maps to its unchanged ordinary parent chunk.

Run `voyage-context-4` at 1,024 dimensions on VPS2 under the shared embedding lock, an 8 GB memory
limit, zero swap, a 250 percent CPU quota, four threads, and lowered priority. Save vectors and
row-level results privately. Query embeddings use the same model and dimension.

For each query, rank all auxiliary views by cosine, then deduplicate by parent `(source, ordinal)`,
keeping the highest scoring view and deterministic source then ordinal tie breaks. Return the top
20 parent chunks.

## Frozen comparison arms

Report three arms:

1. `dense`, the current production dense top 20
2. `atomic`, the parent-deduplicated auxiliary top 20
3. `equal_rrf`, reciprocal rank fusion of `dense` and `atomic` with constant 60 and equal weights

RRF is computed over parent identities. Ties break by the best contributing rank, then dense rank,
then source and ordinal. No score normalization or learned weight is allowed.

For every arm report exact span and gold source reach at ranks 1, 3, 5, 10, and 20. For atomic and
RRF relative to dense, report changed rank one rows, exact and gold gains and losses, and changed
selection precision.

## Frozen decision

Return `PROMISING_ATOMIC_FACT_PILOT` only if either atomic or equal RRF satisfies every condition:

1. exact rank one improves by at least two rows
2. gold rank one improves by at least two rows
3. exact rank one losses are at most one
4. gold rank one losses are at most one
5. changed rank one precision is at least 0.50 for both exact and gold
6. exact and gold reach at rank 20 do not decline
7. at least one of exact or gold reach at rank 20 improves by at least one row

Choose the qualifying arm with more exact rank one, then more gold rank one, then more exact rank 20,
then prefer atomic over RRF. If neither qualifies, return `STOP_ATOMIC_FACT_RETRIEVAL_PILOT` and do
not tune fusion, rendering, eligibility, or Voyage grouping on these rows.

A promising result authorizes only prospective gold accumulation and an isolated product shadow.
It does not authorize serving.

## Prediction

I predict the construction gate will pass and equal RRF will return
`PROMISING_ATOMIC_FACT_PILOT`. The mechanism targets a measured limitation: current dense top 20
missed exact chunks even after every tested selector preserved or damaged the fixed candidate pool.
Atomic views change candidate membership by embedding one complete fact rather than an 800 character
multi-fact chunk. Equal RRF preserves the original query ranking while admitting complementary
parents, so I expect at least two exact and two gold rank one gains with no more than one loss each.
