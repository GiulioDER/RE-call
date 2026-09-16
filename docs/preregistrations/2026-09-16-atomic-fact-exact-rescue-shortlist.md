# Atomic fact exact rescue shortlist

Status: preregistered before implementing or timing the masked maximum candidate path.

## Purpose and evidence boundary

Measure whether the dense-preserving atomic rescue can select exactly the same sixth parent as the
existing full atomic ranking without sorting every atomic view. The candidate computes all cosine
scores, excludes every atomic view whose parent is already present in dense top five, and returns
the maximum remaining parent with the existing deterministic source, parent ordinal, and view
ordinal tie ordering.

This is a latency and semantic-equivalence experiment. It reuses the frozen 31-row blind query pool
only as realistic query vectors and dense exclusions. It does not re-estimate retrieval quality,
tune a trigger, add another rescue slot, or authorize a serving change. A pass authorizes an
off-by-default aggregate production shadow of this exact selector.

## Frozen lineage and inputs

| Field | Frozen value |
| --- | --- |
| Tenant | `memory` |
| Generation | `gen_e5c95bffed8c41bb9c05680325fb8d60` |
| Calibration | `cal_aa2d53e5d051418c8d7b28534b862afc` |
| Pipeline | `57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86` |
| Corpus | `0587af4766daf313067eb07a72f7489765ff4b0192a96bf9c0d1142b24d5c0e7` |
| Embedding profile | `voyage-context-4-v1` |
| Vector dimension | 1,024 |
| Production chunks | 11,376 |
| Private query pool SHA256 | `97f77c71c1feb278b9d5297511e8b4fb913002208eaae2b3bbd2dc65d578fd18` |
| Query rows | 31 |

The runner must rebuild the atomic view matrix from this exact active generation and refuse a pool
hash, row count, lineage, dimension, manifest object, source root, vector count, nonfinite value, or
duplicate dense parent mismatch. Raw questions, sources, candidate identities, scores, and row-level
outcomes remain private. Publish only aggregate counts, timing distributions, hashes, and lineage.

## Frozen arms

For each query, compute the current generation's dense top five outside the timed region and reuse
one normalized query vector for both arms.

1. `full_sort_reference`: compute every atomic cosine score, sort every view by descending score
   followed by source, parent ordinal, and view ordinal, deduplicate parents, then take the first
   parent absent from dense top five.
2. `exact_masked_max`: compute the same score vector, mask all views whose parent is present in
   dense top five, find the highest remaining score, then choose the smallest source, parent
   ordinal, and view ordinal among exact score ties.

The candidate must not use an approximate index, score threshold, reduced view set, gold label,
query-specific tuning, cached result, or changed numeric precision.

## Frozen correctness checks

1. Selected source and parent ordinal must match the reference on all 31 real queries.
2. Selected view score must match the reference exactly after conversion to Python `float`.
3. Repeating the candidate selection must return the same identity and score on every query.
4. A deterministic adversarial fixture must prove dense-parent masking, duplicate-view parent
   handling, and source, parent ordinal, and view ordinal tie ordering.
5. Both arms must complete with zero nonfinite scores, missing candidates, shape errors, or
   duplicate-parent errors.

## Frozen timing protocol

Build and normalize the matrix, embed queries, compute dense top five, and touch all arrays before
timing. Time only selection from an already normalized matrix and query vector, including score
matrix multiplication, exclusion masking, deterministic tie resolution, and candidate creation.

Run five untimed warmup queries. Then run 20 repetitions of all 31 queries for each arm, 620 timed
samples per arm. Alternate which arm runs first by query and repetition parity. Use a monotonic
nanosecond clock. Report sample count, median, p95, p99, maximum, reference-to-candidate p95 ratio,
and total elapsed time. Do not remove outliers.

Document and query embedding run on VPS2 under the shared
`/home/sentiment/recall-repos/.locks/embed.lock`. Before launch, both the lock and process list must
show no competing embedder. Use a user systemd scope with `MemoryMax=8G`, `MemorySwapMax=0`,
`CPUQuota=250%`, nice level 15, four embedding threads, and one embedding process.

The run is an isolated read-only shadow. It must not modify the active generation, calibration,
production route, trust policy, or served evidence.

## Prediction and decision

I predict `PROMISING_EXACT_RESCUE_SHORTLIST`. I expect exact selection equivalence on every real
query, candidate p95 at or below 25 milliseconds, candidate p99 at or below 50 milliseconds, and a
reference-to-candidate p95 speedup of at least four times.

Return `PROMISING_EXACT_RESCUE_SHORTLIST` only if:

1. Every frozen integrity, lineage, correctness, and determinism check passes.
2. Candidate p95 is at most 25 milliseconds.
3. Candidate p99 is at most 50 milliseconds.
4. Reference p95 divided by candidate p95 is at least 4.0.
5. The active production generation and its full binding are unchanged after the run.

Otherwise return `STOP_EXACT_RESCUE_SHORTLIST`. A stop rejects this exact masked maximum
implementation, but it does not invalidate the already measured retrieval-quality gain or close a
future approximate-index implementation.
