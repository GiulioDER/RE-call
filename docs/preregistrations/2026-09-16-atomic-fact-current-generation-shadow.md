# Atomic fact current generation production shadow

Status: preregistered before building the current generation auxiliary index or running any
candidate retrieval.

## Purpose

Measure whether the atomic fact representation that passed the 22 row pilot is operationally
small and fast enough to run beside the current production memory generation. This experiment is
an isolated read only shadow. It cannot change retrieval, trust, abstention, or evidence returned
by the production MCP service.

This run has two deliberately separate interpretations:

1. The engineering measurements are new evidence about the current production corpus.
2. Retrieval quality on the existing 50 query production set is diagnostic only because that set
   has already been evaluated and cannot provide prospective confirmation.

No result from this run can authorize serving atomic candidates. A later sealed prospective set
from sources absent from the 22 row pilot is required for that decision.

## Frozen lineage and inputs

The run is pinned to tenant `memory` and the production lineage observed before preregistration:

| Field | Frozen value |
| --- | --- |
| Generation | `gen_83393e5c58524eecb9e390cb367e4d61` |
| Calibration | `cal_86a67f38755c45f38456d6b2445767cc` |
| Pipeline | `57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86` |
| Corpus | `5357c7fb736dfd8add6182e41f97932a8bd6c103daae0f9bf8d984f113f87c42` |
| Embedding profile | `voyage-context-4-v1` |
| Vector dimension | 1,024 |
| Query file | `docs/preregistrations/2026-09-13-memory-queries-source-gold.json` |
| Query file SHA256 | `06e5cfb2a345d3108ee5ae9e2d0bc2cd74fba455d46f56658bf496b2447e088f` |
| Query rows | 50 |

The seven source roots, manifest byte verification, basename exclusions, UTF-8 BOM and NUL
handling, atomic parser, rendered view format, and parent mapping are unchanged from the committed
22 row pilot. Any necessary correction to those rules requires a committed addendum before the
affected measurement.

The runner must refuse a lineage, query hash, dimension, source root, manifest byte, parent text,
or vector count mismatch.

## Frozen arms

1. `dense`: production generation dense top 20 using the one frozen query embedding.
2. `atomic`: cosine search over every atomic Context 4 view, followed by deterministic parent
   deduplication and truncation to 20 parents.

Query embedding is performed once per query and reused by both arms. There is no RRF, graph
expansion, selector, reranker, reader, rendering change, threshold, or tuned fallback in this run.

The optimized atomic matrix implementation must match the committed scalar reference ordering on
a deterministic test fixture before measurement. Ties are ordered by source, parent ordinal, and
view ordinal exactly as in the pilot.

## Frozen execution controls

All document embedding runs on VPS2 under the shared
`/home/sentiment/recall-repos/.locks/embed.lock`. Before launch, both the lock and process list must
show no competing embedder. The process runs in a user systemd scope with `MemoryMax=8G`,
`MemorySwapMax=0`, `CPUQuota=250%`, nice level 15, and four embedding threads.

The auxiliary artifact and per query rows are private and mode 0600 outside the repository. Only
aggregate counts, timings, hashes, and metrics may be committed. Production routing and the active
generation must remain unchanged.

Timing uses a monotonic clock. The atomic matrix is loaded before query timing. Five deterministic
warmup rankings use the first five query vectors and are excluded. Each of the 50 query vectors is
then ranked five times. Candidate metrics use the first repetition; latency uses all 250 measured
rankings. Ranking latency includes matrix scoring, deterministic ordering, parent deduplication,
and construction of the top 20 parent identities. It excludes query embedding and artifact load.

Resident memory is the process maximum RSS increase from immediately before loading the atomic
artifact to immediately after loading and touching the full matrix and metadata. Disk size is the
sum of the private vector and metadata artifacts.

## Frozen measurements

Engineering measurements:

1. Manifest objects, ordinary chunks, sources with views, zero view sources, and atomic views.
2. Build wall time, artifact bytes, vector count, and vector dimension.
3. Atomic ranking latency p50, p95, p99, and maximum across 250 repetitions.
4. Atomic artifact resident RSS increase.
5. Query errors, nonfinite scores, duplicate parents, and repeat determinism failures.
6. Rank one movement, dense and atomic overlap at ranks 1, 5, 10, and 20.
7. Number of diagnostic query rows for which every labelled gold source has zero atomic views.

Consumed set diagnostics, reported but excluded from the production decision:

1. Dense and atomic gold source reach at ranks 1, 3, 5, 10, and 20.
2. Rank one gold gains and losses.
3. Per row candidate identities retained only in the private artifact.

## Predictions and decision gates

Prediction: the current corpus will produce fewer than 7,500 atomic views, occupy no more than 64
MiB on disk and 96 MiB additional resident memory, and rank at p95 no slower than 100 ms. I expect
the zero view source fraction to remain below 12 percent. These are capacity predictions, not
quality predictions.

The engineering shadow passes only if all of the following hold:

1. Every frozen lineage and integrity check passes.
2. There are zero build errors, query errors, nonfinite scores, duplicate parents, and repeat
   determinism failures.
3. Atomic view count is at most 7,500.
4. Private artifact disk size is at most 64 MiB.
5. Additional resident memory is at most 96 MiB.
6. Atomic ranking p95 is at most 100 ms and p99 is at most 150 ms.
7. Zero view sources are at most 12 percent of included sources.
8. The production generation and serving checkout are unchanged after the run.

If every gate passes, the result authorizes implementation of an off by default, fail open, low
rate live shadow that records only aggregate or hashed diagnostics and never changes served
evidence. It does not authorize active retrieval.

If a resource or latency gate fails, stop before live integration and preregister an optimized
storage or search design. If an integrity or determinism gate fails, stop and treat the artifact as
invalid. Regardless of the verdict, the consumed 50 query quality diagnostics cannot be used to
tune rendering, fusion, or thresholds.

## Prospective quality track

In parallel with engineering validation, future eligible sources not present in the 22 row pilot
will accumulate into a sealed confirmation set. A query writer may see the fact content but not the
source title, headings, field label, rendered view, or retrieval output. Exact answer span and gold
source are frozen before retrieval. No prospective quality result is included in this current
generation engineering run.
