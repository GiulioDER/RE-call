# Atomic fact release confirmation retrieval

Status: preregistered after the 96 row pool was frozen and before embedding any confirmation
question or reading any dense or atomic ranking.

## Purpose

Decide whether dense-five plus one atomic rescue has enough prospective retrieval benefit to enter
the production-readiness path. This is the first confirmation-sized quality measurement for the
feature. The earlier 22 and 31 row results selected the representation and rescue design; they are
not pooled with this set and do not contribute to the gates below.

A pass authorizes implementation of a generation-bound, off-by-default production shadow. It does
not by itself change served results. Serving requires a separately frozen live shadow and rollback
gate.

## Frozen lineage and inputs

| Field | Frozen value |
| --- | --- |
| Tenant | `memory` |
| Generation | `gen_dff506e12f494965af9f109671a99e63` |
| Calibration | `cal_6171177aeb614d288baa4e28600caca1` |
| Pipeline | `57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86` |
| Corpus | `522e9d1c506142d8a6a5a1f5be9073ff9409132d356e29f61fbf8e7d6d2044a0` |
| Chunks | 11,385 |
| Embedding profile | `voyage-context-4-v1` |
| Vector dimension | 1,024 |
| Private pool SHA256 | `414b441fd95ccc0de7a6ede329515941c09f8aef8e9fc8e20ccbdfc1d7514f0f` |
| Answerable rows | 96 |

The runner must refuse a pool hash, row count, query identifier, source hash, answer span hash,
gold parent, generation, calibration, pipeline, corpus, chunk count, dimension, source root,
manifest object, or atomic vector count mismatch.

## Frozen arms

Embed each question once with Voyage Context 4 and reuse that vector for both arms.

1. `dense6`: the first six distinct parents from the certified generation's dense retrieval.
2. `dense5_atomic1`: preserve the first five dense parents in their original order, then append
   the exact highest-scoring atomic parent whose `(source, ordinal)` is absent from dense top five.

The atomic candidate uses the already measured exact masked maximum selector. It must agree in
identity and score with the full deterministic atomic sort for all 96 rows. There is no trigger,
threshold, fusion weight, second atomic slot, graph expansion, reranker, reader, query rewrite, or
gold-dependent ranking.

## Frozen measurements

For each arm, measure exact gold parent and gold source reach at rank six. Also report dense reach
at ranks one, three, and five, and the number of times the atomic candidate differs from dense rank
six.

For exact parent and gold source separately, report paired gains, losses, ties, net gain, absolute
percentage-point change, and the one-sided exact McNemar binomial probability under equal gain and
loss probability. When there are `g` gains and `l` losses, calculate
`sum(Binomial(g + l, 0.5) >= g)`; return one when there are no discordant rows.

Report gain and loss counts by production root as a diagnostic. Do not add a root-specific gate
after seeing the outcomes.

Raw questions, answer spans, source identities, candidate identities, scores, and row-level
outcomes remain private. Publish only aggregate metrics and hashes.

## Frozen quality gates

Return `PASS_ATOMIC_RELEASE_CONFIRMATION` only if all conditions hold:

1. Exact parent net gain is at least six rows, exact losses are at most one, and the one-sided
   paired probability is at most 0.05.
2. Gold source net gain is at least six rows, gold losses are at most one, and the one-sided paired
   probability is at most 0.05.
3. Exact masked maximum matches the full-sort reference in identity and score on all 96 rows.
4. Every candidate score is finite, every arm has six distinct parent identities, all 96 rows are
   measured, and the active generation remains unchanged.

Otherwise return `STOP_ATOMIC_RELEASE_LANE`. Do not tune and rerun on this pool.

The six-row minimum is a 6.25 percentage-point prospective uplift. Combined with the paired
probability gate, it prevents a statistically favorable but operationally trivial promotion. The
loss ceiling protects dense depth six rather than allowing a large two-way churn to masquerade as
net improvement.

## Frozen execution controls

Document and query embedding run only on VPS2, as one process under the shared
`/home/sentiment/recall-repos/.locks/embed.lock` flock. Immediately before launch, verify the lock
holder and process list. Run through a transient systemd scope with `MemoryMax=8G`,
`MemorySwapMax=0`, `CPUQuota=250%`, nice level 15, and four embedding threads. A missing lock,
resource-policy mismatch, second embedding process, or lineage mismatch is a hard refusal.

Private vectors and row-level results remain outside the repository with restricted permissions.
The production serving route must remain unchanged.

## Production path after a pass

After a quality pass, implement the exact selector behind an off-by-default generation-bound
production shadow. The shadow must load one matrix per active generation, preserve dense top five,
emit aggregate overlap and latency only, fail independently, and never alter served candidates.
Preregister load time, resident memory, p95 and p99 added latency, generation rollover, concurrency,
and failure isolation gates. Only a passing shadow may authorize an explicit serving flag with a
one-command rollback.

## Prediction

I predict a pass with at least eight exact and gold gains and no more than one loss for either
label. The consumed 31 row set produced four exact and four gold gains with zero losses; the 96 row
set was constructed from a much larger development-disjoint population and is large enough to
test whether that signal generalizes.
