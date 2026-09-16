# Atomic fact dense-preserving rescue on the blind source census

Status: preregistered before embedding the census queries or reading any dense or atomic ranking.

## Purpose and evidence boundary

Measure whether one atomic fact candidate is a better sixth candidate than ordinary dense depth
on the 31 accepted rows from the blind source census.

The census missed its construction gate by one row, so this is an exploratory diagnostic. A pass
can justify engineering a faster atomic candidate path and a larger future confirmation set. It
cannot authorize production routing, change serving, or establish a general retrieval uplift.

## Frozen lineage and inputs

| Field | Frozen value |
| --- | --- |
| Tenant | `memory` |
| Generation | `gen_55487101e0d2421594b31f5c9519070f` |
| Calibration | `cal_d86150126d3f46c8aefa910c3a0529a6` |
| Pipeline | `57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86` |
| Corpus | `ac4aab3af34f68e9f168af75f9219883a363fd555012b96d888d18a4a0344eaa` |
| Embedding profile | `voyage-context-4-v1` |
| Vector dimension | 1,024 |
| Private query pool SHA256 | `97f77c71c1feb278b9d5297511e8b4fb913002208eaae2b3bbd2dc65d578fd18` |
| Answerable query rows | 31 |

The active RE-call memory service reported this certified lineage and a fresh 11,370 chunk corpus
immediately before preregistration. The seven source roots, manifest byte checks, atomic parser,
rendered view construction, and deterministic tie ordering are unchanged from the current
generation shadow.

The runner must refuse a pool hash, row count, lineage, dimension, source hash, answer span hash,
gold parent, manifest object, or vector count mismatch.

## Frozen arms

Embed each question once with Voyage Context 4 and reuse that vector for every arm.

1. `dense5`: the production generation's first five dense parents.
2. `dense6`: the production generation's first six dense parents.
3. `atomic5`: the first five distinct parents from cosine ranking over all atomic views.
4. `dense5_atomic1`: preserve `dense5` in its original order, then append the highest-ranked
   atomic parent whose `(source, ordinal)` identity is absent from `dense5`.

Every arm has deterministic source, parent ordinal, and view ordinal tie ordering. The main
comparison is `dense5_atomic1` against `dense6`, which gives both arms exactly six parent
candidates. There is no fusion score, threshold, graph expansion, reranker, selector, reader,
query rewrite, or use of gold during ranking.

## Frozen measurements

For each arm, report exact gold parent and gold source reach. Report ranks one, three, and five for
the native dense and atomic rankings, plus rank six for `dense6` and `dense5_atomic1`.

For the paired rank-six comparison, report:

1. exact parent gains, losses, ties, and net gain
2. gold source gains, losses, ties, and net gain
3. how often the appended atomic parent differs from dense rank six
4. precision of the appended atomic parent among rows missed by `dense5`
5. the atomic top-20 rescue ceiling among rows missed by `dense5`

Raw questions, answer spans, source identities, candidate identities, scores, and row-level
outcomes remain private. Publish only aggregates and hashes.

## Frozen execution controls

Document and query embedding run on VPS2 under the shared
`/home/sentiment/recall-repos/.locks/embed.lock`. Before launch, both the lock and process list must
show no competing embedder. Use a user systemd scope with `MemoryMax=8G`, `MemorySwapMax=0`,
`CPUQuota=250%`, nice level 15, four embedding threads, and one embedding process.

The run is an isolated read-only shadow. It must not modify the active generation, calibration,
production route, trust policy, or served evidence.

## Prediction and decision

I predict `PROMISING_EXPLORATORY_ATOMIC_RESCUE`. I expect the atomic sixth candidate to recover at
least two more exact gold parents and at least two more gold sources than dense rank six, with no
more than one paired loss on either metric.

Return `PROMISING_EXPLORATORY_ATOMIC_RESCUE` only if:

1. Every frozen integrity and lineage check passes with zero query, vector, duplicate-parent, or
   nonfinite-score errors.
2. `dense5_atomic1` has exact parent net gain of at least two over `dense6`, with at most one exact
   loss.
3. `dense5_atomic1` has gold source net gain of at least two over `dense6`, with at most one gold
   loss.
4. The active production generation and its full binding are unchanged after the run.

Otherwise return `STOP_ATOMIC_RESCUE_LANE`. A stop closes this atomic representation as a current
retrieval improvement path unless a materially different representation, not a fusion-weight or
threshold retune, supplies new evidence.
