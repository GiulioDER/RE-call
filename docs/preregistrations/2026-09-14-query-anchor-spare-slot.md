# Pre-registration: query anchor spare slot admission

Date: 2026-09-14.

## Question

Can direct lexical compatibility between a query and the first guarded proposal recover exact gold
evidence while rejecting controls whose requested anchor is absent from the corpus?

## Prior evidence and scope

The complete `extractive_strict_v1` holdout is consumed development data. It produced zero exact
span gains and two control activations because source agreement did not establish query support.
Cheap relevance, reranker, entailment, margin, and fusion scores have already failed as general
near-miss answerability classifiers. This experiment does not reopen that claim. It tests the
narrower proposition that an admitted source must contain the rare lexical anchors supplied by the
query.

## New untouched pool

Build a new pool from the same two memory source roots and the same deterministic exact-span rules
used by `2026-09-14-guarded-spare-slot-extractive-pool.json`. Exclude every source present in:

1. the 72-query source-admission development trace;
2. the 500-query guarded spare-slot fresh pool;
3. the consumed 500-query extractive holdout;
4. repository index and execution-log files in the existing builder skip list.

Use seed `query-anchor-spare-slot-v1`. Freeze 80 answerable queries and 80 matched controls. Return
`INSUFFICIENT_POOL` without retrieval if fewer than 80 unique answerable questions remain.

Each answerable item freezes the canonical source, raw source SHA256, chunk ordinal, normalized
exact answer span, and answer-span SHA256 before retrieval. Each control inserts one deterministic
lowercase alphabetic nonce into the positive question subject using the template
`According to the <nonce> revision, <positive question with lowercase first character>`. Construct
the nonce from the seed, pair index, and source digest. It must contain 10 through 14 alphabetic
characters, must not share a fixed human-readable prefix across controls, and must be absent from
every candidate source. Controls freeze `NOT_FOUND` as the exact answer and have no gold source.

Sort the complete 160-query pool by the seeded SHA256 order and commit its digest before policy
development.

## Development features

Use the consumed 250-pair extractive cohort only. Re-run its queries against the pinned generation
used by the completed result and inspect only the first proposal from the original
`guarded_spare_slot` ordering. Do not apply `extractive_strict_v1`.

Tokenize the query and every candidate chunk with lowercase matches of `[a-z0-9]+` and retain
unique tokens of length at least four after removing the exact stopword list frozen in the feature
implementation. For each retained query token, calculate source document frequency across the
retrieval pool. Sort anchors by ascending document frequency, then token SHA256, and keep at most
three. For the proposed source, join every chunk from that source present in the retrieval pool and
measure exact token coverage of the selected anchors.

The development artifact may contain numeric features and gold booleans but must remain outside the
repository. Compare only monotone, interpretable rules based on selected-anchor count, covered
anchor count, coverage fraction, minimum anchor document frequency, proposal lane, and proposal
position. The policy must inspect the first proposal only and append at most one item. It must not
use the known control nonce generator, answer span, source label, query identifier, or answerability
label at serving time.

Before holdout retrieval, append and commit the exact selected rule, development counts, prediction,
and promotion gate below the frozen marker. Do not inspect new holdout queries, sources, spans, or
retrieval outcomes during policy development.

## Primary metrics and gate

For answerable queries, report base and candidate exact-span coverage, source hit, gains, losses,
and additions containing the frozen span. For controls, report activations and added items. Report
complete base-prefix preservation, maximum additions per query, and exact-span precision across all
additions.

Promotion requires at least one exact-span gain, zero exact-span losses, zero source-hit losses,
zero control activations, complete base-prefix preservation, and no more than one addition per
query. A retrieval pass does not authorize active serving or prove general near-miss answerability.

<!-- frozen_above -->

## Pool freeze

Built 2026-09-14 before development feature capture. The registered exclusions left 116 eligible
sources and the builder froze 80 answerable plus 80 matched alphabetic-anchor controls. The pool
SHA256 is `6dd9485b12bf0c88d166cc29114cd03ada1e732e47b11592a4725e91d7324e68`.

Rebuild from the repository root with:

```powershell
python scripts/build_query_anchor_spare_slot_pool.py --source-root "recall=C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-recall\memory" --source-root "sentiment-agent=C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-progetto-sentimental\memory" --trace docs/results/2026-09-13-live-source-admission-trace-capture.json --old-pool docs/preregistrations/2026-09-14-guarded-spare-slot-fresh-pool.json --consumed-pool docs/preregistrations/2026-09-14-guarded-spare-slot-extractive-pool.json --output docs/preregistrations/2026-09-14-query-anchor-spare-slot-pool.json
Get-FileHash docs/preregistrations/2026-09-14-query-anchor-spare-slot-pool.json -Algorithm SHA256
```

The pool is sealed. Do not inspect its individual questions, sources, answer spans, or retrieval
outcomes until the exact anchor policy is appended and committed below.

## Development recapture repair

Attempted 2026-09-14. Three 125-query shards completed. The fourth stopped at its 121st query
because the locally recomputed guarded ordering differed from the repeated live shadow ordering.
The new holdout remained sealed.

The repaired recapture must write a checkpoint after every query. An exact guarded-order parity
mismatch records only the query index, answerability class, and mismatch status. It must not retain
features or labels from that row and must continue. Every other error remains fatal. Policy fitting
requires at least 450 exact parity rows across the 500-query consumed cohort and at least five
first-proposal exact-span additions. Otherwise return `INSUFFICIENT_DEVELOPMENT_PARITY` and do not
run the new holdout.
