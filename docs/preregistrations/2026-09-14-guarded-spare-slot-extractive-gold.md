# Pre-registration: guarded spare-slot extractive gold

Date: 2026-09-14.

## Question

Can a stricter version of the guarded spare-slot rescue recover exact fact-bearing evidence on an
untouched memory holdout while avoiding activation on matched absent-identifier controls?

## Why this replaces the retired review

The prior fresh screen supplied no required facts before treatment and asked a human reviewer to
reconstruct an answer key from unfamiliar links and retrieved evidence. That instrument is retired
as `INVALID_REVIEW_INSTRUMENT`. Its aggregate source audit is development evidence only.

This experiment freezes an answer span copied from the canonical source before any retrieval run.
The primary quality metric is whether admitted evidence contains that exact span. No human label and
no model judge participates in the primary metric.

## Isolation boundary

Candidate sources must be absent from all of these inputs:

1. the 72-query source-admission development trace;
2. the 500-query guarded spare-slot fresh pool;
3. repository index and execution-log files named by the pool builder's skip list.

The new pool is frozen before the stricter policy is fitted. Policy development may use the retired
43-query source-audit cohort and the 72-query development trace, but it may not read, score, or tune
against the new holdout after the pool digest is committed.

## Deterministic extractive gold construction

Use seed `guarded-spare-slot-extractive-v1` and negative prefix `ZXQXACT`. Enumerate Markdown memory
sources from the same two local source roots used by the prior pool. Verify that the negative prefix
is absent from every candidate source.

For each source, strip leading YAML front matter for parsing but preserve the raw file for hashing
and production `recall.index.chunk_text` chunking. Select one answer span by this fixed priority:

1. the value of the first eligible structured paragraph beginning with `FACT`, `VERDICT`, `RESULT`,
   `OUTCOME`, `DECISION`, `APPLY`, `WHY`, `STATUS`, `OBJECTIVE`, or `ROOT CAUSE` followed by a colon;
2. otherwise, the first eligible prose paragraph following a level-two or deeper Markdown heading;
3. otherwise, the first eligible prose paragraph after the title.

An eligible span is 40 through 320 characters after whitespace normalization, contains at least
eight alphanumeric words, contains no URL, table row, fenced code, or Markdown heading, is not
already contained in the question, occurs exactly once in the source, and lies wholly inside one
default `chunk_text` chunk. Freeze the raw source SHA256, source path, chunk ordinal, exact normalized
answer span, and answer-span SHA256.

Construct the question from the source title and the selected field or heading. Structured fields
use a fixed field-specific template. Heading spans use `What does <title> record under <heading>?`.
Fallback spans use `What key statement is recorded in <title>?`. Reject duplicate normalized
questions.

For every answerable item, construct a matched unanswerable item by inserting the unique
`ZXQXACT-NNNN` identifier into the same question subject. Its frozen answer is exactly `NOT_FOUND`
and its gold source list is empty. Sort all rows by the seeded SHA256 order.

Target 250 answerable and 250 matched unanswerable queries. Return `INSUFFICIENT_POOL` and do not run
retrieval if fewer than 250 eligible sources remain.

## Development and policy freeze

The retired cohort may be recaptured with private per-candidate rank, cosine, source-support, lane,
redundancy, and source-concentration fields. Label only whether each addition matches the already
frozen gold source. Compare small interpretable rules, prioritizing:

1. append at most one item rather than filling every spare slot;
2. require independent dense and sparse support;
3. reject source concentration and additions redundant with the base prefix;
4. raise evidence thresholds only when the development precision gain is material.

Before holdout retrieval, append the exact selected rule, prediction, cohort requirement, and
promotion gate to this file and commit it. Do not edit any text above this marker after the first
pool build or any registered number after measurement.

## Holdout metrics

For triggered answerable queries, report base and candidate exact-span coverage, source hit,
fact gains, fact losses, and additions containing the frozen span. For controls, report activation
and every added item as non-gold. Report added-item precision across the whole triggered cohort and
base-prefix preservation.

The primary success condition is at least one exact-span gain, zero exact-span losses, zero source
hit losses, and zero control activations. A candidate that activates on any matched absent-identifier
control is not safe for promotion. The candidate must also preserve the complete base prefix and add
at most one item per query.

This retrieval result does not by itself license active serving. An end-to-end answer experiment is
separate and may run only after the retrieval gate passes. That stage must require the reader to
return an exact span from evidence or the exact token `NOT_FOUND`; it may not use a permissive LLM
judge as the primary scorer.

<!-- frozen_above -->

## Pool build attempt one repair

Attempted 2026-09-14. The builder stopped before writing any pool artifact because 14 duplicate
normalized questions appeared among the first 250 ordered source candidates. The candidate set was
large enough, but the implementation skipped duplicates without continuing through the remaining
ordered candidates.

The repair changes only the deterministic selection loop: continue through the already ordered
eligible candidates until 250 unique positive and matched negative pairs have been collected, or
return `INSUFFICIENT_POOL` after exhausting all eligible candidates. The seed, source exclusions,
span rules, target size, metrics, and gates are unchanged.

## Extractive pool freeze

Built 2026-09-14. The deterministic builder found 366 eligible sources after the registered
exclusions and froze 250 answerable plus 250 matched unanswerable queries. The pool SHA256 is
`66ec82a058c9b06cf80314a1001779a1608e5144e097ace28b91664a48ede855`.

Rebuild from the repository root:

```powershell
python scripts/build_guarded_spare_slot_extractive_pool.py --source-root "recall=C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-recall\memory" --source-root "sentiment-agent=C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-progetto-sentimental\memory" --trace docs/results/2026-09-13-live-source-admission-trace-capture.json --old-pool docs/preregistrations/2026-09-14-guarded-spare-slot-fresh-pool.json --output docs/preregistrations/2026-09-14-guarded-spare-slot-extractive-pool.json
Get-FileHash docs/preregistrations/2026-09-14-guarded-spare-slot-extractive-pool.json -Algorithm SHA256
```

The holdout is sealed for policy development. Do not inspect individual questions, spans, sources,
or retrieval outcomes before the selected stricter rule is appended and committed below.

## Development recapture attempt one repair

Attempted 2026-09-14 against the original pinned generation. The first ten retired cohort queries
reproduced their frozen base and candidate hashes. Query eleven did not, so the harness stopped and
wrote no feature artifact. The extractive holdout was not queried or inspected.

Repeated live retrieval is not interchangeable with the frozen capture. The repaired development
recapture must continue after a mismatch, record only an aggregate mismatch receipt for that query,
and admit numeric feature rows only when the complete base and candidate hash sequence matches the
frozen capture exactly. Policy fitting requires at least 30 of 43 parity rows and at least five
gold-source additions among those rows. Otherwise return `INSUFFICIENT_DEVELOPMENT_PARITY` and do
not run the extractive holdout.

## Frozen strict policy and holdout gate

Frozen 2026-09-14 before any extractive holdout retrieval. Development recapture met the registered
cohort requirement with 42 of 43 exact parity rows and 10 gold-source additions across all guarded
positions. Apply policy `extractive_strict_v1` to the first guarded spare-slot proposal only. Admit
that proposal if and only if all of these conditions hold:

1. its lane is `dual_leg`;
2. its source sparse rank is at most 2;
3. its source margin is at least 0.0;
4. its cross-leg fraction is at least 0.5.

If the first proposal fails, preserve the base unchanged. Do not consider a later proposal. If it
passes, append that one item only. Preserve the complete base prefix in either case.

On the private development rows, this exact rule admitted 4 gold additions, 0 answerable non-gold
additions, and 0 controls. Five gold additions after the first position were deliberately not
considered. The aggregate receipt is
`docs/results/2026-09-14-guarded-spare-slot-strict-development.json`. Reproduce it with:

```powershell
python scripts/summarize_guarded_spare_slot_strict_dev.py --features C:\Users\gde00\.codex\evals\guarded-spare-slot-2026-09-14\dev-features.json --output docs/results/2026-09-14-guarded-spare-slot-strict-development.json
```

Prediction: on the sealed 250-pair holdout, the strict candidate will produce at least one exact
span gain, zero exact span losses, zero source-hit losses, zero control activations, a preserved
base prefix on every query, and no more than one addition per query.

Promotion requires every predicted safety condition plus positive exact-span gain. Added-item
precision is descriptive and does not override any zero-tolerance safety gate. If the frozen source
SHA256 values do not match the indexed corpus used for the run, return `HOLDOUT_LINEAGE_INVALID`
rather than score the affected queries.

## Holdout result

Measured 2026-09-14 on the complete sealed 250 answerable plus 250 matched control cohort. The
decision was `FAIL_CONTROL_ACTIVATION`. Base and candidate both covered 149 exact spans and hit 171
gold sources. The candidate produced 0 exact-span gains, 0 source-hit gains, 0 losses, and 2 total
additions. Both additions were control activations and neither contained gold. The base prefix was
preserved for all 500 queries and the maximum addition count was 1.

The strict source-only gate is closed. Do not retune its thresholds on this consumed holdout. The
result supports a change in signal class: a later experiment may use source agreement to propose a
candidate, but admission must measure compatibility between the actual query anchors and candidate
text before adding evidence.

The full aggregate-safe capture is
`docs/results/2026-09-14-guarded-spare-slot-extractive-holdout.json`. Reproduce against the pinned
lineage with:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/guarded-spare-slot-89caaf43'
$env:RECALL_SOURCE_COMMIT='98b6ee9d'
$env:RECALL_POLICY_COMMIT='2d36eabd'
python -u scripts/run_live_guarded_spare_slot_extractive_holdout.py --query-pool docs/preregistrations/2026-09-14-guarded-spare-slot-extractive-pool.json --artifact docs/results/2026-09-13-source-conditioning-model.json --inventory-receipt docs/results/2026-09-14-guarded-spare-slot-extractive-inventory.json --output docs/results/2026-09-14-guarded-spare-slot-extractive-holdout.json --generation-id gen_5a945edfbc644e5db77906c06658dc49 --calibration-id cal_e4c81ba2db404eafbeb59f297f3c1dd2 --pipeline-fingerprint 57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86 --corpus-fingerprint d5570385d78065192b724d6f29ad18e3c996ff2ab82a679cac1913e558511a07
```
