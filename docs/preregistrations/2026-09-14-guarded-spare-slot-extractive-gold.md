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
