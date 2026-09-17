# Atomic fact blind source census

Status: preregistered before any question is generated from the 45 source population.

## Purpose

Build an exploratory blind quality set from every eligible source left by the stopped 80 source
holdout. The earlier protocol remains stopped and is not relabelled. This protocol changes the
estimand from an 80 source confirmation set to an exhaustive census of the feasible untouched
population.

The result can prioritize the next retrieval design and provide real paired numbers. Its effective
sample size is too small to authorize production serving by itself.

No retrieval runs in this phase. A later preregistration must freeze the completed pool SHA256
before any dense or atomic candidate is read.

## Frozen population

The production lineage, seven source roots, six exclusion inputs, index and date exclusions, fact
parser, normalization, and global source-uniqueness rule are exactly those in
`2026-09-16-atomic-fact-blind-source-holdout-construction.md`.

The stopped inventory established these fixed input facts before query generation:

1. 1,482 excluded sources.
2. 86 parsed unexcluded sources.
3. 45 eligible sources with a globally source-unique atomic fact.
4. Candidate root distribution: 28 `agent-memory-bench`, 14 `recall`, and one each from
   `ai-boost-av-safety`, `sentiment-agent`, and `cca-demos`.
5. The canonical ordered candidate manifest SHA256 is
   `edfd11a46216b4b10b3b063c65623456bfd75876c37667f43e57505c616d1b60`.

The canonical manifest is UTF-8 JSON with sorted object keys and compact separators. Each ordered
row contains source, source SHA256, gold parent ordinal, answer span SHA256, and deterministic
order. Source identities and answer spans remain private.

The builder must reconstruct exactly 45 candidates and the exact manifest hash or refuse before a
model call.

## Frozen question generation

Call OpenRouter model `deepseek/deepseek-v4-flash` exactly once for each of the 45 ordered
candidates, at temperature zero and maximum 120 completion tokens. Use the exact system prompt,
user template, retry policy, forbidden words, JSON shape, question length, control character,
answer copy, five token overlap, and normalized uniqueness checks frozen in the stopped protocol.

The writer sees only the exact fact content. It never sees the source path, source hash, title,
heading hierarchy, field label, rendered view, parent chunk, candidate order, or retrieval output.

Retain every valid unique question. Do not stop at a target count. Do not retry a validation
rejection, alter its prompt, substitute a source, or discard a valid question to improve root
balance. A provider exception after the repository retry policy is a hard stop with no usable
pool.

## Frozen privacy and audit rules

The row-level pool is outside the repository under
`C:\Users\gde00\.codex\evals\atomic-fact-blind-holdout-2026-09-16`. Its NTFS access list must not
grant `Everyone`, `BUILTIN\Users`, or `Authenticated Users`. Access inherited by the current user,
`SYSTEM`, and `Administrators` is permitted. Publish only a boolean restricted-ACL verdict, never
the local account or machine name.

Re-audit every accepted row against the frozen source SHA256, answer span SHA256, parent ordinal,
global source uniqueness, question validation, and exclusion set. The public artifact may contain
only hashes, aggregate counts, root distribution, rejection categories, provider usage, and gate
results.

## Frozen gates

Return `READY_TO_PREREGISTER_EXPLORATORY_RETRIEVAL` only when all conditions hold:

1. The reconstructed candidate count is exactly 45 and its canonical manifest SHA256 matches.
2. All six exclusion input hashes match and no accepted source is excluded.
3. Exactly 45 successful model completions are accounted for, one per candidate, with no provider
   exception.
4. At least 32 questions pass every frozen validation rule.
5. Every accepted row passes source, answer span, parent ordinal, and global uniqueness audit.
6. Accepted questions are normalized-unique.
7. At least three production roots are represented, and no root supplies more than 80 percent of
   accepted rows.
8. The private artifact remains outside the repository and passes the restricted NTFS ACL check.

Otherwise return `STOP_BLIND_SOURCE_CENSUS`. Do not regenerate, resample, loosen validation, or
change the model on this population.

## Prediction

I predict `READY_TO_PREREGISTER_EXPLORATORY_RETRIEVAL`, with at least 36 accepted questions. The
prompt and output schema are simple for the selected model, while the four source-independent
validation filters leave room for some copied phrases or forbidden structural wording to be
rejected. I expect at least three roots to survive because five roots are present before generation.
