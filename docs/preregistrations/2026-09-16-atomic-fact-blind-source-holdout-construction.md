# Atomic fact blind source holdout construction

Status: preregistered before candidate selection, question generation, or pool construction.

## Purpose

Construct a private, source-disjoint retrieval quality set for the atomic fact experiment. The
question writer sees only one exact fact span. It never sees the source path, document title,
heading hierarchy, field label, rendered atomic view, parent chunk, retrieval candidates, or
scores.

This is a prospective query construction over an existing source holdout, not a temporal future
source test. The representation parser has previously processed the corpus globally, but none of
the selected sources may have supplied gold to the consumed atomic pilot or the earlier production
query pools.

No retrieval runs in this phase. A separate preregistration must freeze the completed private pool
hash before any dense or atomic result is measured.

## Frozen production lineage

The source snapshot is the production memory lineage active before preregistration:

| Field | Frozen value |
| --- | --- |
| Generation | `gen_55487101e0d2421594b31f5c9519070f` |
| Calibration | `cal_d86150126d3f46c8aefa910c3a0529a6` |
| Pipeline | `57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86` |
| Corpus | `ac4aab3af34f68e9f168af75f9219883a363fd555012b96d888d18a4a0344eaa` |
| Embedding profile | `voyage-context-4-v1` |
| Vector dimension | 1,024 |

The seven roots are `recall`, `sentiment-agent`, `ai-boost-av-safety`, `ai-boost-cad`, `steel`,
`cca-demos`, and `agent-memory-bench`, using the synchronized production memory mirrors under
`C:\Users\gde00\.claude\recall-vps2\memstores`.

Every selected source must still exist when the pool is written. Its SHA256, gold parent ordinal,
exact answer span, and answer span SHA256 are private row-level fields. The public receipt may
contain only aggregate counts, hashes, provider usage, validation failures by category, and root
distribution.

## Frozen exclusions

Exclude every source named by these inputs:

1. `docs/preregistrations/2026-09-14-guarded-spare-slot-extractive-pool.json`, SHA256
   `66ec82a058c9b06cf80314a1001779a1608e5144e097ace28b91664a48ede855`.
2. `docs/preregistrations/2026-09-14-query-anchor-spare-slot-pool.json`, SHA256
   `6dd9485b12bf0c88d166cc29114cd03ada1e732e47b11592a4725e91d7324e68`.
3. `docs/preregistrations/2026-09-14-guarded-spare-slot-fresh-pool.json`, SHA256
   `af7c74d4d2b232cb79de72d5fdfd67e1ccf1d6ad814fb634ba61f98999632c75`.
4. `docs/results/2026-09-13-live-source-admission-trace-capture.json`, SHA256
   `facdac77945c820c80af76e8adaa3fd106f34598c88dd56a291d001f3fa2bfd9`.
5. The private 22 row atomic pilot pool, SHA256
   `720272504e128d861b09b7d29df8570eadfe3362865a256951c0b404bf448aa9`.
6. `docs/preregistrations/2026-09-13-memory-queries-source-gold.json`, SHA256
   `06e5cfb2a345d3108ee5ae9e2d0bc2cd74fba455d46f56658bf496b2447e088f`.

For the sixth input, exclude `relevant_files` only from answerable rows. For pool inputs, exclude
`gold_sources`. For the trace, exclude every source in every captured candidate pool.

Also exclude aggregate and index basenames `MEMORY.md`, `project_index.md`, `feedback_index.md`,
`archived_index.md`, `closed_hypotheses_index.md`, and `EXECUTION_LOG.md`. Exclude every basename
beginning `2026-09-15` or `2026-09-16`, so records created during this research sequence cannot
become gold.

Any input hash mismatch is a hard refusal. Exclusions may not be weakened after construction
starts.

## Frozen fact candidates

Use `build_source_views` from the production aligned atomic audit without modifying its parser,
fact limits, parent mapping, or rendering ladder. Candidate answer spans therefore remain exact
extractive facts of 40 through 320 collapsed characters, at least eight words, mapped wholly into
one ordinary parent chunk.

Across all eligible source views, retain only an answer span whose normalized content occurs in
exactly one source. Keep the first such view in document order for each source. Order sources by
`SHA256("atomic-fact-blind-source-holdout-v1" + "\0" + source)`, then by source.

Attempt at most the first 240 ordered source candidates. Stop after 80 accepted rows. A rejected
question causes deterministic progression to the next source and never a prompt rewrite or retry
with additional metadata.

## Frozen question writer

Use OpenRouter model `deepseek/deepseek-v4-flash`, temperature zero, maximum 120 completion tokens,
and the repository retry policy. The system prompt requires exactly one JSON object with one
`question` string. The user message contains only the exact fact content inside a delimited data
block.

The writer is instructed to ask the natural question a user might ask when they need that fact,
without copying the answer, referring to stored material, or inventing context. It is explicitly
forbidden from using the words `document`, `memory`, `memo`, `source`, `title`, `heading`, `field`,
`record`, `recorded`, `text`, `passage`, and `according`.

An output is accepted only when all of these conditions hold:

1. The response parses as exactly one JSON object with exactly one string field named `question`.
2. The collapsed question contains 6 through 28 word tokens and ends with one question mark.
3. It contains none of the forbidden words, case-insensitively as whole words.
4. It does not contain the normalized answer span.
5. It shares no contiguous sequence of five normalized word tokens with the answer span.
6. Its normalized form is unique among accepted questions.
7. It contains no URL, Markdown code fence, newline, NUL, or control character.

The prompt, system prompt, model identifier, seed, exclusion hashes, and provider metadata are
recorded in the private artifact. Their hashes and aggregate usage are recorded publicly.

## Frozen construction gates

Return `READY_TO_PREREGISTER_BLIND_RETRIEVAL` only if all conditions hold:

1. Exactly 80 accepted rows come from exactly 80 distinct sources.
2. Every exclusion input matches its frozen hash, and no selected source is excluded.
3. Every selected source and answer span matches its stored hash and parent ordinal on re-audit.
4. Every answer span is globally source-unique under the frozen normalization.
5. Every accepted question passes all frozen validation rules.
6. At least three production roots are represented and no root supplies more than 80 percent of
   rows.
7. Provider metadata reports exactly one successful completion per accepted row plus any rejected
   attempts, with no unreported generation error.
8. The private pool is mode 0600 and remains outside the repository.

Otherwise return `STOP_BLIND_HOLDOUT_CONSTRUCTION`. Do not reduce the target, loosen validation,
alter the prompt, change the model, or resample under this protocol.

## Prediction

I predict `READY_TO_PREREGISTER_BLIND_RETRIEVAL`. The source exclusions remove prior gold rather
than most of the production corpus, and the seven roots contain substantially more than 80 sources
with exact atomic facts. I expect question validation to reject some outputs, especially because
the five token overlap rule is strict, but fewer than the 240 attempt ceiling.
