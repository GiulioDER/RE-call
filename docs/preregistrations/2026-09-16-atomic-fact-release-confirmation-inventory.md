# Atomic fact release confirmation inventory

Status: preregistered before constructing the candidate inventory, generating questions, or
measuring retrieval.

## Purpose

Determine whether the current production memory snapshot contains enough atomic-development-
disjoint facts to build a 96 source prospective confirmation set. This inventory performs no
model calls, embeddings, or retrieval. A later committed preregistration must freeze the private
pool hash and quality gates before any candidate result is measured.

The earlier blind holdout excluded every source that appeared anywhere in historical candidate
pools, including unrelated negative candidates. That rule left only 45 eligible sources and made
the intended 80 source confirmation impossible. This protocol instead excludes every source or
fact population that materially influenced the atomic representation or the dense-five plus one
atomic rescue design. A source is not excluded merely because an older retriever once returned it
as a negative.

## Frozen source snapshot

Use the seven synchronized production memory roots under
`C:\Users\gde00\.claude\recall-vps2\memstores`: `recall`, `sentiment-agent`,
`ai-boost-av-safety`, `ai-boost-cad`, `steel`, `cca-demos`, and `agent-memory-bench`.

The serving lineage observed immediately before preregistration is:

| Field | Frozen value |
| --- | --- |
| Generation | `gen_dff506e12f494965af9f109671a99e63` |
| Calibration | `cal_6171177aeb614d288baa4e28600caca1` |
| Pipeline | `57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86` |
| Corpus | `522e9d1c506142d8a6a5a1f5be9073ff9409132d356e29f61fbf8e7d6d2044a0` |
| Embedding profile | `voyage-context-4-v1` |

The inventory records a canonical source snapshot hash from source identifier, source SHA256,
and ordered Markdown path. Any later pool construction must reconstruct that hash exactly or
stop. It may not silently use a refreshed mirror.

## Frozen atomic-development exclusions

The inventory verifies these exact inputs before use:

1. The 250 row extractive answerable pool at
   `docs/preregistrations/2026-09-14-guarded-spare-slot-extractive-pool.json`, SHA256
   `66ec82a058c9b06cf80314a1001779a1608e5144e097ace28b91664a48ede855`. Exclude every
   answerable row's `gold_sources`; this population drove the atomic parser and rendering audit.
2. The private 22 source atomic pilot pool, SHA256
   `720272504e128d861b09b7d29df8570eadfe3362865a256951c0b404bf448aa9`. Exclude every
   `gold_sources` value; this population selected atomic retrieval as the promising lane.
3. The 50 query production memory source-gold set at
   `docs/preregistrations/2026-09-13-memory-queries-source-gold.json`, SHA256
   `06e5cfb2a345d3108ee5ae9e2d0bc2cd74fba455d46f56658bf496b2447e088f`. Exclude
   `relevant_files` from answerable rows; this population selected dense-five plus one atomic
   rescue over pure replacement.
4. The complete 45 source population from the prior blind census, including its 14 rejected
   question candidates. Reconstruct it with the old six-input exclusion rule and require candidate
   manifest SHA256
   `edfd11a46216b4b10b3b063c65623456bfd75876c37667f43e57505c616d1b60` before adding all
   45 source identifiers to the new exclusion set. The private accepted census pool SHA256 is
   `97f77c71c1feb278b9d5297511e8b4fb913002208eaae2b3bbd2dc65d578fd18` and is checked as
   an additional receipt, but it is not sufficient by itself because it contains only 31 sources.

The old six-input reconstruction additionally verifies these historical artifacts:

| Input | SHA256 |
| --- | --- |
| Query anchor spare slot pool | `6dd9485b12bf0c88d166cc29114cd03ada1e732e47b11592a4725e91d7324e68` |
| Guarded spare slot fresh pool | `af7c74d4d2b232cb79de72d5fdfd67e1ccf1d6ad814fb634ba61f98999632c75` |
| Live source admission trace | `facdac77945c820c80af76e8adaa3fd106f34598c88dd56a291d001f3fa2bfd9` |

The new exclusion set does not inherit the historical trace's negative candidates, nor the query
anchor and fresh pool gold sources, unless a source is also present in one of the four atomic-
development populations above. This boundary is fixed before the inventory count is known.

Also exclude aggregate and index basenames `MEMORY.md`, `project_index.md`, `feedback_index.md`,
`archived_index.md`, `closed_hypotheses_index.md`, and `EXECUTION_LOG.md`. Exclude every basename
beginning `2026-09-15` or `2026-09-16`, preventing records created during atomic research from
becoming confirmation gold.

## Frozen fact eligibility

Use the unchanged production-aligned `build_source_views` parser. For every non-skipped source,
record all eligible atomic views before applying development exclusions. A candidate source must:

1. not be in the frozen atomic-development exclusion set;
2. contain an atomic view with an exact extractive content span of 40 through 320 collapsed
   characters and at least eight words;
3. map that span wholly to one ordinary parent chunk;
4. have at least one normalized atomic content span that occurs in exactly one source across the
   entire non-skipped seven-root snapshot, including excluded development sources.

Keep the first globally source-unique view in document order for each candidate source. Order the
candidates by `SHA256("atomic-fact-release-confirmation-v1" + "\0" + source)`, then by source.

The private canonical manifest contains source, source SHA256, gold parent ordinal, answer span
SHA256, construction, and deterministic order. It remains outside the repository. The public
receipt contains only hashes and aggregate counts by root and construction.

## Frozen inventory gates

Return `READY_TO_BUILD_ATOMIC_RELEASE_CONFIRMATION` only if all conditions hold:

1. The old blind population reconstructs exactly 45 candidates with its frozen manifest hash.
2. Every input hash matches, and the private accepted census pool is a subset of the reconstructed
   45 source population.
3. At least 160 eligible candidate sources remain after the atomic-development exclusions.
4. At least three production roots are represented, and no root supplies more than 80 percent of
   candidates.
5. Every candidate's source hash, answer span hash, parent ordinal, and global source uniqueness
   pass a second audit.
6. The private manifest remains outside the repository and its Windows ACL does not grant
   `Everyone`, `BUILTIN\Users`, or `Authenticated Users`.

Otherwise return `STOP_ATOMIC_RELEASE_CONFIRMATION_INVENTORY`. Do not weaken exclusions, reduce
the future 96 row target, or redefine global uniqueness under this protocol.

## Next protocol if ready

If the inventory passes, preregister construction of exactly 96 accepted questions from at most
160 ordered candidates. The independent writer will see only the exact fact span. The completed
private pool hash, question validation, paired retrieval metrics, statistical test, latency gates,
and production decision rule must be committed before retrieval or embedding begins.

## Prediction

I predict the inventory will pass with at least 160 candidates. The earlier 45 source ceiling was
mainly caused by excluding 1,482 sources from unrelated historical negative pools, not by a lack of
extractive atomic facts in the production corpus.
