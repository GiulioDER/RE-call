# Production aligned atomic fact auxiliary view fresh audit

Status: predicted, not yet measured.

Registered 2026-09-16 before building the fresh pool, implementing the revised construction, or
calculating any aggregate from eligible sources.

## Question

Does the atomic fact auxiliary retrieval view pass its construction and safety gates on a fresh,
source-disjoint set when field parsing and structural degradation follow the production context
contract?

This is the preregistered reentry after
`2026-09-16-atomic-fact-auxiliary-view-audit.md` stopped at 248 of 250 gold rows. It does not revise
or rescore that consumed pool.

No model, embedding, retrieval, or external inference runs in this phase.

## Frozen source roots and exclusions

The source roots are:

1. `recall` at
   `C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-recall\memory`
2. `sentiment-agent` at
   `C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-progetto-sentimental\memory`

Exclude every source present in these committed inputs:

1. `docs/preregistrations/2026-09-14-guarded-spare-slot-extractive-pool.json`, SHA256
   `66ec82a058c9b06cf80314a1001779a1608e5144e097ace28b91664a48ede855`
2. `docs/preregistrations/2026-09-14-query-anchor-spare-slot-pool.json`, SHA256
   `6dd9485b12bf0c88d166cc29114cd03ada1e732e47b11592a4725e91d7324e68`
3. `docs/preregistrations/2026-09-14-guarded-spare-slot-fresh-pool.json`, SHA256
   `af7c74d4d2b232cb79de72d5fdfd67e1ccf1d6ad814fb634ba61f98999632c75`
4. `docs/results/2026-09-13-live-source-admission-trace-capture.json`, SHA256
   `facdac77945c820c80af76e8adaa3fd106f34598c88dd56a291d001f3fa2bfd9`

Also exclude index files named `MEMORY.md`, `project_index.md`, `feedback_index.md`,
`archived_index.md`, `closed_hypotheses_index.md`, and `EXECUTION_LOG.md`. Exclude a source whose
basename begins `2026-09-15` or `2026-09-16`, so memos written during the current research sequence
cannot become its gold.

The pool seed is the literal UTF-8 string `atomic-fact-fresh-audit-v1`. Sort eligible source
candidates by `SHA256(seed + "\0" + source)` and then by source. Select the first 250 candidates
whose normalized questions are unique. If fewer than 250 remain, stop with
`INSUFFICIENT_FRESH_POOL` and do not weaken exclusions or reduce the target on this protocol.

The row-level pool is private under
`C:\Users\gde00\.codex\evals\atomic-fact-fresh-2026-09-16`. Publish only its SHA256 and aggregate
counts.

## Frozen fact extraction

Parse each source with `recall.document.parse_document`. Ordinary parent chunks are
`recall.index.chunk_text(parsed.human_body)` using default settings. A heading is Markdown level
one through six and updates the nearest heading hierarchy. A heading is not a fact.

For every nonheading paragraph, first separate an optional explicit field label. The accepted
labels remain `FACT`, `VERDICT`, `RESULT`, `OUTCOME`, `DECISION`, `APPLY`, `WHY`, `STATUS`,
`OBJECTIVE`, and `ROOT CAUSE`. Eligibility applies to the complete fact content after label
separation:

1. 40 through 320 characters after whitespace collapse
2. at least eight words under `\b[\w'-]+\b`
3. no HTTP URL, fenced code marker, or Markdown table pipe
4. exact normalized fact content occurs once in the parsed human body
5. the complete original paragraph maps wholly into exactly one ordinary parent chunk

Every eligible view records source, parent ordinal, title, heading hierarchy, optional field, and
complete fact content. The source path itself is not embedding text.

The pool gold for a source is its first eligible fact in document order. Query construction is the
same frozen title, field, heading, and fallback template family used by
`scripts/build_guarded_spare_slot_extractive_pool.py`. The pool contains answerable rows only.

## Frozen production aligned rendering

Document title uses `recall.context.document_title` and its production 256 character cap. Heading
hierarchy uses the production 512 character full cap and 256 character degraded cap. Structural
fields have control characters removed and whitespace collapsed. Complete fact content is never
shortened.

Render the first representation at or below 800 characters from this ordered ladder:

1. title, full section, optional field, content
2. title, section shortened to 256 characters, optional field, content
3. title, optional field, content
4. optional field, content
5. content only

The first fitting rung is the view. Since fact content is at most 320 characters, the final rung
always fits. Record the chosen rung. This mirrors the production principle that section detail is
reduced before title and complete current content is preserved at every rung.

## Measurements

Report source integrity, eligible and selected source counts, construction family counts, ordinary
chunks, auxiliary views, sources with zero views, auxiliary row growth, rendered character growth,
views per source, views per parent chunk, rendering rung counts, title, heading, and field coverage,
view and content length distributions, exact normalized fact duplicates, exact normalized rendered
collisions, and views above 800 characters.

For every gold row, require exactly one auxiliary view with content exactly equal to the gold fact
and the same parent ordinal. Report coverage by field, heading, and fallback construction family.

All distributions use nearest rank quantiles at minimum, p50, p90, p95, and maximum. Duplicate and
collision counts are excess rows beyond the first member of an exact normalized group.

## Frozen gates

Return `GO_BUILD_ATOMIC_FACT_CONTEXT4_SHADOW` only when all conditions hold:

1. exactly 250 fresh source-disjoint rows are built and every frozen input hash matches
2. all selected source files exist and match the hashes written into the private pool
3. all 250 gold facts occur in exactly one auxiliary view and map to the frozen parent ordinal
4. field, heading, and fallback are all represented and each has 100 percent gold coverage
5. every selected source has at least one view
6. zero rendered views exceed 800 characters
7. within-source exact fact duplicates are at most 1 percent of views
8. cross-source exact rendered collisions are at most 1 percent of views
9. auxiliary row growth is at most 2.0 times ordinary chunk count
10. rendered character growth is at most 1.5 times ordinary chunk characters
11. p95 views per parent chunk is at most 4 and maximum views per parent chunk is at most 8

Otherwise return `STOP_PRODUCTION_ALIGNED_ATOMIC_FACT_VIEW` and list every failed gate. Do not tune
the extraction, ladder, thresholds, seed, exclusions, or sample size on this pool.

A pass authorizes implementation of one isolated Context 4 shadow generation only. It does not
authorize serving and does not establish a retrieval improvement. Retrieval quality must be tested
later on a second untouched source-disjoint exact-span set.

## Prediction

I predict `GO_BUILD_ATOMIC_FACT_CONTEXT4_SHADOW`. The first audit already passed every growth,
duplicate, collision, and crowding gate. Separating field content before eligibility should restore
the two missed field facts, while the production ordered degradation ladder should eliminate the
eleven oversized views without truncating content.
