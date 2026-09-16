# Extractive gold shape and source split audit

Status: predicted, not yet measured.

Registered 2026-09-16 before implementing the audit harness or calculating any aggregate from the
frozen pool.

## Question

Can the existing automatic extractive pool support a leakage resistant, source disjoint training
experiment for a task specific query to evidence selector?

This audit separates two decisions. The answerable rows may be valid supervision for ranking even
if the matched absent identifier controls are invalid supervision for a null class. No model,
embedding, retrieval, or inference runs in this phase.

## Frozen inputs

The primary input is
`docs/preregistrations/2026-09-14-guarded-spare-slot-extractive-pool.json`, SHA256
`66ec82a058c9b06cf80314a1001779a1608e5144e097ace28b91664a48ede855`. It declares 250
answerable rows and 250 matched controls. It was produced by
`scripts/build_guarded_spare_slot_extractive_pool.py` and is already consumed development data.
This audit cannot create a product quality claim or authorize serving.

The source roots are frozen as:

1. `recall` at
   `C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-recall\memory`
2. `sentiment-agent` at
   `C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-progetto-sentimental\memory`

The consumed comparison pool is
`docs/preregistrations/2026-09-14-query-anchor-spare-slot-pool.json`, SHA256
`6dd9485b12bf0c88d166cc29114cd03ada1e732e47b11592a4725e91d7324e68`. It is used only to
measure exact source path and source content overlap. No result from that later cohort is rescored.

The split seed is the literal UTF-8 string
`extractive-selector-source-split-v1`. Group every answerable row by `source_sha256`. Assign the
whole group, including its numeric matched control, using the first eight bytes of
`SHA256(seed + "\0" + source_sha256)` interpreted as an unsigned big endian integer modulo 100:

1. buckets 0 through 69 are `train`
2. buckets 70 through 84 are `validation`
3. buckets 85 through 99 are `internal_test`

No row may be reassigned to balance a split.

## Integrity measurements

Verify the primary and comparison input hashes before aggregation. For every numeric pair, require
one answerable row and one control. Report missing, duplicate, or malformed pairs.

For each answerable row, resolve its one gold source under the declared source root. Verify the
source file SHA256, reconstruct the default `recall.index.chunk_text` chunks, and require the frozen
answer span to occur after whitespace normalization in exactly one chunk at the recorded ordinal.
Report missing sources, source hash mismatches, ordinal mismatches, and nonunique span containment.

Report these answerable distributions:

1. answer character and word count at minimum, p10, p25, p50, p75, p90, and maximum, using nearest
   rank quantiles
2. construction counts for `extractive_field`, `extractive_heading`, and `extractive_fallback`
3. field template counts derived from the fixed question prefixes in the pool builder
4. source root counts and filename family counts
5. gold chunk ordinal counts
6. exact character clearance from each answer span to the left and right chunk boundary, plus the
   count whose minimum clearance is below 40 characters
7. unique normalized questions, source paths, source content hashes, and answer span hashes
8. exact source path and content hash overlap with the consumed comparison pool

Report split row counts, construction counts per split, source family counts per split, and every
pairwise overlap among splits for normalized questions, source paths, source content hashes, and
answer span hashes.

Normalization is Unicode NFKC, case folding, then collapsing all whitespace to one ASCII space.
Word counts use the builder's Unicode word expression `\b[\w'-]+\b`.

## Control leakage measurements

The pool declares the control prefix `ZXQXACT`. Measure two frozen label only rules over all 500
queries:

1. predict unanswerable when the query contains `ZXQXACT-` followed by four decimal digits
2. predict unanswerable when the query begins with `According to memory item `

Report accuracy, balanced accuracy, answerable false positive count, and unanswerable false
negative count for each rule. Also report the count of normalized query tokens that occur in every
control and no answerable row, excluding punctuation and tokens shorter than three characters.

These are leakage diagnostics, not learned baselines. Do not inspect individual errors to invent a
new rule.

## Frozen gates and decisions

The positive ranking supervision passes only if all of the following hold:

1. exactly 250 complete answerable and control pairs exist
2. all 250 answerable sources exist and match their frozen SHA256
3. every answer span occurs in exactly one reconstructed chunk at its frozen ordinal
4. there are at least 240 distinct answerable source content hashes
5. train has at least 150 answerable rows, validation at least 25, and internal test at least 25
6. source path and source content overlap are both zero across all split pairs
7. normalized question overlap is zero across all split pairs
8. exact source path and source content overlap with the comparison pool are both zero
9. every split contains at least two answerable construction families

The current controls pass for explicit null training only if both frozen lexical rules have balanced
accuracy at most 0.60 and the class exclusive universal token count is zero.

Return one decision:

1. `GO_FULL_SELECTOR_WITH_NULL` when both the positive and control gates pass
2. `GO_POSITIVE_SELECTOR_ONLY` when the positive gate passes and the control gate fails
3. `STOP_CURRENT_POOL` when the positive gate fails

For `GO_POSITIVE_SELECTOR_ONLY`, the next experiment may train ranking relevance from exact bearing
chunks and same query real dense top 20 hard negatives, but it must exclude these controls and must
not claim null handling. A separate natural unanswerable set is required before training or testing
an explicit null class.

For `GO_FULL_SELECTOR_WITH_NULL`, the next experiment may use the controls only as paired query
level null examples and must still use real dense top 20 hard negatives for chunk relevance.

For `STOP_CURRENT_POOL`, do not train on this pool. Repair gold or build a new source disjoint pool
before spending model compute.

## Prediction

I predict the positive ranking supervision will pass and the control gate will fail, producing
`GO_POSITIVE_SELECTOR_ONLY`. The controls were deliberately generated with a synthetic absent
identifier and a control only introductory phrase, so a full null training decision would otherwise
confuse construction leakage with answerability.
