# Atomic fact Context 4 implementation clarification

Status: registered before document embedding and before any retrieval outcome.

This note clarifies the source iteration and byte handling for
`2026-09-16-atomic-fact-exhaustive-22-pilot.md`. It does not change the frozen queries,
lineage, embedding profile, retrieval arms, fusion, metrics, thresholds, or decision rule.

## Observed preflight failures

The first runner attempt stopped before document embedding because text mode newline conversion
changed CRLF chunk boundaries relative to the pinned production generation. The embedder
constructor issued its fixed dimension probe, but no auxiliary document vector, evaluated query
vector, ranking, label, or retrieval outcome was produced.

After decoding the verified manifest bytes without newline conversion, a full read only preflight
found 1,583 manifest objects, zero missing objects, and two byte mismatches. The mismatches were
`recall/MEMORY.md` and `recall/project_index.md`. Both basenames belong to the frozen `SKIP_NAMES`
set in `scripts/run_production_atomic_fact_fresh_audit.py`. All 22 private gold sources matched
their frozen pool hashes, the pinned manifest hashes, and current bytes.

## Frozen clarification

The auxiliary source iterator applies the preregistered `SKIP_NAMES` set to every one of the seven
roots before byte verification or view extraction. This excludes every aggregate or index object
by the same basename rule, not only the two observed mismatches. All other manifest objects remain
included, including the sources whose basenames begin `2026-09-15` or `2026-09-16`: the date rule
prevented current research memos from becoming gold but did not remove them from the production
search corpus.

Every included object must still exist and match its pinned manifest size and SHA256. Source bytes
are decoded directly so CRLF is preserved for `chunk_text`. Every view must occur inside the
unchanged pinned parent chunk. Any failure stops the run before document embedding.

The public aggregate must report both the verified included object count and the excluded index
source count. No source identity or row level result is published.
