# Pre registration: can a chunker be identified from stored chunks alone?

**Date:** 2026-08-18   **Status:** predicted, not yet measured

## The question

`docs/UNCALIBRATED_FIRST_RUN_DESIGN.md` section 6b records one genuinely open engineering gap: the
embedder attestation re embeds the **stored chunk text**, so it verifies the embedder and can never
verify the chunker that produced that text. This asks whether a chunker attestation is possible at
all, and if so what shape it must take.

The legacy `chunks` table records **no chunker identity**: not the algorithm, not `max_chars`, not
`overlap`. `_index_fingerprint` has no chunker term either (`recall/index.py:442-452`). So an
adopted generation's `ChunkerIdentity` is an assertion with nothing behind it, and the calibration
binds to a `pipeline_fingerprint` that includes it.

**Q1 (identifiability).** Re chunking a verified source from disk with a candidate configuration
and comparing to the stored chunks exactly: does the shipped default configuration reproduce the
stored chunks, and for what fraction of sources?

**Q2 (uniqueness).** Is the identification *unique*? If more than one candidate reproduces the same
stored chunks, the chunker is under determined on this corpus and an attestation can only report a
set, never a value.

**Q3 (cost).** Is exhaustive re chunking affordable, so the check can cover every source rather
than a sample? This is the question that decides whether the chunker attestation needs the sample
size machinery the embedder attestation needs (decision 9 of the design).

## What I predict

**Q1.** The shipped default `chunk_text(max_chars=800, overlap=80)` reproduces the stored chunks
**exactly for at least 98 percent of sources**. The corpus was indexed by `recall index` with no
chunker arguments, so the defaults are what ran. Failures, if any, will be sources whose body
derivation moved, not sources whose packing differs.

**Q2. The identification will NOT be unique, and `overlap` is the parameter I expect to be
unidentifiable.** `overlap` only affects a paragraph longer than `max_chars`, which is force split
(`recall/index.py:228-230`). Memo prose is mostly short paragraphs, so I predict **fewer than 10
percent of sources contain any force split**, and on the rest `overlap=80` and `overlap=0` produce
identical output. `max_chars` I expect to be identifiable on most sources, because packing to a
limit leaves a visible signature wherever two paragraphs did not fit together.

**Q3.** Exhaustive re chunking of all 1,080 sources completes in **under 30 seconds** for a single
candidate, against 1.82 s measured for hashing the same corpus. Pure Python string work, no model,
no database round trip per source. So exhaustive coverage is affordable and the chunker attestation
needs no sampling rule.

## What would falsify this

- **Q1** is falsified if under 98 percent of sources reproduce. That would mean the stored chunks
  cannot be re derived from the file plus a known configuration, and a chunker attestation is
  impossible rather than merely expensive. The design section would have to say so.
- **Q2** is falsified if exactly one candidate in the fixed set reproduces every source, i.e. the
  chunker is fully identified. That is the *better* outcome and would make the attestation stronger
  than predicted; it is registered as a falsifier because I expect the weaker result and must not
  be able to claim the stronger one after the fact.
- **Q3** is falsified if a single candidate takes over 30 seconds, in which case the exhaustive
  claim fails and the design must size a sample exactly as decision 9 does for the embedder.

⚠️ **The candidate set is FIXED here, before measuring, and must not be widened afterwards.**
It is exactly four: `(text, 800, 80)`, `(text, 800, 0)`, `(code, 800, n/a)`, `(text, 1200, 80)`.
Widening the search until something fits would find a configuration that never ran, which is
curve fitting wearing a verification's clothes. If none of the four reproduces a source, the
registered answer is "not identifiable", not "search harder".

## How it will be measured

Corpus: the same remote `memory` tenant used in
`docs/preregistrations/2026-08-18-uncalibrated-first-run.md`, 1,080 sources and 8,716 chunks, whose
`content_hash` values were already verified 1,080 of 1,080 against bytes on disk.

1. Pull `(source, metadata->>'file', metadata->>'ord', text, metadata->>'content_hash')` for the
   tenant, read only.
2. **Validate the local copy first.** For every source, sha256 the local file exactly as
   `recall/index.py:671` and `:690` do for markdown and compare to the stored `content_hash`. A
   source whose local copy does not match is excluded and counted, because re chunking a different
   file would measure nothing. This is the apparatus check, and it runs before any comparison.
3. Derive the body with `parse_frontmatter` (`recall/frontmatter.py:186`), which is what the
   indexer hands the chunker for markdown.
4. For each of the four fixed candidates, re chunk and compare the resulting list against the
   stored chunks ordered by `ord`, as an **exact string list equality**.
5. Report, per candidate: sources reproduced, sources differing, and the first differing ordinal.
6. Report the count of sources containing at least one force split, which is what makes `overlap`
   identifiable.
7. Time step 4 for one candidate with a monotonic clock, excluding the database pull.

## What I already know

- `chunk_text(text, max_chars=800, overlap=80)` and `chunk_code(text, max_chars=800)`
  (`recall/index.py:223`, `:233`). `ChunkerKind` is `Literal["text", "code"]` (`recall/index.py:59`).
- `ChunkerIdentity` carries `algorithm`, `schema_version` and a frozen `configuration`
  (`recall/lineage.py:132`), so a chunker is fully specified by a small tuple.
- `text_start` / `text_end` in chunk metadata are **not independent evidence**: `structure_chunks`
  derives them by searching for the chunk text inside the body (`recall/context.py:205`), so they
  move with the chunk text rather than constraining it.
- The frontmatter pairing rule changed once, and the fix carries a version marker
  (`recall/generations.py:109`), which is a documented case of the same bytes yielding a different
  body. This corpus was indexed after that change.

## Confounds I can name now

Every source here is markdown, so nothing measures the `content_blocks` extraction path that
`bd582316` added for other media types, where `text_start` and `text_end` are recorded as `None`
(`recall/index.py:800`). A corpus of PDFs would additionally depend on the extractor being
deterministic across versions, which is not tested here and is a strictly harder problem.

The corpus is my own memo prose, which is unusually uniform in paragraph length, and paragraph
length is exactly what determines whether force splitting occurs. So the Q2 result about `overlap`
may be a property of this corpus rather than of the chunker.

An exact string list comparison is deliberately stricter than retrieval cares about. Two chunkings
differing by one trailing newline would retrieve near identically and be reported as a mismatch
here. That strictness is the point for an attestation, but it means the reproduction rate is a
lower bound on practical equivalence.
