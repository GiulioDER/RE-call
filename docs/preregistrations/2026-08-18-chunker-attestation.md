# Pre registration: can a chunker be identified from stored chunks alone?

**Date:** 2026-08-18   **Status:** measured 2026-08-18. Q1 and Q3 confirmed, Q2 falsified: the
chunker turned out to be **fully identified**, which is the better outcome and the one registered
as a falsifier so it could not be claimed after the fact. Predictions and falsifiers below are
unedited; the result is appended below the horizontal rule.

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

---

## Result

Measured 2026-08-18 against the remote `memory` tenant, 1,080 sources and 8,716 stored chunks.
**Q1 and Q3 confirmed, Q2 falsified**, and the falsification is the one that changes the design.

### Apparatus check, before any comparison

| | sources |
|---|---:|
| in the tenant | 1,080 |
| local copy absent | 18 |
| local copy present but hash differs from what was indexed | 4 |
| **usable** | **1,058** |

The 22 exclusions are local snapshot divergence, not corpus damage: the same corpus verified
**1,080 of 1,080** against the bytes on its own host in the earlier run. This is the apparatus check
doing exactly what it exists for, and every figure below is over the 1,058, not the 1,080.

### Q1 and Q2: which candidates reproduce the stored chunks exactly

| Candidate | Reproduced | % | Seconds |
|---|---:|---:|---:|
| **`chunk_text(800, overlap=80)`** | **1,058 / 1,058** | **100.00** | 0.16 |
| `chunk_text(800, overlap=0)` | 535 | 50.57 | 0.20 |
| `chunk_code(800)` | 31 | 2.93 | 0.24 |
| `chunk_text(1200, overlap=80)` | 43 | 4.06 | 0.13 |

**Q1 confirmed, above its bar:** 100.00 percent against a registered 98 percent.

⛔ **Q2 falsified: the identification IS unique.** Exactly one candidate reproduces every source,
so the chunker is fully determined on this corpus rather than under determined as predicted.

**Why the prediction was wrong, which is the useful part.** I argued `overlap` would be
unidentifiable because it only bites on a paragraph longer than `max_chars`, and predicted **fewer
than 10 percent** of sources would contain one. Measured: **523 of 1,058, or 49.43 percent.** Memo
prose is not the short paragraph prose I assumed. My model of the corpus was wrong, not my model of
the chunker.

🔑 **An exact internal cross check fell out of it.** `overlap=0` reproduced **535** sources and
523 sources contain a force split, and 535 + 523 = 1,058 exactly. The two numbers were computed by
independent code paths, one comparing chunk lists and the other measuring paragraph lengths, and
they partition the corpus with no remainder. So `overlap` is identifiable on precisely the sources
where it can act, and on no others, which is what the parameter means.

### Q3: cost

**0.16 seconds for a full pass over 1,058 sources**, against a registered bar of 30 and against
1.82 s for merely hashing the same corpus. All four candidates together run in under a second.

**Exhaustive coverage is therefore free, and the chunker attestation needs no sampling rule at
all.** This is a real asymmetry with the embedder attestation, which needs one only because
inference is expensive: chunking is pure string work, so the chunker check can be complete where
the embedder check can only be a sample.

### Controls

1. **Discrimination is not assumed.** Three of the four candidates fail, at 50.57, 2.93 and 4.06
   percent. A comparison stuck on "equal" would have returned 100 percent four times.
2. **Planted corruption, because the above is not a mutation test.** Two mutations injected into
   the stored data: a **one character append** to a single chunk, and an **ordinal swap** between
   two adjacent chunks of a different source. The detector moved 1,058 to **1,056** and named both
   sources. It sees a content change and an ordering change, which are the two failure modes an
   attestation exists to catch.

### Verdict against the prediction

| Registered prediction | Measured | Verdict |
|---|---|---|
| default reproduces ≥98 percent of sources | 100.00 percent | **confirmed** |
| identification NOT unique, `overlap` unidentifiable | unique; `overlap` identifiable on 49.43 percent | ⛔ **falsified** |
| fewer than 10 percent of sources force split | 49.43 percent | ⛔ **falsified** |
| exhaustive pass under 30 seconds | 0.16 s | **confirmed** |

### Scope, so this is not over read

- **One corpus, all markdown.** Nothing here measures the `content_blocks` extraction path
  `bd582316` added for other media types, where `text_start` and `text_end` are stored as `None`
  (`recall/index.py:800`). A PDF corpus would additionally require the extractor to be
  deterministic across versions, which is a strictly harder problem and is untested.
- **Reproduction shows an observationally equivalent chunker, not the identical one.** A different
  implementation producing identical output on these 1,058 sources is indistinguishable here. That
  is the same standard the embedder attestation meets with cosine 1.0, and it should be claimed no
  more strongly.
- **The candidate set was four and was fixed before measuring.** It was not widened, and the
  100 percent is a result about those four, not a search over all possible chunkers.
- The comparison is exact string list equality, which is stricter than retrieval cares about. Two
  chunkings differing by a trailing newline would retrieve near identically and be counted as a
  mismatch, so the reproduction rate is a lower bound on practical equivalence.

## Correction, appended 2026-08-18 after `79a0d6ed`

**Appended, not edited.** The text above stands as written and measured. This records that one
premise it rests on stopped being literally true a few hours after it was registered, and states
exactly how much of the result that costs, which is nothing.

**What changed.** `79a0d6ed` (#381) widened `_index_fingerprint` (`recall/index.py:420`) to hash
`EmbeddingProfile.fingerprint()` instead of `embedding_profile_id(embedder)`. That fingerprint
covers `chunker_version` (`recall/embeddings.py:412`), so the sentence "`_index_fingerprint` has no
chunker term either" is now imprecise: a field of that NAME is in the hash.

**What did not change, which is every conclusion drawn from it.** `chunker_version` is a field of
the EMBEDDING profile, defaulted to `chunk-text-v1` at both of its definitions and set by nothing
else in the tree. The `Indexer`'s actual chunker (`recall/index.py:535`) never reaches it, and
neither `max_chars` nor `overlap` appears in an `EmbeddingProfile` at all. So the term is inert with
respect to the thing it is named after. Measured against `79a0d6ed`, holding the embedder and the
file fixed and varying only the chunker:

| chunking | chunks produced | index fingerprint |
|---|---|---|
| `chunk_text(max_chars=800, overlap=80)` | 1 | `78e3179317b6a7d556b9…` |
| `chunk_text(max_chars=60, overlap=10)` | 4 | `78e3179317b6a7d556b9…` |
| `chunk_code` | 1 | `78e3179317b6a7d556b9…` |

Three genuinely different chunkings, one fingerprint. So all three load bearing claims survive
unchanged, and they should be read as being about the chunker CONFIGURATION rather than about the
absence of any field:

- an adopted generation's `ChunkerIdentity` is still an assertion with nothing behind it;
- re indexing still does not repair a chunker change, because the skip guard still reports the file
  unchanged;
- the attestation must still IDENTIFY by re deriving and re chunking, because there is still no
  stated chunker configuration to verify against.

**The correct statement going forward**, replacing the one sentence: *the index fingerprint carries
a coarse `chunker_version` string from the embedding profile, and no chunker configuration; no
change to the chunker actually in use moves it.*

⚠️ **What this DOES change is a future assumption, not a past result.** `chunker_version` is now
key material for the skip guard, so if anything ever begins setting it per chunker, a chunker change
would start forcing a re index. Nothing sets it today. Anyone adding that should know it is now a
re embed trigger and not merely a label.
