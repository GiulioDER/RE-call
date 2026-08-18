# Pre registration: is non markdown extraction reproducible enough to attest?

**Date:** 2026-08-18   **Status:** predicted, not yet measured

## The question

`docs/UNCALIBRATED_FIRST_RUN_DESIGN.md` section 6c scoped the chunker attestation to markdown and
said the `content_blocks` extraction path was out of scope "until that is measured". This measures
it.

The chunker attestation works because markdown body derivation is pure Python inside this
repository: `parse_frontmatter` plus `chunk_text`. Non markdown is not. `extract_document`
(`recall/extraction.py:156`) dispatches to **six third party libraries** (`pdfplumber`,
`python-docx`, `openpyxl`, `xlrd`, `python-pptx`, `beautifulsoup4`, plus `oxmsg` for `.msg`) and,
for five suffixes, to an **external LibreOffice binary** (`recall/extraction.py:565`).

**Q1 (run to run determinism).** Does `extract_document` produce byte identical output across two
independent processes on one machine, for every format that can be exercised here?

**Q2 (environment leakage).** Does any extractor put environment varying content into the extracted
text or metadata: a temporary directory path, an absolute path, a user name, or a timestamp? Any of
those makes output differ between machines even at identical library versions, which would defeat a
re extraction comparison outright.

**Q3 (external binary surface).** How many of the supported extensions depend on a binary that is
not a Python dependency and therefore cannot be pinned by the lockfile?

## What I predict

**Q1. Every testable format is byte identical across processes.** Extraction reads content rather
than rendering it, so there is no sampling, no layout engine output and no randomness in the path.

**Q2. No format leaks environment varying content: 0 of N.** The LibreOffice path is the one I would
bet against, because it writes a temporary profile and converts through an intermediate file, but
what is read back is the converted document's text and I expect the temp path to stay out of it.

**Q3. Five reachable suffixes** (`.doc`, `.odt`, `.ods`, `.odp`, `.ppt`) route to LibreOffice, out
of the 24 in `DOCUMENT_EXTENSIONS`. A sixth, `.msg`, appears in that branch but is **unreachable**:
`extract_document` matches `.msg` earlier at `recall/extraction.py:176`.

## What would falsify this

- **Q1** is falsified by any format differing across processes. That would mean re extraction cannot
  be compared at all for that format, and the attestation for it is impossible rather than merely
  environment dependent.
- **Q2** is falsified by any format leaking. That is the more damaging outcome, because it survives
  identical library versions and breaks comparison **between** machines, which is exactly the case an
  adoption path faces.
- **Q3** is a code fact rather than a prediction and is stated so it can be checked, not scored.

⚠️ **The design conclusion does not depend on Q1 passing, and I am committing to that here so a
pass cannot be read as more than it is.** Run to run determinism on one machine at one set of
library versions is **necessary and not sufficient**. Even a clean sweep leaves the attestation
unable to distinguish "the extractor version changed" from "the file is wrong", because nothing
records which extractor ran. What Q1 and Q2 decide is whether an extraction attestation is possible
*at all*; what makes it *sound* is a recorded identity, and that is a separate change.

## How it will be measured

1. Generate one small file per format that can be produced deterministically here: `.txt`, `.md`,
   `.csv`, `.tsv`, `.html`, `.eml`, `.docx`, `.xlsx`, `.pptx`, and `.odt` / `.ods` / `.odp` / `.doc`
   / `.ppt` by LibreOffice conversion from the modern equivalents. `.msg` is skipped because `oxmsg`
   is not installed here, and its absence is reported rather than worked around.
2. For each, run `extract_document` in **two separate Python processes** and canonically digest the
   whole `ExtractedDocument`: `text`, `media_type`, `metadata`, and every block's `text`, `kind` and
   `metadata`. Compare digests.
3. Scan each extraction's text and metadata for the temporary directory path used during the run,
   the OS user name, any absolute path, and the current date in several formats.
4. Report per format: deterministic yes or no, leakage yes or no, and whether it routed through
   LibreOffice.
5. **Apparatus check, run first:** deliberately corrupt one extraction result and confirm the
   digest comparison reports a difference. A comparison that cannot fail would report every format
   as deterministic.

## What I already know

- `STRUCTURED_DOCUMENT_VERSION = "table-row-groups-v1"` (`recall/extraction.py:130`) versions
  recall's own **block shape** and is carried in `_index_fingerprint`. It says nothing about the
  third party libraries, so a `pdfplumber` upgrade changes extracted text without changing any
  recorded version.
- Chunk metadata on this path does carry `source_format`, `extraction` and `media_type`, via
  `{**extracted.metadata, "media_type": ...}` (`recall/index.py:762`), but **no library versions and
  no LibreOffice version**.
- The precedent for recording them exists and is deliberate: `EmbeddingProfile.dependencies`
  (`recall/embeddings.py:414`) carries the inference library version as key material, and its
  docstring states that a `fastembed` upgrade costs a re embed on purpose, "because ONNX runtime
  changes are free to move the last bits of a vector and a cache cannot tell". The identical
  argument applies to an extractor.
- `text_start` and `text_end` are stored as `None` on this path (`recall/index.py:816`), so the
  offset cross check available for markdown does not exist here.

## Confounds I can name now

One machine, one operating system, one set of library versions, one LibreOffice build. Determinism
across **versions** is the question that actually matters for an adoption path and it is **not**
tested here; testing it needs two installed versions of each library, which is a different
experiment. So a clean Q1 is the weakest of the three possible good outcomes.

Generated fixtures are small and simple. A real corpus has scanned PDFs, multi column layouts,
embedded fonts and tables that straddle pages, all of which exercise far more extractor code than
anything generated here. A determinism result on simple inputs does not transfer to those.

`.msg` is untested because `oxmsg` is absent, and `.epub` and `.rtf` are generated by hand rather
than by an authoring tool, so they may be unrepresentatively regular.
