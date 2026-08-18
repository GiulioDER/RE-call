# Pre registration: is non markdown extraction reproducible enough to attest?

**Date:** 2026-08-18   **Status:** measured 2026-08-18. Q1 and Q2 confirmed, 17 of 17 formats
deterministic with no environment leakage. ⚠️ **A clean sweep here does NOT make an extraction
attestation sound**, which was committed in writing before the run. Predictions and falsifiers below
are unedited; the result is appended below the horizontal rule.

## The question

`docs/UNCALIBRATED_FIRST_RUN_DESIGN.md` section 6c scoped the chunker attestation to markdown and
said the `content_blocks` extraction path was out of scope "until that is measured". This measures
it.

The chunker attestation works because markdown body derivation is pure Python inside this
repository: `parse_frontmatter` plus `chunk_text`. Non markdown is not. `extract_document`
(`recall/extraction.py:164`) dispatches to **six third party libraries** (`pdfplumber`,
`python-docx`, `openpyxl`, `xlrd`, `python-pptx`, `beautifulsoup4`, plus `oxmsg` for `.msg`) and,
for five suffixes, to an **external LibreOffice binary** (`recall/extraction.py:573`).

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
`extract_document` matches `.msg` earlier at `recall/extraction.py:184`.

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

- `STRUCTURED_DOCUMENT_VERSION = "table-row-groups-v1"` (`recall/extraction.py:138`) versions
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

---

## Result

Measured 2026-08-18 on one Windows machine, LibreOffice 26.2.5.2, `pdfplumber` 0.11.10,
`python-docx` 1.2.0, `openpyxl` 3.1.5, `xlrd` 2.0.2, `python-pptx` 1.0.2, `beautifulsoup4` 4.15.0.
**Q1 and Q2 both confirmed. Neither makes an attestation sound**, which was stated before measuring
and is restated below because a clean sweep is exactly the result most likely to be over read.

### ⚠️ Two citations above the rule have since moved

The registered sections are left exactly as written, because a prediction is a historical record.
Two of their citations no longer resolve to what they described, both because `recall/index.py`
grew after registration:

| Cited | Was | Is now |
|---|---|---|
| `recall/index.py:782` | the `{**extracted.metadata, "media_type": ...}` dict | `recall/index.py:809` |
| `recall/index.py:836` | `text_start` / `text_end` stored as `None` | `recall/index.py:861` |

Neither claim changed; only the line numbers did. This is the fifth such drift in this document's
short life, which is why the design now states requirements as behaviour rather than as the presence
or absence of a field.

### Apparatus check, run first

A deliberately corrupted extraction digest was reported as differing. Without that, every
"deterministic" verdict below would be the output of a comparison that cannot fail.

### Q1 and Q2

| | value |
|---|---:|
| formats exercised | **17** |
| byte identical across two independent processes | **17 of 17** |
| leaking a temp path, working directory, user name or today's date | **0 of 17** |
| failed to extract | 0 |
| routed through LibreOffice | 5 (`.doc`, `.odt`, `.ods`, `.odp`, `.ppt`) |

Every format tested: `.csv`, `.doc`, `.docx`, `.eml`, `.epub`, `.html`, `.md`, `.odp`, `.ods`,
`.odt`, `.pdf`, `.ppt`, `.pptx`, `.rtf`, `.tsv`, `.txt`, `.xlsx`. The five LibreOffice formats,
which I named as the ones worth betting against, were as stable as the pure Python ones.

### Q3, a code fact rather than a prediction

Five reachable suffixes route to LibreOffice. A sixth, `.msg`, appears in that branch at
`recall/extraction.py:195` but is **unreachable**, because `extract_document` matches `.msg`
earlier at `:188`. So a deployment without `python-oxmsg` gets an extraction error where the code
appears to offer a LibreOffice fallback.

🔁 **Fixed upstream by `64ffee52` (#389), which this finding prompted, and the citation above is now
a record of the measured tree rather than of the code.** The branch reads
`if suffix in LIBREOFFICE_EXTENSIONS:`, and that constant is
`frozenset({".doc", ".odt", ".ods", ".odp", ".ppt"})` (`recall/extraction.py:133`): `.msg` is no
longer in it, so the dead offer is gone rather than merely unreachable. Repointing the line number
alone would have sent a reader to a branch where `.msg` does not appear, which is why this note
exists instead of a silent renumber. The Q3 finding itself stands as measured.

### ⚠️ What this result is NOT

Registered in advance, and it holds: **run to run determinism on one machine at one set of library
versions is necessary and not sufficient.** Nothing here tests determinism across *versions*, which
is the question an adoption path actually faces, and testing it needs two installed versions of each
library. A clean sweep is the weakest of the three good outcomes that were possible.

Three further limits, none of them small:

- **The fixtures are trivial**: extracted text ran from 6 to 124 characters. A real corpus has
  scanned PDFs, multi column layouts, embedded fonts and tables straddling pages, all of which
  exercise far more extractor code than anything here. A determinism result on a 6 character HTML
  extraction is close to no evidence about PDF extraction in the field.
- **`.msg` is untested** because `python-oxmsg` is not installed here, and the alias suffixes
  (`.docm`, `.dotx`, `.dotm`, `.xlsm`, `.xltx`, `.xltm`, `.pptm`, `.potx`, `.potm`) were not
  exercised individually.
- **One operating system.** The LibreOffice result in particular is a Windows result.

### An implementation constraint found while designing around this

`soffice --version` exits 0 and prints **nothing** on Windows. A naive implementation of an
extraction identity would therefore record an empty LibreOffice version and compare equal across
upgrades, which is precisely the failure the identity exists to prevent. The version is obtainable
by other means: the binary's file version metadata reports **26.2.5.2**, and a `version.ini` sits in
the install directory.

### Verdict against the prediction

| Registered prediction | Measured | Verdict |
|---|---|---|
| every testable format byte identical across processes | 17 of 17 | **confirmed** |
| no format leaks environment varying content | 0 of 17 | **confirmed** |
| five reachable suffixes route to LibreOffice, `.msg` unreachable | as stated | **confirmed as a code fact** |
| (pre committed) a pass does not make attestation sound | unchanged | **holds** |
