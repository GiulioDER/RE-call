# Pre registration: multi format ingestion verification

**Date:** 2026-08-17   **Status:** measured

## The question

Does the current RE call ingestion path correctly extract and type every supported document family,
preserve table provenance, reject malformed input safely, and exclude known secret files from the
default scan?

## What I predict

All valid core fixtures will extract successfully, every extracted table chunk will retain headers
and numeric metadata, malformed fixtures will raise a typed extraction error rather than silently
indexing empty content, and the safe default scan will exclude known secret names and unsupported
binary files.

## What would falsify this

Any valid core fixture is silently dropped or produces no searchable text, any table chunk loses its
headers or source metadata, malformed input returns an apparently successful empty document, or the
default scan admits a known secret fixture.

## How it will be measured

Run an in memory live fixture matrix through `recall.extraction.extract_document` and
`chunk_extracted_document`, run malformed and boundary probes, run `candidate_files` against a
temporary corpus, and run static checks with mypy, Ruff, and Python compilation. The matrix will
cover text, Markdown, JSON, YAML, TOML, XML, CSV, TSV, HTML, RTF, EML, PDF, DOCX, XLSX, and PPTX.

## What I already know

The implementation uses a single text embedding space, typed table blocks, and a structured document
chunking version. The live PostgreSQL index path cannot be exercised unless the current environment
provides psycopg and a pgvector database.

## Confounds I can name now

Optional parser packages, LibreOffice availability, malformed fixture validity, and PostgreSQL
availability can distinguish an implementation failure from an environment failure. I will report
those separately.

## Result (2026-08-18)

**Status:** measured

The first live matrix exposed three implementation defects. Plain text bypassed the extracted
character limit, long table headers could produce chunks larger than the requested cap, and the
safe default scan admitted hidden filenames. I fixed all three and reran the probes.

Measured:

* 30 valid files passed across modern formats and aliases: text, Markdown, JSON, YAML, TOML, XML,
  CSV, TSV, HTML, RTF, EML, PDF, DOCX, XLS, XLSX, and PPTX.
* 12 malformed or unavailable files raised `DocumentExtractionError`, including all seven legacy
  LibreOffice dependent formats tested with invalid payloads.
* Empty text, oversized text, long table headers, table headers and numeric metadata, and safe
  default discovery all passed after the fixes.
* The live normal `Indexer.index_path` path indexed two files into a recording store and preserved
  table metadata and provenance.
* Focused regression checks: 127 passed, 11 skipped. Static typing, Ruff, compilation, and diff
  checks passed.
* The final repository-wide run passed 4,928 tests and skipped 530 environment-dependent tests.
  The one environment-sensitive Voyage test was made deterministic by clearing the local
  credential in its test precondition.

**Prediction gap:** The prediction was partially falsified by the three boundary defects. The
format parsers and typed failure behavior matched the prediction once those defects were repaired.

The remaining infrastructure gap is a real PostgreSQL plus pgvector run. `RECALL_TEST_DSN` is
unset in this checkout, so database backed tests skip rather than claiming live index or query
coverage.

## Gap closure (2026-08-18)

The environmental gaps were closed with scoped live fixtures.

* A disposable PostgreSQL plus pgvector container ran the database-backed ingestion and query
  coverage. The focused run passed 79 tests. The final full live database run passed 5,430 tests
  and skipped 31. An earlier full run exposed one real boundary issue: the default recursive scan
  admitted the generated `queries.json` artifact into sparse coverage. I excluded that known
  artifact, reran the affected test, and the complete final suite passed.
* LibreOffice was discovered at the standard Windows installation path. A unique temporary
  LibreOffice profile is now used for every conversion, preventing profile-lock and stale-process
  interference. Valid DOC, ODT, ODS, ODP, and PPT fixtures extracted successfully.
* EPUB now uses a native ZIP, OPF, and XHTML extraction path and passed a valid fixture. MSG uses
  the optional `extract-msg` parser, passed a real valid fixture, and rejected malformed input with
  a typed extraction error. The optional dependency is declared in the `documents` extra.
* Static checks passed: Ruff, mypy, Python compilation, and diff checks. The legacy-format test
  module passed 3 tests.

The final full live suite was rerun after the `queries.json` exclusion, so no unverified ingestion
defect remains in the covered paths.
