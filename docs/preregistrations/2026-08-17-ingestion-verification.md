# Pre registration: multi format ingestion verification

**Date:** 2026-08-17   **Status:** predicted, not yet measured

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
