from pathlib import Path

import pytest

from recall.extraction import (
    MAX_EXTRACTED_CHARACTERS,
    DocumentExtractionError,
    chunk_extracted_document,
    extract_document,
)
from recall.index import candidate_files


def test_csv_table_chunks_repeat_headers_and_record_numeric_values(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text(
        "Year,Revenue,Margin\n2023,42,12.5%\n2024,51,14.0%\n",
        encoding="utf-8",
    )

    document = extract_document(path, path.read_bytes())
    chunks = chunk_extracted_document(document, max_chars=90)

    assert len(chunks) >= 2
    text, metadata = chunks[0]
    assert "| Year | Revenue | Margin |" in text
    assert metadata["content_kind"] == "table"
    assert "42" in metadata["numeric_values"]
    assert metadata["table_headers"] == ["Year", "Revenue", "Margin"]


def test_default_index_scan_accepts_documents_but_excludes_known_secret_files(tmp_path: Path) -> None:
    (tmp_path / "memo.md").write_text("memo", encoding="utf-8")
    (tmp_path / "report.pdf").write_bytes(b"placeholder")
    (tmp_path / "tokens.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"not indexed")

    files = candidate_files(tmp_path)

    assert {path.name for path in files} == {"memo.md", "report.pdf"}


def test_default_index_scan_excludes_hidden_filenames(tmp_path: Path) -> None:
    (tmp_path / ".hidden.txt").write_text("private", encoding="utf-8")
    (tmp_path / "queries.json").write_text('{"answer_key": "private"}', encoding="utf-8")
    (tmp_path / "visible.txt").write_text("visible", encoding="utf-8")

    assert [path.name for path in candidate_files(tmp_path)] == ["visible.txt"]


def test_default_index_scan_requires_explicit_opt_in_for_configuration_formats(
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text('{"tenant": "default"}', encoding="utf-8")

    assert candidate_files(tmp_path) == []
    assert candidate_files(tmp_path, glob="**/*.json") == [settings]


def test_text_extraction_rejects_empty_and_oversized_sources(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    with pytest.raises(DocumentExtractionError, match="did not contain extractable"):
        extract_document(empty, b"")

    huge = tmp_path / "huge.txt"
    payload = b"x" * (MAX_EXTRACTED_CHARACTERS + 1)
    huge.write_bytes(payload)
    with pytest.raises(DocumentExtractionError, match="more than"):
        extract_document(huge, payload)


def test_table_chunks_never_exceed_cap_with_long_headers(tmp_path: Path) -> None:
    path = tmp_path / "long.csv"
    path.write_text("H" * 100 + ",Revenue\nEU,123.45\n", encoding="utf-8")

    document = extract_document(path, path.read_bytes())
    chunks = chunk_extracted_document(document, max_chars=40, overlap=0)

    assert chunks
    assert all(len(text) <= 40 for text, _ in chunks)
