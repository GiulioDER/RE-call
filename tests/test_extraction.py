from __future__ import annotations

from pathlib import Path

import pytest

from recall.extraction import DocumentExtractionError, extract_document, extraction_supported


def test_text_and_csv_are_normalized_to_searchable_utf8(tmp_path: Path) -> None:
    text_path = tmp_path / "memo.md"
    text_path.write_bytes("# Decision\n\nUse the safer path.".encode("utf-8"))
    text = extract_document(text_path, text_path.read_bytes())
    assert text.text.startswith("# Decision")
    assert text.metadata["source_format"] == "md"

    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("Metric,Value\nRevenue,42\n", encoding="utf-8")
    table = extract_document(csv_path, csv_path.read_bytes())
    assert "| Metric | Value |" in table.text
    assert "| Revenue | 42 |" in table.text
    assert table.metadata["table_count"] == 1


def test_docx_tables_are_extracted(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    path = tmp_path / "report.docx"
    document = docx.Document()
    document.add_paragraph("Quarterly report")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Revenue"
    table.cell(1, 1).text = "42"
    document.save(path)

    result = extract_document(path, path.read_bytes())
    assert "Quarterly report" in result.text
    assert "| Revenue | 42 |" in result.text
    assert result.metadata["table_count"] == 1


def test_xlsx_sheets_are_extracted(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "metrics.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Summary"
    sheet.append(["Metric", "Value"])
    sheet.append(["Revenue", 42])
    workbook.save(path)

    result = extract_document(path, path.read_bytes())
    assert "## Sheet: Summary" in result.text
    assert "| Revenue | 42 |" in result.text
    assert result.metadata["table_count"] == 1


def test_supported_office_extensions_are_not_silently_dropped() -> None:
    for suffix in (
        ".pdf",
        ".doc",
        ".docm",
        ".docx",
        ".xls",
        ".xlsm",
        ".xlsx",
        ".ppt",
        ".pptm",
        ".pptx",
        ".odt",
        ".tsv",
    ):
        assert extraction_supported("source" + suffix)


def test_invalid_pdf_reports_an_extraction_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a PDF")
    with pytest.raises(DocumentExtractionError, match="could not extract PDF"):
        extract_document(path, path.read_bytes())
