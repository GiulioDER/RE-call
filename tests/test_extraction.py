from __future__ import annotations

from pathlib import Path
from typing import Any

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


def _text_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with one line of Helvetica text per page, with no PDF library."""
    count = len(pages)
    font_id = 3 + 2 * count
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids ["
        + b" ".join(f"{3 + 2 * index} 0 R".encode() for index in range(count))
        + f"] /Count {count} >>".encode(),
    ]
    for index, line in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({line}) Tj ET".encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * index} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_pdf_pages_are_closed_as_they_are_read_so_memory_stays_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each PDF page must release its parse caches before the next page is read.

    pdfplumber keeps every parsed page's layout and object caches until the file closes, so a
    loop that never calls ``page.close()`` holds all of them at once and its memory grows with
    page count: a 198 page 10-K peaked at 1.50 GB RSS and was OOM-killed under a 1.5 GB cap
    (2026-09-29, pdfplumber 0.11.10), against 157 MB for a 468 page PDF with the close.

    Asserting only that ``close()`` was called would prove nothing, because ``PDF.close()``
    closes every page on exit from the ``with`` block anyway. The invariant is ORDER: when page
    N is opened for reading, pages 1..N-1 are already closed, so at most one page is live.

    Red proof, 2026-09-29: with the ``page.close()`` line in ``recall.extraction._extract_pdf``
    deleted, this test fails on the ``live_when_read`` assertion (page 2 is read while page 1
    is still open), then passes with the line restored.
    """
    page_module = pytest.importorskip("pdfplumber.page")
    page_class = page_module.Page
    open_pages: set[int] = set()
    live_when_read: list[tuple[int, list[int]]] = []

    real_extract_text = page_class.extract_text
    real_close = page_class.close

    def spy_extract_text(self: Any, *args: Any, **kwargs: Any) -> Any:
        live_when_read.append((self.page_number, sorted(open_pages - {self.page_number})))
        open_pages.add(self.page_number)
        return real_extract_text(self, *args, **kwargs)

    def spy_close(self: Any) -> None:
        open_pages.discard(self.page_number)
        real_close(self)

    monkeypatch.setattr(page_class, "extract_text", spy_extract_text)
    monkeypatch.setattr(page_class, "close", spy_close)

    lines = [f"Page {number} holds fact {number * 7}" for number in range(1, 6)]
    path = tmp_path / "report.pdf"
    path.write_bytes(_text_pdf(lines))
    document = extract_document(path, path.read_bytes())

    assert [number for number, _ in live_when_read] == [1, 2, 3, 4, 5]
    assert live_when_read == [(number, []) for number in range(1, 6)], (
        "a page was read while an earlier page still held its caches: "
        f"{live_when_read}"
    )
    for number, line in enumerate(lines, start=1):
        assert f"## Page {number}\n\n{line}" in document.text
    assert document.metadata["source_format"] == "pdf"
