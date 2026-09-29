from __future__ import annotations

import gc
import weakref
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


def _biff8_workbook(sheets: list[list[list[float]]]) -> bytes:
    """Build a bare BIFF8 workbook stream of NUMBER cells, which xlrd reads without an OLE wrapper.

    No XLS writer is a dependency of this project, and xlrd parses a stream with no compound
    document container through the same sheet loading code as a wrapped one.
    """
    import struct

    def record(kind: int, payload: bytes = b"") -> bytes:
        return struct.pack("<HH", kind, len(payload)) + payload

    def bof(stream_type: int) -> bytes:
        return record(0x0809, struct.pack("<HHHHII", 0x0600, stream_type, 0, 0, 0, 0))

    bodies = []
    for rows in sheets:
        width = max(len(row) for row in rows)
        dimensions = record(0x0200, struct.pack("<IIHHH", 0, len(rows), 0, width, 0))
        cells = b"".join(
            record(0x0203, struct.pack("<HHHd", row_index, column, 0, value))
            for row_index, row in enumerate(rows)
            for column, value in enumerate(row)
        )
        bodies.append(bof(0x0010) + dimensions + cells + record(0x000A))
    names = [f"Sheet{index}".encode("latin-1") for index in range(len(sheets))]
    globals_length = len(bof(0x0005)) + sum(4 + 8 + len(name) for name in names) + 4
    parts, offset = [bof(0x0005)], globals_length
    for name, body in zip(names, bodies, strict=True):
        parts.append(record(0x0085, struct.pack("<IBB", offset, 0, 0) + bytes([len(name), 0]) + name))
        offset += len(body)
    parts.append(record(0x000A))
    return b"".join(parts + bodies)


def test_xls_sheets_are_read_one_at_a_time_so_memory_stays_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While any XLS sheet is being read, no other sheet is loaded or even alive.

    Invariant: `_extract_spreadsheet` parses a sheet, reads it and releases it before parsing the
    next, so one parsed sheet exists at a time. `xlrd.Book.sheets()` parses every sheet before
    returning and the workbook held all of them to the end; measured 2026-09-29, see the pull
    request for the numbers.

    Two conditions, because either alone lets memory grow:

    - the workbook no longer holds the sheet (`Book.sheet_loaded`), which `unload_sheet` controls;
    - the sheet object is gone. An xlrd 2.0 `Sheet` is in a reference cycle (`self.put_cell` is a
      bound method of itself), so an unloaded sheet otherwise waits for the cyclic collector, which
      in measurement left about two more sheets resident. The collector is disabled here, so only
      reference counting can free a sheet and the check is deterministic.

    Red proof, 2026-09-29, for
    `tests/test_extraction.py::test_xls_sheets_are_read_one_at_a_time_so_memory_stays_bounded`:

    - against the unfixed code at `8f9e504f` (`for sheet in workbook.sheets()`): fails the
      loaded-count assertion, every read sees 3 sheets loaded;
    - mutation deleting only `workbook.unload_sheet(index)` in `_xls_sheet_section`: fails the
      loaded-count assertion, reads see 1, 2 and 3 sheets loaded;
    - mutation deleting only `vars(sheet).pop("put_cell", None)`: fails the alive-count assertion,
      reads see 1, 2 and 3 sheets alive.

    Green with the fix restored. Failure texts are recorded in the pull request.
    """
    xlrd = pytest.importorskip("xlrd")
    path = tmp_path / "ledger.xls"
    path.write_bytes(
        _biff8_workbook(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ]
        )
    )

    books: list[Any] = []
    parsed: list[weakref.ref[Any]] = []
    seen: list[tuple[str, int, int]] = []
    real_open_workbook = xlrd.open_workbook
    real_get_sheet = xlrd.book.Book.get_sheet
    real_row_values = xlrd.sheet.Sheet.row_values

    def open_workbook(*args: Any, **kwargs: Any) -> Any:
        book = real_open_workbook(*args, **kwargs)
        books.append(book)
        return book

    def get_sheet(self: Any, sh_number: int, *args: Any, **kwargs: Any) -> Any:
        sheet = real_get_sheet(self, sh_number, *args, **kwargs)
        parsed.append(weakref.ref(sheet))
        return sheet

    def row_values(self: Any, rowx: int, *args: Any, **kwargs: Any) -> Any:
        book = books[-1]
        loaded = sum(book.sheet_loaded(index) for index in range(book.nsheets))
        alive = sum(ref() is not None for ref in parsed)
        seen.append((self.name, loaded, alive))
        return real_row_values(self, rowx, *args, **kwargs)

    monkeypatch.setattr(xlrd, "open_workbook", open_workbook)
    monkeypatch.setattr(xlrd.book.Book, "get_sheet", get_sheet)
    monkeypatch.setattr(xlrd.sheet.Sheet, "row_values", row_values)

    collector_was_enabled = gc.isenabled()
    gc.disable()
    try:
        result = extract_document(path, path.read_bytes())
    finally:
        if collector_was_enabled:
            gc.enable()

    # The spy must have seen every sheet being read, or the checks below are vacuous.
    assert sorted({name for name, _, _ in seen}) == ["Sheet0", "Sheet1", "Sheet2"]
    assert [loaded for _, loaded, _ in seen] == [1] * len(seen), seen
    assert [alive for _, _, alive in seen] == [1] * len(seen), seen
    assert result.metadata["table_count"] == 3
    assert result.text == (
        "## Sheet: Sheet0\n\n| 1.0 | 2.0 |\n| --- | --- |\n| 3.0 | 4.0 |\n\n"
        "## Sheet: Sheet1\n\n| 5.0 | 6.0 |\n| --- | --- |\n\n"
        "## Sheet: Sheet2\n\n| 7.0 | 8.0 |\n| --- | --- |\n| 9.0 | 10.0 |\n| 11.0 | 12.0 |"
    )


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
