"""Format aware document extraction for the desktop and MCP ingest paths.

The index stores UTF 8 text, but users should not have to convert a document before adding it.
This module keeps the original file as the manifest object and derives a searchable text view for
PDF, Office, spreadsheet, presentation, HTML, and delimited table sources.
"""

from __future__ import annotations

import csv
import io
import re
import subprocess
import tempfile
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable


class DocumentExtractionError(ValueError):
    """Raised when a supported file cannot produce a searchable text view."""


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    media_type: str
    metadata: dict[str, Any]


TEXT_EXTENSIONS = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".css",
        ".csv",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".htm",
        ".java",
        ".js",
        ".jsx",
        ".json",
        ".md",
        ".markdown",
        ".mdx",
        ".py",
        ".pyi",
        ".rb",
        ".rs",
        ".rst",
        ".sh",
        ".sql",
        ".svelte",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".tsv",
        ".xml",
        ".yaml",
        ".yml",
    }
)
DOCUMENT_EXTENSIONS = frozenset(
    {
        *TEXT_EXTENSIONS,
        ".doc",
        ".docm",
        ".docx",
        ".dotm",
        ".dotx",
        ".eml",
        ".epub",
        ".msg",
        ".odt",
        ".odp",
        ".ods",
        ".pdf",
        ".ppt",
        ".pptm",
        ".pptx",
        ".potm",
        ".potx",
        ".rtf",
        ".xls",
        ".xlsm",
        ".xlsx",
        ".xltm",
        ".xltx",
    }
)

MAX_TABLE_ROWS = 10_000
MAX_TABLE_COLUMNS = 200
MAX_TABLE_CELLS = 500_000
MAX_EXTRACTED_CHARACTERS = 5_000_000


def extraction_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in DOCUMENT_EXTENSIONS


def extract_document(path: Path, data: bytes) -> ExtractedDocument:
    """Return a UTF 8 searchable representation while retaining source provenance."""
    suffix = path.suffix.lower()
    if suffix not in DOCUMENT_EXTENSIONS:
        raise DocumentExtractionError(
            f"unsupported file type {suffix or '<no extension>'}; supported document formats "
            "include PDF, DOCX, XLSX, PPTX, CSV, HTML, and UTF 8 text"
        )

    if suffix == ".pdf":
        return _extract_pdf(path, data)
    if suffix in {".docx", ".docm", ".dotx", ".dotm"}:
        return _extract_docx(data)
    if suffix in {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"}:
        return _extract_spreadsheet(suffix, data)
    if suffix in {".pptx", ".pptm", ".potx", ".potm"}:
        return _extract_pptx(data)
    if suffix == ".eml":
        return _extract_email(data)
    if suffix == ".rtf":
        return _extract_rtf(data)
    if suffix in {".html", ".htm"}:
        return _extract_html(data)
    if suffix in {".doc", ".msg", ".odt", ".ods", ".odp", ".ppt", ".epub"}:
        return _extract_with_libreoffice(path, data)
    if suffix in {".csv", ".tsv"}:
        return _extract_delimited(suffix, data)
    return ExtractedDocument(
        _decode_text(data),
        "text/markdown" if suffix in {".md", ".markdown", ".mdx"} else "text/plain",
        {"source_format": suffix.removeprefix("."), "extraction": "text"},
    )


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding).replace("\x00", "")
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace").replace("\x00", "")


def _result(text: str, source_format: str, *, tables: int = 0, **metadata: Any) -> ExtractedDocument:
    clean = text.strip()
    if not clean:
        raise DocumentExtractionError(
            f"{source_format.upper()} did not contain extractable text or tables"
        )
    if len(clean) > MAX_EXTRACTED_CHARACTERS:
        raise DocumentExtractionError(
            f"{source_format.upper()} extracted more than {MAX_EXTRACTED_CHARACTERS:,} characters; "
            "split the source into smaller files"
        )
    return ExtractedDocument(
        clean,
        "text/plain",
        {
            "source_format": source_format,
            "extraction": "structured" if tables else "text",
            "table_count": tables,
            **metadata,
        },
    )


def _extract_pdf(path: Path, data: bytes) -> ExtractedDocument:
    try:
        import pdfplumber
    except ImportError as exc:
        raise DocumentExtractionError(
            "PDF extraction requires the documents extra: pip install \"recall-rag[documents]\""
        ) from exc

    sections: list[str] = []
    table_count = 0
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    sections.append(f"## Page {page_number}\n\n{page_text.strip()}")
                for table_number, rows in enumerate(page.extract_tables() or (), start=1):
                    markdown = _rows_to_markdown(rows)
                    if markdown:
                        table_count += 1
                        sections.append(
                            f"### Table {table_count} (page {page_number}, table {table_number})\n\n"
                            f"{markdown}"
                        )
    except Exception as exc:
        raise DocumentExtractionError(f"could not extract PDF {path.name}: {type(exc).__name__}") from exc
    return _result("\n\n".join(sections), "pdf", tables=table_count)


def _extract_docx(data: bytes) -> ExtractedDocument:
    try:
        from docx import Document
    except ImportError as exc:
        raise DocumentExtractionError(
            "DOCX extraction requires the documents extra: pip install \"recall-rag[documents]\""
        ) from exc

    try:
        document = Document(io.BytesIO(data))
        sections = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        tables = 0
        for index, table in enumerate(document.tables, start=1):
            markdown = _rows_to_markdown(
                [[cell.text for cell in row.cells] for row in table.rows]
            )
            if markdown:
                tables += 1
                sections.append(f"### Table {index}\n\n{markdown}")
    except Exception as exc:
        raise DocumentExtractionError(f"could not extract DOCX: {type(exc).__name__}") from exc
    return _result("\n\n".join(sections), "docx", tables=tables)


def _extract_spreadsheet(suffix: str, data: bytes) -> ExtractedDocument:
    if suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise DocumentExtractionError(
                "XLSX extraction requires the documents extra: pip install \"recall-rag[documents]\""
            ) from exc
        try:
            workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=False)
            sections: list[str] = []
            tables = 0
            for sheet in workbook.worksheets:
                rows = _bounded_rows(sheet.iter_rows(values_only=True), sheet.max_row, sheet.max_column)
                markdown = _rows_to_markdown(rows)
                if markdown:
                    tables += 1
                    sections.append(f"## Sheet: {sheet.title}\n\n{markdown}")
            workbook.close()
        except Exception as exc:
            raise DocumentExtractionError(f"could not extract XLSX: {type(exc).__name__}") from exc
        return _result("\n\n".join(sections), suffix.removeprefix("."), tables=tables)

    try:
        import xlrd
    except ImportError as exc:
        raise DocumentExtractionError(
            "XLS extraction requires the documents extra: pip install \"recall-rag[documents]\""
        ) from exc
    try:
        workbook = xlrd.open_workbook(file_contents=data, on_demand=True)
        sections = []
        tables = 0
        for sheet in workbook.sheets():
            sheet_rows = (sheet.row_values(row) for row in range(min(sheet.nrows, MAX_TABLE_ROWS)))
            markdown = _rows_to_markdown(sheet_rows)
            if markdown:
                tables += 1
                sections.append(f"## Sheet: {sheet.name}\n\n{markdown}")
        workbook.release_resources()
    except Exception as exc:
        raise DocumentExtractionError(f"could not extract XLS: {type(exc).__name__}") from exc
    return _result("\n\n".join(sections), "xls", tables=tables)


def _extract_pptx(data: bytes) -> ExtractedDocument:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise DocumentExtractionError(
            "PPTX extraction requires the documents extra: pip install \"recall-rag[documents]\""
        ) from exc
    try:
        presentation = Presentation(io.BytesIO(data))
        sections: list[str] = []
        tables = 0
        for slide_number, slide in enumerate(presentation.slides, start=1):
            slide_parts: list[str] = []
            for shape in slide.shapes:
                if getattr(shape, "has_table", False):
                    markdown = _rows_to_markdown(
                        [[cell.text for cell in row.cells] for row in shape.table.rows]
                    )
                    if markdown:
                        tables += 1
                        slide_parts.append(markdown)
                elif getattr(shape, "has_text_frame", False):
                    text = shape.text.strip()
                    if text:
                        slide_parts.append(text)
            if slide_parts:
                sections.append(f"## Slide {slide_number}\n\n" + "\n\n".join(slide_parts))
    except Exception as exc:
        raise DocumentExtractionError(f"could not extract PPTX: {type(exc).__name__}") from exc
    return _result("\n\n".join(sections), "pptx", tables=tables)


def _extract_delimited(suffix: str, data: bytes) -> ExtractedDocument:
    delimiter = "\t" if suffix == ".tsv" else ","
    try:
        rows = csv.reader(io.StringIO(_decode_text(data)), delimiter=delimiter)
        markdown = _rows_to_markdown(rows)
    except csv.Error as exc:
        raise DocumentExtractionError(f"could not parse {suffix[1:].upper()}: {exc}") from exc
    return _result(markdown, suffix.removeprefix("."), tables=1)


def _extract_html(data: bytes) -> ExtractedDocument:
    text = _decode_text(data)
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        parser = _VisibleTextParser()
        parser.feed(text)
        return _result(parser.text, "html")
    soup = BeautifulSoup(text, "html.parser")
    sections = [value.strip() for value in soup.stripped_strings]
    tables = 0
    for index, table in enumerate(soup.find_all("table"), start=1):
        markdown = _rows_to_markdown(
            [[cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])] for row in table.find_all("tr")]
        )
        if markdown:
            tables += 1
            sections.append(f"Table {index}\n{markdown}")
    return _result("\n".join(sections), "html", tables=tables)


def _extract_email(data: bytes) -> ExtractedDocument:
    try:
        message = BytesParser(policy=policy.default).parsebytes(data)
        sections = [f"Subject: {message.get('subject', '')}", f"From: {message.get('from', '')}"]
        if message.is_multipart():
            bodies = [part.get_content() for part in message.walk() if part.get_content_type() == "text/plain"]
            sections.extend(str(body) for body in bodies)
        else:
            sections.append(str(message.get_content()))
    except Exception as exc:
        raise DocumentExtractionError(f"could not extract EML: {type(exc).__name__}") from exc
    return _result("\n\n".join(sections), "eml")


def _extract_rtf(data: bytes) -> ExtractedDocument:
    text = _decode_text(data)
    text = re.sub(r"\\'[0-9a-fA-F]{2}", "", text)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    text = text.replace("{", "").replace("}", "")
    return _result(text, "rtf")


def _extract_with_libreoffice(path: Path, data: bytes) -> ExtractedDocument:
    executable = "soffice"
    with tempfile.TemporaryDirectory(prefix="recall-office-") as directory:
        source = Path(directory) / path.name
        source.write_bytes(data)
        if path.suffix.lower() == ".ppt":
            try:
                subprocess.run(
                    [executable, "--headless", "--convert-to", "pptx", "--outdir", directory, str(source)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except FileNotFoundError as exc:
                raise DocumentExtractionError(
                    "PPT needs LibreOffice for extraction; install LibreOffice or use PPTX"
                ) from exc
            except subprocess.SubprocessError as exc:
                raise DocumentExtractionError(
                    f"LibreOffice could not convert {path.name}: {type(exc).__name__}"
                ) from exc
            converted_pptx = source.with_suffix(".pptx")
            if not converted_pptx.exists():
                raise DocumentExtractionError(f"LibreOffice produced no PPTX for {path.name}")
            return _extract_pptx(converted_pptx.read_bytes())
        try:
            subprocess.run(
                [executable, "--headless", "--convert-to", "txt:Text", "--outdir", directory, str(source)],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError as exc:
            raise DocumentExtractionError(
                f"{path.suffix.upper()} needs LibreOffice for extraction; install LibreOffice "
                "or use the modern Office format"
            ) from exc
        except subprocess.SubprocessError as exc:
            raise DocumentExtractionError(
                f"LibreOffice could not extract {path.name}: {type(exc).__name__}"
            ) from exc
        converted = source.with_suffix(".txt")
        if not converted.exists():
            raise DocumentExtractionError(f"LibreOffice produced no text for {path.name}")
        return _result(_decode_text(converted.read_bytes()), path.suffix.removeprefix("."))


def _bounded_rows(rows: Iterable[Iterable[Any]], max_row: int | None, max_column: int | None) -> list[list[Any]]:
    row_limit = min(max_row or MAX_TABLE_ROWS, MAX_TABLE_ROWS)
    column_limit = min(max_column or MAX_TABLE_COLUMNS, MAX_TABLE_COLUMNS)
    result: list[list[Any]] = []
    for row_index, row in enumerate(rows):
        if row_index >= row_limit or len(result) * column_limit >= MAX_TABLE_CELLS:
            break
        result.append(list(row)[:column_limit])
    return result


def _rows_to_markdown(rows: Iterable[Iterable[Any]]) -> str:
    normalized = [[_cell(value) for value in row] for row in rows]
    normalized = [row for row in normalized if any(value for value in row)]
    if not normalized:
        return ""
    width = min(max(len(row) for row in normalized), MAX_TABLE_COLUMNS)
    normalized = [(row + [""] * width)[:width] for row in normalized]
    header = normalized[0]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return "\n".join(lines)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text.replace("|", "\\|")


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if value:
            self.parts.append(value)

    @property
    def text(self) -> str:
        return "\n".join(self.parts)
