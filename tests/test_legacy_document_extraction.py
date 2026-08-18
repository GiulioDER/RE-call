from __future__ import annotations

import io
import subprocess
import zipfile
from pathlib import Path

import pytest

from recall.extraction import (
    DocumentExtractionError,
    _libreoffice_executable,
    extract_document,
)


def _epub_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container version="1.0" '
            'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        archive.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" '
            'version="2.0" unique-identifier="BookId"><metadata '
            'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>EPUB test</dc:title>'
            '<dc:language>en</dc:language><dc:identifier id="BookId">urn:test</dc:identifier>'
            '</metadata><manifest><item id="page" href="page.xhtml" '
            'media-type="application/xhtml+xml"/></manifest><spine>'
            '<itemref idref="page"/></spine></package>',
        )
        archive.writestr(
            "OEBPS/page.xhtml",
            '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
            '<h1>EPUB-TEST</h1><p>Searchable EPUB body.</p></body></html>',
        )
    return buffer.getvalue()


def test_epub_extracts_spine_text_without_libreoffice(tmp_path: Path) -> None:
    path = tmp_path / "book.epub"
    path.write_bytes(_epub_bytes())

    document = extract_document(path, path.read_bytes())

    assert "EPUB-TEST" in document.text
    assert document.metadata["source_format"] == "epub"


def test_epub_rejects_excessive_uncompressed_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "large.epub"
    path.write_bytes(_epub_bytes())
    monkeypatch.setattr("recall.extraction.MAX_EPUB_TOTAL_BYTES", 64, raising=False)

    with pytest.raises(DocumentExtractionError, match="EPUB archive is too large"):
        extract_document(path, path.read_bytes())


def test_malformed_msg_fails_with_typed_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.msg"
    path.write_bytes(b"not an Outlook message")

    with pytest.raises(DocumentExtractionError, match="MSG"):
        extract_document(path, path.read_bytes())


@pytest.mark.skipif(_libreoffice_executable() is None, reason="LibreOffice is not installed")
def test_libreoffice_converts_legacy_office_sources(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    workbook_module = pytest.importorskip("openpyxl")
    pptx_module = pytest.importorskip("pptx")
    executable = _libreoffice_executable()
    assert executable is not None

    sources = tmp_path / "sources"
    output = tmp_path / "output"
    profile = tmp_path / "lo-profile"
    sources.mkdir()
    output.mkdir()
    profile.mkdir()

    document = docx.Document()
    document.add_paragraph("LEGACY-DOC-TEST")
    document.save(sources / "source.docx")

    workbook = workbook_module.Workbook()
    workbook.active.append(["Region", "Revenue"])
    workbook.active.append(["EU", 123.45])
    workbook.save(sources / "source.xlsx")

    presentation = pptx_module.Presentation()
    presentation.slides.add_slide(presentation.slide_layouts[5]).shapes.title.text = "LEGACY-PPT-TEST"
    presentation.save(sources / "source.pptx")

    cases = [
        ("source.docx", "doc", "LEGACY-DOC-TEST"),
        ("source.docx", "odt", "LEGACY-DOC-TEST"),
        ("source.xlsx", "ods", "123.45"),
        ("source.pptx", "ppt", "LEGACY-PPT-TEST"),
        ("source.pptx", "odp", "LEGACY-PPT-TEST"),
    ]
    for source_name, output_format, expected in cases:
        subprocess.run(
            [
                executable,
                f"-env:UserInstallation={profile.as_uri()}",
                "--headless",
                "--convert-to",
                output_format,
                "--outdir",
                str(output),
                str(sources / source_name),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        converted = output / f"{Path(source_name).stem}.{output_format}"
        document = extract_document(converted, converted.read_bytes())
        assert expected in document.text
