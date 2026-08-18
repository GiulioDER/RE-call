from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from recall.extraction import (
    DOCUMENT_EXTENSIONS,
    LIBREOFFICE_EXTENSIONS,
    DocumentExtractionError,
    _libreoffice_executable,
    extract_document,
)


# The compound file magic that opens every real .msg, so the fixture below is rejected for
# the reason under test rather than for being obviously not a document.
CFBF_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


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


def test_msg_without_oxmsg_names_the_documents_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment missing the optional extra must be told which extra to install.

    MSG has no LibreOffice fallback, because LibreOffice ships no MAPI import filter. Measured
    2026-08-18 against LibreOffice 25.8 on a genuine ``.msg``: ``--convert-to txt:Text`` exits 1
    with "source file could not be loaded" and writes no output file. Routing MSG through
    ``_extract_with_libreoffice`` therefore cannot succeed, and would only replace the actionable
    message asserted below with one blaming LibreOffice. Re-measure with
    ``scripts/check_libreoffice_msg.py``.
    """
    path = tmp_path / "note.msg"
    path.write_bytes(CFBF_MAGIC + bytes(64))
    monkeypatch.setitem(sys.modules, "oxmsg", None)

    with pytest.raises(DocumentExtractionError) as exc_info:
        extract_document(path, path.read_bytes())

    message = str(exc_info.value)
    assert "recall-rag[documents]" in message, message  # the extra to install
    assert "pip install" in message, message  # the command that installs it
    assert "libreoffice" not in message.lower(), message  # never blame the wrong dependency


def test_libreoffice_dispatch_excludes_msg() -> None:
    """The dispatch table must not advertise a fallback that can never be reached.

    ``.msg`` is matched earlier by the oxmsg branch, so listing it here was dead code that told a
    reader a fallback existed. The test above covers the behaviour; this one pins the table a
    reader actually consults.
    """
    assert ".msg" in DOCUMENT_EXTENSIONS  # MSG is still a supported format...
    assert ".msg" not in LIBREOFFICE_EXTENSIONS  # ...just not one LibreOffice can read
    assert LIBREOFFICE_EXTENSIONS == {".doc", ".odt", ".ods", ".odp", ".ppt"}


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
