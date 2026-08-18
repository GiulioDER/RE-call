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


# The repository-wide limit is `timeout = 120` (`pyproject.toml`), and this test sat under four
# seconds of it: measured 2026-08-18 on Windows 11 with LibreOffice 25.8, it passes in 116.17s run
# alone. It failed twice in one session for no reason except that other `soffice.exe` processes
# held the CPU, and a `timeout_method = "thread"` expiry prints a faulthandler dump with no
# pass/fail summary line, which reads as a hang rather than as a failure.
#
# 600 restores roughly a 5x margin. It is the escape hatch `pyproject.toml` names for a test with a
# legitimate reason to run long, and the reason here is that the body starts LibreOffice **ten**
# times, not five: five conversions below, plus one inside each `extract_document` call, since
# `_extract_with_libreoffice` shells out again for every legacy format. Measured split, same run,
# fixture conversion against extraction: 21.52/17.06, 10.70/12.84, 4.18/9.70, 7.57/8.62,
# 4.21/7.78 seconds, totalling 48.18s and 56.00s.
#
# The two cheaper-looking fixes were measured and rejected rather than skipped:
#   * Profile reuse is already here. All five conversions below share one `-env:UserInstallation`,
#     and the numbers show it paying: the first launch carries the 21.5s profile bootstrap and the
#     rest fall to 4-10s. What stays cold is the extraction half, because `_extract_with_libreoffice`
#     builds a fresh profile in a fresh `TemporaryDirectory` per call. That is production code and
#     a shared profile there needs an answer for concurrent extractions locking it, so it is not a
#     change to make from a flaky test.
#   * Passing several sources to one `--convert-to` cannot help, because grouped by target format
#     every input is a singleton: doc<-docx, odt<-docx, ods<-xlsx, ppt<-pptx, odp<-pptx.
#
# The per-conversion `subprocess.run(timeout=120)` below is deliberately left alone, so a single
# genuinely stuck LibreOffice still fails fast instead of idling toward the new ceiling.
#
# What the margin is really for: this test's duration is set by machine load, not by the test.
# Samples taken the same day on the same machine span 53.86, 53.93, 57.26 (three consecutive runs
# on an idle box), 75.64 (single test, `-k`), and 116.17 seconds (the run that motivated this),
# a 2.15x spread against a ceiling that used to sit 3.83s above the worst of them. Three quick
# green runs do not sample the failure mode, because the failure mode is a busy machine.
#
# Re-measure: `python -m pytest tests/test_legacy_document_extraction.py -k libreoffice --durations=0`
#
# ----------------------------------------------------------------------------------------------
# CORRECTION, appended 2026-08-18. Every figure above is left exactly as measured; this says which
# of them the code moved out from under, and which still stand.
#
# The paragraph above that parks profile reuse as "not a change to make from a flaky test" has been
# acted on. `_extract_with_libreoffice` no longer builds a profile per call: it checks one out of a
# small process-wide pool. The answer to "concurrent extractions locking it" turned out not to be a
# lock at all, because a single locked profile measured *worse* than the old always-cold code at
# four way concurrency, 12.89s against 8.18s. Distinct profiles used concurrently are fine, so the
# pool keeps the reuse and the parallelism both. Record, with the falsified prediction that produced
# that design: `docs/preregistrations/2026-08-18-libreoffice-profile-reuse.md`.
#
# What still stands: the body really does start LibreOffice **ten** times. The five fixture
# conversions above are unchanged, and each `extract_document` below still shells out once.
#
# What moved: the extraction half is no longer cold. Only the first of the five pays a profile
# bootstrap, so measured 2026-08-18 on this machine the whole test now runs in **36.25, 35.36 and
# 35.09 seconds**, against the 53.86, 53.93 and 57.26 recorded above for the same three-consecutive-
# idle-runs condition. The 48.18s / 56.00s split above is therefore historical: the 48.18s fixture
# half is untouched, the 56.00s extraction half is the part that shrank.
#
# ⚠️ **600 is deliberately NOT lowered, and the arithmetic above is why.** The margin was never
# sized against the duration, it was sized against the 2.15x load-driven spread, and a pool of
# LibreOffice profiles does nothing about a busy machine. The same multiplier applied to the new
# median still lands near 76s, which clears the repository-wide 120 comfortably, but that comparison
# is against three green runs on an idle box, and the paragraph above already says why three green
# runs do not sample this failure mode. Lowering the ceiling is its own change and wants its own
# measurement under load, not an inference from a faster median.
# ----------------------------------------------------------------------------------------------
@pytest.mark.timeout(600)
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
