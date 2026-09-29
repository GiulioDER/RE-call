"""A NUL in an extracted document's BLOCKS must be stripped, not only one in its text.

Found 2026-09-29 on master `1170cc28`, indexing the PageIndex-OSS-Benchmark PDFs: `mi_phone.pdf`
extracted to text carrying a NUL. `Indexer.index_path` stripped `extracted.text` with `_strip_nul`,
but a non-markdown document is chunked from `extracted.blocks`, which were never stripped, so the
store's own guard raised `chunk ... contains a NUL (0x00) byte` and the whole directory failed to
index, not only the one file. Generation build had the same gap for documents with table blocks,
the only case in which it chunks `verified.blocks` rather than the stripped text.

Invariant: no chunk text and no chunk metadata string derived from an extracted block carries a
NUL, the strip is logged with the file and the count, and security redaction still sees the
stripped text (a NUL splitting an email address must not let the address through).

Red proof, 2026-09-29, each test run against the unfixed consumer and seen to fail in its own
assertion (not at import, collection or setup):

- `test_a_nul_in_an_extracted_block_never_reaches_a_chunk`: against `recall/index.py` at
  `8c4f15dd` (blocks unstripped), fails on `assert "\\x00" not in chunk.text`.
- `test_redaction_sees_block_text_after_the_nul_is_stripped`: mutation that redacts the blocks
  BEFORE stripping them in `Indexer.index_path`, fails on the address appearing in a chunk.
- `test_generation_build_strips_a_nul_from_extracted_blocks`: against `recall/generations.py`
  `_decoded_secure_text` at `8c4f15dd` (only the decoded text stripped), fails on the block
  assertion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import recall.index as index_module
from recall.embeddings import HashingEmbedder
from recall.extraction import ExtractedBlock, ExtractedDocument
from recall.generations import _decoded_secure_text
from recall.index import Indexer
from recall.manifest import ManifestObjectV1, VerifiedObject
from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy


class _Store:
    """Accepts anything, so what reaches it can be inspected rather than refused."""

    def __init__(self) -> None:
        self.chunks: list = []

    def source_content_hashes(self):
        return {}

    def replace_sources(self, sources, chunks, embeddings):
        self.chunks.extend(chunks)
        return len(chunks)

    def analyze_if_stale(self, modified):
        return True

    def delete_sources(self, sources):
        return 0


def _document_with_nul_blocks(email: str = "no address here") -> ExtractedDocument:
    """Five NULs across the blocks: narrative text, a table cell, and the heading, which is carried
    in both blocks' metadata and in the table block's text."""
    heading = "## Res\x00ults"
    narrative = f"revenue grew\x00 in 2024, contact {email}"
    table = "| Year | Revenue |\n| --- | --- |\n| 2024 | 5\x001 |"
    metadata = {"source_format": "pdf", "extraction": "structured", "table_count": 1}
    return ExtractedDocument(
        f"{heading}\n\n{narrative}\n\n{table}",
        "text/plain",
        metadata,
        (
            ExtractedBlock(narrative, "text", {**metadata, "content_kind": "text", "heading": heading}),
            ExtractedBlock(
                f"{heading}\n\n{table}",
                "table",
                {**metadata, "content_kind": "table", "table_index": 1, "heading": heading},
            ),
        ),
    )


def _index_one_pdf(tmp_path: Path, monkeypatch, document: ExtractedDocument, **indexer_kwargs):
    root = tmp_path / "corpus"
    (root / "finance").mkdir(parents=True)
    pdf = root / "finance" / "report.pdf"
    pdf.write_bytes(b"%PDF-1.7 synthetic, extraction is replaced below")
    monkeypatch.setattr(index_module, "extract_document", lambda path, data: document)
    store = _Store()
    Indexer(store, HashingEmbedder(dim=64), **indexer_kwargs).index_path(root)
    return store


def test_a_nul_in_an_extracted_block_never_reaches_a_chunk(tmp_path, monkeypatch, caplog) -> None:
    with caplog.at_level("WARNING", logger="recall.index"):
        store = _index_one_pdf(tmp_path, monkeypatch, _document_with_nul_blocks())

    assert store.chunks, "the document must still produce chunks"
    for chunk in store.chunks:
        assert "\x00" not in chunk.text
        assert "\\u0000" not in json.dumps(chunk.metadata), "a NUL in metadata breaks jsonb too"
    assert any("revenue grew in 2024" in chunk.text for chunk in store.chunks)
    assert any(chunk.metadata.get("heading") == "## Results" for chunk in store.chunks)
    block_warnings = [r.getMessage() for r in caplog.records if "block" in r.getMessage()]
    assert any(
        "report.pdf" in message and "stripped 5 NUL" in message for message in block_warnings
    ), f"the block strip must name the file and the count, got {block_warnings}"


def test_redaction_sees_block_text_after_the_nul_is_stripped(tmp_path, monkeypatch) -> None:
    policy = SourceSecurityPolicy(
        (SourceRule("finance", principals=frozenset({"alice"}), redactions=("email",)),)
    )
    store = _index_one_pdf(
        tmp_path,
        monkeypatch,
        # The NUL sits right after the `@`: anywhere in the local part, the pattern still matches
        # the tail (`ce@example.com`) and redaction succeeds whatever the order, so the test
        # would pass against the bug it exists to catch.
        _document_with_nul_blocks(email="alice@\x00example.com"),
        security_policy=policy,
        security_context=AccessContext("alice", "tenant"),
    )

    assert store.chunks
    for chunk in store.chunks:
        assert "alice@example.com" not in chunk.text, "stripping must happen before redaction"
    assert any("[REDACTED:email]" in chunk.text for chunk in store.chunks)


@pytest.mark.parametrize("with_policy", [False, True])
def test_generation_build_strips_a_nul_from_extracted_blocks(with_policy: bool) -> None:
    document = _document_with_nul_blocks()
    data = document.text.encode("utf-8")
    entry = ManifestObjectV1(
        "s3://approved/corpora/tenant/finance/report.pdf",
        "v1",
        "application/pdf",
        len(data),
        hashlib.sha256(data).hexdigest(),
    )
    verified = VerifiedObject(entry, data, dict(document.metadata), document.blocks)
    policy = (
        SourceSecurityPolicy((SourceRule("finance", principals=frozenset({"alice"})),))
        if with_policy
        else None
    )
    context = AccessContext("alice", "tenant") if with_policy else None

    text, blocks = _decoded_secure_text(entry, "finance/report.pdf", verified, policy, context)

    assert "\x00" not in text
    assert blocks, "the blocks must survive, only their NULs removed"
    for block in blocks:
        assert "\x00" not in block.text
        assert "\\u0000" not in json.dumps(block.metadata)
