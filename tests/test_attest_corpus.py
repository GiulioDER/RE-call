"""Tests for `scripts/attest_corpus.py`.

Every check here is written so it FAILS when the behaviour it names is removed. The attestation
tool exists to catch corruption, so a test suite that passes against a broken comparison would be
the exact defect the tool is meant to prevent, one level up.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "attest_corpus", Path(__file__).resolve().parents[1] / "scripts" / "attest_corpus.py"
)
attest = importlib.util.module_from_spec(_SPEC)
sys.modules["attest_corpus"] = attest
assert _SPEC.loader is not None
_SPEC.loader.exec_module(attest)


# ---- content hash rule ---------------------------------------------------------------
def test_hash_rule_matches_the_indexer(tmp_path: Path) -> None:
    """The one test that makes the duplicated hash rule safe.

    `scripts/attest_corpus.py` reimplements the rule that lives inline in `recall.index`, because
    the original is not a callable. If the indexer's rule changes and this one does not, every
    source in a real corpus reports `changed` and the tool becomes a liar. Pinning them against
    each other is what makes that drift loud.
    """
    from recall.index import _strip_nul

    md = tmp_path / "a.md"
    md.write_text("---\nkey: v\n---\n\nBody text.\n", encoding="utf-8")
    raw = _strip_nul(md.read_text(encoding="utf-8-sig"), md)
    indexer_rule = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert attest.content_hash_for(md) == indexer_rule


def test_hash_rule_branches_on_media_type(tmp_path: Path) -> None:
    """Markdown hashes decoded text; anything else hashes raw bytes.

    A CRLF file is the discriminating case: decoded-and-normalised differs from raw bytes, so a
    tool that used one rule for both would refuse every markdown file written on Windows.
    """
    body = b"line one\r\nline two\r\n"
    md, binary = tmp_path / "a.md", tmp_path / "a.pdf"
    md.write_bytes(body)
    binary.write_bytes(body)
    assert attest.content_hash_for(binary) == hashlib.sha256(body).hexdigest()
    assert attest.content_hash_for(md) != hashlib.sha256(body).hexdigest()
    assert attest.content_hash_for(md) == hashlib.sha256(b"line one\nline two\n").hexdigest()


# ---- census --------------------------------------------------------------------------
def test_census_separates_all_five_dispositions(tmp_path: Path) -> None:
    good, changed = tmp_path / "good.md", tmp_path / "changed.md"
    good.write_text("hello\n", encoding="utf-8")
    changed.write_text("hello\n", encoding="utf-8")
    rows = [
        ("good.md", attest.content_hash_for(good)),
        ("changed.md", "0" * 64),
        ("gone.md", "0" * 64),
        ("nohash.md", ""),
    ]
    census = attest.run_census(rows, tmp_path)
    assert census.verified == ["good.md"]
    assert census.changed == ["changed.md"]
    assert census.missing == ["gone.md"]
    assert census.no_hash == ["nohash.md"]
    assert census.total == 4


def test_census_detects_a_one_byte_change(tmp_path: Path) -> None:
    """The mutation that matters: the file moves under a stored hash."""
    f = tmp_path / "a.md"
    f.write_text("hello\n", encoding="utf-8")
    stored = attest.content_hash_for(f)
    assert attest.run_census([("a.md", stored)], tmp_path).verified == ["a.md"]
    f.write_text("hello!\n", encoding="utf-8")
    assert attest.run_census([("a.md", stored)], tmp_path).changed == ["a.md"]


# ---- chunker identification ----------------------------------------------------------
def _source(name: str, body: str, candidate: str):
    return (name, body, attest.CHUNKER_CANDIDATES[candidate](body))


def test_chunker_identifies_the_configuration_that_produced_the_chunks() -> None:
    body = "\n\n".join(f"Paragraph {i} with some words in it." for i in range(6))
    result = attest.identify_chunker([_source("a.md", body, "text/800/80")])
    assert result["verdict"] in {"identified", "ambiguous"}
    assert "text/800/80" in result["identifies"]


def test_chunker_reports_not_identifiable_when_nothing_reproduces() -> None:
    """A corpus chunked by something outside the candidate set must NOT be forced to fit."""
    result = attest.identify_chunker([("a.md", "one\n\ntwo\n\nthree", ["deliberately", "wrong"])])
    assert result["verdict"] == "not_identifiable"
    assert result["identifies"] == []


def test_chunker_distinguishes_ambiguous_from_identified() -> None:
    """Several candidates agreeing is not a failure and must not be reported as one."""
    body = "short one paragraph"          # no force split, so overlap cannot be identified
    result = attest.identify_chunker([_source("a.md", body, "text/800/80")])
    assert len(result["identifies"]) > 1
    assert result["verdict"] == "ambiguous"


def test_chunker_detects_an_ordinal_swap() -> None:
    """Reordering is a real corruption mode, and only a LIST comparison catches it.

    ⚠️ The first version of this test used four short paragraphs, which pack into ONE chunk at
    `max_chars=800`. Its swap was guarded by `if len(chunks) > 1`, so the assertion never ran and
    a `sorted(produced) == sorted(stored)` mutant survived the whole suite. The paragraphs below
    are deliberately long enough to force separate chunks, and the count is asserted
    UNCONDITIONALLY so the test cannot silently stop exercising the swap.
    """
    body = "\n\n".join(f"Paragraph {i}. " + ("filler words here " * 30) for i in range(3))
    chunks = attest.CHUNKER_CANDIDATES["text/800/80"](body)
    assert len(chunks) >= 2, f"fixture must produce multiple chunks, got {len(chunks)}"
    assert chunks[0] != chunks[1]

    swapped = list(chunks)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert attest.identify_chunker([("a.md", body, swapped)])["verdict"] == "not_identifiable"
    # and the unswapped original still identifies, so the assertion above is about ORDER
    assert "text/800/80" in attest.identify_chunker([("a.md", body, chunks)])["identifies"]


# ---- sample size ---------------------------------------------------------------------
def test_sample_size_reproduces_the_documented_table() -> None:
    assert attest.sample_size_for(0.10) == 29
    assert attest.sample_size_for(0.05) == 59
    assert attest.sample_size_for(0.01) == 299
    assert attest.sample_size_for(0.05, confidence=0.99) == 90


def test_sample_size_shows_why_twenty_was_too_small() -> None:
    """20 samples cannot detect 10 percent contamination at 95 percent confidence.

    The exact boundary is 1 - 0.05**(1/20) = 0.13911, so 20 samples buy detection of roughly a
    seventh of the corpus and nothing finer. Asserted at the boundary rather than at a rounded
    figure: `sample_size_for(0.139)` is 21, and writing the assertion loosely hid that.
    """
    boundary = 1 - 0.05 ** (1 / 20)
    assert attest.sample_size_for(boundary * 1.001) == 20
    assert attest.sample_size_for(0.139) == 21          # just under the boundary
    assert attest.sample_size_for(0.10) == 29 > 20      # 10 percent is out of reach at n=20


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_sample_size_refuses_impossible_fractions(bad: float) -> None:
    with pytest.raises(ValueError):
        attest.sample_size_for(bad)


# ---- embedder attestation ------------------------------------------------------------
def test_embedder_attestation_passes_when_vectors_reproduce() -> None:
    stored = [[1.0, 0.0], [0.0, 1.0]]
    result = attest.attest_embedder(
        [("a", stored[0]), ("b", stored[1])], embed=lambda texts: list(stored)
    )
    assert result["verdict"] == "pass"
    assert result["at_or_above_bar"] == 2
    # The control must NOT also read 1.0, or the comparison proves nothing.
    assert result["control_offdiagonal_max"] < 0.9999


def test_embedder_attestation_aborts_on_a_single_failure() -> None:
    """One failure aborts, deliberately: a partial pass cannot be scoped to safe sources."""
    stored = [[1.0, 0.0], [0.0, 1.0]]
    result = attest.attest_embedder(
        [("a", stored[0]), ("b", stored[1])],
        embed=lambda texts: [[1.0, 0.0], [1.0, 0.0]],   # second vector is wrong
    )
    assert result["verdict"] == "abort"
    assert result["at_or_above_bar"] == 1


def test_embedder_attestation_reports_no_samples_rather_than_passing() -> None:
    """An empty sample must not read as a clean result."""
    result = attest.attest_embedder([], embed=lambda texts: [])
    assert result["verdict"] == "no_samples"
    assert result["verdict"] != "pass"


# ---- regressions from the review of this file's first draft ---------------------------
def test_txt_is_hashed_as_raw_bytes_not_decoded_text(tmp_path: Path) -> None:
    """`.txt` is NOT markdown to the indexer (recall/index.py:694), so it hashes raw bytes.

    The first draft of this tool listed `.txt` and `.rst` as text, which would have reported
    every such source as `changed` on any real corpus.
    """
    body = b"alpha\r\nbeta\r\n"
    for suffix in (".txt", ".rst", ".pdf"):
        f = tmp_path / f"a{suffix}"
        f.write_bytes(body)
        assert attest.content_hash_for(f) == hashlib.sha256(body).hexdigest(), suffix


def test_chunker_excludes_non_markdown_rather_than_failing_it() -> None:
    """A PDF cannot be re-chunked by these chunkers, so it is out of scope, not a mismatch."""
    body = "one\n\ntwo"
    result = attest.identify_chunker([
        ("a.md", body, attest.CHUNKER_CANDIDATES["text/800/80"](body)),
        ("b.pdf", body, ["something else entirely"]),
    ])
    assert result["out_of_scope_non_markdown"] == 1
    assert result["sources"] == 1
    assert "text/800/80" in result["identifies"]


def test_chunker_counts_out_of_scope_when_given_a_generator() -> None:
    """`sources` may be a generator; iterating it twice would report zero out-of-scope."""
    body = "one\n\ntwo"
    gen = (s for s in [
        ("a.md", body, attest.CHUNKER_CANDIDATES["text/800/80"](body)),
        ("b.pdf", body, ["x"]),
    ])
    assert attest.identify_chunker(gen)["out_of_scope_non_markdown"] == 1


def test_embedder_attestation_refuses_a_short_vector_list() -> None:
    """zip() would have truncated silently, scoring only the vectors that happened to arrive."""
    with pytest.raises(ValueError, match="2 texts"):
        attest.attest_embedder([("a", [1.0]), ("b", [1.0])], embed=lambda texts: [[1.0]])


@pytest.mark.parametrize("bad", ["chunks; DROP TABLE x", "a-b", "a b", "a.b", ""])
def test_table_name_must_be_a_plain_identifier(bad: str) -> None:
    with pytest.raises(ValueError, match="identifiers only"):
        attest._safe_table(bad)


def test_table_name_accepts_the_real_tables() -> None:
    for good in ("chunks", "recall_chunks_v1", "cal_ab12"):
        assert attest._safe_table(good) == good
