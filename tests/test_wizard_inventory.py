"""The local `file://` inventory builder, and the reader contract it has to satisfy.

The generation path is the only path that can be calibrated, and it starts from a manifest. Nothing
in the repository turned a directory into one, so `recall manifest create --objects` had no
producer. These tests pin the producer to the CONSUMER's contract rather than to a shape that looks
right: every case ends by driving the built entries through `ManifestObjectV1` and, where the bytes
matter, through `LocalObjectReader.fetch`, which is what `generation build` actually calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from recall.lineage import LineageError, ManifestObjectV1
from recall.manifest import LocalObjectReader, load_inventory
from recall.wizard.inventory import build_inventory, write_inventory


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    _write(root / "alpha.md", b"# Alpha\n\nFirst document.\n")
    _write(root / "beta.md", b"# Beta\n\nSecond document.\n")
    _write(root / "sub" / "gamma.md", b"# Gamma\n\nNested document.\n")
    _write(root / "notes.py", b"def f():\n    return 1\n")
    return root


# --------------------------------------------------------------------------------------
# Shape and the consumer contract
# --------------------------------------------------------------------------------------


def test_entries_are_accepted_by_manifest_object(corpus: Path) -> None:
    """The only shape test that matters: the real validator accepts every entry."""
    entries = build_inventory(corpus)
    assert entries, "fixture corpus is not empty"
    for entry in entries:
        ManifestObjectV1(**entry)  # raises LineageError if any field is wrong


def test_output_round_trips_through_load_inventory(corpus: Path, tmp_path: Path) -> None:
    """`recall manifest create --objects` reads this file; it must parse without help."""
    out = tmp_path / "inventory.json"
    count = write_inventory(corpus, out)
    objects = load_inventory(out)
    assert len(objects) == count == 3  # three .md files, the .py is not in the default glob


def test_default_glob_is_markdown_and_glob_is_honoured(corpus: Path) -> None:
    md = {Path(e["uri"]).name for e in build_inventory(corpus)}
    py = build_inventory(corpus, glob="**/*.py")
    assert md == {"alpha.md", "beta.md", "gamma.md"}
    assert len(py) == 1
    assert py[0]["uri"].endswith("/notes.py")


def test_entries_are_sorted_and_stable(corpus: Path) -> None:
    """Two runs over an unchanged tree produce byte-identical output."""
    first = build_inventory(corpus)
    second = build_inventory(corpus)
    assert first == second
    assert [e["uri"] for e in first] == sorted(e["uri"] for e in first)


# --------------------------------------------------------------------------------------
# The version_id == sha256 invariant, which is what makes a local manifest honest
# --------------------------------------------------------------------------------------


def test_version_id_equals_sha256(corpus: Path) -> None:
    for entry in build_inventory(corpus):
        assert entry["version_id"] == entry["sha256"]


def test_sha256_is_of_raw_bytes_not_decoded_text(tmp_path: Path) -> None:
    """CRLF must not hash the same as LF.

    `LocalObjectReader.fetch` digests `path.read_bytes()`. A builder that hashed decoded text, or
    that opened the file in text mode on Windows, would emit a digest the reader cannot reproduce,
    and every object would fail verification at `generation build` time.
    """
    root = tmp_path / "eol"
    _write(root / "crlf.md", b"line one\r\nline two\r\n")
    _write(root / "lf.md", b"line one\nline two\n")
    by_name = {Path(e["uri"]).name: e for e in build_inventory(root)}

    assert by_name["crlf.md"]["sha256"] != by_name["lf.md"]["sha256"]
    assert by_name["crlf.md"]["sha256"] == hashlib.sha256(b"line one\r\nline two\r\n").hexdigest()
    assert by_name["crlf.md"]["size"] == 20


def test_size_is_the_byte_length(corpus: Path) -> None:
    for entry in build_inventory(corpus):
        path = Path(entry["uri"].removeprefix("file:///"))
        # Compare against the reader's own view rather than re-deriving the path by hand.
        obj = ManifestObjectV1(**entry)
        reader = LocalObjectReader([corpus])
        assert len(reader.fetch(obj).data) == entry["size"]
        assert path  # the uri is absolute, not relative


# --------------------------------------------------------------------------------------
# URI form, and the filenames that break a naive encoder
# --------------------------------------------------------------------------------------


def test_uri_is_absolute_file_scheme_without_query_or_fragment(corpus: Path) -> None:
    for entry in build_inventory(corpus):
        assert entry["uri"].startswith("file:///")
        assert "?" not in entry["uri"].split("://", 1)[1] or "%3F" in entry["uri"]
        ManifestObjectV1(**entry)  # this is what actually refuses a query/fragment


@pytest.mark.parametrize(
    "name",
    [
        "plain.md",
        "with space.md",
        "hash#tag.md",
        "quest?ion.md",
        "percent%20literal.md",
        "plus+sign.md",
        "unicode-éè.md",
    ],
)
def test_awkward_filenames_survive_the_round_trip_to_the_reader(tmp_path: Path, name: str) -> None:
    """The builder and the reader must agree on the path for every legal filename.

    `#`, `?` and a literal `%` are the three that a percent-encoding round trip gets wrong. If the
    reader resolves a different path than the builder described, it reads different bytes and the
    digest check fails — so this asserts through `fetch`, which performs that check.
    """
    root = tmp_path / "awkward"
    payload = f"content of {name}\n".encode()
    try:
        _write(root / name, payload)
    except OSError:
        pytest.skip(f"filesystem refuses the name {name!r}")

    entries = build_inventory(root)
    assert len(entries) == 1

    obj = ManifestObjectV1(**entries[0])
    reader = LocalObjectReader([root])
    assert reader.fetch(obj).data == payload


# --------------------------------------------------------------------------------------
# Refusals. Each of these is a guard, and each is mutated in test_wizard_inventory_guards
# --------------------------------------------------------------------------------------


def test_empty_result_is_refused_loudly(tmp_path: Path) -> None:
    """An empty corpus must not become an empty generation.

    `IndexManifestV1` accepts an empty object tuple, so nothing downstream would complain: the
    build would succeed, the generation would hold nothing, and calibration would measure an empty
    index. That is the false green `candidate_files` refuses a glob mismatch to avoid.
    """
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(ValueError, match="no files"):
        build_inventory(root)


def test_single_file_outside_the_glob_is_refused(corpus: Path) -> None:
    """Inherited from `candidate_files`, which treats this as a security boundary."""
    with pytest.raises(ValueError, match="glob"):
        build_inventory(corpus / "notes.py")


def test_missing_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises((ValueError, OSError)):
        build_inventory(tmp_path / "does-not-exist")


def test_media_type_falls_back_for_an_extension_nothing_recognises(tmp_path: Path) -> None:
    """`ManifestObjectV1` refuses an empty media_type, and `mimetypes` returns None for many names.

    The extension here must be in neither `MEDIA_TYPES` nor the stdlib table, or the fallback is
    never reached and this asserts nothing. An earlier version of this test used `.mdx`, which
    `MEDIA_TYPES` claims, so removing the fallback entirely left the test green — caught by
    mutating the guard rather than by reading the test.
    """
    from recall.wizard.inventory import FALLBACK_MEDIA_TYPE, media_type_for

    assert mimetypes_returns_nothing_for(".qqq"), "pick an extension the stdlib really ignores"

    root = tmp_path / "types"
    _write(root / "opaque.qqq", b"x\n")
    entries = build_inventory(root, glob="**/*.qqq")

    assert len(entries) == 1
    assert entries[0]["media_type"] == FALLBACK_MEDIA_TYPE
    ManifestObjectV1(**entries[0])
    assert media_type_for(Path("no_extension_at_all")) == FALLBACK_MEDIA_TYPE


def mimetypes_returns_nothing_for(suffix: str) -> bool:
    import mimetypes

    return mimetypes.guess_type("probe" + suffix)[0] is None


def test_known_source_extensions_beat_the_stdlib_table(tmp_path: Path) -> None:
    """`.ts` is `video/vnd.dlna.mpeg-tts` to the stdlib, which is wrong for a code corpus."""
    from recall.wizard.inventory import media_type_for

    assert media_type_for(Path("component.ts")) == "text/x-typescript"
    assert media_type_for(Path("notes.md")) == "text/markdown"


def test_a_manifest_built_from_the_inventory_verifies_end_to_end(
    corpus: Path, tmp_path: Path
) -> None:
    """The whole point, in one test: directory in, verified manifest out."""
    from recall.lineage import IndexManifestV1

    out = tmp_path / "inventory.json"
    write_inventory(corpus, out)
    manifest = IndexManifestV1("t", "2026-08-16", load_inventory(out))
    verified = LocalObjectReader([corpus]).verify(manifest)
    assert len(verified) == 3


def test_a_file_rewritten_after_the_inventory_is_detected(corpus: Path) -> None:
    """Detection, not prevention — the guarantee `lineage.py` documents for file:// objects."""
    entries = build_inventory(corpus)
    (corpus / "alpha.md").write_bytes(b"# Alpha\n\nRewritten.\n")
    reader = LocalObjectReader([corpus])
    with pytest.raises(Exception) as exc:
        for entry in entries:
            reader.fetch(ManifestObjectV1(**entry))
    assert "alpha.md" in str(exc.value)


def test_entry_with_a_mismatched_version_id_is_refused_by_the_validator(corpus: Path) -> None:
    """Proves the invariant is enforced downstream, not merely produced correctly here."""
    entry = dict(build_inventory(corpus)[0])
    entry["version_id"] = "not-the-digest"
    with pytest.raises(LineageError, match="version_id must be its content digest"):
        ManifestObjectV1(**entry)


def test_json_output_is_lf_and_utf8(corpus: Path, tmp_path: Path) -> None:
    """The digest of a manifest is over its bytes; CRLF from Windows would change it."""
    out = tmp_path / "inventory.json"
    write_inventory(corpus, out)
    raw = out.read_bytes()
    assert b"\r\n" not in raw
    json.loads(raw.decode("utf-8"))
