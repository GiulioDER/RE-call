"""The prompt-time hook's per-file memo cache (PERF-001 of the 2026-11-04 recall_hooks hunt).

The invariant every test here serves: `load_memos` returns exactly what a full read of the store
returns, whatever state the cache file is in, and it reads a memo file only when that file's name,
`st_mtime_ns` or `st_size` differs from its cache entry. The failure each one is meant to catch is
named in its docstring, with the mutation it was watched to go red against.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from recall_hooks import prompt_time

MEMOS = {
    # A plain one-line description.
    "embedding-runs-remotely": (
        "---\nname: embedding-runs-remotely\ndescription: Embedding runs on the test host only.\n"
        "---\n\nNever two embedding processes at once.\n"
    ),
    # A block-scalar description, which `_description` joins.
    "suite-runs-in-parallel": (
        "---\nname: suite-runs-in-parallel\ndescription: >-\n  The pytest suite takes fifty\n"
        "  minutes serial.\n---\n\nFourteen minutes under four xdist workers.\n"
    ),
    # No frontmatter at all: the summary falls back to the title, the case a cache that stored
    # the raw description instead of `summary or title` would get wrong.
    "no-frontmatter-memo": "Just a body about parquet files and the workstation.\n",
    # Non-ASCII in the summary and body, which the cache file must round-trip.
    "unicode-memo": (
        "---\nname: unicode-memo\ndescription: ⛔ Never restart the service 🔑 during a run\n"
        "---\n\nCafé, naïve, 🔁 corrected 2026-11-04.\n"
    ),
    # An index file, which is never a memo.
    "MEMORY": "hub index pointing at embedding pyarrow parquet suite\n",
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A store and an isolated hook config home, so the cache lands under `tmp_path` only."""

    monkeypatch.setenv("RECALL_HOOK_CONFIG_HOME", str(tmp_path / "hook-home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-home"))
    store = tmp_path / "store"
    store.mkdir()
    for stem, text in MEMOS.items():
        (store / f"{stem}.md").write_text(text, encoding="utf-8", newline="\n")
    return store


def full_read(store: Path, monkeypatch) -> list[tuple[str, str, list[str]]]:
    """`load_memos` with the cache neither read nor written: the uncached path, every file read."""

    with monkeypatch.context() as patch:
        patch.setattr(prompt_time, "_read_cache", lambda path, tag: {})
        patch.setattr(prompt_time, "_write_cache", lambda path, tag, entries: None)
        return prompt_time.load_memos(store)


def cache_entries(store: Path) -> dict:
    return json.loads(prompt_time.cache_path(store).read_text(encoding="utf-8"))["entries"]


def count_memo_reads(monkeypatch, store: Path) -> list[str]:
    """Every read of a file inside `store`, by name, from here on."""

    reads: list[str] = []
    real = Path.read_text

    def counting(self, *args, **kwargs):
        if self.parent == store:
            reads.append(self.name)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting)
    return reads


def test_cached_output_equals_the_uncached_output(env, monkeypatch):
    """A warm cache returns, for every memo, the same `(stem, summary, tokens)` as a full read.

    Red proof: `tests/test_prompt_time_cache.py::test_cached_output_equals_the_uncached_output`
    against a mutation of `load_memos` that caches `summary` instead of `summary or title`; it
    failed on `warm == expected` for `no-frontmatter-memo`, whose cached summary was ''.
    """

    expected = full_read(env, monkeypatch)
    cold = prompt_time.load_memos(env)
    assert prompt_time.cache_path(env).is_file()
    warm = prompt_time.load_memos(env)

    assert cold == expected
    assert warm == expected
    assert {stem for stem, _, _ in warm} == set(MEMOS) - {"MEMORY"}


def test_a_second_call_reads_no_unchanged_memo(env, monkeypatch):
    """After one call has written the cache, the next reads no memo file at all.

    Red proof: `tests/test_prompt_time_cache.py::test_a_second_call_reads_no_unchanged_memo`
    against a mutation of `load_memos` whose cache-hit test compared the entry to
    `(found.st_mtime_ns, found.st_size + 1)`, so nothing ever hit; it failed on `reads == []`
    with all four memos read.
    """

    prompt_time.load_memos(env)
    reads = count_memo_reads(monkeypatch, env)
    prompt_time.load_memos(env)

    assert reads == []


@pytest.mark.parametrize("what_changes", ["size", "mtime"])
def test_an_edited_memo_is_read_again(env, monkeypatch, what_changes):
    """A memo whose size alone, or whose mtime alone, changed is re-read, and only that memo.

    Red proof: `tests/test_prompt_time_cache.py::test_an_edited_memo_is_read_again[mtime]`
    against a mutation of `load_memos` keying the hit on `st_size` only (`entry[1] ==
    found.st_size`); it failed on `reads == [memo.name]`, with no read. `[size]` against the
    mirror mutation keying on `st_mtime_ns` only, failing the same assertion.
    """

    memo = env / "embedding-runs-remotely.md"
    before = memo.stat()
    prompt_time.load_memos(env)

    old = memo.read_text(encoding="utf-8")
    if what_changes == "size":
        new = old.replace("Never two", "Never three")
        memo.write_text(new, encoding="utf-8", newline="\n")
        # The same mtime as before, so the size is the only signal left.
        os.utime(memo, ns=(before.st_atime_ns, before.st_mtime_ns))
    else:
        new = old.replace("Never two", "Often two")
        assert len(new) == len(old)
        memo.write_text(new, encoding="utf-8", newline="\n")
        os.utime(memo, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
    assert (memo.stat().st_size == before.st_size) == (what_changes == "mtime")

    reads = count_memo_reads(monkeypatch, env)
    memos = prompt_time.load_memos(env)

    assert reads == [memo.name]
    tokens = dict((stem, toks) for stem, _, toks in memos)["embedding-runs-remotely"]
    word = "three" if what_changes == "size" else "often"
    assert prompt_time.stem(word) in tokens
    assert memos == full_read(env, monkeypatch)


def test_a_deleted_memo_drops_out_of_the_result_and_the_cache(env, monkeypatch):
    """A memo deleted after the cache was written is not returned, and leaves the cache file.

    Red proof: `tests/test_prompt_time_cache.py::test_a_deleted_memo_drops_out_of_the_result_and_the_cache`
    against a mutation of `load_memos` rewriting the cache only when a memo was read (`if changed:`,
    dropping the key comparison); it failed on the cache assertion, the deleted entry still there.
    """

    prompt_time.load_memos(env)
    (env / "suite-runs-in-parallel.md").unlink()
    memos = prompt_time.load_memos(env)

    assert "suite-runs-in-parallel" not in {stem for stem, _, _ in memos}
    assert "suite-runs-in-parallel.md" not in cache_entries(env)
    assert memos == full_read(env, monkeypatch)


def _poison_tokens(store: Path, tag: str | None = None) -> None:
    """Rewrite the cache so every entry still matches its file's stamp but holds wrong tokens."""

    path = prompt_time.cache_path(store)
    document = json.loads(path.read_text(encoding="utf-8"))
    for entry in document["entries"].values():
        entry[3] = "poison poison"
    if tag is not None:
        document["tag"] = tag
    path.write_text(json.dumps(document), encoding="ascii")


@pytest.mark.parametrize(
    "corruption",
    ["garbage", "truncated", "wrong-shape", "wrong-type", "wrong-tag", "too-large"],
)
def test_a_bad_cache_gives_the_full_read_and_no_exception(env, monkeypatch, corruption):
    """A cache that is unreadable, from another version, or past the size bound is ignored.

    Red proof: `[wrong-tag]` against a mutation of `_read_cache` without the tag check; it failed on
    `memos == expected`, every memo carrying the poisoned tokens. `[too-large]` against a mutation
    without the `st_size > MAX_CACHE_BYTES` check, failing the same assertion. `[garbage]` against
    a mutation of `_read_cache` narrowing its `except` to `OSError`, failing with
    `UnicodeDecodeError` raised out of `load_memos`, which is the defect that case exists to catch
    (a hook that raises).
    """

    expected = full_read(env, monkeypatch)
    prompt_time.load_memos(env)
    path = prompt_time.cache_path(env)
    if corruption == "garbage":
        path.write_bytes(b"\x00\xff not json")
    elif corruption == "truncated":
        path.write_bytes(path.read_bytes()[:40])
    elif corruption == "wrong-shape":
        path.write_text(json.dumps({"tag": prompt_time._cache_tag(), "entries": [1, 2]}))
    elif corruption == "wrong-type":
        document = json.loads(path.read_text(encoding="utf-8"))
        for entry in document["entries"].values():
            entry[3] = ["poison"]
        path.write_text(json.dumps(document))
    elif corruption == "wrong-tag":
        _poison_tokens(env, tag="0:another-version")
    else:
        _poison_tokens(env)
        monkeypatch.setattr(prompt_time, "MAX_CACHE_BYTES", path.stat().st_size - 1)

    memos = prompt_time.load_memos(env)

    assert memos == expected


def test_a_cache_past_the_bound_is_not_written(env, monkeypatch):
    """With the bound below what the store needs, no cache file appears and no temporary is left.

    Red proof: `tests/test_prompt_time_cache.py::test_a_cache_past_the_bound_is_not_written`
    against a mutation of `_write_cache` without the `len(text) > MAX_CACHE_BYTES` return; it failed
    on `not path.exists()`.
    """

    monkeypatch.setattr(prompt_time, "MAX_CACHE_BYTES", 64)
    memos = prompt_time.load_memos(env)
    path = prompt_time.cache_path(env)

    assert not path.exists()
    assert not path.parent.exists() or not list(path.parent.glob("*.tmp"))
    assert memos == full_read(env, monkeypatch)


def test_the_cache_lives_under_the_hook_config_home(env, tmp_path):
    """Beside `auditor_brief`'s cache, under `claude_config_home()`, and written atomically: the
    temporary file is in the same directory and none is left behind. Nonbehavioral placement
    check, exempt from red proof: it pins a path, and a wrong path fails every other test here."""

    prompt_time.load_memos(env)
    path = prompt_time.cache_path(env)

    assert path.parent == tmp_path / "hook-home" / "cache"
    assert path.name.startswith(prompt_time.CACHE_PREFIX)
    assert list(path.parent.glob("*.tmp")) == []
