"""The hosted sync planner: what gets uploaded, decided before anything is sent.

Almost all of this is pure and needs neither a database nor a network. The one exception is the
test that matters most — that the client's digest equals the digest the SERVER stores — which
indexes a real file and compares. That agreement is the whole contract, and if it breaks nothing
raises: every sync would upload the entire corpus while appearing to work.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from recall_hooks.hosted import Change, Limits, digest_of, plan, scan

from .conftest import TEST_DSN, requires_db


def _write(root: Path, name: str, body: str, newline: str = "\n") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline=newline)
    return path


def _change(name: str, sha: str, size: int = 10) -> Change:
    return Change(name=name, path=Path(name), sha256=sha, size=size)


# --------------------------------------------------------------------------- digest


def test_markdown_is_hashed_as_decoded_text_not_as_bytes(tmp_path: Path) -> None:
    """The server hashes markdown as UTF-8 text after stripping a BOM and NULs.

    Hashing the raw bytes instead would be the silent failure this module exists to avoid: every
    file would mismatch its server entry forever and every sync would re-upload everything.
    """
    body = "# Note\n\nSome prose.\n"
    path = _write(tmp_path, "a.md", body)
    got = digest_of(path)
    assert got is not None
    assert got[0] == hashlib.sha256(body.encode("utf-8")).hexdigest()


def test_crlf_and_lf_agree(tmp_path: Path) -> None:
    """Text mode normalises newlines, so the same content on Windows and POSIX is one digest.

    Without this a Windows client would re-upload its whole corpus to a Linux-indexed corpus, and
    the symptom would be cost rather than an error.
    """
    body = "# Note\n\nLine one.\nLine two.\n"
    lf = digest_of(_write(tmp_path, "lf.md", body, newline="\n"))
    crlf = digest_of(_write(tmp_path, "crlf.md", body, newline="\r\n"))
    assert lf is not None and crlf is not None
    assert lf[0] == crlf[0]


def test_a_bom_does_not_change_the_digest(tmp_path: Path) -> None:
    """`utf-8-sig` strips it server-side, so the client must too."""
    body = "# Note\n\nProse.\n"
    plain = tmp_path / "plain.md"
    plain.write_bytes(body.encode("utf-8"))
    bommed = tmp_path / "bom.md"
    bommed.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    a, b = digest_of(plain), digest_of(bommed)
    assert a is not None and b is not None
    assert a[0] == b[0]


def test_nul_characters_are_stripped_before_hashing(tmp_path: Path) -> None:
    """One file in 792 carried stray NULs and took down a whole indexing run; the server strips
    them, so a client that did not would disagree about exactly those files."""
    clean = "# Note\n\nProse.\n"
    dirty = tmp_path / "dirty.md"
    dirty.write_bytes(clean.replace("Prose", "Pro\x00se").encode("utf-8"))
    got = digest_of(dirty)
    assert got is not None
    assert got[0] == hashlib.sha256("# Note\n\nProse.\n".encode("utf-8")).hexdigest()


def test_a_non_markdown_file_is_hashed_as_raw_bytes(tmp_path: Path) -> None:
    raw = b"\x89PNG\r\n\x1a\n binary-ish"
    path = tmp_path / "x.bin"
    path.write_bytes(raw)
    got = digest_of(path)
    assert got is not None
    assert got[0] == hashlib.sha256(raw).hexdigest()


def test_an_unreadable_file_is_none_rather_than_an_exception(tmp_path: Path) -> None:
    """Hooks must never take a session down, and a corpus routinely holds one unreadable thing."""
    assert digest_of(tmp_path / "does-not-exist.md") is None


# --------------------------------------------------------------------------- scan


def test_scan_prefixes_each_root_so_two_roots_cannot_collide(tmp_path: Path) -> None:
    """Both memory roots are scanned, and identical layouts under each stay distinct.

    A project can have two memory directories and they are both real on machines today. Without the
    prefix, `a.md` under each becomes one name and each sync overwrites the other's file.
    """
    p, w = tmp_path / "p", tmp_path / "w"
    _write(p, "a.md", "from project")
    _write(w, "a.md", "from worktree")
    found = scan([("project", p), ("worktree", w)])
    assert sorted(found) == ["project/a.md", "worktree/a.md"]
    assert found["project/a.md"].sha256 != found["worktree/a.md"].sha256


def test_scan_keeps_nested_paths_and_skips_a_missing_root(tmp_path: Path) -> None:
    p = tmp_path / "p"
    _write(p, "notes/deep/a.md", "x")
    found = scan([("project", p), ("worktree", tmp_path / "absent")])
    assert list(found) == ["project/notes/deep/a.md"]


# --------------------------------------------------------------------------- plan


def test_an_unchanged_corpus_uploads_nothing() -> None:
    local = {"project/a.md": _change("project/a.md", "aaa")}
    result = plan(local, {"file:///stage/project/a.md": "aaa"})
    assert result.upload == []
    assert result.unchanged == 1


def test_a_changed_file_is_uploaded_and_an_unchanged_neighbour_is_not() -> None:
    local = {
        "project/a.md": _change("project/a.md", "NEW"),
        "project/b.md": _change("project/b.md", "same"),
    }
    remote = {"file:///s/project/a.md": "OLD", "file:///s/project/b.md": "same"}
    result = plan(local, remote)
    assert [c.name for batch in result.upload for c in batch] == ["project/a.md"]
    assert result.unchanged == 1


def test_a_file_the_server_has_never_seen_is_uploaded() -> None:
    result = plan({"project/new.md": _change("project/new.md", "x")}, {})
    assert [c.name for b in result.upload for c in b] == ["project/new.md"]


def test_an_empty_remote_digest_is_treated_as_unknown_and_re_uploaded() -> None:
    """The server reports an empty hash for a row indexed before content hashing existed.

    Empty means "I cannot tell you what I hold", and reading that as "unchanged" would make a
    stale copy permanent. Re-uploading is the recoverable direction.
    """
    result = plan({"project/a.md": _change("project/a.md", "aaa")},
                  {"file:///s/project/a.md": ""})
    assert [c.name for b in result.upload for c in b] == ["project/a.md"]
    assert result.unchanged == 0


def test_an_oversized_file_is_named_rather_than_silently_dropped() -> None:
    """It will never fix itself, and a sync that omits it leaves the user believing it is stored."""
    limits = Limits(max_file_bytes=100)
    local = {
        "project/big.md": _change("project/big.md", "x", size=101),
        "project/ok.md": _change("project/ok.md", "y", size=10),
    }
    result = plan(local, {}, limits)
    assert result.oversize == ["project/big.md"]
    assert [c.name for b in result.upload for c in b] == ["project/ok.md"]


def test_batches_respect_both_the_file_count_and_the_byte_cap() -> None:
    limits = Limits(max_files=2, max_bytes=1000, max_file_bytes=1000)
    local = {f"project/{i}.md": _change(f"project/{i}.md", str(i), size=10) for i in range(5)}
    result = plan(local, {}, limits)
    assert [len(b) for b in result.upload] == [2, 2, 1]

    heavy = {f"project/h{i}.md": _change(f"project/h{i}.md", str(i), size=600) for i in range(3)}
    by_bytes = plan(heavy, {}, limits)
    assert all(sum(c.size for c in b) <= limits.max_bytes for b in by_bytes.upload)


def test_a_single_file_over_the_batch_byte_cap_still_gets_its_own_batch() -> None:
    """Otherwise it would be dropped by the batching loop rather than by the oversize check, and
    the two have different remedies."""
    limits = Limits(max_files=10, max_bytes=100, max_file_bytes=10_000)
    local = {"project/a.md": _change("project/a.md", "x", size=500)}
    result = plan(local, {}, limits)
    assert [c.name for b in result.upload for c in b] == ["project/a.md"]


def test_the_plan_is_ordered_so_two_runs_agree() -> None:
    local = {f"project/{c}.md": _change(f"project/{c}.md", c) for c in "dcba"}
    first = plan(local, {})
    second = plan(local, {})
    names = [c.name for b in first.upload for c in b]
    assert names == [c.name for b in second.upload for c in b]
    assert names == sorted(names)


def test_deletion_is_off_by_default() -> None:
    """"Absent locally" and "should be erased" are different claims.

    A client that has never seen a directory must not conclude the server should forget it, so the
    caller has to ask.
    """
    result = plan({}, {"file:///s/project/gone.md": "aaa"})
    assert result.forget == []


def test_deletion_names_what_the_client_no_longer_has_when_asked() -> None:
    remote = {"file:///s/project/gone.md": "aaa", "file:///s/project/kept.md": "bbb"}
    local = {"project/kept.md": _change("project/kept.md", "bbb")}
    result = plan(local, remote, forget_missing=True)
    assert result.forget == ["file:///s/project/gone.md"]


def test_the_server_source_is_returned_verbatim_for_forget() -> None:
    """`recall_forget` takes the source exactly as the inventory reported it; prettifying it here
    would produce a string the server cannot match."""
    source = "file:///C:/staging/uploads/acme/sync-memory/project/gone.md"
    result = plan({}, {source: "aaa"}, forget_missing=True)
    assert result.forget == [source]


def test_matching_is_by_suffix_because_the_server_prefixes_a_staging_path() -> None:
    """The client knows `project/a.md`; the server knows a URI ending in it. Requiring equality
    would make every file look new forever."""
    local = {"project/a.md": _change("project/a.md", "same")}
    result = plan(local, {"file:///var/uploads/t/sync-memory/project/a.md": "same"})
    assert result.upload == []
    assert result.unchanged == 1


@pytest.mark.parametrize("bad", [0, -1])
def test_limits_of_zero_or_less_do_not_produce_an_infinite_plan(bad: int) -> None:
    """A degenerate limit must not loop forever or silently drop everything; each file still lands
    in a batch of its own."""
    local = {f"project/{i}.md": _change(f"project/{i}.md", str(i), size=1) for i in range(3)}
    result = plan(local, {}, Limits(max_files=bad, max_bytes=bad, max_file_bytes=10_000))
    assert sum(len(b) for b in result.upload) == 3


# --------------------------------------------------------- the contract, against a real server


@requires_db
def test_the_client_digest_equals_what_the_server_stores(tmp_path: Path) -> None:
    """The whole contract, checked against a real index rather than against my reading of it.

    Every test above asserts `digest_of` matches what I believe `recall/index.py` does. This one
    indexes real files and compares against what the store actually reports — the same values
    `recall_inventory` returns to a client. If the two ever diverge, nothing raises: the client
    would upload its entire corpus on every sync and call it success.

    Deliberately includes the awkward cases, because they are where the two implementations would
    drift apart first: CRLF, a BOM, and a stray NUL.
    """
    import uuid

    from recall.embeddings import HashingEmbedder
    from recall.index import Indexer
    from recall.store import PgVectorStore

    root = tmp_path / "corpus"
    root.mkdir()
    (root / "plain.md").write_bytes("# A\n\nProse enough to chunk.\n".encode("utf-8"))
    (root / "crlf.md").write_bytes("# B\r\n\r\nWindows line endings here.\r\n".encode("utf-8"))
    (root / "bom.md").write_bytes(b"\xef\xbb\xbf" + "# C\n\nWith a byte order mark.\n".encode())
    (root / "nul.md").write_bytes("# D\n\nHas a \x00 stray NUL.\n".encode("utf-8"))

    embedder = HashingEmbedder(dim=64)
    store = PgVectorStore(TEST_DSN, dim=embedder.dim, tenant=f"digest-{uuid.uuid4().hex[:8]}")
    store.check_schema()
    try:
        Indexer(store, embedder).index_path(root, glob="*.md")
        stored = store.source_raw_hashes()
    finally:
        store.close()

    assert stored, "nothing was indexed, so this proves nothing"
    for source, server_hash in stored.items():
        client = digest_of(Path(source))
        assert client is not None, f"client could not read {source}"
        assert client[0] == server_hash, (
            f"digest disagreement on {Path(source).name}: client {client[0][:12]} vs server "
            f"{server_hash[:12]}. A client hashing this way would re-upload the whole corpus "
            f"on every sync and never report a problem."
        )


# --------------------------------------------------------------------- flattening and classifying


def test_a_task_group_error_is_flattened_to_the_cause() -> None:
    """The SDK runs its session in a task group, so EVERY server error arrives wrapped.

    Unflattened, every hosted failure message is "unhandled errors in a TaskGroup (1
    sub-exception)" — a sentence that names the plumbing, hides the cause, and leaves the
    classifier nothing to read. This was met for real while driving a server by hand.
    """
    from recall_hooks.hosted import legible

    wrapped = BaseExceptionGroup("unhandled errors in a TaskGroup", [ValueError("the real cause")])
    assert legible(wrapped) == "ValueError: the real cause"


def test_nested_groups_are_flattened_too() -> None:
    from recall_hooks.hosted import legible

    inner = BaseExceptionGroup("inner", [KeyError("k")])
    outer = BaseExceptionGroup("outer", [inner, ValueError("v")])
    flat = legible(outer)
    assert "KeyError" in flat and "ValueError: v" in flat
    assert "TaskGroup" not in flat


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("HTTPStatusError: 401 Unauthorized", "auth"),
        ("invalid_token: Authentication required", "auth"),
        ("AuthError: no credential; run `recall-hooks login`", "auth"),
        ("RateLimited: index_bytes budget exhausted for tenant", "quota"),
        ("429 Too Many Requests", "quota"),
        ("UploadError: duplicate file name 'a.md' in one upload", "refusal"),
        ("UploadError: upload exceeds the 50 MiB request limit", "refusal"),
        ("ValueError: category must be documents, code, or memory", "refusal"),
        ("ConnectTimeout: timed out", "network"),
        ("gaierror: getaddrinfo failed", "network"),
        ("SSLCertVerificationError: certificate verify failed", "network"),
    ],
)
def test_each_failure_is_classified_by_the_remedy_it_needs(message: str, kind: str) -> None:
    """The four kinds exist because they want genuinely different handling.

    auth must not retry in a loop, quota must back off for a long time, refusal must be surfaced
    immediately and name the file, network should retry quietly. Classifying wrongly means
    applying the wrong one, which is worse than not classifying at all.
    """
    from recall_hooks.hosted import classify

    assert classify(message) == kind


def test_an_unrecognised_failure_is_treated_as_network() -> None:
    """Because retry-quietly is the policy that is safe to apply to something nobody has
    classified yet — unlike surfacing it loudly or backing off for an hour."""
    from recall_hooks.hosted import classify

    assert classify("SomethingNobodyPredicted: a novel disaster") == "network"


def test_auth_wins_over_quota_when_a_message_mentions_both() -> None:
    """A 401 arriving inside a quota-shaped message is still an authentication problem, and the
    remedies differ: signing in again versus waiting an hour."""
    from recall_hooks.hosted import classify

    assert classify("401 Unauthorized while checking the index_bytes budget") == "auth"


# --------------------------------------------------------------------------- inventory guard


def test_a_truncated_inventory_is_refused_rather_than_diffed(monkeypatch) -> None:
    """⛔ The diff reads "absent from the inventory" as "the server does not have it".

    A silently short listing would therefore re-upload the tail of the corpus and, once deletion
    is enabled, forget it. The tool reports truncation for exactly this reason, so using a
    truncated listing is the one thing this wrapper must not do.
    """
    from recall_hooks import hosted

    monkeypatch.setattr(
        hosted, "call_tool",
        lambda *a, **k: {"entries": [{"source": "s", "sha256": "h"}], "truncated": True},
    )
    with pytest.raises(hosted.SyncError) as caught:
        hosted.remote_inventory("https://e/mcp", {})
    assert caught.value.kind == "refusal"
    assert "truncated" in str(caught.value)


def test_an_inventory_becomes_a_source_to_digest_map(monkeypatch) -> None:
    from recall_hooks import hosted

    monkeypatch.setattr(
        hosted, "call_tool",
        lambda *a, **k: {
            "entries": [{"source": "file:///s/a.md", "sha256": "aaa"},
                        {"source": "file:///s/b.md", "sha256": ""}],
            "truncated": False,
        },
    )
    got = hosted.remote_inventory("https://e/mcp", {})
    assert got == {"file:///s/a.md": "aaa", "file:///s/b.md": ""}


def test_the_transport_is_not_imported_until_it_is_used() -> None:
    """`mcp` and `httpx2` are heavy, and only the async SessionEnd path ever reaches them.

    Paying that import at module scope would charge every session launch for a code path most
    sessions never take, which is the cost this whole package exists to avoid.
    """
    import subprocess

    code = (
        "import sys, recall_hooks.hosted; "
        "print(all(m not in sys.modules for m in ('recall', 'mcp', 'httpx2')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "True", out.stdout + out.stderr


# --------------------------------------------------------------- the screen, as WIRED not merely present
#
# A guard that exists and is never called is the failure mode this repository keeps meeting, so
# these drive `sync_memory_roots` end to end with the transport stubbed and assert on the payload
# that would have left the machine.


HOSTED_CFG = {
    "endpoint": "https://mcp.example.test/mcp",
    "tenant": "t",
    "account": "screen@example.test",
}

# Split inside the prefix so no literal in this file matches the screen. See
# tests/test_hosted_screening.py for why that matters.
FAKE_KEY = "AK" + "IA" + "ZXCVBNMASDFGHJKL"


def _sync_with(monkeypatch, tmp_path, files: dict[str, str]):
    """Run a sync over `files`, returning (outcome, names actually sent)."""
    from recall_hooks import credentials as cred
    from recall_hooks import hosted

    root = tmp_path / "memory"
    root.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        _write(root, name, body)

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setattr(cred, "headers", lambda _c: {"Authorization": "Bearer x"})
    monkeypatch.setattr(hosted, "remote_inventory", lambda *_a, **_k: {})

    sent: list[str] = []

    def fake_call(_endpoint, _head, _tool, params, **_kw):
        sent.extend(f["name"] for f in params["files"])
        return {"ok": True}

    monkeypatch.setattr(hosted, "call_tool", fake_call)
    outcome = hosted.sync_memory_roots([("worktree", root)], HOSTED_CFG)
    return outcome, sent


def test_a_memo_holding_a_credential_never_reaches_the_transport(monkeypatch, tmp_path):
    """⛔ The point of the whole module: the bytes must not leave the machine.

    Asserted on the PAYLOAD rather than on the outcome, because an outcome saying `withheld=1`
    while the file is still in the request would be a passing test and a leak.
    """
    outcome, sent = _sync_with(
        monkeypatch,
        tmp_path,
        {"clean.md": "# Clean\n\nOrdinary prose.\n", "leaky.md": f"# Leak\n\nkey = {FAKE_KEY}\n"},
    )
    assert sent == ["worktree/clean.md"]
    assert outcome.withheld == 1
    assert outcome.uploaded == 1


def test_the_withheld_file_is_left_exactly_as_it_was(monkeypatch, tmp_path):
    body = f"# Leak\n\nkey = {FAKE_KEY}\n"
    _outcome, _sent = _sync_with(monkeypatch, tmp_path, {"leaky.md": body})
    kept = (tmp_path / "memory" / "leaky.md").read_text(encoding="utf-8")
    assert kept == body, "the sync never rewrites, moves or deletes anything under a memory root"


def test_the_manifest_records_what_was_withheld(monkeypatch, tmp_path):
    """SessionStart reads this. Without it the refusal is invisible until somebody looks for a
    memo that is not there."""
    from recall_hooks import hosted

    _sync_with(monkeypatch, tmp_path, {"leaky.md": f"key = {FAKE_KEY}\n"})
    manifest = hosted.read_manifest(HOSTED_CFG)
    assert list(manifest["withheld"]) == ["worktree/leaky.md"]
    assert "line 1" in manifest["withheld"]["worktree/leaky.md"][0]
    assert FAKE_KEY not in str(manifest), "the manifest must not become a place the key is stored"


def test_a_withheld_file_is_reported_even_when_auth_fails(monkeypatch, tmp_path):
    """⚠️ The screen runs BEFORE the credential for this reason. If it ran after, every early
    return on an auth or network failure would drop the one finding that needs a person."""
    from recall_hooks import credentials as cred
    from recall_hooks import hosted

    root = tmp_path / "memory"
    root.mkdir(parents=True)
    _write(root, "leaky.md", f"key = {FAKE_KEY}\n")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))

    def no_credential(_config):
        raise cred.AuthError("no stored credential")

    monkeypatch.setattr(cred, "headers", no_credential)
    outcome = hosted.sync_memory_roots([("worktree", root)], HOSTED_CFG)
    assert outcome.kind == "auth"
    assert outcome.withheld == 1


def test_the_screen_failing_to_load_uploads_NOTHING(monkeypatch, tmp_path):
    """⛔ Fails CLOSED, against this package's usual habit, and the difference is the point.

    Everywhere else an ImportError means the thing that failed to load is a FEATURE, and skipping
    it is safe. Here it is a GUARD, and skipping a guard is the failure it was written to prevent.
    """
    monkeypatch.setitem(sys.modules, "recall_hooks.screening", None)
    outcome, sent = _sync_with(monkeypatch, tmp_path, {"clean.md": "# Clean\n\nProse.\n"})
    assert sent == [], "a clean file is not worth uploading past a guard that could not run"
    assert outcome.uploaded == 0

