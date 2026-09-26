"""Hardening of the hosted sync client, salvaged from commit 140dbbd5 and reviewed before landing.

Almost every test here drives `sync_memory_roots`, the function `SessionEnd` actually calls, with
only the network stubbed, and asserts on the PAYLOAD that would have left the machine or on the
sentence SessionStart would print. A helper that is correct and never called is the failure this
repository keeps meeting, so the entry point is what is exercised wherever that is possible.

Red proof, per CLAUDE.md: every behaviour test below names, in its docstring, the production line
it protects and the mutation that was applied to watch it fail in its own assertion. The full
table, with the failure each produced, is in the pull request that landed this file.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

import recall_hooks
from recall_hooks import credentials as cred
from recall_hooks import hosted

HOSTED = {"endpoint": "https://mcp.example.test/mcp", "tenant": "t", "account": "a@example.test"}

# Split inside the vendor prefix so no literal in this file matches the screen; see
# tests/test_hosted_screening.py for why.
FAKE_KEY = "AK" + "IA" + "ZXCVBNMASDFGHJKL"


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RECALL_HOOK_CONFIG_HOME", raising=False)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))


def _write(root: Path, name: str, body: bytes | str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(body, str):
        body = body.encode("utf-8")
    path.write_bytes(body)
    return path


class Wire:
    """What would have left the machine: every `recall_ingest` payload, decoded."""

    def __init__(self) -> None:
        self.sent: dict[str, bytes] = {}
        self.calls: list[str] = []
        self.headers_asked = 0


def _sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    files: dict[str, bytes | str],
    *,
    config: dict[str, Any] | None = None,
    inventory: Any = None,
    **kwargs: Any,
) -> tuple[hosted.SyncOutcome, Wire, Path]:
    import base64

    root = tmp_path / "memory"
    root.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        _write(root, name, body)
    wire = Wire()

    def fake_headers(_config: dict[str, Any]) -> dict[str, str]:
        wire.headers_asked += 1
        return {"Authorization": "Bearer x"}

    def fake_inventory(*_a: Any, **_k: Any) -> dict[str, str]:
        wire.calls.append("recall_inventory")
        return inventory() if callable(inventory) else {}

    def fake_call(_endpoint: str, _head: Any, tool: str, params: dict[str, Any], **_k: Any) -> Any:
        wire.calls.append(tool)
        for item in params["files"]:
            wire.sent[item["name"]] = base64.b64decode(item["content_b64"])
        return {"ok": True}

    monkeypatch.setattr(cred, "headers", fake_headers)
    monkeypatch.setattr(hosted, "remote_inventory", fake_inventory)
    monkeypatch.setattr(hosted, "call_tool", fake_call)
    outcome = hosted.sync_memory_roots([("worktree", root)], config or HOSTED, **kwargs)
    return outcome, wire, root


# ------------------------------------------------------------------ one read, and the bytes sent


def test_bytes_that_change_between_two_reads_are_never_sent(monkeypatch, tmp_path):
    """⛔ The digest must be taken over the buffer that is SENT, not over a second read.

    The salvaged `_bytes_if_unchanged` read the file to send it and read it AGAIN to hash it. A
    file that holds a credential for the first of those reads and its screened content for the
    second passes the check and ships the credential. Simulated by making the upload-time read
    return different bytes from every other read.

    Red proof: mutated `_bytes_if_unchanged` to hash `_read_once(change.path)` (the salvaged
    second read) instead of `_digest_raw(change.path, data)`; this failed at
    `assert "worktree/a.md" not in wire.sent`.
    """
    clean = b"# A\n\nOrdinary prose.\n"
    leaky = f"# A\n\nkey = {FAKE_KEY}\n".encode()
    reads = {"n": 0}
    real = Path.read_bytes

    def flaky(self: Path) -> bytes:
        if self.name != "a.md":
            return real(self)
        reads["n"] += 1
        # read 1 is the scan, read 2 is the upload's buffer, anything after is a re-check.
        return leaky if reads["n"] == 2 else clean

    _write(tmp_path / "memory", "a.md", clean)
    monkeypatch.setattr(Path, "read_bytes", flaky)
    outcome, wire, _root = _sync(monkeypatch, tmp_path, {})
    assert "worktree/a.md" not in wire.sent, "bytes no screen ever saw were uploaded"
    assert outcome.pending == 1, "a file dropped at upload time stays pending for the next run"


def test_a_memo_rewritten_after_the_scan_is_held_back(monkeypatch, tmp_path):
    """The window the single read exists to close: rewritten between the scan and the upload.

    Red proof: mutated `_bytes_if_unchanged` to return `data` without comparing digests; this
    failed at `assert "worktree/a.md" not in wire.sent` because the credential was uploaded.
    """
    def rewrite_during_inventory() -> dict[str, str]:
        _write(tmp_path / "memory", "a.md", f"# A\n\nkey = {FAKE_KEY}\n")
        return {}

    outcome, wire, _root = _sync(
        monkeypatch, tmp_path, {"a.md": "# A\n\nclean\n", "b.md": "# B\n\nclean\n"},
        inventory=rewrite_during_inventory,
    )
    assert "worktree/a.md" not in wire.sent
    assert "worktree/b.md" in wire.sent
    assert outcome.pending == 1


def test_a_file_that_vanishes_is_left_pending_not_counted(monkeypatch, tmp_path):
    """⛔ Confirmation is over what was SENT. Counting the whole batch removed a file that never
    went into the request from the only record of unfinished work, and reported it uploaded.

    Red proof: mutated the confirmation loop `for change in sent:` to `for change in batch:`; this
    failed at `assert outcome.uploaded == 1` (it reported 2).
    """
    def vanish() -> dict[str, str]:
        (tmp_path / "memory" / "gone.md").unlink()
        return {}

    outcome, wire, _root = _sync(
        monkeypatch, tmp_path, {"gone.md": "# G\n\nx\n", "kept.md": "# K\n\ny\n"},
        inventory=vanish,
    )
    assert list(wire.sent) == ["worktree/kept.md"]
    assert outcome.uploaded == 1
    assert outcome.pending == 1
    assert list(hosted.read_manifest(HOSTED)["pending"]) == ["worktree/gone.md"]


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"# A\r\nB\r\n", id="crlf"),
        pytest.param(b"# A\rB\rC", id="lone-cr"),
        pytest.param(b"# A\r\r\nB", id="cr-crlf"),
        pytest.param(b"\xef\xbb\xbf# A\r\nB\x00C\n", id="bom-crlf-nul"),
    ],
)
def test_hashing_the_buffer_agrees_with_the_servers_read_text(tmp_path, raw):
    """`_digest_raw` replaces a `read_text` call, so it must reproduce text mode exactly.

    The server's expression is `read_text(encoding="utf-8-sig")` then NUL stripping, and a lone
    `\\r` is a newline to text mode. Disagreeing on it re-uploads that file on every sync.

    Red proof: removed `.replace("\\r", "\\n")` from `_digest_raw`; the `lone-cr` and `cr-crlf`
    cases failed at the digest equality assertion.
    """
    path = _write(tmp_path, "m.md", raw)
    server = path.read_text(encoding="utf-8-sig").replace("\x00", "")
    got = hosted.digest_of(path)
    assert got is not None
    assert got[0] == hashlib.sha256(server.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------ wire size, not text size


def test_batches_are_budgeted_on_the_bytes_that_go_on_the_wire(tmp_path):
    """⛔ CRLF files are larger on the wire than their normalised text, and the server meters the
    wire. Budgeting on the text built a batch the server refused whole, every session.

    Drives the real `scan`, which is where `wire_size` is measured, then `plan`.

    Red proof: mutated `plan`'s `too_big` to add `change.size` instead of `wire`; this failed at
    `assert len(decided.upload) == 2` because both files fitted one batch by text length.
    """
    root = tmp_path / "memory"
    body = b"x\r\n" * 100  # 300 bytes on the wire, 200 as text
    _write(root, "a.md", body)
    _write(root, "b.md", body)
    local = hosted.scan([("worktree", root)])
    assert {c.wire_size for c in local.values()} == {300}
    assert {c.size for c in local.values()} == {200}
    decided = hosted.plan(local, {}, hosted.Limits(max_files=10, max_bytes=500, max_file_bytes=1000))
    assert len(decided.upload) == 2


def test_the_per_file_cap_is_on_the_wire_size_too(tmp_path):
    """Red proof: mutated the oversize test to `change.size > limits.max_file_bytes`; this failed
    at `assert decided.oversize == ["worktree/a.md"]`."""
    root = tmp_path / "memory"
    _write(root, "a.md", b"x\r\n" * 100)
    local = hosted.scan([("worktree", root)])
    decided = hosted.plan(local, {}, hosted.Limits(max_file_bytes=250))
    assert decided.oversize == ["worktree/a.md"]


# ------------------------------------------------------------------------------ names and emptiness


def _server_refuses(name: str) -> bool:
    from recall.desktop.uploads import UploadError, _safe_relative_name

    try:
        _safe_relative_name(name)
    except UploadError:
        return True
    return False


NAMES = [
    "worktree/a.md",
    "worktree/notes/deep/a.md",
    "worktree/aux.md",
    "worktree/CON.md",
    "worktree/com1.txt.md",
    "worktree/lpt9.md",
    "worktree/console.md",
    "worktree/a<b.md",
    "worktree/a:b.md",
    'worktree/a"b.md',
    "worktree/a|b.md",
    "worktree/a?b.md",
    "worktree/a*b.md",
    "worktree/tab\there.md",
    "worktree/trailing.",
    "worktree/trailing /a.md",
    "worktree/" + "x" * 256 + ".md",
    "worktree/" + "é" * 126 + ".md",
    "/".join(["worktree"] + ["d"] * 6 + ["a.md"]),
    "/".join(["worktree"] + ["d"] * 7 + ["a.md"]),
]


@pytest.mark.parametrize("name", NAMES)
def test_the_client_name_rule_agrees_with_the_server(name):
    """⛔ The drift test the salvaged docstring said "belongs beside this" and did not write.

    A name the client accepts and the server refuses costs the whole batch, every session, with
    the offending file never named. So the client must refuse exactly what the server refuses.

    Red proof: emptied `_RESERVED_STEMS`; the `aux`, `CON`, `com1` and `lpt9` cases failed here.
    """
    assert bool(hosted.unsafe_name_reason(name)) == _server_refuses(name)


def test_the_client_is_stricter_than_the_server_only_about_a_backslash():
    """The one deliberate difference, pinned so it cannot widen unnoticed: the server splits on a
    backslash and would store the file under a name this client never plans with."""
    assert hosted.unsafe_name_reason("worktree/a\\b.md")
    assert not _server_refuses("worktree/a\\b.md")


def test_a_name_the_server_refuses_costs_that_file_not_the_batch(monkeypatch, tmp_path):
    """Through the real entry point: `aux.md` is skipped and named, its neighbour still ships.

    Red proof: deleted the `if reason:` early `continue` in `scan`; this failed at
    `assert "worktree/aux.md" not in wire.sent`.
    """
    outcome, wire, _root = _sync(
        monkeypatch, tmp_path, {"aux.md": "# A\n\nx\n", "ok.md": "# O\n\ny\n"}
    )
    assert "worktree/aux.md" not in wire.sent
    assert "worktree/ok.md" in wire.sent
    assert outcome.skipped == 1
    assert "worktree/aux.md" in hosted.read_manifest(HOSTED)["skipped"]
    notice = recall_hooks._unsynced_notice(HOSTED)
    assert "worktree/aux.md" in notice and "can never be uploaded" in notice


def test_a_zero_byte_memo_is_skipped_and_not_announced(monkeypatch, tmp_path):
    """The server refuses an empty `content_b64` and takes the whole batch with it.

    Red proof: deleted the `if not wire:` skip in `scan`; this failed at
    `assert "worktree/empty.md" not in wire.sent`.
    """
    outcome, wire, _root = _sync(monkeypatch, tmp_path, {"empty.md": b"", "ok.md": "# O\n\ny\n"})
    assert "worktree/empty.md" not in wire.sent
    assert outcome.skipped == 1
    assert recall_hooks._unsynced_notice(HOSTED) == "", "an empty file loses nothing; no warning"


def test_a_whitespace_only_memo_is_still_sent(monkeypatch, tmp_path):
    """⛔ Changed from the salvaged commit, which skipped it. The server indexes a blank file as a
    zero-chunk REPLACEMENT so the old version's rows are removed; skipping it here would leave a
    memo the user blanked answering from the server with its old content.

    Red proof: restored the salvaged condition `not size or (text is not None and not
    text.strip())`; this failed at `assert "worktree/blank.md" in wire.sent`.
    """
    _outcome, wire, _root = _sync(monkeypatch, tmp_path, {"blank.md": " \n\n  \n"})
    assert "worktree/blank.md" in wire.sent


# -------------------------------------------------------------------------- matching the inventory


def test_suffix_matching_is_anchored_on_a_path_boundary():
    """`.../myproject/notes.md` is not `project/notes.md`. Unanchored, the local memo was declared
    already stored against an unrelated file and never uploaded.

    Red proof: made `stored_digest` fall back to the old unanchored
    `next((d for s, d in remote.items() if s.endswith(name)), None)`; this failed at
    `assert [c.name for b in decided.upload for c in b] == ["project/notes.md"]`.
    """
    local = {"project/notes.md": hosted.Change("project/notes.md", Path("x"), "mine", 1)}
    decided = hosted.plan(local, {"file:///s/myproject/notes.md": "mine"})
    assert [c.name for b in decided.upload for c in b] == ["project/notes.md"]


def test_an_exact_source_beats_a_suffix_whatever_the_listing_order():
    """Red proof: restored the salvaged index, which interleaved exact sources and suffixes with
    `setdefault` in listing order; this failed at `assert decided.unchanged == 1`."""
    local = {"project/a.md": hosted.Change("project/a.md", Path("x"), "same", 1)}
    remote = {"stage/project/a.md": "OTHER", "project/a.md": "same"}
    decided = hosted.plan(local, remote)
    assert decided.unchanged == 1
    assert decided.upload == []


def test_forget_uses_the_same_anchored_rule_as_matching():
    """A memo deleted under one root was kept on the server as long as another root held a file
    of the same basename.

    Red proof: restored master's basename comparison in the `forget_missing` loop; this failed at
    `assert decided.forget == ["file:///s/project/notes.md"]` (it was empty).
    """
    local = {"worktree/notes.md": hosted.Change("worktree/notes.md", Path("x"), "h", 1)}
    remote = {"file:///s/project/notes.md": "h", "file:///s/worktree/notes.md": "h"}
    decided = hosted.plan(local, remote, forget_missing=True)
    assert decided.forget == ["file:///s/project/notes.md"]


# ------------------------------------------------------------------------------------ the endpoint


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://mcp.example.test/mcp",
        "http://localhost:8000/mcp",
        "http://127.0.0.1/mcp",
        "http://[::1]:8000/mcp",
        "HTTPS://MCP.EXAMPLE.TEST/mcp",
    ],
)
def test_https_and_loopback_http_are_accepted(endpoint):
    """Red proof: restored the salvaged hand-sliced host parse; `http://[::1]:8000/mcp` failed
    here, because the slice stopped at the first `:` inside the brackets."""
    assert hosted.insecure_endpoint_reason(endpoint) == ""


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://mcp.example.test/mcp",
        "http://localhost:1@evil.example.test/mcp",
        "http://127.0.0.1:80@evil.example.test/",
        "http://localhost.evil.example.test/mcp",
        "ftp://mcp.example.test/mcp",
        "https:///mcp",
        "mcp.example.test/mcp",
    ],
)
def test_plaintext_to_anything_but_loopback_is_refused(endpoint):
    """⛔ `http://localhost:1@evil/` is USERINFO `localhost:1` and host `evil` to every HTTP client.

    Red proof: restored the salvaged hand-sliced host parse; the two `@evil` cases failed here
    (it returned "" and would have sent the bearer and the corpus in plaintext to `evil`).
    """
    assert hosted.insecure_endpoint_reason(endpoint) != ""


def test_a_plaintext_endpoint_is_refused_before_any_credential_or_network_and_is_reported(
    monkeypatch, tmp_path
):
    """Through the entry point, and to the sentence a person reads.

    Red proof, two mutations: (1) deleted the `insecure_endpoint_reason` early return in
    `sync_memory_roots`; this failed at `assert wire.headers_asked == 0`. (2) deleted
    `manifest["blocked"] = scheme_problem`; this failed at the `"not running"` notice assertion,
    because `_index_and_refresh` discards the outcome and nothing else would ever say it.
    """
    config = {**HOSTED, "endpoint": "http://localhost:1@evil.example.test/mcp"}
    outcome, wire, _root = _sync(monkeypatch, tmp_path, {"a.md": "# A\n\nx\n"}, config=config)
    assert outcome.kind == "refusal"
    assert wire.headers_asked == 0, "the bearer token must not even be fetched"
    assert wire.calls == []
    notice = recall_hooks._unsynced_notice(config)
    assert "hosted memory sync is not running" in notice
    assert "evil" not in notice and "localhost:1" not in notice, "the endpoint is never echoed"


def test_blocked_is_cleared_once_the_endpoint_is_fixed(monkeypatch, tmp_path):
    """Red proof: deleted `manifest.pop("blocked", None)`; this failed at the final assertion,
    because the old refusal kept printing after the transport answered."""
    hosted.write_manifest(HOSTED, {"pending": {}, "blocked": "an old refusal"})
    _outcome, _wire, _root = _sync(monkeypatch, tmp_path, {"a.md": "# A\n\nx\n"})
    assert "blocked" not in hosted.read_manifest(HOSTED)


# ------------------------------------------------------------------------------------ the transport


def test_a_missing_transport_is_blocking_and_names_an_extra_that_installs_it(monkeypatch):
    """⛔ The salvaged message said `pip install 'recall-rag[hosted]'`. That is the SERVER's extra
    and installs no `mcp`, so following the advice left the install exactly as broken.

    Red proof, two mutations: (1) restored `recall-rag[hosted]` in the message; this failed at
    `assert "mcp" in extras[extra]`. (2) deleted the `except ImportError` branch in `call_tool`;
    this failed at `assert caught.value.blocking` (it classified as a non-blocking `network`).
    """
    import re
    import tomllib

    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "httpx2", None)
    with pytest.raises(hosted.SyncError) as caught:
        hosted.call_tool("https://mcp.example.test/mcp", {}, "recall_inventory", {})
    assert caught.value.blocking
    assert caught.value.kind == "refusal"

    extra = re.search(r"recall-rag\[(\w+)\]", caught.value.message)
    assert extra, caught.value.message
    root = Path(__file__).resolve().parent.parent
    extras = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ][extra.group(1)]
    names = [re.split(r"[<>=!\[ ;]", dep, maxsplit=1)[0].lower() for dep in extras]
    assert "mcp" in names and "httpx2" in names, f"[{extra.group(1)}] does not install the client"


def test_a_missing_transport_reaches_session_start_on_the_first_run(monkeypatch, tmp_path):
    """Nothing is pending on a first run, so the pending-and-error rule kept this silent forever.

    Red proof: deleted `if exc.blocking: manifest["blocked"] = exc.message` in
    `sync_memory_roots`; this failed at the notice assertion.
    """
    root = tmp_path / "memory"
    _write(root, "a.md", "# A\n\nx\n")
    monkeypatch.setattr(cred, "headers", lambda _c: {"Authorization": "Bearer x"})
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "httpx2", None)
    outcome = hosted.sync_memory_roots([("worktree", root)], HOSTED)
    assert outcome.kind == "refusal"
    notice = recall_hooks._unsynced_notice(HOSTED)
    assert "not running" in notice and "recall-rag[mcp]" in notice


def test_a_stop_request_inside_the_transport_is_not_turned_into_a_sync_error(monkeypatch):
    """Red proof: deleted the `except (KeyboardInterrupt, SystemExit): raise` branch in
    `call_tool`; this failed at `pytest.raises(KeyboardInterrupt)` with a SyncError instead."""
    import asyncio

    def interrupted(coro: Any) -> Any:
        coro.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(asyncio, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        hosted.call_tool("https://mcp.example.test/mcp", {}, "recall_inventory", {})


# -------------------------------------------------------------------------------------- the deadline


def test_the_deadline_can_fire_through_the_clock_seam(monkeypatch, tmp_path):
    """⛔ The old `now: float` seam made elapsed time identically zero, so no test could ever see
    the one bound on a hung sync fire.

    Red proof: deleted the `if tick() - started > deadline: break`; this failed at
    `assert wire.sent == {}`.
    """
    ticks = iter([0.0, 1000.0, 1000.0, 1000.0])
    outcome, wire, _root = _sync(
        monkeypatch, tmp_path, {"a.md": "# A\n\nx\n"},
        config={**HOSTED, "sync": {"timeout_s": 5}}, clock=lambda: next(ticks),
    )
    assert wire.sent == {}
    assert outcome.pending == 1 and outcome.kind == "network"


def test_a_nan_timeout_is_not_an_unbounded_sync(monkeypatch, tmp_path):
    """`float("nan")` parses, and `elapsed > nan` is always False.

    Red proof: deleted the `math.isfinite` guard; this failed at `assert wire.sent == {}`.
    """
    ticks = iter([0.0, 1000.0, 1000.0, 1000.0])
    _outcome, wire, _root = _sync(
        monkeypatch, tmp_path, {"a.md": "# A\n\nx\n"},
        config={**HOSTED, "sync": {"timeout_s": "nan"}}, clock=lambda: next(ticks),
    )
    assert wire.sent == {}


@pytest.mark.parametrize("block", [None, "2m", {"timeout_s": "2m"}, {"timeout_s": None}])
def test_a_malformed_sync_block_does_not_raise(monkeypatch, tmp_path, block):
    """The contract is "never raises", and nothing in the repository writes this block.

    Red proof: replaced the defensive read with master's
    `float(config.get("sync", {}).get("timeout_s", 120))`; the `None` and `"2m"` cases failed at
    `pytest.fail` with the AttributeError the function raised.
    """
    try:
        outcome, wire, _root = _sync(
            monkeypatch, tmp_path, {"a.md": "# A\n\nx\n"}, config={**HOSTED, "sync": block}
        )
    except Exception as exc:  # noqa: BLE001 - the assertion IS that nothing escapes
        pytest.fail(f"sync_memory_roots raised {type(exc).__name__}: {exc}")
    assert "worktree/a.md" in wire.sent


# ------------------------------------------------------------------------ oversize, and the manifest


def test_an_oversize_memo_is_recorded_and_announced(monkeypatch, tmp_path):
    """`plan` always computed `oversize`; the caller read it nowhere, so the memo vanished.

    Red proof: deleted `manifest["oversize"] = list(decided.oversize)`; this failed at the
    notice assertion.
    """
    outcome, wire, _root = _sync(
        monkeypatch, tmp_path, {"big.md": "# B\n\n" + "x" * 100, "ok.md": "# O\n\ny\n"},
        limits=hosted.Limits(max_file_bytes=50),
    )
    assert list(wire.sent) == ["worktree/ok.md"]
    assert outcome.oversize == 1
    assert "worktree/big.md" in recall_hooks._unsynced_notice(HOSTED)


def test_a_manifest_that_is_not_utf8_does_not_raise_out_of_session_start(monkeypatch, capsys):
    """`UnicodeDecodeError` is a `ValueError`, not a `JSONDecodeError`, and SessionStart reaches
    this with no guard of its own.

    Red proof: narrowed `read_manifest`'s handler back to `(OSError, JSONDecodeError)`; this
    failed at `pytest.fail` with the UnicodeDecodeError raised out of `session_start`.
    """
    path = hosted.manifest_path(HOSTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'{"pending": {"\xff": ""}}')
    monkeypatch.setattr(recall_hooks, "load_config", lambda: HOSTED)
    try:
        assert recall_hooks.session_start({"cwd": "/tmp/proj"}) == 0
    except Exception as exc:  # noqa: BLE001 - the assertion IS that nothing escapes
        pytest.fail(f"session_start raised {type(exc).__name__}: {exc}")


def test_a_manifest_with_wrongly_typed_values_does_not_raise(monkeypatch):
    """Red proof: deleted the per-key type coercion loop in `read_manifest`; this failed at
    `pytest.fail` with the TypeError `sorted(5)` raised inside `_unsynced_notice`."""
    path = hosted.manifest_path(HOSTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pending": 3, "withheld": 5, "last_error": "x"}), encoding="utf-8")
    try:
        recall_hooks._unsynced_notice(HOSTED)
    except Exception as exc:  # noqa: BLE001 - the assertion IS that nothing escapes
        pytest.fail(f"_unsynced_notice raised {type(exc).__name__}: {exc}")


def test_an_unserialisable_manifest_neither_raises_nor_leaks_a_temp_file(tmp_path):
    """Red proof: narrowed `write_manifest`'s handler back to `except OSError`; this failed at
    `pytest.fail` with the TypeError from `json.dump`."""
    try:
        hosted.write_manifest(HOSTED, {"pending": {"a": object()}})
    except Exception as exc:  # noqa: BLE001 - the assertion IS that nothing escapes
        pytest.fail(f"write_manifest raised {type(exc).__name__}: {exc}")
    leftovers = [p.name for p in hosted.manifest_path(HOSTED).parent.glob(".*.tmp")]
    assert leftovers == []


def test_the_manifest_home_expands_a_tilde_in_claude_config_dir(monkeypatch, tmp_path):
    """Master re-derived the directory without `expanduser`, so `CLAUDE_CONFIG_DIR=~/alt` wrote a
    literal `~` directory beside the hook's cwd, which is somebody's checkout.

    Red proof: replaced `claude_config_home()` in `manifest_path` with master's
    `Path(os.environ["CLAUDE_CONFIG_DIR"])`; this failed at `assert "~" not in ...parts`.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "~/alt")
    path = hosted.manifest_path(HOSTED)
    assert "~" not in path.parts
    assert path.parent == tmp_path / "alt"


# ------------------------------------------------------------------------------------ the screen


def test_a_screen_that_raises_anything_uploads_nothing(monkeypatch, tmp_path):
    """⛔ Fails CLOSED for any exception, not only an ImportError.

    Red proof: narrowed `_screened`'s handler back to `except ImportError`; this failed at
    `pytest.fail` with the RuntimeError escaping `sync_memory_roots`.
    """
    from recall_hooks import screening

    def broken(_changes: Any) -> Any:
        raise RuntimeError("a future pattern failed to compile")

    monkeypatch.setattr(screening, "screen", broken)
    try:
        outcome, wire, _root = _sync(monkeypatch, tmp_path, {"a.md": "# A\n\nclean\n"})
    except Exception as exc:  # noqa: BLE001 - the assertion IS that nothing escapes
        pytest.fail(f"sync_memory_roots raised {type(exc).__name__}: {exc}")
    assert wire.sent == {}
    assert outcome.uploaded == 0
    assert "could not run" in hosted.read_manifest(HOSTED)["screen_error"]


def test_the_screen_inspects_the_buffer_that_was_hashed_not_a_fresh_read(tmp_path):
    """`screen` must use `Change.text` when it is present.

    Red proof: made `screen` always call `screen_file(Path(change.path))`; this failed at
    `assert withheld` (the file on disk is clean, the scanned buffer is not).
    """
    from recall_hooks import screening

    clean = _write(tmp_path, "a.md", "# A\n\nclean\n")
    change = hosted.Change("worktree/a.md", clean, "h", 1, text=f"key = {FAKE_KEY}\n")
    _allowed, withheld = screening.screen([change])
    assert withheld
