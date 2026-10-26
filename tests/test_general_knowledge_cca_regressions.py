"""PR992 CCA regressions, each proved red against 15b0501e before its repair.

SEC-001 covers the public failure hook and persistent diagnostic output. BUG-001 covers
concurrent successful session claims. PERF-001 and PERF-002 cover bounded reader and cleanup
work without making latency claims. CODE-001 is a behavior-preserving extraction and DOC-001
is prose only; their existing behavioral controls are retained.
"""
from __future__ import annotations

import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from recall_hooks import general_knowledge as gk
from tests.test_general_knowledge_hook import example_entry, write_index


@pytest.mark.parametrize(
    ("error", "secret"),
    [
        ("HTTPError: Unauthorized https://example.invalid/?api_key=violetorchid", "violetorchid"),
        ("HTTPError: denied https://alice:orchid@localhost/", "orchid"),
        ("HTTPError: denied postgres://alice:orchid@db/", "orchid"),
        ("HTTPError Authorization: Bearer lilacgarden", "lilacgarden"),
        ("HTTPError Authorization: Basic orchidcredential", "orchidcredential"),
        ("Authentication error password=meadowflower", "meadowflower"),
        ('Authentication error {"access_token": "daffodil"}', "daffodil"),
    ],
)
def test_failure_hook_never_persists_credentials(tmp_path, monkeypatch, error, secret):
    """SEC-001: public failure logging must remove credentials, even without an index.

    Red at 15b0501e: assertion `secret not in logged` fails with the synthetic secret present.
    The Basic case is also red against the first SEC-001 repair, which handled Bearer only.
    Production target: post_tool_use_failure -> failure_signature -> _normalise -> log_event.
    """
    monkeypatch.setenv("RECALL_PROJECT_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("RECALL_GENERAL_KNOWLEDGE_HOME", str(tmp_path / "no-index"))
    log = tmp_path / "events.jsonl"
    monkeypatch.setenv("RECALL_GENERAL_KNOWLEDGE_LOG", str(log))
    assert gk.post_tool_use_failure({
        "tool_name": "Bash", "tool_input": {"command": "curl https://example.invalid"},
        "error": error, "session_id": "example-session", "cwd": str(tmp_path),
    }) == 0
    logged = log.read_text(encoding="utf-8")
    assert secret not in logged, "failure diagnostics persisted the credential"
    assert json.loads(logged)["event"] == "failure"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not Windows ACLs")
def test_diagnostic_log_is_private(tmp_path):
    """SEC-001: creating a diagnostic log must not grant group/other access.

    Red at 15b0501e with umask 022: mode is 0644 instead of 0600. Target: log_event.
    """
    path = tmp_path / "events.jsonl"
    previous = os.umask(0o022)
    try:
        gk.log_event({"event": "failure", "signature": "example"}, path)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("second_id", ["x", "y"])
def test_concurrent_claims_never_duplicate_or_lose_ids(tmp_path, monkeypatch, second_id):
    """BUG-001: concurrent successful callers serialize the entire read/update operation.

    Red at 15b0501e: both callers return x, or saved state loses x/y. A barrier at the existing
    write boundary exposes the race. Its bounded wait permits the first serialized writer to
    finish after repair; timeout is orchestration, never the failure assertion.
    """
    barrier = threading.Barrier(2)
    original = gk._write_claimed

    def synchronized_write(path, claimed):
        try:
            barrier.wait(timeout=0.15)
        except threading.BrokenBarrierError:
            pass
        original(path, claimed)

    monkeypatch.setattr(gk, "_write_claimed", synchronized_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda name: gk.claim_for_session("s1", [name], tmp_path), ["x", second_id]))
    delivered = [name for result in results for name in result]
    assert len(delivered) == len(set(delivered)), "concurrent callers delivered an entry twice"
    saved = json.loads((tmp_path / "state" / "s1.json").read_text(encoding="utf-8"))
    assert set(saved) == {"x", second_id}, "concurrent callers lost a successful claim"


def test_oversized_index_is_declined(tmp_path):
    """PERF-001: a valid index exceeding the 1 MiB input budget is declined.

    Red at 15b0501e: load_index returns the oversized entry. Production target: load_index.
    """
    write_index(tmp_path, [example_entry(body="x" * (2 * 1024 * 1024))])
    assert gk.load_index(tmp_path) == [], "oversized index was admitted"


def test_excessive_entry_count_is_declined(tmp_path):
    """PERF-001: the 256-entry admission budget is enforced before mapping entries.

    Red at 15b0501e: load_index admits all 257 entries. Target: load_index.
    """
    write_index(tmp_path, [example_entry(id=f"example-{i}") for i in range(257)])
    assert gk.load_index(tmp_path) == [], "too many entries were admitted"


@pytest.mark.parametrize("oversized", [
    {"body": "x" * 16385},
    {"triggers": [{"command": "pytest"}] * 33},
    {"symptoms": ["failure"] * 33},
])
def test_oversized_entry_does_not_cost_other_entries(tmp_path, oversized):
    """PERF-001: reject oversized fields/lists while preserving a normal neighboring entry.

    Red at 15b0501e: both entries are returned, rather than only the control. Target: _entry.
    """
    write_index(tmp_path, [example_entry(id="oversized", **oversized), example_entry(id="control")])
    assert [entry.id for entry in gk.load_index(tmp_path)] == ["control"]


def test_cleanup_is_not_repeated_for_every_claim(tmp_path, monkeypatch):
    """PERF-002: successful claims in different sessions share the cleanup throttle.

    Red at 15b0501e: two calls scan twice. Target: claim_for_session -> _prune_state.
    """
    original = gk.os.scandir
    scans = []

    def count(path):
        scans.append(path)
        return original(path)

    monkeypatch.setattr(gk.os, "scandir", count)
    assert gk.claim_for_session("s1", ["x"], tmp_path) == ["x"]
    assert gk.claim_for_session("s2", ["y"], tmp_path) == ["y"]
    assert len(scans) == 1, "every claim repeated directory-wide cleanup"


def test_cleanup_examines_a_bounded_batch(tmp_path, monkeypatch):
    """PERF-002: one public claim checks metadata of at most 128 state entries during cleanup.

    Red at 15b0501e: all 301 files receive metadata checks. Names are enumerated hourly, not bounded.
    Target: _prune_state. The retained candidates and expensive file checks remain bounded.
    """
    state = tmp_path / "state"
    state.mkdir()
    for index in range(300):
        (state / f"example-{index}.json").write_text("[]", encoding="utf-8")
    original = gk.os.scandir
    examined = []

    class CountedEntry:
        def __init__(self, item):
            self.item = item
            self.name = item.name
            self.path = item.path

        def is_file(self, **kwargs):
            return self.item.is_file(**kwargs)

        def stat(self, **kwargs):
            examined.append(self.name)
            return self.item.stat(**kwargs)

    class CountingScan:
        def __init__(self, path):
            self.iterator = original(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.iterator.close()

        def __iter__(self):
            for item in self.iterator:
                yield CountedEntry(item)

    monkeypatch.setattr(gk.os, "scandir", CountingScan)
    assert gk.claim_for_session("s1", ["x"], tmp_path) == ["x"]
    assert len(examined) <= 128, "one cleanup checked metadata for the entire retained session directory"


def test_cleanup_reaches_old_files_after_retained_batches(tmp_path, monkeypatch):
    """FIX-005 L6: retained entries must not starve old files beyond the first batch."""
    state = tmp_path / "state"
    state.mkdir()
    for index in range(300):
        (state / f"a{index:03}.json").write_text("[]", encoding="utf-8")
    expired = state / "z-old.json"
    expired.write_text("[]", encoding="utf-8")
    old = gk.time.time() - gk.STATE_MAX_AGE_SECONDS - 1
    os.utime(expired, (old, old))
    original = gk.os.scandir

    class OrderedScan:
        def __init__(self, path):
            with original(path) as listing:
                self.entries = sorted(listing, key=lambda item: item.name)

        def __enter__(self):
            return iter(self.entries)

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(gk.os, "scandir", OrderedScan)
    for index in range(4):
        stamp = tmp_path / ".state-cleanup"
        if stamp.exists():
            previous = gk.time.time() - gk.PRUNE_INTERVAL_SECONDS - 1
            os.utime(stamp, (previous, previous))
        assert gk.claim_for_session(f"cleanup-{index}", ["x"], tmp_path) == ["x"]
    assert not expired.exists(), "retained first-batch files starved later expired files"
