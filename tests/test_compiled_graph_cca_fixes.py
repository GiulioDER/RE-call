"""CCA run 20261126T220805Z on PR 1518: the compiled graph's forget, spend and hook defects.

Fake transport and fake embedder; no network, no paid call. Each test names the finding it pins.
Red proof is recorded per test under "Red proof" at the end of this docstring.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from recall.compiled_graph import service
from recall.compiled_graph.budget import Budget
from recall.compiled_graph.provider import MAX_OUTPUT_TOKENS, TEMPLATE_SLACK_TOKENS
from tests.test_compiled_graph_service import BINDING, FakeEmbedder, Transport, _chunks, _ensure, _env


def _artifacts(root: Path, tenant: str = "t1") -> list[Path]:
    return [path for path in service.tenant_folder(root, tenant).glob("cg-*") if path.is_dir()]


def _quotes(root: Path, text: bytes = b"Alice owns the release checklist.") -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and text in path.read_bytes()]


class _Reject(Transport):
    """A billed 200 whose output was cut off at the token cap: the same answer every time."""

    def __call__(self, url: str, body: bytes, headers: Mapping[str, str], timeout: float):  # type: ignore[no-untyped-def]  # matches provider.Transport
        self.calls += 1
        reply = {"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
        return 200, {}, json.dumps(reply).encode()


# --- STAKES-002: a billed rejection is not paid for again, and any stop backs off ---------------


def test_a_billed_rejection_is_not_sent_again_on_the_next_write(tmp_path: Path) -> None:
    env = _env(tmp_path)
    first = _Reject()
    _ensure(env, first)
    assert first.calls == 1, "negative control: the first build sends the source"
    again = _Reject()
    service.ensure_compiled_graph(
        "t1", _chunks(), {"generation_id": "g2", "corpus_fingerprint": "cf2"}, FakeEmbedder(),
        env=env, transport=again, sleep=lambda _: None,
    )
    assert again.calls == 0, "the next write's build paid again for a source the provider already refused"


def test_a_budget_stop_does_not_block_the_next_build(tmp_path: Path) -> None:
    """FA-1-7: the cache keeps what a build paid for, so the next one goes on from there."""
    env = {**_env(tmp_path), "RECALL_COMPILED_GRAPH_MAX_USD": "0.000001"}
    assert "stopped on budget" in _ensure(env, Transport())
    env["RECALL_COMPILED_GRAPH_MAX_USD"] = "1"
    message = service.ensure_compiled_graph(
        "t1", _chunks(), {"generation_id": "g2", "corpus_fingerprint": "cf2"}, FakeEmbedder(),
        env=env, transport=Transport(), sleep=lambda _: None,
    )
    assert "built" in message and "backoff" not in message, message


def test_force_clears_a_backoff_without_paying_again_for_cached_refusals(tmp_path: Path) -> None:
    """FA-1-7: only --retry-rejected sends a cached refusal again."""
    env = _env(tmp_path)
    _ensure(env, _Reject())
    binding = {"generation_id": "g2", "corpus_fingerprint": "cf2"}
    forced = _Reject()
    service.ensure_compiled_graph(
        "t1", _chunks(), binding, FakeEmbedder(), env=env, transport=forced, force=True, sleep=lambda _: None
    )
    assert forced.calls == 0, "--force paid again for a refusal the cache holds"
    retried = _Reject()
    service.ensure_compiled_graph(
        "t1", _chunks(), {"generation_id": "g3", "corpus_fingerprint": "cf3"}, FakeEmbedder(),
        env=env, transport=retried, retry_rejected=True, sleep=lambda _: None,
    )
    assert retried.calls == 1, "negative control: --retry-rejected sends it again"


class _ProxyPage(Transport):
    """A 200 whose body is not the provider's JSON: an intermediary's page, which may not recur."""

    def __call__(self, url: str, body: bytes, headers: Mapping[str, str], timeout: float):  # type: ignore[no-untyped-def]  # matches provider.Transport
        self.calls += 1
        return 200, {}, b"<html>bad gateway</html>"


def test_a_malformed_reply_body_is_not_cached_as_a_refusal(tmp_path: Path) -> None:
    """FA-1-8: only a refusal the same request would get again is cached."""
    env = _env(tmp_path)
    _ensure(env, _ProxyPage())
    again = _ProxyPage()
    service.ensure_compiled_graph(
        "t1", _chunks(), {"generation_id": "g2", "corpus_fingerprint": "cf2"}, FakeEmbedder(),
        env=env, transport=again, sleep=lambda _: None,
    )
    assert again.calls == 1, "a proxy's page was cached as the provider's permanent refusal"


def test_the_dry_run_does_not_price_cached_refusals(tmp_path: Path) -> None:
    """FA-1-9."""
    env = _env(tmp_path)
    _ensure(env, _Reject())
    estimate = service.estimate_compile("t1", _chunks(), env)
    assert "0 source(s) to compile" in estimate and "1 refused before" in estimate, estimate


# --- SEC-002: an absolute root, and a forget that finds the root a build recorded ---------------


def test_a_relative_root_is_refused_before_anything_is_sent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    env = {**_env(tmp_path), "RECALL_COMPILED_GRAPH_ARTIFACT_ROOT": "relative-root"}
    transport = Transport()
    message = _ensure(env, transport)
    assert "absolute" in message, message
    assert transport.calls == 0 and not (tmp_path / "relative-root").exists()


# --- STAKES-005: the hook never raises into the write -------------------------------------------


class _StoreDouble:
    tenant = "t1"

    def iter_chunks(self, batch_size: int = 1000):  # type: ignore[no-untyped-def]  # mirrors the store
        return iter(_chunks())

    def generation_binding(self) -> dict[str, str]:
        return dict(BINDING)

    def sources_present(self, sources: list[str]) -> set[str]:
        return set(sources)

    def record_compiled_graph_root(self, root: Path) -> None:
        return None


def test_a_malformed_mode_does_not_raise_into_the_write(tmp_path: Path) -> None:
    env = {**_env(tmp_path), "RECALL_COMPILED_GRAPH_MODE": "on"}
    message = service.ensure_for_store(_StoreDouble(), FakeEmbedder(), env)
    assert message.startswith("compiled graph not built"), message


def test_a_second_413_is_a_rejection_not_an_exception(tmp_path: Path) -> None:
    message = _ensure(_env(tmp_path), Transport(413))
    assert "built" in message and "1 source(s) rejected" in message, message


# --- SEC-004: a forget erases only its own tenant ------------------------------------------------


def test_a_forget_leaves_another_tenants_graph(tmp_path: Path) -> None:
    env = _env(tmp_path)
    for tenant in ("t1", "t2"):
        service.ensure_compiled_graph(
            tenant, _chunks(), BINDING, FakeEmbedder(), env=env, transport=Transport(), sleep=lambda _: None
        )
    assert _artifacts(tmp_path, "t2"), "negative control: t2 has an artifact"
    service.erase_forgotten_sources(["a.md"], tenant="t1", env=env)
    assert _artifacts(tmp_path, "t1") == []
    assert _artifacts(tmp_path, "t2"), "t1's forget removed t2's artifact"


# --- DAT-003: staging leftovers and unreadable artifacts are erased, and one does not stop the rest


def test_erasure_removes_staging_leftovers_and_unreadable_artifacts(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _ensure(env, Transport())
    folder = service.tenant_folder(tmp_path, "t1")
    leftover = folder / ".cg-staging-killed"
    leftover.mkdir()
    (leftover / "records.jsonl").write_text('{"source": "a.md", "fields": ["Alice owns the release checklist."]}\n')
    damaged = folder / "cg-damaged"
    damaged.mkdir()
    (damaged / "records.jsonl").write_text("not json\n")
    removed, _ = service.erase_forgotten_sources(["a.md"], tenant="t1", env=env)
    assert removed == 3
    assert not leftover.exists() and not damaged.exists() and _artifacts(tmp_path) == []
    assert _quotes(tmp_path) == []


# --- PERF-003 and PERF-001: superseded artifacts are pruned, unchanged records are not re-embedded


class _CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return super().embed(texts)


def test_superseded_artifacts_are_pruned_and_unchanged_records_reuse_their_vectors(tmp_path: Path) -> None:
    env = _env(tmp_path)
    embedders = []
    for corpus in ("cf1", "cf2", "cf3"):
        embedder = _CountingEmbedder()
        embedders.append(embedder)
        service.ensure_compiled_graph(
            "t1", _chunks(), {"generation_id": "g1", "corpus_fingerprint": corpus}, embedder,
            env=env, transport=Transport(), owner="s1", sleep=lambda _: None,
        )
    assert embedders[0].texts, "negative control: the first build embeds its records"
    assert embedders[1].texts == [] and embedders[2].texts == [], "unchanged records were embedded again"
    assert len(_artifacts(tmp_path)) == 1, "superseded artifacts were kept"


# --- DOC-003: the backoff message names the command that clears it ------------------------------


def test_the_backoff_message_names_force(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _ensure(env, Transport(402))
    assert "`recall graph compile --force`" in _ensure(env, Transport())


# --- CODE-001: the dry run prices what the build reserves ---------------------------------------


def test_the_dry_run_prices_the_request_the_build_reserves(tmp_path: Path) -> None:
    env = _env(tmp_path)
    sent: list[int] = []

    class Recording(Transport):
        def __call__(self, url: str, body: bytes, headers: Mapping[str, str], timeout: float):  # type: ignore[no-untyped-def]  # matches provider.Transport
            sent.append(len(body))
            return super().__call__(url, body, headers, timeout)

    estimate = service.estimate_compile("t1", _chunks(), env)
    _ensure(env, Recording())
    budget = Budget.from_env(env)
    reserved = budget._cost(sent[0] + TEMPLATE_SLACK_TOKENS, MAX_OUTPUT_TOKENS)
    assert f"worst case USD {reserved:.4f}" in estimate, (estimate, reserved)


# --- Fix audit pass 1: erasure under bad roots, damaged files, pruning -------------------------


def test_a_relative_configured_root_does_not_stop_erasure_under_recorded_roots(tmp_path: Path) -> None:
    """FA-1-2: the forget's own root is unusable; the root a build recorded is still erased."""
    env = _env(tmp_path)
    _ensure(env, Transport())
    forgetting = {**env, "RECALL_COMPILED_GRAPH_ARTIFACT_ROOT": "relative-root"}
    removed, unreached = service.erase_forgotten_sources(
        ["a.md"], tenant="t1", env=forgetting, recorded_roots=[str(tmp_path)]
    )
    assert removed == 1 and unreached == ["relative-root"]
    assert _quotes(tmp_path) == []


def test_a_recorded_unc_root_is_reported_and_never_probed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FA-1-1: a record naming //host/share must not open a connection to that host."""
    probed: list[str] = []
    real_is_dir = Path.is_dir

    def is_dir(self: Path) -> bool:
        probed.append(str(self))
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    _, unreached = service.erase_forgotten_sources(
        ["a.md"], tenant="t1", env=_env(tmp_path), recorded_roots=["//host/share/compiled"]
    )
    assert unreached == ["//host/share/compiled"]
    assert not [path for path in probed if path.replace("\\", "/").startswith("//")], probed


def test_a_non_utf8_artifact_and_a_locked_cache_do_not_stop_the_rest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FA-1-5: every artifact is visited, then the failure is raised."""
    import sqlite3

    from recall.compiled_graph.cache import CompileCache

    env = _env(tmp_path)
    _ensure(env, Transport())
    folder = service.tenant_folder(tmp_path, "t1")
    torn = folder / "cg-torn"
    torn.mkdir()
    (torn / "records.jsonl").write_bytes(b"\xff\xfe not utf-8\n")

    def locked(self: CompileCache, sources: object) -> int:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(CompileCache, "erase_sources", locked)
    with pytest.raises(OSError, match="database is locked"):
        service.erase_forgotten_sources(["a.md"], tenant="t1", env=env)
    assert not torn.exists() and _artifacts(tmp_path) == [], "artifacts after the failure were not visited"


def _build(env: Mapping[str, str], generation: str, *, owner: str = "s1", serving: str | None = None) -> str:
    return service.ensure_compiled_graph(
        "t1", _chunks(), {"generation_id": generation, "corpus_fingerprint": generation}, FakeEmbedder(),
        env=env, transport=Transport(), owner=owner, sleep=lambda _: None,
        serving_binding={"generation_id": serving, "corpus_fingerprint": serving} if serving else None,
    )


def _generations(root: Path) -> set[str]:
    from recall.compiled_graph.artifact import stored_lineage

    return {str((stored_lineage(path) or {}).get("generation")) for path in _artifacts(root)}


def test_pruning_keeps_the_serving_artifact_and_other_stores_artifacts(tmp_path: Path) -> None:
    """FA-1-6: an unpromoted candidate must not cost the active generation its artifact."""
    env = _env(tmp_path)
    _build(env, "g1")
    _build(env, "g2", serving="g1")
    _build(env, "g3", serving="g1")
    assert _generations(tmp_path) == {"g1", "g3"}, "the serving g1 artifact was pruned, or g2 kept"
    _build(env, "other", owner="s2")
    _build(env, "g4")
    assert "other" in _generations(tmp_path), "one store's build pruned another store's artifact"


def test_a_prune_that_fails_still_reports_the_published_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FA-1-11."""
    env = _env(tmp_path)
    _build(env, "g1")

    def refuse(path: object, *args: object, **kwargs: object) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(service.shutil, "rmtree", refuse)
    assert _build(env, "g2").startswith("compiled graph built"), "a failed prune hid a published artifact"


# --- Fix audit pass 2: the erasure counter, cached model refusals, roots, owner ---------------


def _counter_bump(folder: Path) -> None:
    """What a forget's erasure does to the folder's counter."""
    import uuid

    (folder / service.EPOCH_NAME).write_text(uuid.uuid4().hex, encoding="utf-8")


def test_a_forget_inside_the_commit_window_refuses_the_write(tmp_path: Path) -> None:
    """FA-2-3: presence is checked, then a forget deletes the source and erases, then the lock is taken."""
    folder = tmp_path / "tenant"
    folder.mkdir()
    present = iter([True, False])

    def all_present(sources: object) -> bool:
        answer = next(present)
        if answer:
            _counter_bump(folder)  # the forget's erasure, after this check and before the lock
        return answer

    ran: list[bool] = []
    assert service._committer(folder, all_present)(["a.md"], lambda: ran.append(True)) is False
    assert ran == [], "a write was committed after a forget erased its source"


def test_an_unrelated_forget_does_not_drop_a_present_source(tmp_path: Path) -> None:
    """FA-2-1: the counter moved, but the source is still held, so the write is retried, not dropped."""
    folder = tmp_path / "tenant"
    folder.mkdir()
    calls = [0]

    def all_present(sources: object) -> bool:
        calls[0] += 1
        if calls[0] == 1:
            _counter_bump(folder)  # a forget of some other source
        return True

    ran: list[bool] = []
    assert service._committer(folder, all_present)(["a.md"], lambda: ran.append(True)) is True
    assert ran == [True]


def test_forgets_that_never_stop_raise_rather_than_drop(tmp_path: Path) -> None:
    folder = tmp_path / "tenant"
    folder.mkdir()

    def all_present(sources: object) -> bool:
        _counter_bump(folder)
        return True

    with pytest.raises(service.CommitContended):
        service._committer(folder, all_present)(["a.md"], lambda: None)


class _Prose(Transport):
    """The model answered in prose inside a well-formed, billed envelope."""

    def __call__(self, url: str, body: bytes, headers: Mapping[str, str], timeout: float):  # type: ignore[no-untyped-def]  # matches provider.Transport
        self.calls += 1
        reply = {"choices": [{"message": {"content": "I cannot help with that."}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
        return 200, {}, json.dumps(reply).encode()


def test_a_billed_prose_reply_is_not_paid_for_again(tmp_path: Path) -> None:
    """FA-2-2."""
    env = _env(tmp_path)
    _ensure(env, _Prose())
    again = _Prose()
    service.ensure_compiled_graph(
        "t1", _chunks(), {"generation_id": "g2", "corpus_fingerprint": "cf2"}, FakeEmbedder(),
        env=env, transport=again, sleep=lambda _: None,
    )
    assert again.calls == 0, "a billed prose reply was paid for again on the next build"


class _UpstreamError(Transport):
    """Generation was cut short upstream: a 200 whose choice ends with finish_reason "error"."""

    def __call__(self, url: str, body: bytes, headers: Mapping[str, str], timeout: float):  # type: ignore[no-untyped-def]  # matches provider.Transport
        self.calls += 1
        reply = {"choices": [{"message": {"content": ""}, "finish_reason": "error"}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 0}}
        return 200, {}, json.dumps(reply).encode()


def test_a_reply_cut_short_upstream_is_not_cached_as_a_refusal(tmp_path: Path) -> None:
    """FA-3-1, owner decision 2026-11-27: only a model that finished ("stop") is a repeatable refusal.

    Red proof: against 909cd5632, which cached every content refusal inside a parsed envelope,
    the second build made 0 calls.
    """
    env = _env(tmp_path)
    _ensure(env, _UpstreamError())
    again = _UpstreamError()
    service.ensure_compiled_graph(
        "t1", _chunks(), {"generation_id": "g2", "corpus_fingerprint": "cf2"}, FakeEmbedder(),
        env=env, transport=again, sleep=lambda _: None,
    )
    assert again.calls == 1, "a reply cut short upstream was cached as the source's permanent refusal"


def test_a_failed_erasure_still_names_the_roots_it_could_not_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FA-2-12 and FA-2-6: the failure keeps the unreached roots, and an OSError from the cache
    erase does not stop the folder's artifacts being removed."""
    from recall.compiled_graph.cache import CompileCache

    env = _env(tmp_path)
    _ensure(env, Transport())

    def held_open(self: CompileCache, sources: object) -> int:
        raise PermissionError("the cache file is open in another process")

    monkeypatch.setattr(CompileCache, "erase_sources", held_open)
    with pytest.raises(service.CompiledGraphErasureIncomplete) as raised:
        service.erase_forgotten_sources(["a.md"], tenant="t1", env=env, recorded_roots=["//host/share"])
    assert raised.value.unreached == ["//host/share"]
    assert _artifacts(tmp_path) == [], "an OSError from the cache erase skipped the folder's artifacts"


class _OtherEmbedder(FakeEmbedder):
    name = "other-4"


def test_the_serving_artifact_is_kept_when_the_candidate_re_embeds(tmp_path: Path) -> None:
    """FA-2-5: the active generation's artifact was built with another embedder than the candidate's."""
    env = _env(tmp_path)
    _build(env, "g1")
    service.ensure_compiled_graph(
        "t1", _chunks(), {"generation_id": "g2", "corpus_fingerprint": "g2"}, _OtherEmbedder(),
        env=env, transport=Transport(), owner="s1", sleep=lambda _: None,
        serving_binding={"generation_id": "g1", "corpus_fingerprint": "g1"},
    )
    assert _generations(tmp_path) == {"g1", "g2"}, "the serving generation's artifact was pruned"


def test_the_owner_of_a_postgres_artifact_does_not_depend_on_its_password() -> None:
    """FA-2-9."""
    one = service._store_owner(SimpleNamespace(_dsn="postgresql://u:secret@db.example/recall"))  # type: ignore[arg-type]  # only _dsn is read
    two = service._store_owner(SimpleNamespace(_dsn="postgresql://u:rotated@db.example/recall"))  # type: ignore[arg-type]  # only _dsn is read
    assert one == two


def test_the_dry_run_prices_refusals_when_they_will_be_retried(tmp_path: Path) -> None:
    """FA-2-11."""
    env = _env(tmp_path)
    _ensure(env, _Reject())
    estimate = service.estimate_compile("t1", _chunks(), env, retry_rejected=True)
    assert "1 source(s) to compile" in estimate, estimate


# --- ENV-004: the wizard builds the compiled graph with the rescue off --------------------------


def test_the_wizard_promotion_builds_the_compiled_graph_with_the_rescue_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from recall.wizard.headless import _RealServices

    monkeypatch.setenv("RECALL_COMPILED_GRAPH_MODE", "active")
    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_MODE", raising=False)
    services = SimpleNamespace(
        atomic_env={}, config=SimpleNamespace(resolved_dsn="postgresql://x"), embedder=lambda: None
    )
    assert _RealServices._rescue_builder(services, "t1") is not None  # type: ignore[arg-type]  # only the attributes it reads
    monkeypatch.setenv("RECALL_COMPILED_GRAPH_MODE", "off")
    assert _RealServices._rescue_builder(services, "t1") is None, "negative control: both off builds nothing"


def test_the_wizard_compiled_graph_builder_never_builds_a_rescue_the_wizard_turned_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L5.5 regression diff: the shell says the rescue is active, the wizard turned it off (no root).

    The builder must build only the compiled graph; through the rescue's own gate it raised and
    failed the install.
    """
    import recall.compiled_graph.service as graph_service
    from recall.wizard.headless import _RealServices

    monkeypatch.setenv("RECALL_COMPILED_GRAPH_MODE", "active")
    monkeypatch.setenv("RECALL_ATOMIC_RESCUE_MODE", "active")
    monkeypatch.delenv("RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT", raising=False)
    built: list[str] = []
    monkeypatch.setattr(graph_service, "ensure_for_generation", lambda dsn, tenant, gen, make, env: built.append(gen) or "")
    services = SimpleNamespace(
        atomic_env={}, config=SimpleNamespace(resolved_dsn="postgresql://x"), embedder=lambda: None
    )
    builder = _RealServices._rescue_builder(services, "t1")  # type: ignore[arg-type]  # only the attributes it reads
    assert builder is not None
    assert builder("gen_1") is None
    assert built == ["gen_1"]
