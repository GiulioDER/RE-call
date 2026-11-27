"""Building the compiled graph for a store: mode, lineage, backoff, erasure, never raising into a write.

Fake transport and fake embedder; no network. Red proof, recorded 2026-11-26 (production symbols
in ``recall.compiled_graph.service``):

- ``test_mode_by_tenant_overrides``: the per-tenant map ignored fails on the override.
- ``test_a_malformed_tenant_entry_is_refused`` (CCA ENV-002): against the pre-fix
  ``compiled_graph_mode`` fails with no exception, the tenant served the global ``active``.
- ``test_a_tenant_name_may_hold_a_colon`` (CCA FA-1-2): against the pass-1 parser, which split on
  the first colon, fails with ``ValueError`` on ``team:active``.
- ``test_a_bad_mode_never_fails_the_write_or_the_promotion`` (CCA FA-1-1): against the pass-1
  ``ensure_for_store`` and ``ensure_for_generation`` fails with the raised ``ValueError``.
- ``test_the_server_refuses_a_bad_tenant_map_at_startup`` (CCA FA-1-1): against the pass-1
  ``_validate_runtime_options`` fails with no exception.
- ``test_a_partial_erase_serves_nothing`` (CCA FA-1-3): against the pass-1
  ``erase_forgotten_sources``, which left the manifest, fails with no exception, the cached
  records served.
- ``test_a_relative_artifact_root_is_refused`` (CCA ENV-003): against the pre-fix
  ``artifact_root``, which resolved a relative root against each process's cwd, fails with no
  exception.
- ``test_body_from_chunks_keeps_offsets``: chunks concatenated without their offsets fails on the
  slice equality.
- ``test_a_build_reports_sources_without_offsets`` and
  ``test_a_dry_run_reports_sources_without_offsets`` (recorded 2026-11-27): with
  ``recall.compiled_graph.build.without_offsets`` mutated to ``return False``, both fail on the
  count assertion, the build and the dry run each reporting ``0`` skipped for a source whose only
  chunk has ``text_start=None``. Before the fix such a source was skipped without being counted.
- ``test_a_source_whose_offsets_place_nothing_is_counted`` (recorded 2026-11-27, CCA CODE-001):
  against ``without_offsets`` judged by ``promotion._offsets`` (commit adce9b281), it fails on the
  count assertion: ``text_start=-1`` passes ``_offsets`` but ``body_from_chunks`` places nothing,
  so the source was skipped and reported as ``0`` skipped.
- ``test_off_or_rootless_builds_nothing``: the mode check removed fails on the call count.
- ``test_build_writes_an_artifact_the_loader_serves``: the loader lineage built without the model
  fails with ``ArtifactMissing``.
- ``test_existing_artifact_is_not_rebuilt``: the in-force check removed fails on the call count.
- ``test_credit_stop_sets_backoff_and_never_raises``: the backoff file not written fails on the
  second call count.
- ``test_forced_build_clears_backoff``: ``force`` ignored fails on the call count.
- ``test_config_error_is_reported_not_raised``: the error re-raised fails with an exception.
- ``test_erase_forgotten_sources_removes_artifacts_and_cache``: artifacts kept on forget fails on
  the quote check.
- ``test_a_forget_keeps_another_tenants_artifacts`` (recorded 2026-11-27): the erase globbing every
  tenant folder again (``root.glob("tenant-*/cg-*/records.jsonl")``) fails on the other tenant's
  artifact count.
- ``test_artifact_dropped_when_a_source_vanished_during_build``: the post-write presence check
  removed fails on ``ArtifactMissing``.
- ``test_generation_build_fixes_the_generation_first``: ``set_fixed_generation`` removed fails on
  ``store.fixed``.
- ``test_generation_build_never_raises_into_the_promotion``: the broad catch narrowed to
  ``KeyError`` fails with ``RuntimeError``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from recall.compiled_graph.anchors import build_source_anchors
from recall.compiled_graph.artifact import ArtifactMissing, load_artifact
from recall.compiled_graph.service import (
    body_from_chunks,
    compiled_graph_mode,
    ensure_compiled_graph,
    erase_forgotten_sources,
    estimate_compile,
    serving_lineage,
    tenant_folder,
)
from recall.types import Chunk

BODY = "Alice owns the release checklist. " + "x" * 2_000 + " The deploy uses port 18015."
BINDING = {"generation_id": "g1", "corpus_fingerprint": "cf1"}


class FakeEmbedder:
    dim = 4
    name = "fake-4"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(len(text)), 0.0, 0.0] for text in texts]


class Transport:
    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.calls = 0

    def __call__(self, url: str, body: bytes, headers: Mapping[str, str], timeout: float):  # type: ignore[no-untyped-def]  # matches provider.Transport
        self.calls += 1
        if self.status != 200:
            return self.status, {}, b"{}"
        anchor = build_source_anchors("a.md", BODY)[0].id
        record = {"kind": "fact", "summary": "Alice owns the release checklist.", "detail": "",
                  "entities": ["Alice"], "evidence_anchor_ids": [anchor]}
        content = json.dumps({"records": [record]})
        reply = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
        return 200, {}, json.dumps(reply).encode()


def _chunks() -> list[Chunk]:
    return [
        Chunk(id=f"a.md:{i}", source="a.md", text=BODY[start : start + 1_000],
              metadata={"file": "a.md", "text_start": start, "text_end": min(start + 1_000, len(BODY))})
        for i, start in enumerate(range(0, len(BODY), 1_000))
    ]


def _env(root: Path, mode: str = "active") -> dict[str, str]:
    return {
        "RECALL_COMPILED_GRAPH_MODE": mode,
        "RECALL_COMPILED_GRAPH_ARTIFACT_ROOT": str(root),
        "RECALL_COMPILED_GRAPH_API_KEY": "sk-test",
        "RECALL_COMPILED_GRAPH_MODEL": "openai/gpt-6-luna-pro",
        "RECALL_COMPILED_GRAPH_MAX_USD": "1",
        "RECALL_COMPILED_GRAPH_COST_PER_1K_INPUT": "0.001",
        "RECALL_COMPILED_GRAPH_COST_PER_1K_OUTPUT": "0.002",
    }


def _ensure(env: Mapping[str, str], transport: Transport, **kwargs: object) -> str:
    return ensure_compiled_graph(
        "t1", _chunks(), BINDING, FakeEmbedder(), env=env, transport=transport,
        sleep=lambda _: None, **kwargs,  # type: ignore[arg-type]  # forwarding keyword options under test
    )


def test_mode_by_tenant_overrides() -> None:
    env = {"RECALL_COMPILED_GRAPH_MODE": "shadow",
           "RECALL_COMPILED_GRAPH_MODE_BY_TENANT": "t1:active, t2:off"}
    assert compiled_graph_mode(env, "t1") == "active"
    assert compiled_graph_mode(env, "t2") == "off"
    assert compiled_graph_mode(env, "t3") == "shadow"
    assert compiled_graph_mode({}, "t1") == "off"
    with pytest.raises(ValueError):
        compiled_graph_mode({"RECALL_COMPILED_GRAPH_MODE": "on"}, "t1")


@pytest.mark.parametrize("entries", ["t1=off", "t1 off", ":off", "t2:on, t1:off"])
def test_a_malformed_tenant_entry_is_refused(entries: str) -> None:
    env = {"RECALL_COMPILED_GRAPH_MODE": "active", "RECALL_COMPILED_GRAPH_MODE_BY_TENANT": entries}
    with pytest.raises(ValueError, match="RECALL_COMPILED_GRAPH_MODE_BY_TENANT"):
        compiled_graph_mode(env, "t1")


def test_a_tenant_name_may_hold_a_colon() -> None:
    env = {"RECALL_COMPILED_GRAPH_MODE_BY_TENANT": "org:team:active, t2:off"}
    assert compiled_graph_mode(env, "org:team") == "active"
    assert compiled_graph_mode(env, "t2") == "off"


def test_a_bad_mode_never_fails_the_write_or_the_promotion(tmp_path: Path) -> None:
    from recall.compiled_graph import service

    class Store:
        tenant = "t1"

    env = {**_env(tmp_path), "RECALL_COMPILED_GRAPH_MODE_BY_TENANT": "t2=off"}
    assert service.ensure_for_store(Store(), FakeEmbedder(), env).startswith("compiled graph not built")  # type: ignore[arg-type]  # only the tenant is read before the refusal
    message = service.ensure_for_generation("postgresql://x", "t1", "g7", FakeEmbedder, env)
    assert message.startswith("compiled graph not built")


def test_the_server_refuses_a_bad_tenant_map_at_startup() -> None:
    from recall_mcp.settings import _validate_runtime_options

    with pytest.raises(ValueError, match="RECALL_COMPILED_GRAPH_MODE_BY_TENANT"):
        _validate_runtime_options({"RECALL_COMPILED_GRAPH_MODE_BY_TENANT": "t2=off"})


def test_a_partial_erase_serves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from recall.compiled_graph import service

    env = _env(tmp_path)
    _ensure(env, Transport())
    lineage = serving_lineage("t1", BINDING, FakeEmbedder(), env)
    assert lineage is not None
    load_artifact(tenant_folder(tmp_path, "t1"), lineage)

    def stops_part_way(path: Path) -> None:
        (Path(path) / "records.jsonl").unlink()
        raise PermissionError("a member is held open")

    monkeypatch.setattr(service.shutil, "rmtree", stops_part_way)
    with pytest.raises(service.CompiledGraphErasureIncomplete):
        erase_forgotten_sources(["a.md"], tenant="t1", env=env)
    with pytest.raises(ArtifactMissing):
        load_artifact(tenant_folder(tmp_path, "t1"), lineage)


def test_a_relative_artifact_root_is_refused(tmp_path: Path) -> None:
    from recall.compiled_graph import service
    from recall_mcp.settings import _validate_runtime_options

    env = {**_env(tmp_path), "RECALL_COMPILED_GRAPH_ARTIFACT_ROOT": "graphs"}
    with pytest.raises(ValueError, match="RECALL_COMPILED_GRAPH_ARTIFACT_ROOT"):
        service.artifact_root(env)
    with pytest.raises(ValueError, match="RECALL_COMPILED_GRAPH_ARTIFACT_ROOT"):
        _validate_runtime_options({"RECALL_COMPILED_GRAPH_ARTIFACT_ROOT": "graphs"})


def test_body_from_chunks_keeps_offsets() -> None:
    chunks = _chunks()
    body = body_from_chunks(chunks)
    for chunk in chunks:
        assert body[chunk.metadata["text_start"] : chunk.metadata["text_end"]] == chunk.text
    gap = [Chunk(id="g:0", source="g", text="abc", metadata={"text_start": 5, "text_end": 8})]
    assert body_from_chunks(gap) == "     abc"


def _chunks_with_an_unplaced_source() -> list[Chunk]:
    unplaced = Chunk(id="c.py:0", source="c.py", text="def f():\n    return 3",
                     metadata={"file": "c.py", "text_start": None, "text_end": None})
    return [*_chunks(), unplaced]


def test_a_build_reports_sources_without_offsets(tmp_path: Path) -> None:
    message = ensure_compiled_graph(
        "t1", _chunks_with_an_unplaced_source(), BINDING, FakeEmbedder(), env=_env(tmp_path),
        transport=Transport(), sleep=lambda _: None,
    )
    assert "built" in message, message
    assert "1 source(s) skipped because no chunk carries character offsets" in message, message


def test_a_dry_run_reports_sources_without_offsets(tmp_path: Path) -> None:
    estimate = estimate_compile("t1", _chunks_with_an_unplaced_source(), _env(tmp_path))
    assert "1 source(s) to compile" in estimate, estimate
    assert "1 skipped because no chunk carries character offsets" in estimate, estimate


def test_a_source_whose_offsets_place_nothing_is_counted(tmp_path: Path) -> None:
    unplaceable = Chunk(id="d.md:0", source="d.md", text="hello",
                        metadata={"file": "d.md", "text_start": -1, "text_end": 5})
    estimate = estimate_compile("t1", [*_chunks(), unplaceable], _env(tmp_path))
    assert "1 skipped because no chunk carries character offsets" in estimate, estimate


def test_off_or_rootless_builds_nothing(tmp_path: Path) -> None:
    transport = Transport()
    assert _ensure(_env(tmp_path, "off"), transport) == ""
    env = _env(tmp_path)
    del env["RECALL_COMPILED_GRAPH_ARTIFACT_ROOT"]
    assert "RECALL_COMPILED_GRAPH_ARTIFACT_ROOT is not set" in _ensure(env, transport)
    assert transport.calls == 0


def test_build_writes_an_artifact_the_loader_serves(tmp_path: Path) -> None:
    env = _env(tmp_path)
    message = _ensure(env, Transport())
    assert "built" in message
    lineage = serving_lineage("t1", BINDING, FakeEmbedder(), env)
    assert lineage is not None
    artifact = load_artifact(tenant_folder(tmp_path, "t1"), lineage)
    assert artifact.records[0].edges == ("a.md:0", "a.md:1")


def test_existing_artifact_is_not_rebuilt(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _ensure(env, Transport())
    again = Transport()
    assert "in force" in _ensure(env, again)
    assert again.calls == 0


def test_credit_stop_sets_backoff_and_never_raises(tmp_path: Path) -> None:
    env = _env(tmp_path)
    refused = Transport(402)
    assert "credit" in _ensure(env, refused)
    assert refused.calls == 1
    later = Transport()
    assert "backoff" in _ensure(env, later)
    assert later.calls == 0


def test_forced_build_clears_backoff(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _ensure(env, Transport(402))
    forced = Transport()
    assert "built" in _ensure(env, forced, force=True)
    assert forced.calls == 1


def test_config_error_is_reported_not_raised(tmp_path: Path) -> None:
    env = _env(tmp_path)
    del env["RECALL_COMPILED_GRAPH_API_KEY"]
    message = _ensure(env, Transport())
    assert "RECALL_COMPILED_GRAPH_API_KEY" in message
    del env["RECALL_COMPILED_GRAPH_MODEL"]
    assert "RECALL_COMPILED_GRAPH_MODEL" in _ensure(env, Transport())


def test_erase_forgotten_sources_removes_artifacts_and_cache(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _ensure(env, Transport())
    removed, unreached = erase_forgotten_sources(["a.md"], tenant="t1", env=env)
    assert removed >= 1 and unreached == []
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert b"Alice owns the release checklist." not in path.read_bytes(), path


def test_a_forget_keeps_another_tenants_artifacts(tmp_path: Path) -> None:
    import numpy as np

    from recall.compiled_graph.artifact import ArtifactRecord, CompiledGraphArtifact, write_artifact

    record = ArtifactRecord("a.md#0", "a.md", "fact", ("x",), ("Alice",), ("c1",))
    for tenant in ("t1", "t2"):
        write_artifact(tenant_folder(tmp_path, tenant),
                       CompiledGraphArtifact({"tenant": tenant}, (record,), np.ones((1, 4), dtype=np.float32)))
    assert erase_forgotten_sources(["a.md"], tenant="t1", env=_env(tmp_path)) == (1, [])
    assert list(tenant_folder(tmp_path, "t1").glob("cg-*")) == []
    assert len(list(tenant_folder(tmp_path, "t2").glob("cg-*/records.jsonl"))) == 1


def test_artifact_dropped_when_a_source_vanished_during_build(tmp_path: Path) -> None:
    env = _env(tmp_path)
    message = _ensure(env, Transport(), all_present=lambda sources: False)
    assert "not built" in message
    lineage = serving_lineage("t1", BINDING, FakeEmbedder(), env)
    assert lineage is not None
    with pytest.raises(ArtifactMissing):
        load_artifact(tenant_folder(tmp_path, "t1"), lineage)


class _GenerationStoreDouble:
    """Stands in for ``GenerationStore``: records the generation fixed before chunks are read."""

    opened: list["_GenerationStoreDouble"] = []

    def __init__(self, dsn: str, dim: int, *, tenant: str) -> None:
        self.tenant = tenant
        self.fixed: str | None = None
        self.read_after_fix = False
        _GenerationStoreDouble.opened.append(self)

    def __enter__(self) -> "_GenerationStoreDouble":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def set_fixed_generation(self, generation_id: str) -> None:
        self.fixed = generation_id

    def iter_chunks(self, batch_size: int = 1000):  # type: ignore[no-untyped-def]  # mirrors the store
        self.read_after_fix = self.fixed is not None
        return iter(_chunks())

    def generation_binding(self) -> dict[str, str]:
        return {"generation_id": str(self.fixed), "corpus_fingerprint": "cf"}

    def sources_present(self, sources: list[str]) -> set[str]:
        return set(sources)

    def record_compiled_graph_root(self, root: Path) -> None:
        return None


def test_generation_build_fixes_the_generation_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from recall.compiled_graph import service

    monkeypatch.setattr("recall.generation_store.GenerationStore", _GenerationStoreDouble)
    monkeypatch.setattr(service, "urllib_transport", Transport())
    _GenerationStoreDouble.opened.clear()
    message = service.ensure_for_generation("postgresql://x", "t1", "g7", FakeEmbedder, _env(tmp_path))
    (store,) = _GenerationStoreDouble.opened
    assert store.fixed == "g7" and store.read_after_fix
    assert "built for g7" in message


def test_generation_build_never_raises_into_the_promotion(tmp_path: Path) -> None:
    from recall.compiled_graph import service

    def broken() -> FakeEmbedder:
        raise RuntimeError("model not available")

    message = service.ensure_for_generation("postgresql://x", "t1", "g7", broken, _env(tmp_path))
    assert message == "compiled graph not built for g7: RuntimeError"
    assert service.ensure_for_generation("postgresql://x", "t1", "g7", broken, _env(tmp_path, "off")) == ""
