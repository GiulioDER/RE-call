"""The search step: off changes nothing, shadow reports only, active promotes into ranks 9 and 10.

Red proof, recorded 2026-11-26 (production symbols in ``recall.compiled_graph.search`` and
``recall.retriever.HybridRetriever.search``):

- ``test_off_builds_no_step``: the mode check removed fails on ``step is None``.
- ``test_active_promotes_linked_tail_into_nine_and_ten``: seeds taken in artifact order instead of
  by cosine fails on the promoted ids.
- ``test_shadow_reports_and_keeps_the_order``: the shadow branch returning the promotion fails on
  the order.
- ``test_scoped_search_is_skipped``: the scope check removed fails on ``skipped:scoped``.
- ``test_missing_and_wrong_lineage_are_reported`` (RF5): every load error read as ``missing``
  fails on ``skipped:lineage``.
- ``test_unconfigured_search_is_reported_as_such`` (CCA ENV-001): against the pre-fix
  ``load_serving_artifact`` and step fails on ``skipped:damaged``.
- ``test_unreadable_artifact_skips_the_graph`` (CCA BUG-001): against the pre-fix
  ``load_artifact`` fails with the raw ``PermissionError`` out of the step.
- ``test_seeds_match_a_stable_full_sort_without_sorting_every_record`` (CCA PERF-002): against
  the pre-fix ``_seeds``, which sorted all 300 scores, fails on the largest array sorted.
- ``test_the_graph_and_a_dense_rescue_both_apply`` (CCA CODE-002): the ``graph_transform`` call
  removed from ``HybridRetriever.search`` fails on ``graph_calls``; it replaces a test that only
  re-checked the dense and post-fusion refusal.
- ``test_every_skip_reason_is_documented`` (CCA CODE-003): ``unconfigured`` removed from
  ``recall_mcp.models._COMPILED_GRAPH_DESCRIPTION`` fails on the description's reasons.
- ``test_step_error_keeps_the_exact_order``: the ``except`` in the transform removed fails with
  the raised error.
- ``test_retriever_applies_the_graph_after_fusion_and_records_the_stage``: the
  ``graph_transform`` call removed from ``HybridRetriever.search`` fails on the hit order.
- ``test_an_unreadable_policy_skips_the_graph_at_search`` (recorded 2026-11-28, SEC-001): before
  the policy digest joined ``serving_lineage`` it failed on ``step is None`` (the artifact loaded
  with the policy unreadable).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from recall.compiled_graph.artifact import ArtifactRecord, CompiledGraphArtifact, write_artifact
from recall.compiled_graph.search import compiled_graph_step
from recall.compiled_graph.service import serving_lineage, tenant_folder
from recall.types import Chunk, ScoredChunk

BINDING = {"generation_id": "g1", "corpus_fingerprint": "cf1"}


class FakeEmbedder:
    dim = 4
    name = "fake-4"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class Store:
    tenant = "t1"

    def generation_binding(self) -> dict[str, str]:
        return dict(BINDING)


def _env(root: Path, mode: str = "active") -> dict[str, str]:
    return {"RECALL_COMPILED_GRAPH_MODE": mode, "RECALL_COMPILED_GRAPH_ARTIFACT_ROOT": str(root),
            "RECALL_COMPILED_GRAPH_MODEL": "openai/gpt-6-luna-pro"}


def _pool(n: int = 30) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=Chunk(id=f"c{i:02d}", source="a.md", text="t"), score=1.0 - i / 100)
            for i in range(n)]


def _write(root: Path, env: dict[str, str]) -> None:
    lineage = serving_lineage("t1", BINDING, FakeEmbedder(), env)
    assert lineage is not None
    records = (
        ArtifactRecord("a.md#0", "a.md", "fact", ("far",), (), ("c21",)),
        ArtifactRecord("a.md#1", "a.md", "fact", ("near",), (), ("c25",)),
    )
    # Record 1 points along the query; record 0 is orthogonal, so cosine order is 1 then 0.
    vectors = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    write_artifact(tenant_folder(root, "t1"), CompiledGraphArtifact(lineage, records, vectors))


def _ids(hits: list[ScoredChunk]) -> list[str]:
    return [hit.chunk.id for hit in hits]


def test_off_builds_no_step(tmp_path: Path) -> None:
    step, outcome = compiled_graph_step(Store(), FakeEmbedder(), _env(tmp_path, "off"), scope_is_empty=True)
    assert step is None and outcome.note == ""


def test_active_promotes_linked_tail_into_nine_and_ten(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _write(tmp_path, env)
    step, outcome = compiled_graph_step(Store(), FakeEmbedder(), env, scope_is_empty=True)
    assert step is not None
    hits = step([1.0, 0.0, 0.0, 0.0], _pool())
    assert _ids(hits)[:8] == _ids(_pool())[:8]
    assert _ids(hits)[8:10] == ["c25", "c21"]
    assert outcome.note == "active:promoted=2"


def test_shadow_reports_and_keeps_the_order(tmp_path: Path) -> None:
    env = _env(tmp_path, "shadow")
    _write(tmp_path, env)
    step, outcome = compiled_graph_step(Store(), FakeEmbedder(), env, scope_is_empty=True)
    assert step is not None
    assert _ids(step([1.0, 0.0, 0.0, 0.0], _pool())) == _ids(_pool())
    assert outcome.note == "shadow:would_promote=2"


def test_scoped_search_is_skipped(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _write(tmp_path, env)
    step, outcome = compiled_graph_step(Store(), FakeEmbedder(), env, scope_is_empty=False)
    assert step is None and outcome.note == "skipped:scoped"


def test_missing_and_wrong_lineage_are_reported(tmp_path: Path) -> None:
    env = _env(tmp_path)
    step, outcome = compiled_graph_step(Store(), FakeEmbedder(), env, scope_is_empty=True)
    assert step is None and outcome.note == "skipped:missing"
    _write(tmp_path, env)
    folder = tenant_folder(tmp_path, "t1")
    (built,) = list(folder.glob("cg-*"))
    manifest = built / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace('"cf1"', '"cf0"'), encoding="utf-8")
    step, outcome = compiled_graph_step(Store(), FakeEmbedder(), env, scope_is_empty=True)
    assert step is None and outcome.note == "skipped:lineage"


@pytest.mark.parametrize("unset", ["RECALL_COMPILED_GRAPH_MODEL", "RECALL_COMPILED_GRAPH_ARTIFACT_ROOT"])
def test_unconfigured_search_is_reported_as_such(tmp_path: Path, unset: str) -> None:
    env = _env(tmp_path)
    _write(tmp_path, env)
    del env[unset]
    step, outcome = compiled_graph_step(Store(), FakeEmbedder(), env, scope_is_empty=True)
    assert step is None and outcome.note == "skipped:unconfigured"


def test_unreadable_artifact_skips_the_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from recall.compiled_graph import artifact as artifact_module

    env = _env(tmp_path)
    _write(tmp_path, env)

    def unreadable(path: Path) -> str:
        raise PermissionError(str(path))

    monkeypatch.setattr(artifact_module, "_sha256", unreadable)
    step, outcome = compiled_graph_step(Store(), FakeEmbedder(), env, scope_is_empty=True)
    assert step is None and outcome.note == "skipped:damaged"


def test_step_error_keeps_the_exact_order(tmp_path: Path) -> None:
    env = _env(tmp_path)
    _write(tmp_path, env)
    step, outcome = compiled_graph_step(Store(), FakeEmbedder(), env, scope_is_empty=True)
    assert step is not None
    pool = _pool()
    assert _ids(step([1.0, 0.0], pool)) == _ids(pool)
    assert outcome.note == "skipped:error"


def test_retriever_applies_the_graph_after_fusion_and_records_the_stage(tmp_path: Path) -> None:
    from recall.lite import LiteStore
    from recall.retriever import HybridRetriever

    store = LiteStore(tmp_path / "memory.db", dim=4)
    chunks = [Chunk(id=f"c{i:02d}", source="a.md", text=f"text {i}", metadata={"file": "a.md", "ord": i})
              for i in range(30)]
    store.upsert(chunks, [[1.0, i / 10, 0.0, 0.0] for i in range(30)])

    def graph(query_vector: list[float], ranked: list[ScoredChunk]) -> list[ScoredChunk]:
        return list(reversed(ranked))

    try:
        plain = HybridRetriever(cast(Any, store), FakeEmbedder(), use_sparse=False, candidate_k=30, env={})
        baseline = plain.search("q", k=30)
        assert "compiled_graph" not in baseline.diagnostics.stage_ms
        retriever = HybridRetriever(cast(Any, store), FakeEmbedder(), use_sparse=False, candidate_k=30,
                                    graph_transform=graph, env={})
        result = retriever.search("q", k=30)
        assert _ids(result.hits) == list(reversed(_ids(baseline.hits)))
        assert "compiled_graph" in result.diagnostics.stage_ms
    finally:
        store.close()


def test_the_graph_and_a_dense_rescue_both_apply(tmp_path: Path) -> None:
    from recall.lite import LiteStore
    from recall.retriever import HybridRetriever

    store = LiteStore(tmp_path / "memory.db", dim=4)
    chunks = [Chunk(id=f"c{i:02d}", source="a.md", text=f"text {i}", metadata={"file": "a.md", "ord": i})
              for i in range(12)]
    store.upsert(chunks, [[1.0, i / 10, 0.0, 0.0] for i in range(12)])
    dense_calls: list[int] = []
    graph_calls: list[int] = []

    def dense(query_vector: list[float], dense_hits: list[ScoredChunk]) -> list[ScoredChunk]:
        dense_calls.append(len(dense_hits))
        return dense_hits

    def graph(query_vector: list[float], ranked: list[ScoredChunk]) -> list[ScoredChunk]:
        graph_calls.append(len(ranked))
        return list(reversed(ranked))

    try:
        retriever = HybridRetriever(cast(Any, store), FakeEmbedder(), use_sparse=False, candidate_k=12,
                                    dense_transform=dense, graph_transform=graph, env={})
        result = retriever.search("q", k=12)
        assert dense_calls and graph_calls
        assert _ids(result.hits)[0] == "c11"
    finally:
        store.close()


def test_seeds_match_a_stable_full_sort_without_sorting_every_record(monkeypatch: pytest.MonkeyPatch) -> None:
    from recall.compiled_graph import search as search_module

    rng = np.random.default_rng(7)
    # Rounded to one decimal so many records tie, including at the SEED_K-th place.
    vectors = np.round(rng.random((300, 4)), 1).astype(np.float32)
    records = tuple(ArtifactRecord(f"a.md#{i}", "a.md", "fact", (), (), (f"c{i}",)) for i in range(300))
    artifact = CompiledGraphArtifact({}, records, vectors)
    query = [0.5, 0.25, 0.125, 0.0]
    # The pre-fix expression exactly: a float32 query over a Python float norm keeps float32.
    query32 = np.asarray(query, dtype=np.float32)
    scores = vectors @ (query32 / float(np.linalg.norm(query32)))
    expected = [records[i].key for i in np.argsort(-scores, kind="stable")[:20]]
    sorted_sizes: list[int] = []
    real_argsort = np.argsort

    def recording(values: Any, *args: Any, **kwargs: Any) -> Any:
        sorted_sizes.append(len(values))
        return real_argsort(values, *args, **kwargs)

    monkeypatch.setattr(search_module.np, "argsort", recording)
    assert [key for key, _ in search_module._seeds(artifact, query)] == expected
    assert max(sorted_sizes) < 300


def test_every_skip_reason_is_documented() -> None:
    import re

    from recall_mcp.models import _COMPILED_GRAPH_DESCRIPTION

    root = Path(__file__).resolve().parents[1]
    emitted = set(re.findall(r'"skipped:(\w+)"', (root / "recall/compiled_graph/search.py").read_text(encoding="utf-8")))
    described = re.search(r"skipped: and why \(([^)]*)\)", _COMPILED_GRAPH_DESCRIPTION)
    documented = re.search(r"`skipped:` and why \(([^)]*)\)", (root / "docs/COMPILED_GRAPH.md").read_text(encoding="utf-8"))
    assert described is not None and documented is not None
    assert {reason.strip() for reason in described.group(1).split(",")} == emitted
    assert {reason.strip(" `") for reason in documented.group(1).split(",")} == emitted


def test_an_unreadable_policy_skips_the_graph_at_search(tmp_path: Path) -> None:
    policy = tmp_path / "policy.json"
    rule = {"prefix": "a.md", "principals": [], "purposes": ["retrieval", "compile"], "redactions": []}
    policy.write_text(json.dumps({"default_deny": True, "rules": [rule]}), encoding="utf-8")
    env = {**_env(tmp_path), "RECALL_SOURCE_POLICY_FILE": str(policy)}
    _write(tmp_path, env)
    step, _ = compiled_graph_step(Store(), FakeEmbedder(), env, scope_is_empty=True)
    assert step is not None
    gone = {**env, "RECALL_SOURCE_POLICY_FILE": str(tmp_path / "missing.json")}
    step, outcome = compiled_graph_step(Store(), FakeEmbedder(), gone, scope_is_empty=True)
    assert step is None and outcome.note == "skipped:unconfigured"
