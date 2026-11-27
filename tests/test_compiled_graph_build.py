"""The build loop: compile what is not cached, stop whole on credit or budget, never half-write.

Every test drives a fake transport and a fake embedder. Red proof, recorded 2026-11-26
(production symbols in ``recall.compiled_graph.build``):

- ``test_build_compiles_embeds_and_links``: edges computed against another source's chunks fails
  on the record edges.
- ``test_cached_source_is_not_sent``: the cache lookup skipped fails on the call count.
- ``test_empty_body_sends_nothing``: anchors not checked fails on the call count.
- ``test_credit_stop_returns_nothing_and_keeps_paid_work``: the stop path returning the partial
  artifact fails on ``artifact is None``.
- ``test_budget_stop_is_reported``: ``BudgetExhausted`` treated as a source rejection fails on
  ``stopped``.
- ``test_repeated_rejection_stops_the_build``: the consecutive-rejection stop removed fails on the
  call count.
- ``test_oversized_payload_is_refit_once``: no refit fails on the second payload's size.
- ``test_chunks_without_offsets_are_counted_once_per_source``: counting inside the record loop
  fails on ``edgeless_chunks == 1`` (2).
- ``test_source_forgotten_during_compile_is_not_cached``: the ``still_present`` check removed
  fails on the cache row (recorded before CCA run 20261126T220805Z replaced that check with
  ``commit``, which the test now refuses).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from recall.compiled_graph.anchors import ANCHOR_CHARS
from recall.compiled_graph.budget import Budget
from recall.compiled_graph.build import SourceInput, build_compiled_graph
from recall.compiled_graph.cache import CompileCache, cache_key
from recall.compiled_graph.provider import ProviderConfig
from recall.types import Chunk

CONFIG = ProviderConfig(api_key="sk-test", model="openai/gpt-6-luna-pro")
LINEAGE = {"tenant": "t", "generation": "g1", "calibration": "c1", "embedder": "fake-4"}
BODY_A = "Alice owns the release checklist. " + "x" * 2_000 + " The deploy uses port 18015."


class FakeEmbedder:
    dim = 4
    name = "fake-4"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[float(len(text)), 1.0, 0.0, 0.0] for text in texts]


class Transport:
    def __init__(self, replies: Mapping[str, tuple[int, bytes]]) -> None:
        self.replies = replies
        self.calls: list[dict[str, object]] = []

    def __call__(self, url: str, body: bytes, headers: Mapping[str, str], timeout: float):  # type: ignore[no-untyped-def]  # matches provider.Transport
        request = json.loads(body)
        self.calls.append(request)
        stored = json.loads(request["messages"][1]["content"][len("<stored_data>") : -len("</stored_data>")])
        status, reply = self.replies.get(stored["source"], (200, b""))
        return status, {}, reply


def _ok(records: list[dict[str, object]]) -> tuple[int, bytes]:
    content = json.dumps({"records": records})
    body = {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1_000, "completion_tokens": 100},
    }
    return 200, json.dumps(body).encode()


def _chunks(source: str, body: str, size: int = 1_000) -> tuple[Chunk, ...]:
    return tuple(
        Chunk(id=f"{source}:{i}", source=source, text=body[start : start + size],
              metadata={"file": source, "text_start": start, "text_end": min(start + size, len(body))})
        for i, start in enumerate(range(0, len(body), size))
    )


def _budget(max_usd: float = 10.0) -> Budget:
    return Budget(max_usd=max_usd, cost_in_per_1k=0.001, cost_out_per_1k=0.002)


def _alice(anchor_id: str) -> dict[str, object]:
    return {"kind": "fact", "summary": "Alice owns the release checklist.", "detail": "",
            "entities": ["Alice"], "evidence_anchor_ids": [anchor_id]}


def _run(sources: list[SourceInput], transport: Transport, tmp_path: Path, **kwargs):  # type: ignore[no-untyped-def]  # returns build_compiled_graph's tuple
    cache = kwargs.pop("cache", None) or CompileCache(tmp_path / "compile-cache.sqlite")
    return build_compiled_graph(
        sources, FakeEmbedder(), CONFIG, transport, cache, kwargs.pop("budget", None) or _budget(),
        lineage=LINEAGE, sleep=lambda _: None, **kwargs,
    )


def test_build_compiles_embeds_and_links(tmp_path: Path) -> None:
    from recall.compiled_graph.anchors import build_source_anchors

    anchors = build_source_anchors("a.md", BODY_A)
    artifact, report = _run(
        [SourceInput("a.md", BODY_A, _chunks("a.md", BODY_A)),
         SourceInput("b.md", "unrelated text", _chunks("b.md", "unrelated text"))],
        Transport({"a.md": _ok([_alice(anchors[0].id)]), "b.md": _ok([])}), tmp_path,
    )
    assert artifact is not None
    assert report.stopped == ""
    assert report.sources_compiled == 2
    (record,) = artifact.records
    assert record.source == "a.md"
    assert record.edges == ("a.md:0", "a.md:1")
    assert record.entities == ("Alice",)
    assert artifact.vectors.shape == (1, 4)
    assert artifact.lineage["model"] == CONFIG.model


def test_cached_source_is_not_sent(tmp_path: Path) -> None:
    cache = CompileCache(tmp_path / "compile-cache.sqlite")
    cache.put(cache_key(CONFIG, "a.md", hashlib.sha256(BODY_A.encode()).hexdigest()), "a.md", [])
    transport = Transport({})
    artifact, report = _run([SourceInput("a.md", BODY_A, _chunks("a.md", BODY_A))], transport, tmp_path, cache=cache)
    assert transport.calls == []
    assert report.sources_cached == 1
    assert artifact is not None and artifact.records == ()


def test_empty_body_sends_nothing(tmp_path: Path) -> None:
    transport = Transport({})
    artifact, report = _run([SourceInput("e.md", "   \n", ())], transport, tmp_path)
    assert transport.calls == []
    assert artifact is not None and report.calls == 0


def test_credit_stop_returns_nothing_and_keeps_paid_work(tmp_path: Path) -> None:
    cache = CompileCache(tmp_path / "compile-cache.sqlite")
    transport = Transport({"a.md": _ok([]), "b.md": (402, b"{}")})
    sources = [SourceInput("a.md", BODY_A, _chunks("a.md", BODY_A)),
               SourceInput("b.md", "body b", _chunks("b.md", "body b")),
               SourceInput("c.md", "body c", _chunks("c.md", "body c"))]
    artifact, report = _run(sources, transport, tmp_path, cache=cache)
    assert artifact is None
    assert report.stopped == "credit"
    assert len(transport.calls) == 2
    assert cache.get(cache_key(CONFIG, "a.md", hashlib.sha256(BODY_A.encode()).hexdigest())) == []
    resumed = Transport({"b.md": _ok([]), "c.md": _ok([])})
    artifact, report = _run(sources, resumed, tmp_path, cache=cache)
    assert artifact is not None
    assert len(resumed.calls) == 2
    assert report.sources_cached == 1


def test_budget_stop_is_reported(tmp_path: Path) -> None:
    transport = Transport({"a.md": _ok([])})
    artifact, report = _run([SourceInput("a.md", BODY_A, _chunks("a.md", BODY_A))], transport,
                            tmp_path, budget=_budget(max_usd=0.001))
    assert artifact is None
    assert report.stopped == "budget"
    assert transport.calls == []


def test_repeated_rejection_stops_the_build(tmp_path: Path) -> None:
    sources = [SourceInput(f"{i}.md", f"body {i}", _chunks(f"{i}.md", f"body {i}")) for i in range(6)]
    transport = Transport({f"{i}.md": (404, b"{}") for i in range(6)})
    artifact, report = _run(sources, transport, tmp_path)
    assert artifact is None
    assert report.stopped == "provider"
    assert len(transport.calls) == 3


def test_one_rejection_skips_only_that_source(tmp_path: Path) -> None:
    sources = [SourceInput(f"{i}.md", f"body {i}", _chunks(f"{i}.md", f"body {i}")) for i in range(3)]
    transport = Transport({"0.md": _ok([]), "1.md": (404, b"{}"), "2.md": _ok([])})
    artifact, report = _run(sources, transport, tmp_path)
    assert artifact is not None
    assert report.sources_rejected == 1
    assert report.stopped == ""


def test_oversized_payload_is_refit_once(tmp_path: Path) -> None:
    body = "y" * (ANCHOR_CHARS * 6)
    big = json.dumps({"error": {"message": "maximum context length exceeded"}}).encode()

    class Shrinking(Transport):
        def __call__(self, url, body, headers, timeout):  # type: ignore[no-untyped-def]  # matches provider.Transport
            self.calls.append(json.loads(body))
            return (400, {}, big) if len(self.calls) == 1 else (200, {}, _ok([])[1])

    transport = Shrinking({})
    artifact, _ = _run([SourceInput("y.md", body, _chunks("y.md", body))], transport, tmp_path)
    assert artifact is not None
    first, second = (len(json.dumps(call)) for call in transport.calls)
    assert second < first


def test_source_forgotten_during_compile_is_not_cached(tmp_path: Path) -> None:
    cache = CompileCache(tmp_path / "compile-cache.sqlite")
    transport = Transport({"a.md": _ok([])})
    artifact, _ = _run([SourceInput("a.md", BODY_A, _chunks("a.md", BODY_A))], transport, tmp_path,
                       cache=cache, commit=lambda sources, action: False)
    assert cache.get(cache_key(CONFIG, "a.md", hashlib.sha256(BODY_A.encode()).hexdigest())) is None
    assert artifact is not None and artifact.records == ()


def test_chunks_without_offsets_are_counted_once_per_source(tmp_path: Path) -> None:
    from recall.compiled_graph.anchors import build_source_anchors

    anchors = build_source_anchors("a.md", BODY_A)
    bare = (Chunk(id="a.md:0", source="a.md", text=BODY_A[:1_000], metadata={"file": "a.md"}),)
    artifact, report = _run(
        [SourceInput("a.md", BODY_A, bare)],
        Transport({"a.md": _ok([_alice(anchors[0].id), _alice(anchors[0].id)])}), tmp_path,
    )
    assert artifact is not None and len(artifact.records) == 2
    assert all(record.edges == () for record in artifact.records)
    assert report.edgeless_chunks == 1
