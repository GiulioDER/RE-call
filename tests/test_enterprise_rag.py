from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from benchmarks import enterprise_rag
from benchmarks.enterprise_rag import (
    EnterpriseDoc,
    EnterpriseQuestion,
    QueryCachedEmbedder,
    apply_top_config,
    build_parser,
    doc_chunks,
    expand_retrieval_hits,
    generated_answer,
    index_documents,
    load_documents,
    load_questions,
    reasoning_promotion_gate,
    retrieval_capture_summary,
    reasoning_summary,
    read_answer_rows,
    write_answers_stream,
    write_answers,
)
from recall.cache import EmbeddingCache
from recall.types import ScoredChunk
from recall.types import Chunk


def test_loads_enterprise_release_shapes_from_zip(tmp_path: Path) -> None:
    archive = tmp_path / "docs.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "a.jsonl",
            json.dumps(
                {
                    "doc_id": "dsid_1",
                    "source_type": "slack",
                    "title": "Launch thread",
                    "content": "The launch owner is Mira.",
                }
            )
            + "\n",
        )
        zf.writestr(
            "b.json",
            json.dumps(
                {
                    "doc_id": "dsid_2",
                    "source_type": "confluence",
                    "title": "Runbook",
                    "content": "Rollback uses the silver lane.",
                }
            ),
        )

    docs = list(load_documents([archive]))

    assert [doc.doc_id for doc in docs] == ["dsid_1", "dsid_2"]
    assert docs[0].source_type == "slack"


def test_loads_official_text_zip_shape(tmp_path: Path) -> None:
    archive = tmp_path / "all_documents.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "confluence/applied-ml/dsid_abc123__rollout-playbook.txt",
            "The rollout owner is Mira.",
        )
        zf.writestr(
            "questions.jsonl",
            json.dumps({"question_id": "qst_1", "question": "Ignored?"}) + "\n",
        )

    docs = list(load_documents([archive]))

    assert len(docs) == 1
    assert docs[0] == EnterpriseDoc(
        doc_id="dsid_abc123",
        source_type="confluence",
        title="rollout playbook",
        content="The rollout owner is Mira.",
    )


def test_text_zip_ingestion_strips_nul_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "all_documents.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "github/dsid_nul123__binary-paste.txt",
            "The answer survives.\x00 This byte cannot go into Postgres.",
        )

    docs = list(load_documents([archive]))

    assert "\x00" not in docs[0].content
    assert "The answer survives." in docs[0].content


def test_chunks_preserve_document_id_for_leaderboard_citations() -> None:
    chunks = doc_chunks(
        EnterpriseDoc(
            doc_id="dsid_keep",
            source_type="gmail",
            title="Customer thread",
            content="The answer is in this document.",
        )
    )

    assert chunks
    assert chunks[0].id.startswith("dsid_keep#")
    assert chunks[0].source == "dsid_keep"
    assert chunks[0].metadata["doc_id"] == "dsid_keep"


def test_doc_chunks_accept_benchmark_scale_chunking() -> None:
    doc = EnterpriseDoc(
        doc_id="dsid_long",
        source_type="confluence",
        title="Long policy",
        content=("alpha " * 5000).strip(),
    )

    tiny = doc_chunks(doc, chunk_chars=800, chunk_overlap=80)
    large = doc_chunks(doc, chunk_chars=12_000, chunk_overlap=200)

    assert len(large) < len(tiny)
    assert all(len(chunk.text) <= 12_200 for chunk in large)


def test_index_documents_can_skip_existing_sources() -> None:
    class _FakeStore:
        def __init__(self) -> None:
            self.written: list[Chunk] = []

        def iter_chunks(self) -> list[Chunk]:
            return [
                Chunk(
                    id="dsid_existing#0000",
                    source="dsid_existing",
                    text="old",
                    metadata={},
                )
            ]

        def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
            self.written.extend(chunks)
            return len(chunks)

        def analyze_if_stale(self, modified: int) -> bool:
            return True

    class _FakeEmbedder:
        dim = 2

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    store = _FakeStore()

    stats = index_documents(
        store,  # type: ignore[arg-type]
        _FakeEmbedder(),  # type: ignore[arg-type]
        [
            EnterpriseDoc("dsid_existing", "github", "Old", "Already indexed."),
            EnterpriseDoc("dsid_new", "github", "New", "Needs indexing."),
        ],
        skip_indexed_sources=True,
    )

    assert stats["skipped_documents"] == 1
    assert stats["documents"] == 1
    assert [chunk.source for chunk in store.written] == ["dsid_new"]


def test_loads_questions_and_writes_answer_jsonl(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json.dumps({"question_id": "qst_1", "question": "Who owns launch?"}) + "\n",
        encoding="utf-8",
    )
    questions = load_questions(questions_path)
    out = tmp_path / "answers.jsonl"

    count = write_answers(
        out,
        [{"question_id": questions[0].question_id, "answer": "Mira", "document_ids": ["dsid_1"]}],
        overwrite=False,
    )

    assert count == 1
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "question_id": "qst_1",
        "answer": "Mira",
        "document_ids": ["dsid_1"],
    }


def test_top_config_enables_lexical_splade_voyage_rerank_and_openrouter() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--questions",
            "questions.jsonl",
            "--documents",
            "docs.zip",
            "--out",
            "answers.jsonl",
            "--top-config",
        ]
    )

    apply_top_config(args)

    assert args.embedder == "voyage:voyage-4-large"
    assert args.sparse_backend == "both"
    assert args.backfill_splade is True
    assert args.reranker == "voyage:rerank-2.5"
    assert args.answer_mode == "openrouter"
    assert args.model == "openai/gpt-4o"
    assert args.k == 8
    assert args.candidate_k == 200
    assert args.batch_chunks == 32
    assert args.max_context_chars == 12_000
    assert args.chunk_chars == 12_000
    assert args.chunk_overlap == 200
    assert args.rerank_document_chars == 4_000


def test_generated_answer_prompt_includes_question_type_and_strict_abstention(
    monkeypatch: Any,
) -> None:
    calls = []

    class _FakeLLM:
        def __init__(self, model: str, api_key: str) -> None:
            self.model = model
            self.api_key = api_key

        def complete(self, system: str, user: str) -> str:
            calls.append({"system": system, "user": user, "model": self.model})
            return "10 MiB per file and 50 MiB per request."

    monkeypatch.setattr("benchmarks.llm.OpenRouterLLM", _FakeLLM)
    hits = doc_chunks(
        EnterpriseDoc(
            doc_id="dsid_1",
            source_type="github",
            title="multipart limits",
            content="max_file_size is 10 MiB and max_total_request_size is 50 MiB.",
        )
    )
    scored = [ScoredChunk(chunk=hits[0], score=1.0, indexed_at=datetime(2026, 1, 1, tzinfo=UTC))]

    answer = generated_answer(
        "What are the upload limits?",
        scored,
        model="openai/gpt-4o",
        api_key="test",
        max_chars=1000,
        question_type="basic",
    )

    assert answer == "10 MiB per file and 50 MiB per request."
    assert calls[0]["model"] == "openai/gpt-4o"
    assert "Question type: basic" in calls[0]["user"]
    assert "Source type: github" in calls[0]["user"]
    assert "available documents do not contain the answer" in calls[0]["system"]
    assert "exact facts, quantities, dates, names" in calls[0]["system"]


def test_streaming_answers_append_for_resume_without_diagnostics(tmp_path: Path) -> None:
    out = tmp_path / "answers.jsonl"
    out.write_text(
        json.dumps({"question_id": "qst_1", "answer": "done", "document_ids": ["dsid_1"]})
        + "\n",
        encoding="utf-8",
    )

    count, rows = write_answers_stream(
        out,
        [
            {
                "question_id": "qst_2",
                "answer": "next",
                "document_ids": ["dsid_2"],
                "_diagnostics": {"gap_warning": False},
            }
        ],
        overwrite=False,
        resume=True,
    )

    assert count == 1
    assert rows[0]["question_id"] == "qst_2"
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert written == [
        {"question_id": "qst_1", "answer": "done", "document_ids": ["dsid_1"]},
        {"question_id": "qst_2", "answer": "next", "document_ids": ["dsid_2"]},
    ]
    assert read_answer_rows(out) == written


def test_parser_exposes_resume_and_sparse_device() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--questions",
            "questions.jsonl",
            "--documents",
            "docs.zip",
            "--out",
            "answers.jsonl",
            "--resume",
            "--sparse-device",
            "cuda",
        ]
    )

    assert args.resume is True
    assert args.sparse_device == "cuda"


def test_parser_exposes_reasoning_arm_and_cache() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--questions",
            "questions.jsonl",
            "--documents",
            "docs.zip",
            "--out",
            "answers.jsonl",
            "--reasoning-arm",
            "closed_loop",
            "--reasoning-cache",
            "expansions.json",
            "--embedding-cache",
            "vectors.sqlite",
            "--retrieval-captures",
            "3",
        ]
    )

    assert args.reasoning_arm == "closed_loop"
    assert args.reasoning_cache == Path("expansions.json")
    assert args.embedding_cache == Path("vectors.sqlite")
    assert args.retrieval_captures == 3


def test_reasoning_summary_records_expansion_and_fallback_rates() -> None:
    summary = reasoning_summary(
        [
            {"_diagnostics": {"reasoning": {"expanded": True, "passes": 2, "queries": ["q"]}}},
            {
                "_diagnostics": {
                    "reasoning": {
                        "expanded": False,
                        "passes": 1,
                        "queries": [],
                        "fallback_reason": "provider_failure",
                    }
                }
            },
        ]
    )

    assert summary == {
        "rows": 2,
        "expanded_rows": 1,
        "expanded_rate": 0.5,
        "fallback_rows": 1,
        "fallback_rate": 0.5,
        "passes_total": 3,
        "queries_total": 1,
        "capture_stability_rate": None,
        "model": None,
    }


def test_closed_loop_skips_cheap_provider_after_depth_resolves_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = EnterpriseQuestion(
        question_id="qst_1",
        question="Who owns the project?",
        raw={"expected_doc_ids": ["dsid_2"]},
    )
    initial = ScoredChunk(
        chunk=doc_chunks(
            EnterpriseDoc("dsid_1", "github", "project", "The project is incomplete.")
        )[0],
        score=0.5,
        indexed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    depth = ScoredChunk(
        chunk=doc_chunks(
            EnterpriseDoc("dsid_2", "github", "owner", "Ada owns the project.")
        )[0],
        score=0.9,
        indexed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    class _Store:
        pass

    class _Embedder:
        pass

    calls: list[str] = []

    def fake_retrieve(*_args: Any, **kwargs: Any) -> tuple[list[str], list[ScoredChunk], bool]:
        del kwargs
        calls.append(_args[2])
        return ["dsid_2"], [depth], False

    monkeypatch.setattr(enterprise_rag, "retrieve_docs", fake_retrieve)
    ids, _, diagnostics = expand_retrieval_hits(
        question,
        _Store(),
        _Embedder(),
        initial_hits=[initial],
        initial_gap_warning=True,
        k=8,
        candidate_k=80,
        sparse_backend="lexical",
        sparse_encoder=None,
        reranker=None,
        gap_threshold=0.5,
        arm="closed_loop",
        provider=None,
        expansion_cache=None,
    )

    assert ids == ["dsid_1", "dsid_2"]
    assert calls == ["Who owns the project?"]
    assert diagnostics["provider_skipped_reason"] == "depth_resolved"


def test_query_cached_embedder_reuses_query_vectors(tmp_path: Path) -> None:
    class FakeEmbedder:
        dim = 2
        name = "fake"

        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return [[float(len(text)), 1.0] for text in texts]

    inner = FakeEmbedder()
    with EmbeddingCache(tmp_path / "vectors.sqlite") as cache:
        embedder = QueryCachedEmbedder(inner, cache)
        assert embedder.embed_query("same") == embedder.embed_query("same")
    assert inner.calls == 1


def test_reasoning_promotion_gate_stays_pending_without_independent_metrics() -> None:
    gate = reasoning_promotion_gate(None)
    assert gate["status"] == "pending"
    assert gate["expensive_model_allowed"] is False


def test_reasoning_promotion_gate_requires_all_safety_signals() -> None:
    metrics = {
        "baseline_correctness": 0.50,
        "candidate_correctness": 0.54,
        "useful_expansion_precision": True,
        "stable_repeated_captures": True,
        "validation_failure_rate": 0.01,
        "no_material_false_abstention": True,
        "info_not_found_correctness": 0.95,
    }
    assert reasoning_promotion_gate(metrics)["expensive_model_allowed"] is True

    metrics["candidate_correctness"] = 0.52
    gate = reasoning_promotion_gate(metrics)
    assert gate["status"] == "blocked"
    assert "correctness_delta_at_least_3_points" in gate["failed"]


def test_retrieval_capture_summary_reports_variance() -> None:
    summary = retrieval_capture_summary(
        [
            {"document_ids": ["a", "b"]},
            {"document_ids": ["a", "b"]},
            {"document_ids": ["a", "c"]},
        ]
    )
    assert summary["count"] == 3
    assert summary["stable"] is False
    assert summary["mean_document_jaccard"] == (1.0 + 1 / 3 + 1 / 3) / 3
