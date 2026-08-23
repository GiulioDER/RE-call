"""RE-call adapter for the Bench'd harness (github.com/benchdai/harness).

This file is the source of truth in the RE-call repository; at run time it is copied into a
harness clone as ``benchd_harness/adapters/recall_adapter.py`` and registered in
``benchd_harness/adapters/__init__.py``. It follows the same faithfulness rules as
``recall_interop.memory_benchmarks.RecallBackend`` (the mem0-harness seam), restated where they
apply because each one changes a number.

Bench'd calls, per benchmark item (``benchd_harness/runner.py``):

    reset() -> ingest(turns) -> recall(query) -> str

and the returned string is the ONLY thing their locked answerer sees. Their turn dicts carry
``role``, ``content`` and a ``metadata`` dict; the harness hands the same dicts to every adapter,
so using ``metadata.session_date_time`` (LoCoMo session dates) is fair game and is the only way
temporal questions are winnable. One field is deliberately refused: LongMemEval's
``metadata.has_answer`` is an oracle evidence flag, and reading it would be cheating, so this
adapter never touches it.

Configuration is via environment variables so a run's config is visible in its command line:

    RECALL_BENCHD_DSN          required, pgvector DSN
    RECALL_BENCHD_EMBEDDER     default "voyage:voyage-4" (resolve_embedder spelling; needs
                               VOYAGE_API_KEY); "fastembed" and "hashing" work for plumbing tests
    RECALL_BENCHD_RERANKER     default "none"; "voyage" is rerank-2.5 (reranker_from_name)
    RECALL_BENCHD_SPARSE       default "lexical"; "splade" replaces the lexical leg with learned
                               sparse (local model, so it falls under the VPS2 embedding rules),
                               "both" runs both legs, "none" is dense only
    RECALL_BENCHD_SPARSE_MODEL default "prithivida/Splade_PP_en_v1" (MIT; never naver/splade-v3)
    RECALL_BENCHD_TOP_K        default "10"; chunks retrieved per recall
    RECALL_BENCHD_SYNTH        default "none"; an OpenRouter model id (e.g.
                               "deepseek/deepseek-v4-pro-0813") turns on the synthesis step: the
                               retrieved chunks are distilled into a short evidence digest and
                               THAT becomes the recall string. Uses OPENROUTER_API_KEY.
    RECALL_BENCHD_SYNTH_MAX_TOKENS  default "120"
    RECALL_BENCHD_THRESHOLD    default "0.0"; the calibration threshold handed to the trust
                               layer. 0.0 means retrieval never abstains, which is the optimal
                               setting on a benchmark whose every question is answerable and
                               whose judge scores "insufficient information" as INCORRECT.
    RECALL_BENCHD_ABSTAIN      "suppress" (default) or "honour". Honoured abstention returns ""
                               and forfeits the question by construction; the knob exists so the
                               choice is explicit in the manifest either way.
    RECALL_BENCHD_INGEST_CACHE "1" (default) or "0"; skip re-ingest when the turn list is
                               byte-identical to what is already indexed (Bench'd re-ingests the
                               same conversation for every one of its questions; RE-call's
                               ingestion is deterministic, so the index state is identical)
    RECALL_BENCHD_TABLE        default "benchd_bench_chunks"; never recall_chunks, a benchmark
                               must not touch or be polluted by a real corpus
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter

TENANT = "benchd-bench"

SYNTH_PROMPT = """You are the retrieval layer of a memory system. Below are memory excerpts \
retrieved for a question. Distill them into the shortest complete evidence digest that answers \
the question.

Rules:
- Use ONLY facts present in the excerpts. Never add outside knowledge, never guess.
- Keep every detail the question asks about (names, dates, counts, places, order of events).
- When excerpts carry session dates, keep the dates that matter for the question.
- Output only the digest, no preamble. One to three short sentences.
- Write complete declarative sentences that name what they describe ("The user's name is \
Marcus Chen."), never a bare fragment ("Marcus Chen."): the reader of your digest sees it \
without the question and must still understand what each fact is about.
- If the excerpts contain nothing relevant, output the most nearly relevant facts they do \
contain, verbatim. Do not say the information is missing.

Question: {question}

Memory excerpts:
{memories}"""


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class RecallAdapter(BaseAdapter):
    """RE-call behind Bench'd's ingest/recall interface."""

    def __init__(self, adapter_config: Optional[Dict[str, Any]] = None) -> None:
        self._config = adapter_config or {}
        self._store: Any = None
        self._embedder: Any = None
        self._reranker: Any = None
        self._sparse_encoder: Any = None
        self._workspace: Optional[Path] = None
        self._seq = 0
        self._indexed_hash: Optional[str] = None
        self._pending_reset = False

        self._top_k = int(_env("RECALL_BENCHD_TOP_K", "10"))
        self._granularity = _env("RECALL_BENCHD_GRANULARITY", "session")
        self._abstain = _env("RECALL_BENCHD_ABSTAIN", "suppress")
        self._threshold = float(_env("RECALL_BENCHD_THRESHOLD", "0.0"))
        self._sparse = _env("RECALL_BENCHD_SPARSE", "lexical")
        self._sparse_model = _env("RECALL_BENCHD_SPARSE_MODEL", "prithivida/Splade_PP_en_v1")
        self._reranker_name = _env("RECALL_BENCHD_RERANKER", "none")
        self._synth_model = _env("RECALL_BENCHD_SYNTH", "none")
        self._synth_max_tokens = int(_env("RECALL_BENCHD_SYNTH_MAX_TOKENS", "120"))
        self._cache = _env("RECALL_BENCHD_INGEST_CACHE", "1") == "1"
        self._table = _env("RECALL_BENCHD_TABLE", "benchd_bench_chunks")

    @property
    def name(self) -> str:
        return "re-call"

    @property
    def version(self) -> Optional[str]:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("recall-rag")
        except PackageNotFoundError:
            return "source"

    def describe(self) -> Dict[str, Any]:
        """This run's configuration, for the results artifact."""
        return {
            "embedder": _env("RECALL_BENCHD_EMBEDDER", "voyage:voyage-4"),
            "reranker": self._reranker_name,
            "sparse": self._sparse,
            "sparse_model": self._sparse_model if self._sparse in ("splade", "both") else None,
            "top_k": self._top_k,
            "granularity": self._granularity,
            "synth": self._synth_model,
            "threshold": self._threshold,
            "abstain": self._abstain,
            "ingest_cache": self._cache,
        }

    # ------------------------------------------------------------------ lifecycle

    def setup(self) -> None:
        dsn = self._config.get("dsn") or os.environ.get("RECALL_BENCHD_DSN")
        if not dsn:
            raise RuntimeError("RecallAdapter requires RECALL_BENCHD_DSN")

        from recall.embeddings import resolve_embedder
        from recall.store import PgVectorStore

        self._embedder = resolve_embedder(_env("RECALL_BENCHD_EMBEDDER", "voyage:voyage-4"))
        if self._reranker_name != "none":
            from recall.rerank import reranker_from_name

            self._reranker = reranker_from_name(self._reranker_name)
        if self._sparse in ("splade", "both"):
            from recall.sparse import SpladeEncoder

            self._sparse_encoder = SpladeEncoder.from_pretrained(self._sparse_model)
        if self._synth_model != "none" and not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError("RECALL_BENCHD_SYNTH requires OPENROUTER_API_KEY")

        # Global generation migrations only apply through the default table, so a fresh
        # database must bootstrap `chunks` before any custom bench table (recall/schema.py).
        bootstrap = PgVectorStore(dsn, dim=self._embedder.dim, tenant=TENANT)
        bootstrap.__enter__()
        bootstrap.ensure_schema()
        bootstrap.close()

        self._store = PgVectorStore(
            dsn, dim=self._embedder.dim, tenant=TENANT, table=self._table
        )
        self._store.__enter__()
        self._store.ensure_schema()
        self._wipe()  # a fresh run must not inherit a previous run's rows
        self._workspace = Path(tempfile.mkdtemp(prefix="benchd-recall-"))

    def teardown(self) -> None:
        if self._store is not None:
            try:
                self._store.close()
            finally:
                self._store = None
        if self._workspace is not None:
            shutil.rmtree(self._workspace, ignore_errors=True)
            self._workspace = None

    def reset(self) -> None:
        """Bench'd resets before every item. With the cache on, the wipe is deferred to
        ingest(), which skips it when the incoming turn list is identical to what is indexed."""
        if self._cache:
            self._pending_reset = True
        else:
            self._wipe()
            self._indexed_hash = None

    def _wipe(self) -> None:
        sources = sorted({c.source for c in self._store.iter_chunks()})
        if sources:
            self._store.delete_sources(sources)
        if self._workspace is not None:
            for f in self._workspace.glob("*.md"):
                f.unlink()
        self._seq = 0

    # ------------------------------------------------------------------ ingest

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        blob = json.dumps(
            [
                {
                    "role": t.get("role"),
                    "content": t.get("content"),
                    # has_answer is deliberately excluded from the hash AND from the documents:
                    # it is LongMemEval's oracle flag and no memory system may see it.
                    "session_index": (t.get("metadata") or {}).get("session_index"),
                    "session_date_time": (t.get("metadata") or {}).get("session_date_time"),
                    "speaker": (t.get("metadata") or {}).get("speaker"),
                }
                for t in turns
            ],
            sort_keys=True,
        )
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()

        if self._cache and self._pending_reset and digest == self._indexed_hash:
            self._pending_reset = False
            return
        if self._pending_reset:
            self._wipe()
            self._pending_reset = False
        # Cleared before indexing, set after: if index_path raises mid-item, the runner records
        # the item as an adapter error and moves on, and the next item on the same conversation
        # must see a cache miss, not a hash that claims the partial index is complete.
        self._indexed_hash = None

        docs = (
            self._docs_by_session(turns)
            if self._granularity == "session"
            else self._docs_by_turn(turns)
        )

        assert self._workspace is not None
        paths: List[Path] = []
        for body in docs:
            if not body.strip():
                continue
            self._seq += 1
            path = self._workspace / f"m{self._seq:06d}.md"
            path.write_text(body, encoding="utf-8")
            paths.append(path)

        if paths:
            from recall.index import Indexer

            # files= rather than a workspace re-glob: one pass, and _prune_vanished deletes
            # nothing that is still on disk.
            Indexer(self._store, self._embedder).index_path(self._workspace, files=paths)
            if self._sparse_encoder is not None:
                from recall.sparse import backfill_learned_sparse

                backfill_learned_sparse(self._store, self._sparse_encoder)
        self._indexed_hash = digest

    @staticmethod
    def _line(t: Dict[str, Any]) -> str:
        meta = t.get("metadata") or {}
        who = meta.get("speaker") or t.get("role", "user")
        return f"{who}: {t.get('content', '')}"

    def _docs_by_session(self, turns: List[Dict[str, Any]]) -> List[str]:
        """One document per session, headed by the session date when the dataset provides one."""
        docs: List[str] = []
        current_key: Any = object()
        lines: List[str] = []
        header = ""
        for t in turns:
            meta = t.get("metadata") or {}
            key = meta.get("session_index")
            if key != current_key:
                if lines:
                    docs.append(header + "\n".join(lines))
                current_key = key
                lines = []
                date = meta.get("session_date_time")
                header = f"Session dated {date}:\n" if date else ""
            lines.append(self._line(t))
        if lines:
            docs.append(header + "\n".join(lines))
        return docs

    def _docs_by_turn(self, turns: List[Dict[str, Any]]) -> List[str]:
        """One document per turn, each carrying its own session date, so a retrieved chunk can
        never be separated from the date temporal questions need."""
        docs = []
        for t in turns:
            meta = t.get("metadata") or {}
            date = meta.get("session_date_time")
            prefix = f"[{date}] " if date else ""
            docs.append(prefix + self._line(t))
        return docs

    # ------------------------------------------------------------------ recall

    def recall(self, query: str) -> str:
        raw = self._retrieve(query)
        if not raw:
            return ""
        if self._synth_model == "none":
            return raw
        return self._synthesize(query, raw)

    def _retrieve(self, query: str) -> str:
        """Retrieve top_k chunk texts, joined. Empty string only on honoured abstention."""
        if self._sparse in ("splade", "both", "none"):
            # The learned-sparse and dense-only legs live on HybridRetriever, not on the trust
            # entry point, so those arms search the retriever directly. Abstention does not
            # apply on this path; the retriever always answers.
            from recall.retriever import DEFAULT_CANDIDATE_K, HybridRetriever

            retriever = HybridRetriever(
                self._store,
                self._embedder,
                reranker=self._reranker,
                candidate_k=max(self._top_k, DEFAULT_CANDIDATE_K),
                use_dense=True,
                use_sparse=self._sparse != "none",
                sparse_backend="lexical" if self._sparse == "none" else self._sparse,
                sparse_encoder=self._sparse_encoder,
                retrieval_profile=f"benchd_{self._sparse}",
                index_generation="benchd",
            )
            result = retriever.search(query, k=self._top_k)
            hits = result.hits
        else:
            from recall.calibration import Calibration
            from recall.retriever import DEFAULT_CANDIDATE_K
            from recall.eval._research_trust import research_search

            result = research_search(
                self._store,
                self._embedder,
                query,
                k=self._top_k,
                candidate_k=max(self._top_k, DEFAULT_CANDIDATE_K),
                reranker=self._reranker,
                calibration=Calibration(
                    embedder=getattr(self._embedder, "name", "benchmark"),
                    threshold=self._threshold,
                ),
            )
            if result.abstained and self._abstain == "honour":
                # The answerer will say "Insufficient information in memory." and the locked
                # judge scores that INCORRECT. The suppress default exists because every
                # Bench'd question is answerable, so abstention here is always a forfeit.
                return ""
            hits = result.hits
        return "\n\n".join(hit.chunk.text for hit in hits)

    def _synthesize(self, query: str, memories: str) -> str:
        """Distill retrieved memories into a short evidence digest with the reasoning model.

        On any provider failure the raw memories are returned instead: a degraded answer beats
        a forfeited question, and the failure is visible in the trace as an oversized recall."""
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
            )
            response = client.chat.completions.create(
                model=self._synth_model,
                messages=[
                    {
                        "role": "user",
                        "content": SYNTH_PROMPT.format(question=query, memories=memories),
                    }
                ],
                temperature=0.0,
                max_tokens=self._synth_max_tokens,
            )
            text = (response.choices[0].message.content or "").strip()
            return text or memories
        except Exception:
            return memories
