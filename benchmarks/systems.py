"""`MemorySystem` protocol + the RE-call adapter for the head-to-head benchmark.

The benchmark runs two memory systems through an IDENTICAL LLM generator (see
`benchmarks.pipeline.run_question`) on LOCOMO — only the retrieved memory differs between systems.
`MemorySystem` is the seam that makes that swap possible: `run_question` takes a bare
``retrieve(question) -> str`` callable, and any `MemorySystem.retrieve` satisfies it directly.
"""
from __future__ import annotations

import re
import tempfile
import warnings
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from recall.embeddings import embedding_profile_id

#: Retrieval budget shared by every arm unless ``--k`` overrides it. 5 is the value the harness was
#: originally written with and the number every early result was measured at; it is kept as the
#: default so a rerun without flags reproduces those runs, and it is recorded in the results
#: artifact so a reader never has to guess which budget produced a published number.
DEFAULT_K = 5

#: Benchmark-only table. Never the default `recall_chunks`: a benchmark run must not be able to
#: touch, or be polluted by, a real corpus sharing the same database.
BENCH_TABLE = "bench_locomo_chunks"

#: Prefix of the per-conversation tenant (RE-call) / user id (Mem0). Shared by both adapters so the
#: two arms are isolated the same way and a stray id is recognisable in either store.
TENANT_PREFIX = "bench-"


def resolve_embedder(name: str) -> Any:
    """Benchmark embedder resolver: a superset of `recall.eval.locomo._make_embedder`.

    Adds a ``fastembed:<model>`` route so the RE-call arm can be run on a STRONGER local embedder
    than the default `bge-small` without an API key — the whole point of the retrieval-quality
    ablation (does a better embedder fix single-hop recall?). ``fastembed:BAAI/bge-large-en-v1.5``
    and ``fastembed:mixedbread-ai/mxbai-embed-large-v1`` are 1024-dim CPU models available in the
    installed fastembed. Every other name (``fastembed`` default, ``st:``, ``voyage``, ``hashing``)
    defers unchanged to the shared factory, so existing runs are byte-for-byte unaffected.
    """
    fe_prefix = "fastembed:"
    if name.startswith(fe_prefix):
        from recall.embeddings import FastEmbedEmbedder

        return FastEmbedEmbedder(model_name=name[len(fe_prefix):])
    # `voyage:<model>` pins a CURRENT Voyage model; the bare `voyage` route in the shared factory
    # still resolves to VoyageEmbedder's default, which is the LEGACY voyage-3. Current generation
    # is voyage-4 (voyage-4-large is "best general-purpose"); see benchmarks/VOYAGE_REFERENCE.md.
    voyage_prefix = "voyage:"
    if name.startswith(voyage_prefix):
        from recall.embeddings import VoyageEmbedder

        return VoyageEmbedder(model=name[len(voyage_prefix):])
    # `router:<model>` = an OpenAI-compatible cloud embedder served by OpenRouter, e.g.
    # `router:openai/text-embedding-3-small`. Lets BOTH arms run on Mem0's documented default
    # embedder without an OpenAI account — the key is the OpenRouter one the generator/judge use.
    router_prefix = "router:"
    if name.startswith(router_prefix):
        from recall.embeddings import OpenAICompatEmbedder

        return OpenAICompatEmbedder(model=name[len(router_prefix):])
    from recall.eval.locomo import _make_embedder

    return _make_embedder(name)


def resolve_reranker(name: str) -> Any:
    """The RE-call arm's optional second-stage reranker, or None.

    ``none`` (default) -> no reranker. ``local`` -> the cross-encoder this library actually ships
    (`ms-marco-MiniLM-L-6-v2`, pinned revision), ``local:<model>`` for a different one.
    ``voyage:<model>`` -> Voyage cross-encoder, e.g. ``voyage:rerank-2.5``. RE-call's retriever
    reranks the WHOLE fused candidate pool before truncating to k, so a reranker can rescue an
    answer that ranked below the cutoff — the cat1 failure the benchmark exposed. See
    benchmarks/VOYAGE_REFERENCE.md.

    ``local`` exists because until it did, the only reachable reranker was a cloud API: a harness
    for a library whose stated property is that data never leaves your infrastructure could not
    measure its own shipped reranker without sending every query and every candidate to a third
    party. FINDINGS §11 also measured `ms-marco-MiniLM-L-6-v2` as the right model for this task —
    `bge-reranker-base`, 12x the parameters, is statistically indistinguishable at 6.3x the cost.
    """
    if not name or name == "none":
        return None
    voyage_prefix = "voyage:"
    if name.startswith(voyage_prefix):
        from benchmarks.voyage_rerank import VoyageReranker

        return VoyageReranker(model=name[len(voyage_prefix):])
    local_prefix = "local:"
    if name == "local" or name.startswith(local_prefix):
        from recall.rerank import CrossEncoderReranker

        if name == "local":
            return CrossEncoderReranker()
        # A custom model drops the default revision pin inside CrossEncoderReranker: the pin
        # belongs to the default weights, and reusing it for different ones would name the wrong
        # artifact in every trace.
        return CrossEncoderReranker(model=name[len(local_prefix):])
    raise ValueError(
        f"unknown reranker {name!r} (use 'none', 'local', 'local:<model>' or 'voyage:<model>')"
    )


def sample_id_of(conversation: dict[str, Any]) -> str:
    """The LOCOMO item's `sample_id`, or a loud failure. THE single identity rule for the harness.

    Both adapters scope their memory to ``bench-{sample_id}``, and the run script groups questions
    by the same value. Three separate fallbacks used to exist for a missing `sample_id` — a
    positional `conv{i}` in the run script and a bare `str(None)` in each adapter — so a dataset
    row without one produced the tenant ``bench-None`` in BOTH adapters: every such conversation
    would share one memory, answer each other's questions, and inflate accuracy with no error and
    no visible symptom in the artifact. There is no correct fallback here, so there is none: a
    LOCOMO item with no usable `sample_id` stops the run.
    """
    raw = conversation.get("sample_id")
    text = "" if raw is None else str(raw).strip()
    if not text:
        raise ValueError(
            "LOCOMO item has no usable 'sample_id' — it is the per-conversation memory scope "
            "(tenant/user id) for both arms, and a shared fallback would silently merge "
            "conversations into one memory"
        )
    return text


def tenant_for(conversation: dict[str, Any]) -> str:
    """The memory scope for one LOCOMO item, identical on both arms."""
    return f"{TENANT_PREFIX}{sample_id_of(conversation)}"


@runtime_checkable
class MemorySystem(Protocol):
    """One competitor in the head-to-head benchmark.

    ``ingest`` indexes a single LOCOMO conversation; ``retrieve`` answers one question against
    whatever was last ingested. Both are per-conversation — LOCOMO's conversations are unrelated
    worlds, and the benchmark harness ingests exactly once per conversation before scoring its
    questions.

    Deliberately three members and no more. Both concrete adapters also expose a ``describe()``
    that reports their configuration into the results artifact, but that is NOT part of the
    protocol: the protocol is the seam a test double has to satisfy, and widening it would make
    every stub carry reporting machinery that has nothing to do with the behaviour under test.
    The run script reads `describe` duck-typed, and a system without one simply reports nothing.
    """

    name: str

    def ingest(self, conversation: dict[str, Any]) -> None: ...

    def retrieve(self, question: str) -> str: ...


class RecallSystem:
    """RE-call adapter: index each dialogue turn, retrieve via the trust layer.

    Reuses the exact LOCOMO indexing/search machinery `recall.eval.locomo` already validates —
    `index_conversation` for ingest, `trusted_search` for retrieve — rather than reimplementing
    either. Each conversation gets its own tenant (``bench-{sample_id}``) in a benchmark-only table
    (``bench_locomo_chunks``), mirroring the isolation `recall.eval.locomo.run` already relies on
    to keep one conversation's turns from leaking into another's answers.

    Returns an EMPTY STRING from `retrieve` when the trust layer abstains. That
    abstention-propagates-as-empty-context behaviour is the single thing this whole benchmark
    exists to measure: the downstream generator sees no memories and must emit its own refusal
    token, exactly as it would for a real caller with no LLM in RE-call's path.
    """

    name = "recall"

    def __init__(
        self,
        dsn: str,
        embedder_name: str = "fastembed",
        k: int = DEFAULT_K,
        reranker_name: str = "none",
        table: str = BENCH_TABLE,
    ) -> None:
        self._dsn = dsn
        self._k = k
        self._embedder_name = embedder_name
        self._embedder = resolve_embedder(embedder_name)
        self._reranker_name = reranker_name
        self._reranker = resolve_reranker(reranker_name)
        # Overridable so a side experiment (e.g. the latency benchmark) can use its own table at a
        # different embedder width without colliding with the accuracy runs' `bench_locomo_chunks`.
        self._table = table
        self._tenant: str | None = None

    @property
    def embedder(self) -> Any:
        """The live `Embedder`. Exposed so a caller can build a store at the matching `dim`."""
        return self._embedder

    def describe(self) -> dict[str, Any]:
        """This arm's configuration, for the results artifact. Carries no secret (the DSN, which
        may embed a password, is deliberately not reported)."""
        return {
            "system": self.name,
            "k": self._k,
            "embedder": {
                "name": self._embedder_name,
                "model": embedding_profile_id(self._embedder),
            },
            "reranker": self._reranker_name,
            "table": self._table,
            "tenant": self._tenant,
        }

    def ingest(self, conversation: dict[str, Any]) -> None:
        from recall.eval.locomo import index_conversation
        from recall.store import PgVectorStore

        # `conversation` here is one LOCOMO item as loaded from locomo10.json: it carries
        # `sample_id` and `qa` alongside the nested `conversation` object (`session_N` turns,
        # `speaker_a`/`speaker_b`) that `index_conversation` actually indexes. Passing the outer
        # item straight into `index_conversation` would find zero `session_` keys and silently
        # index nothing.
        self._tenant = tenant_for(conversation)
        inner = conversation["conversation"]
        with PgVectorStore(
            self._dsn, dim=self._embedder.dim, tenant=self._tenant, table=self._table
        ) as store:
            store.ensure_schema()
            self._clear(store)
            index_conversation(store, self._embedder, inner)

    @staticmethod
    def _clear(store: Any) -> int:
        """Delete every row this tenant already holds. Returns the number of sources removed.

        Without this, a SECOND run against the same database silently doubles the corpus instead
        of replacing it. `index_conversation` writes each turn to a fresh `mkdtemp` directory, and
        `Indexer` derives both the chunk id (``md5(f"{abs_path}:{i}")``) and the row's `source`
        from that absolute path — so run two sees paths run one never used. The id differs, so
        ``ON CONFLICT (tenant_id, id)`` never fires; the `source` differs, so `replace_sources`
        matches nothing to delete. Every turn is inserted again, alongside its twin.

        The damage is not cosmetic: top-k then fills with duplicate copies of the same turns, so
        RE-call's effective context SHRINKS with each rerun and the published numbers become a
        function of how many times the harness happened to be run. Clearing at ingest makes a run
        idempotent — the second run measures the same corpus as the first.
        """
        sources = sorted({chunk.source for chunk in store.iter_chunks()})
        if sources:
            store.delete_sources(sources)
        return len(sources)

    def retrieve(self, question: str) -> str:
        from recall.store import PgVectorStore
        from recall.trust import trusted_search

        if self._tenant is None:
            raise RuntimeError("RecallSystem.retrieve() called before ingest()")
        with PgVectorStore(
            self._dsn, dim=self._embedder.dim, tenant=self._tenant, table=self._table
        ) as store:
            result = trusted_search(
                store, self._embedder, question, k=self._k, reranker=self._reranker
            )
            if result.abstained:
                return ""
            return "\n".join(hit.chunk.text for hit in result.hits)

    def ablation_preflight(
        self,
        questions: list[str],
        *,
        sample: int,
        metric_class: str,
        allow_inert: bool,
    ) -> list[dict[str, Any]]:
        """Refuse the run if a configured mechanism changes nothing. Retrieval only — no LLM spend.

        Opens its own store exactly as `retrieve` does, because the store is per-call here and the
        preflight must query the index this arm will actually use.
        """
        from recall.eval.arm_check import DEFAULT_SAMPLE, ablation_verdicts, enforce
        from recall.retriever import DEFAULT_CANDIDATE_K
        from recall.store import PgVectorStore

        if self._tenant is None:
            raise RuntimeError("RecallSystem.ablation_preflight() called before ingest()")
        sampled = questions[: sample or DEFAULT_SAMPLE]
        with PgVectorStore(
            self._dsn, dim=self._embedder.dim, tenant=self._tenant, table=self._table
        ) as store:
            # DEFAULT_CANDIDATE_K, not self._k or a separately-tunable value: `retrieve` above calls
            # `trusted_search` without passing `candidate_k`, so that is the pool this arm actually
            # retrieves at. Measuring a different pool size here would produce a verdict about a
            # configuration the run never uses.
            verdicts = ablation_verdicts(
                store,
                self._embedder,
                sampled,
                k=self._k,
                candidate_k=DEFAULT_CANDIDATE_K,
                reranker=self._reranker,
                use_sparse=True,
            )
        enforce(verdicts, metric_class=metric_class, allow_inert=allow_inert)
        return [v.as_dict() for v in verdicts]


def mem0ai_version() -> str | None:
    """The installed `mem0ai` version, or None when the `bench` extra is absent.

    Reported into the results artifact: Mem0's API and its extraction prompt both move between
    releases, so a number published without the version it was produced against is not
    reproducible. Absence is tolerated because every module here imports `mem0` lazily and the
    offline tests run without the extra installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("mem0ai")
    except PackageNotFoundError:
        return None


#: Embedding width per embedder choice, needed because the vector store is configured EXPLICITLY
#: below and Mem0's Qdrant config defaults to 1536 — which silently mismatches the 384-wide
#: bge-small the controlled arm uses.
_EMBEDDER_DIMS = {"huggingface": 384, "openai": 1536}

#: Output dimension per HuggingFace embedding model, so Mem0's Qdrant collection is created at the
#: right width. Wrong dims silently break similarity. Extend when a new model is used on this arm.
_HF_MODEL_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}

#: Keys whose values are secrets. The results artifact is meant to be published, and `describe()`
#: reports the Mem0 config into it verbatim apart from these.
_SECRET_KEYS = frozenset({"api_key"})


def _redact(value: Any) -> Any:
    """Deep-copy `value`, replacing every secret-named leaf with a placeholder."""
    if isinstance(value, dict):
        return {
            k: ("***redacted***" if k in _SECRET_KEYS and v is not None else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _slug(text: str) -> str:
    """`text` reduced to the characters a Qdrant collection name and a path segment both accept."""
    return re.sub(r"[^0-9A-Za-z_-]+", "_", text).strip("_") or "run"


def mem0_config(
    openrouter_key: str,
    model: str,
    embedder: str = "huggingface",
    openai_key: str | None = None,
    *,
    hf_model: str = "BAAI/bge-small-en-v1.5",
    openai_model: str = "text-embedding-3-small",
    openai_base_url: str | None = None,
    run_id: str = "adhoc",
    storage_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a Mem0 config: LLM via OpenRouter (OpenAI-compatible), embedder local by default.

    Pure function, no `mem0ai` import — testable without the `bench` extra installed. Both knobs
    exist to make the comparison fair rather than flattering:

    - The LLM is routed through OpenRouter at the SAME model the benchmark's own generator/judge
      use (`{"provider": "openai", ...}` because OpenRouter speaks the OpenAI wire format), not
      whatever Mem0 defaults to. A model mismatch there would confound "which memory system is
      better" with "which LLM is better".
    - `embedder="huggingface"` (the default) is the free local `bge-small` model — the same
      substrate class RE-call's `fastembed` embedder uses — so neither system gets an embedding
      quality advantage from a paid API the other doesn't have.
    - `embedder="openai"` is the deliberate exception: Mem0's own documented default embedder is
      OpenAI `text-embedding-3-small`. Keeping it selectable (off by default) supports an ablation
      arm that measures Mem0 "as shipped" rather than only the fairness-controlled arm.

    The **vector store is configured explicitly**, and that is a correctness fix rather than
    tidiness. Left unset, Mem0 falls back to its own on-disk default — one Qdrant database at a
    fixed path, under one fixed collection name — so a second benchmark run reopens the FIRST
    run's collection and adds to it. That is the same silent-accumulation defect
    `RecallSystem._clear` fixes on the other arm, and it has to be fixed on both or the comparison
    is between one system that was reset and one that was not. Path and collection are stamped
    with `run_id`, so every run starts on storage no previous run has ever written to.

    `embedding_model_dims` is set from the chosen embedder for the same reason: Mem0's Qdrant
    config defaults it to 1536, which is right for `text-embedding-3-small` and wrong for the
    384-wide bge-small the controlled arm uses. `history_db_path` is stamped too — it defaults to
    a single `~/.mem0/history.db` shared by every run on the machine.
    """
    llm = {
        "provider": "openai",
        "config": {
            "model": model,
            "api_key": openrouter_key,
            "openai_base_url": "https://openrouter.ai/api/v1",
        },
    }
    if embedder == "openai":
        emb_config: dict[str, Any] = {"model": openai_model, "api_key": openai_key}
        # When set, route the embedder through an OpenAI-compatible proxy (OpenRouter) rather than
        # api.openai.com — the SAME model, a key that is not OpenAI's. `embedding_dims` is left
        # unset so Mem0 does not send the `dimensions` param (some proxies reject it); the native
        # 1536-wide vector matches the Qdrant collection width configured below.
        if openai_base_url:
            emb_config["openai_base_url"] = openai_base_url
        emb = {"provider": "openai", "config": emb_config}
    else:
        emb = {"provider": "huggingface", "config": {"model": hf_model}}
    root = Path(storage_dir) if storage_dir is not None else Path(tempfile.gettempdir())
    run = _slug(run_id)
    workspace = root / "recall-bench-mem0" / run
    vector_store = {
        "provider": "qdrant",
        "config": {
            "collection_name": f"bench_{run}",
            "path": str(workspace / "qdrant"),
            "embedding_model_dims": (
                _EMBEDDER_DIMS["openai"]
                if embedder == "openai"
                else _HF_MODEL_DIMS.get(hf_model, _EMBEDDER_DIMS["huggingface"])
            ),
        },
    }
    return {
        "llm": llm,
        "embedder": emb,
        "vector_store": vector_store,
        "history_db_path": str(workspace / "history.db"),
    }


def _conversation_to_session_messages(conversation: dict[str, Any]) -> list[list[dict[str, str]]]:
    """One LOCOMO conversation as Mem0 chat messages, GROUPED BY SESSION and in walk order.

    Mirrors `recall.eval.locomo.write_conversation_corpus`'s turn walk exactly — same session
    discovery, same sort, same skip rule, same per-turn information — so `RecallSystem` and
    `Mem0System` ingest the identical material in the identical order. Diverge here and the
    benchmark stops comparing memory systems and starts comparing which one got fed more:

    - Sessions are found by the ``session_`` prefix, excluding the sibling ``_date_time`` keys,
      and sorted numerically by the trailing session number (dict/string order would put
      ``session_10`` before ``session_2``).
    - A turn without a ``dia_id`` is skipped, because `write_conversation_corpus` skips it too
      (it uses `dia_id` for the output filename) — keeping it here would feed Mem0 a turn RE-call
      never sees.
    - Every field `_turn_document` writes into an indexed RE-call document is written into the
      message content: speaker, **session date**, turn text, and the image caption line when
      present. The session date is not decoration — `_turn_document`'s own docstring records that
      it is frequently the answer to LOCOMO's temporal (category 2) questions, so a Mem0 message
      without it would hand RE-call an information advantage on a whole question category. It is
      carried as a ``[date]`` prefix INSIDE the content (rather than as message metadata) so it
      survives Mem0's LLM fact-extraction step, which only ever reads the content.

    Grouping is by session because `Mem0System.ingest` issues one ``add()`` per group; see the
    comment there. Role is derived from `speaker_a` vs `speaker_b` (LOCOMO conversations are
    always two-party) so Mem0 sees a real alternating chat transcript rather than every turn
    collapsed onto one role.
    """
    speaker_a = conversation.get("speaker_a")
    sessions = sorted(
        (k for k in conversation if k.startswith("session_") and not k.endswith("date_time")),
        key=lambda k: int(k.split("_")[1]),
    )
    grouped: list[list[dict[str, str]]] = []
    for key in sessions:
        turns = conversation[key]
        if not isinstance(turns, list):
            continue
        # Same default as `write_conversation_corpus`: a session with no date line still gets the
        # same placeholder on both sides, so neither system silently sees a field the other lacks.
        date = conversation.get(f"{key}_date_time", "unknown date")
        session_messages: list[dict[str, str]] = []
        for turn in turns:
            if not turn.get("dia_id"):
                continue
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            content = f"[{date}] {speaker}: {text}"
            caption = turn.get("blip_caption")
            if caption:
                content += f"\n\n[shared an image: {caption}]"
            role = "user" if speaker == speaker_a else "assistant"
            session_messages.append({"role": role, "content": content})
        if session_messages:
            grouped.append(session_messages)
    return grouped


def _conversation_to_messages(conversation: dict[str, Any]) -> list[dict[str, str]]:
    """Flat, ordered view of `_conversation_to_session_messages` — every turn, session order.

    Kept as the single description of "what Mem0 is fed, in what order", so the parity test can
    compare one flat sequence against `write_conversation_corpus`'s corpus without having to know
    how ingestion happens to be chunked.
    """
    return [m for session in _conversation_to_session_messages(conversation) for m in session]


def _read(obj: object, name: str) -> Any:
    """`getattr` that cannot itself raise.

    `getattr(x, name, None)` swallows only `AttributeError`, and `Mem0System.close` probes
    attributes on a third-party object whose properties are free to raise anything. Since `close`
    runs inside `finally` blocks, an escaping probe would replace the exception that explains the
    run. Unreadable has to mean "nothing there", never "a new exception".
    """
    try:
        return getattr(obj, name, None)
    except Exception:  # noqa: BLE001 - a probe that raises has told us nothing, not something
        return None


class Mem0System:
    """Mem0 adapter. Feeds the conversation via `add`, retrieves via `search`. LLM on OpenRouter.

    Mirrors `RecallSystem`'s per-conversation tenancy: each LOCOMO conversation gets its own Mem0
    ``user_id`` (``bench-{sample_id}``), so one conversation's turns cannot leak into another's
    answers the way they could if every conversation shared one Mem0 user. Ingestion is chunked
    one ``add()`` per LOCOMO session — see `ingest`.
    """

    name = "mem0"

    def __init__(
        self,
        openrouter_key: str,
        model: str,
        embedder: str = "huggingface",
        openai_key: str | None = None,
        k: int = DEFAULT_K,
        *,
        hf_model: str = "BAAI/bge-small-en-v1.5",
        openai_model: str = "text-embedding-3-small",
        openai_base_url: str | None = None,
        run_id: str = "adhoc",
        storage_dir: str | Path | None = None,
    ) -> None:
        self._config = mem0_config(
            openrouter_key,
            model,
            embedder,
            openai_key,
            hf_model=hf_model,
            openai_model=openai_model,
            openai_base_url=openai_base_url,
            run_id=run_id,
            storage_dir=storage_dir,
        )
        self._k = k
        self._user: str | None = None
        self._mem: Any = None
        self._closed = False

    def describe(self) -> dict[str, Any]:
        """This arm's configuration, for the results artifact. API keys are redacted."""
        redacted: dict[str, Any] = _redact(self._config)
        return {
            "system": self.name,
            "k": self._k,
            "embedder": redacted["embedder"],
            "vector_store": redacted["vector_store"],
            "llm": redacted["llm"],
            "history_db_path": redacted["history_db_path"],
            "mem0ai_version": mem0ai_version(),
            "user": self._user,
        }

    def _memory(self) -> Any:
        if self._closed:
            # Without this, `_memory()`'s laziness would quietly REBUILD after `close()`, taking
            # the exclusive Qdrant lock again with no `__exit__` left to release it — the very
            # defect `close` exists to fix, restored through a path that reports nothing.
            raise RuntimeError("Mem0System used after close()")
        if self._mem is None:
            from mem0 import Memory

            self._mem = Memory.from_config(self._config)
        return self._mem

    def close(self) -> None:
        """Release the handles this system holds. Idempotent, and safe to call after a failure.

        Not tidiness. In the local Qdrant configuration `mem0_config` pins, a `mem0.Memory` owns
        EXCLUSIVE `portalocker` locks on its vector-store directories, held for as long as the
        client objects live. Until this method existed there was no way to let go of them, so an
        arm that raised mid-ingest kept its lock for the rest of the process and the next
        `Mem0System` on the same path died with

            RuntimeError: Storage folder ... is already accessed by another instance of Qdrant client

        which reads like a code failure and is not one. That is exactly what happened on
        2026-08-05: a run that ran out of OpenRouter credit left a lock held, and the resulting
        error was investigated as a regression. The default `run_id="adhoc"` makes it easy to hit,
        because every system built without an explicit run id shares one path.

        What this releases, having checked rather than assumed:

        - The primary Qdrant client, and with it the lock on this run's vector-store directory.
        - `_telemetry_vector_store`'s client. `Memory.__init__` opens a SECOND Qdrant store
          whenever `MEM0_TELEMETRY` is set, which is its default and which this project never
          turns off. Its path comes from mem0's home directory, not from `run_id`, so EVERY run
          and every concurrent process contends for that one lock — a wider collision than the
          per-run one, and the reason closing only `vector_store` is not enough.
        - `_entity_store`'s client, when it has a distinct one. For Qdrant, mem0 deliberately
          hands it the primary client "to avoid RocksDB lock contention", so the stores are
          deduped by identity and a shared client is closed exactly once.
        - The SQLite history connection, via `Memory.close()`.

        What it does NOT release, so the docstring cannot be read as a guarantee it is not making:
        the LLM client's HTTP connection pool, mem0's process-global PostHog telemetry thread
        (`atexit`-only), and the embedder's weights — which are held in a reference cycle, so
        dropping `_mem` does not free them until a GC pass runs.

        Two details a shorter implementation gets wrong. `Memory.close()` is NOT sufficient: it
        closes the SQLite handle and returns, never touching any vector store, so every lock
        survives it. And `__del__` is not a fallback: at interpreter shutdown qdrant-client's own
        finaliser dies with `ModuleNotFoundError: import of msvcrt halted`, so a lock left to the
        garbage collector can outlive the process's ability to release it.
        """
        memory, self._mem = self._mem, None
        self._closed = True
        if memory is None:
            return
        # Collect first, release second, so one unreadable store cannot hide the others. `_read`
        # is total because these are attributes on a third-party object, and a property is free to
        # raise: this runs inside `finally` blocks and `__exit__`, where a raise would REPLACE the
        # exception that actually explains the run.
        clients: list[tuple[str, Any]] = []
        # `_entity_store`, the PRIVATE name, deliberately. `Memory.entity_store` is a lazy property
        # that CONSTRUCTS the store on first access, so reading the public name here would build a
        # vector store — and take a lock — in the middle of releasing them. Do not "use the public
        # API" here; `test_close_reads_the_private_entity_store_not_the_lazy_property` pins it.
        for store in ("vector_store", "_telemetry_vector_store", "_entity_store"):
            client = _read(_read(memory, store), "client")
            # Identity, not equality: a `VectorStore` may not define `__eq__`, and the entity store
            # legitimately shares the primary client rather than owning a second one.
            if client is not None and not any(client is seen for _, seen in clients):
                clients.append((f"{store}.client", client))
        # Each handle is released independently. A lock is what damages the NEXT run, so letting go
        # of one must never be conditional on another closing cleanly.
        releases = [(name, _read(client, "close")) for name, client in clients]
        releases.append(("Memory.close", _read(memory, "close")))
        for name, release in releases:
            if release is None:
                continue
            try:
                release()
            except Exception as exc:  # noqa: BLE001 - a handle that will not close is not the caller's bug
                # Warn rather than raise, and rather than stay silent. The entire cost of the
                # 2026-08-05 incident was diagnostic; if a release fails, the next run hits the
                # same baffling "already accessed" error, and this line is the breadcrumb that
                # turns a second investigation into a one-line diagnosis.
                #
                # `name` is our own literal, never the handle's `repr`. `repr()` of a bound method
                # embeds `repr(__self__)`, so interpolating the handle handed an arbitrary
                # `__repr__` a path straight out of `close()` — from inside a `finally`, replacing
                # the error that explains the run. The two conditions correlate, too: a handle that
                # will not close is in a broken state, which is when its `__repr__` also raises.
                # `name` is a better breadcrumb anyway: it names the leaked lock.
                try:
                    detail = repr(exc)
                except Exception:  # noqa: BLE001 - an exception whose repr raises is still evidence
                    detail = f"<unprintable {type(exc).__name__}>"
                warnings.warn(f"Mem0System.close: {name} failed: {detail}", stacklevel=2)

    def __enter__(self) -> Mem0System:
        return self

    def __exit__(self, *exc_info: object) -> None:
        # Returns None (falsy), so an exception raised in the body still propagates — including
        # `pytest.skip`, which is a BaseException and therefore reaches here but not an
        # `except Exception` clause.
        self.close()

    def ingest(self, conversation: dict[str, Any]) -> None:
        # `conversation` here is the OUTER LOCOMO item (`sample_id` + nested `conversation`
        # object), matching `RecallSystem.ingest`'s contract — see the comment there.
        self._user = tenant_for(conversation)
        inner = conversation["conversation"]
        memory = self._memory()
        # ONE `add()` PER SESSION, not one per conversation. Mem0 runs an LLM fact-extraction pass
        # inside every `add()` call, so handing it a whole LOCOMO conversation (hundreds of turns,
        # tens of thousands of tokens) in a single call risks context truncation and severe
        # under-extraction — the benchmark would then be measuring a misconfiguration of Mem0
        # rather than Mem0. The LOCOMO session is the fair chunk: it is the unit the dataset itself
        # groups turns into, the unit RE-call's corpus walk iterates, and it bounds each extraction
        # call to a realistic conversation length. Sessions are sent in walk order under the SAME
        # `user_id`, so Mem0 accumulates exactly the same material, in the same sequence, as the
        # single-call version would have — only split at a boundary it can actually digest.
        for session_messages in _conversation_to_session_messages(inner):
            memory.add(session_messages, user_id=self._user)

    def retrieve(self, question: str) -> str:
        if self._user is None:
            raise RuntimeError("Mem0System.retrieve() called before ingest()")
        # mem0ai 2.x's `search` signature: the entity id goes in `filters`, and the budget is
        # `top_k`. The 1.x spelling (`user_id=`, `limit=`) does not merely get ignored on 2.x —
        # `_reject_top_level_entity_params` raises on a top-level `user_id`. Everything else is
        # left at Mem0's own defaults (notably its `threshold`), per the design's fairness rule
        # that each system runs on its recommended retrieval settings.
        res = self._memory().search(question, filters={"user_id": self._user}, top_k=self._k)
        results = res["results"] if isinstance(res, dict) else res
        return "\n".join(r["memory"] for r in results)
