"""`MemorySystem` protocol + the RE-call adapter for the head-to-head benchmark.

The benchmark runs two memory systems through an IDENTICAL LLM generator (see
`benchmarks.pipeline.run_question`) on LOCOMO — only the retrieved memory differs between systems.
`MemorySystem` is the seam that makes that swap possible: `run_question` takes a bare
``retrieve(question) -> str`` callable, and any `MemorySystem.retrieve` satisfies it directly.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MemorySystem(Protocol):
    """One competitor in the head-to-head benchmark.

    ``ingest`` indexes a single LOCOMO conversation; ``retrieve`` answers one question against
    whatever was last ingested. Both are per-conversation — LOCOMO's conversations are unrelated
    worlds, and the benchmark harness ingests exactly once per conversation before scoring its
    questions.
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

    def __init__(self, dsn: str, embedder_name: str = "fastembed", k: int = 5) -> None:
        from recall.eval.locomo import _make_embedder

        self._dsn = dsn
        self._k = k
        self._embedder_name = embedder_name
        self._embedder = _make_embedder(embedder_name)
        self._tenant: str | None = None

    def ingest(self, conversation: dict[str, Any]) -> None:
        from recall.eval.locomo import index_conversation
        from recall.store import PgVectorStore

        # `conversation` here is one LOCOMO item as loaded from locomo10.json: it carries
        # `sample_id` and `qa` alongside the nested `conversation` object (`session_N` turns,
        # `speaker_a`/`speaker_b`) that `index_conversation` actually indexes. Passing the outer
        # item straight into `index_conversation` would find zero `session_` keys and silently
        # index nothing.
        self._tenant = f"bench-{conversation.get('sample_id')}"
        inner = conversation["conversation"]
        with PgVectorStore(
            self._dsn, dim=self._embedder.dim, tenant=self._tenant, table="bench_locomo_chunks"
        ) as store:
            index_conversation(store, self._embedder, inner)

    def retrieve(self, question: str) -> str:
        from recall.store import PgVectorStore
        from recall.trust import trusted_search

        if self._tenant is None:
            raise RuntimeError("RecallSystem.retrieve() called before ingest()")
        with PgVectorStore(
            self._dsn, dim=self._embedder.dim, tenant=self._tenant, table="bench_locomo_chunks"
        ) as store:
            result = trusted_search(store, self._embedder, question, k=self._k)
            if result.abstained:
                return ""
            return "\n".join(hit.chunk.text for hit in result.hits)


def mem0_config(
    openrouter_key: str,
    model: str,
    embedder: str = "huggingface",
    openai_key: str | None = None,
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
        emb = {
            "provider": "openai",
            "config": {"model": "text-embedding-3-small", "api_key": openai_key},
        }
    else:
        emb = {"provider": "huggingface", "config": {"model": "BAAI/bge-small-en-v1.5"}}
    return {"llm": llm, "embedder": emb}


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
        k: int = 5,
    ) -> None:
        self._config = mem0_config(openrouter_key, model, embedder, openai_key)
        self._k = k
        self._user: str | None = None
        self._mem: Any = None

    def _memory(self) -> Any:
        if self._mem is None:
            from mem0 import Memory

            self._mem = Memory.from_config(self._config)
        return self._mem

    def ingest(self, conversation: dict[str, Any]) -> None:
        # `conversation` here is the OUTER LOCOMO item (`sample_id` + nested `conversation`
        # object), matching `RecallSystem.ingest`'s contract — see the comment there.
        self._user = f"bench-{conversation.get('sample_id')}"
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
        res = self._memory().search(question, user_id=self._user, limit=self._k)
        results = res["results"] if isinstance(res, dict) else res
        return "\n".join(r["memory"] for r in results)
