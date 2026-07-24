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


def _conversation_to_messages(conversation: dict[str, Any]) -> list[dict[str, str]]:
    """Map one LOCOMO conversation's turns to Mem0's ``[{"role", "content"}]`` chat shape.

    Mirrors `recall.eval.locomo.write_conversation_corpus`'s turn walk exactly — same session
    discovery, same sort, same skip rule — so `RecallSystem` and `Mem0System` ingest the identical
    set of turns in the identical order. Diverge here and the benchmark stops comparing memory
    systems and starts comparing which one got fed more/different material:

    - Sessions are found by the ``session_`` prefix, excluding the sibling ``_date_time`` keys,
      and sorted numerically by the trailing session number (dict/string order would put
      ``session_10`` before ``session_2``).
    - A turn without a ``dia_id`` is skipped, because `write_conversation_corpus` skips it too
      (it uses `dia_id` for the output filename) — keeping it here would feed Mem0 a turn RE-call
      never sees.

    Content-wise this reuses `_turn_document`'s body convention (``"speaker: text"``, plus the
    image caption line when present) so the same information — not just the same turn count —
    reaches both systems; only the RE-call-specific markdown header/date line is dropped, since
    Mem0 messages have no place for it. Role is derived from `speaker_a` vs `speaker_b` (LOCOMO
    conversations are always two-party) so Mem0 sees a real alternating chat transcript rather
    than every turn collapsed onto one role.
    """
    speaker_a = conversation.get("speaker_a")
    sessions = sorted(
        (k for k in conversation if k.startswith("session_") and not k.endswith("date_time")),
        key=lambda k: int(k.split("_")[1]),
    )
    messages: list[dict[str, str]] = []
    for key in sessions:
        turns = conversation[key]
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if not turn.get("dia_id"):
                continue
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            content = f"{speaker}: {text}"
            caption = turn.get("blip_caption")
            if caption:
                content += f"\n\n[shared an image: {caption}]"
            role = "user" if speaker == speaker_a else "assistant"
            messages.append({"role": role, "content": content})
    return messages


class Mem0System:
    """Mem0 adapter. Feeds the conversation via `add`, retrieves via `search`. LLM on OpenRouter.

    Mirrors `RecallSystem`'s per-conversation tenancy: each LOCOMO conversation gets its own Mem0
    ``user_id`` (``bench-{sample_id}``), so one conversation's turns cannot leak into another's
    answers the way they could if every conversation shared one Mem0 user.
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
        messages = _conversation_to_messages(inner)
        self._memory().add(messages, user_id=self._user)

    def retrieve(self, question: str) -> str:
        if self._user is None:
            raise RuntimeError("Mem0System.retrieve() called before ingest()")
        res = self._memory().search(question, user_id=self._user, limit=self._k)
        results = res["results"] if isinstance(res, dict) else res
        return "\n".join(r["memory"] for r in results)
