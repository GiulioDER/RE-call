"""An assistant's memory surface on a lite store: one fact per memo, written and forgotten by memo id.

The other write tools take a path (`recall_index`) or a source (`recall_forget`), which suits an
agent that edits files. A personal assistant does not edit files: it remembers one fact, later
replaces it, and forgets it when asked. This module is that surface, for a lite server
(`sqlite:///`) whose operator names a memo folder in `RECALL_ASSISTANT_MEMOS`:

- `recall_remember` writes ONE fact as one memo file in that folder, with RE-call frontmatter
  (`valid_from` and `valid_until` as dates, `supersedes` as a file name, an optional structured fact
  as `recall_graph`), and indexes it before it returns, so the next search finds it.
- `supersedes` may name only a memo this surface wrote with the author `assistant`. A memo the person
  wrote, or one seeded from elsewhere, is not the assistant's to replace.
- `recall_forget_memo` forgets one memo of that folder by its id: `forget_memory` on its source
  first (which retracts every ledger fact whose only evidence it was), then the file, so the next
  refresh does not index it again.
- With the surface on, `recall_search`, `recall_evidence`, `recall_current_facts` and
  `recall_apply_fact` answer in the compact view below: the fields an assistant acts on, each hit
  named by its memo id, serialized as `json.dumps` writes them. The view is a projection of the full
  answer and nothing else, so it cannot disagree with it.

A model that learned this surface depends on it field for field, so the tool texts, the parameter
schemas and the view are pinned by `tests/test_assistant_surface.py` against
`tests/fixtures/assistant_surface_contract.json`. Change them only together.

The files themselves (memos, the authorship ledger, a forgotten memo's deletion) are written by
`recall.lite.assistant_memos`, outside this package, which makes no file write call of its own.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from recall.backends import write_session
from recall.errors import RecallError
from recall.lite.assistant_memos import MemoFolder

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.lite import LiteStore

__all__ = [
    "ASSISTANT_ENV",
    "ASSISTANT_TOOLS",
    "FORGET_DESCRIPTION",
    "REMEMBER_DESCRIPTION",
    "STATED_FACT_SCHEMA",
    "AssistantMemos",
    "AssistantSurfaceError",
    "assistant_memos_from_env",
    "dumps",
    "memo_ref",
]

#: The variable that turns the surface on: the folder the assistant's memos are written to.
ASSISTANT_ENV = "RECALL_ASSISTANT_MEMOS"
#: The tools this surface adds. Registered only when the surface is on.
ASSISTANT_TOOLS: frozenset[str] = frozenset({"recall_remember", "recall_forget_memo"})

REMEMBER_DESCRIPTION = (
    "Store ONE fact in the user's long-term memory for future sessions. Put dates in the text as "
    "YYYY-MM-DD with the weekday and the user's time zone. valid_from/valid_until (YYYY-MM-DD) say "
    "when the fact holds; supersedes names a memo_id this fact replaces. fact (optional) is the "
    "same fact as {subject, predicate, object}: recall_evidence then returns cards that "
    "recall_apply_fact can cite."
    " Corrections should pass supersedes and the revision returned by search or a write receipt "
    "as expected_revision. Reuse request_id only to retry the same write. include_receipt returns "
    "write, indexing and calibration readiness separately; old calls still return only memo_id. "
    "A correction target that already has a successor is refused."
)
FORGET_DESCRIPTION = (
    "Delete one memo from long-term memory by memo_id. A fact it alone supported is retracted, "
    "but its value stays in the append-only fact ledger."
)
_STRING = {"type": "string"}
#: The published schema of `recall_remember`'s `fact`: an inline object, not a `$ref`.
STATED_FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"subject": _STRING, "predicate": _STRING, "object": _STRING},
    "required": ["subject", "predicate", "object"],
}

#: The compact view, in the order a model reads it. Absent fields are written as null.
SEARCH_FIELDS = (
    "decision_state", "abstained", "reason", "advice", "evidence", "refresh_warning",
    "unknown_term_check", "unknown_terms", "historical", "query_support", "calibration_coverage",
)
EVIDENCE_FIELDS = (
    "decision_state", "decision", "reason_code", "trust_state", "advice", "refresh_warning",
    "unknown_term_check", "unknown_terms", "query_support", "calibration_coverage",
)

APPLY_FACT_FIELDS = (
    "allowed", "decision_code", "request_id", "fact_id", "retried", "detail", "event_id", "evidence_card_ids",
)


class AssistantSurfaceError(ValueError, RecallError):
    """The assistant surface is configured in a way it cannot serve."""


def dumps(payload: object) -> str:
    """The surface's serialization: `json.dumps` with its defaults, compact and ASCII."""
    return json.dumps(payload)


def memo_ref(value: object) -> str | None:
    """A memo id as a model may write it (`pref-0042`, `pref-0042.md`, padded): the bare id."""
    if not value:
        return None
    return str(value).strip().removesuffix(".md")


def assistant_memos_from_env(env: Mapping[str, str], *, lite: bool) -> AssistantMemos | None:
    """The surface `RECALL_ASSISTANT_MEMOS` asks for, or None when it is unset.

    Refuses rather than serving half of it: the folder must be on a lite store, and inside
    RECALL_INDEX_ROOT, which the server's automatic refresh follows and stored names are keyed to.
    """
    raw = env.get(ASSISTANT_ENV, "").strip()
    if not raw:
        return None
    if not lite:
        raise AssistantSurfaceError(
            f"{ASSISTANT_ENV} is served on a lite store only (a sqlite:/// DSN); unset it, or use "
            "recall_index and recall_forget on this store"
        )
    root = env.get("RECALL_INDEX_ROOT", "").strip()
    if not root:
        raise AssistantSurfaceError(f"{ASSISTANT_ENV} needs RECALL_INDEX_ROOT, the folder it lies in")
    folder, index_root = Path(raw).resolve(), Path(root).resolve()
    if not folder.is_relative_to(index_root):
        raise AssistantSurfaceError(
            f"{ASSISTANT_ENV}={str(folder)!r} is outside RECALL_INDEX_ROOT={str(index_root)!r}; the "
            "server's refresh would not follow it"
        )
    return AssistantMemos(folder, index_root=index_root)


class AssistantMemos(MemoFolder):
    """The memo folder (`recall.lite.assistant_memos`) with its lane-held writes and the view that
    names hits by memo id."""

    def remember(
        self,
        store: LiteStore,
        embedder: Embedder,
        text: str,
        kind: str = "note",
        valid_from: str | None = None,
        valid_until: str | None = None,
        supersedes: str | None = None,
        fact: object = None,
        *,
        author: str = "assistant",
        request_id: str | None = None,
        expected_revision: str | None = None,
        include_receipt: bool = False,
        **refresh_kwargs: Any,
    ) -> dict[str, Any]:
        """`write` then `refresh`, both under the store's write lane."""
        with write_session(store, "remember"):
            result = self.write(
                text, kind, valid_from, valid_until, supersedes, fact, author=author,
                request_id=request_id, expected_revision=expected_revision, include_receipt=include_receipt,
            )
            if "memo_id" in result:
                receipt = result.get("receipt")
                if receipt is None:
                    self.refresh(store, embedder, **refresh_kwargs)
                else:
                    try:
                        self.refresh(store, embedder, **refresh_kwargs)
                        path = self.path_of(result["memo_id"])
                        indexed = path is not None and store.source_raw_hash(str(path)) == receipt["revision"]
                        receipt["index_state"] = "indexed" if indexed else "failed"
                    except Exception as exc:  # BROAD-CATCH: fail-closed receipt; retry can finish indexing
                        receipt.update(index_state="failed", calibration_status="unavailable",
                                       retrieval_ready=False, error_type=type(exc).__name__)
                        return result
                    try:
                        resolution = store.resolve_calibration()
                        receipt["calibration_status"] = resolution.status.value
                        receipt["retrieval_ready"] = indexed and resolution.status.value == "certified"
                    except Exception as exc:  # BROAD-CATCH: fail-closed readiness, keep verified indexing state
                        receipt.update(calibration_status="unavailable", retrieval_ready=False,
                                       error_type=type(exc).__name__)
            return result

    def forget(
        self,
        store: LiteStore,
        embedder: Embedder,
        memo_id: str,
        *,
        security_policy: Any = None,
        security_context: Any = None,
        refresh_context: Any = None,
        **refresh_kwargs: Any,
    ) -> dict[str, Any]:
        """Forget one memo of the folder: its rows and facts first, then its file. {deleted: 0} for
        an id that names no memo here.

        Any memo of the folder may be forgotten, not only one the assistant wrote: a person asking
        their assistant to forget something they put there is the ordinary case, and the tool
        reaches nothing outside the folder. Whether a forget SHOULD have happened (the person
        asked, or a mail did) is the caller's judgement, as with `recall_forget`, whose scope this
        tool needs.

        `security_context` is the caller's, for `forget_memory`, as `recall_forget` passes it;
        `refresh_context` is the one the follow-up refresh indexes under (defaults to the same).

        A forget that found no rows, or whose fact retraction failed, raises rather than delete
        the file: the value would stay served while the caller reads the memo as gone.
        """
        from recall_mcp.lifecycle import forget_memory

        path = self.path_of(memo_id)
        if path is None:
            return {"deleted": 0}
        with write_session(store, "forget"):
            # Under the lane, so a concurrent forget of the same memo cannot pass the check too.
            if not path.exists():
                return {"deleted": 0}
            result = forget_memory(
                store,  # type: ignore[arg-type]  # the shared lifecycle types the Postgres store
                [str(path)],
                security_policy=security_policy,
                security_context=security_context,
            )
            if result.sources_not_found or result.facts_retracted == -1:
                raise RuntimeError(f"forget of {memo_id} incomplete: {result.message}")
            self.delete(memo_id)
            self.refresh(
                store,
                embedder,
                security_policy=security_policy,
                security_context=refresh_context if refresh_context is not None else security_context,
                **refresh_kwargs,
            )
        return {"deleted": 1, "memo_id": memo_id, "facts_retracted": result.facts_retracted}

    # -- the compact view -------------------------------------------------------------------------
    def memo_id(self, source: str) -> str:
        """The id a hit is named by: the file's stem in the memo folder, else its path under the
        index root without `.md`, which no tool of this surface accepts as a memo id. A hit names
        its source absolutely and a card by its file under the index root; both are read here."""
        path = Path(source)
        if not path.is_absolute():
            path = self.index_root / path
        if path.parent == self.folder:
            return path.stem
        if path.is_absolute() and path.is_relative_to(self.index_root):
            return path.relative_to(self.index_root).as_posix().removesuffix(".md")
        return path.stem

    def render_search(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """`recall_search`'s answer in the compact view."""
        return {
            **{k: payload.get(k) for k in SEARCH_FIELDS},
            "hits": [
                {
                    "memo_id": self.memo_id(h["source"]),
                    "chunk_id": h.get("chunk_id"),
                    "verdict": h["verdict"],
                    "text": h["text"],
                    "revision": h.get("revision"),
                    "valid_from": h.get("valid_from"),
                    "valid_until": h.get("valid_until"),
                    "superseded_by": h.get("superseded_by"),
                }
                for h in payload.get("hits", [])
            ],
        }

    def render_evidence(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """`recall_evidence`'s answer in the compact view: the items and cards, not the rendered
        prompts, which repeat the items inside a delimiter."""
        return {
            **{k: payload.get(k) for k in EVIDENCE_FIELDS},
            "items": [
                {
                    "chunk_id": i["chunk_id"],
                    "memo_id": self.memo_id(i["source"]),
                    "text": i["text"],
                    "valid_from": i.get("valid_from"),
                    "valid_until": i.get("valid_until"),
                }
                for i in payload.get("items", [])
            ],
            "cards": [
                {
                    "card_id": c["card_id"],
                    "chunk_id": c["chunk_id"],
                    "memo_id": self.memo_id(c["source"]),
                    "structured_facts": c.get("structured_facts", []),
                    "supersession_links": c.get("supersession_links", []),
                    "valid_from": c.get("valid_from"),
                    "valid_until": c.get("valid_until"),
                }
                for c in payload.get("cards", [])
            ],
        }

    @staticmethod
    def render_apply_fact(payload: Mapping[str, Any]) -> dict[str, Any]:
        """`recall_apply_fact`'s answer, its fields in the order `apply_fact_memory` writes them."""
        return {k: payload[k] for k in APPLY_FACT_FIELDS}

    @staticmethod
    def render_current_facts(payload: Mapping[str, Any]) -> dict[str, Any]:
        """`recall_current_facts`'s answer in the compact view."""
        return {
            "as_of": payload["as_of"],
            "facts": [
                {
                    "fact_id": f["fact_id"],
                    "fact": f["fact"],
                    "evidence_card_ids": f["evidence_card_ids"],
                    "asserted_at": f["asserted_at"],
                    "sources": f.get("sources", []),
                }
                for f in payload["facts"]
            ],
            "contested": payload.get("contested", []),
            "served_state": payload.get("served_state"),
            "served_lookups_failed": payload.get("served_lookups_failed", []),
        }
