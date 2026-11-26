"""An assistant's memory surface: one fact per memo, written and forgotten by memo id.

The other write tools take a path (`recall_index`) or a source (`recall_forget`), which suits an
agent that edits files. A personal assistant does not edit files: it remembers one fact, later
replaces it, and forgets it when asked. This module is that surface, for a server whose operator
names a memo folder in `RECALL_ASSISTANT_MEMOS`:

- `recall_remember` writes ONE fact as one memo file in that folder, with RE-call frontmatter
  (`valid_from` and `valid_until` as dates, `supersedes` as the target's path under the index root,
  an optional structured fact as `recall_graph`). On a lite store (`sqlite:///`) it indexes the
  memo before it returns, so the next search finds it. On PostgreSQL it leaves the memo to the next
  build of the corpus, as production indexes nothing at write time (`recall.runtime_route`).
- `supersedes` may name only a memo this surface wrote with the author `assistant`: of this folder
  by its memo id, or of another assistant's memo folder under the same index root by the id search
  names it by (`agent-b/note-0001`). A memo the person wrote, or one seeded from elsewhere, is not
  the assistant's to replace.
- `recall_forget_memo` forgets one memo of that folder by its id: `forget_memory` on its source
  first (which retracts every ledger fact whose only evidence it was), then the file, so the next
  refresh does not index it again.
- `recall_withdraw` takes a memo of that folder by its id or its file too, and withdraws it the
  same way, file included; any other source goes to the plain tool.
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
from contextlib import AbstractContextManager
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, cast

from recall.backends import write_session
from recall.confinement import within
from recall.errors import RecallError
from recall._env import strict_bool
from recall.declared_keys import STATED_KEY_SUCCESSOR_RULE, settle_precondition, settle_target
from recall.lite import LiteStore
from recall.lite.assistant_memos import ASSISTANT_AUTHOR, MEMO_ID, MemoFolder

from recall_mcp.models import ForgetResult, WithdrawResult

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.store import PgVectorStore

__all__ = [
    "ASSISTANT_ENV",
    "ASSISTANT_TOOLS",
    "FORGET_DESCRIPTION",
    "REMEMBER_DESCRIPTION",
    "REPORT_STALE_DESCRIPTION",
    "REPORT_USE_DESCRIPTION",
    "STATED_FACT_SCHEMA",
    "WITHDRAW_DESCRIPTION",
    "AssistantMemos",
    "AssistantSurfaceError",
    "assistant_memos_from_env",
    "dumps",
    "memo_ref",
]

#: The variable that turns the surface on: the folder the assistant's memos are written to.
ASSISTANT_ENV = "RECALL_ASSISTANT_MEMOS"
#: The operator's half of `recall_remember(settle=true)`: off by default, and a call that asks to
#: settle on a server without it is refused rather than written without the edge.
SETTLE_ENV = "RECALL_ASSISTANT_SETTLE_SUCCESSORS"
#: The refusals of a settled `supersedes` that `remember` turns into a declined settle, by reason:
#: the rule read the target from the index, and the files have moved on since.
_UNSETTLED = {"stale_target": "target_has_successor", "target_missing": "target_missing"}
#: The tools this surface adds. Registered only when the surface is on.
ASSISTANT_TOOLS: frozenset[str] = frozenset({"recall_remember", "recall_forget_memo"})

REMEMBER_DESCRIPTION = (
    "Store ONE fact in the user's long-term memory for future sessions. Put dates in the text as "
    "YYYY-MM-DD with the weekday and the user's time zone. valid_from/valid_until (YYYY-MM-DD) say "
    "when the fact holds; supersedes names a memo_id this fact replaces. fact (optional) is the "
    "same fact as {subject, predicate, object}: recall_evidence then returns cards that "
    "recall_apply_fact can cite."
    " Corrections should pass supersedes and the revision returned by search or a write receipt "
    "as expected_revision (a lowercase SHA-256, exactly as returned). Reuse request_id (at most 128 "
    "characters) only to retry the same write. A refusal carries `refused`, and a refused "
    "precondition also a `conflict` code. include_receipt returns "
    "write, indexing and calibration readiness separately; old calls still return only memo_id. "
    "A correction target that already has a successor is refused."
    " Before stating a fact, look its subject up with recall_current_facts(subject=...) and repeat an "
    "existing subject and predicate exactly; the receipt's fact_key says which key the fact went "
    "under, what else is in force for it, and any supersession left to settle."
    " settle (where the server allows it) writes that supersession into this memo instead, when "
    "the earlier memo is the assistant's and the predicate holds one value at a time; the answer's "
    "settle names the memo it replaced, or why it did not."
)
FORGET_DESCRIPTION = (
    "Delete one memo from long-term memory by memo_id. A fact it alone supported is retracted "
    "and its value erased from the fact ledger."
)
#: `recall_report_use`'s text when served beside this surface, where search names hits by memo_id.
REPORT_USE_DESCRIPTION = (
    "After a task that searched memory, say what the memories did for it. effect is one of "
    "helped, no_difference or misled. List in used the memories you actually relied on, and in "
    "wrong any that were retrieved but wrong, outdated or misleading; name each by the memo_id "
    "search returned it by. Set task_succeeded when you know whether the task itself succeeded, "
    "and query to the search you made. Report once per task, honestly, including when memory did "
    "not help: the counts are only useful if the no-difference cases are in them. Nothing in "
    "memory or search changes. A server with a shared rate limiter requires idempotency_key, as "
    "for every write."
)
#: `recall_withdraw`'s text when served beside this surface, where search names hits by memo_id
#: and shows no source (CCA DOC-001).
WITHDRAW_DESCRIPTION = (
    "Withdraw memory that was a MISTAKE, and give back what it replaced. IRREVERSIBLE. Use this, "
    "not recall_forget_memo, when a memory should never have been believed: a wrong correction, a "
    "mistaken \"this replaces that\". It erases the memory as recall_forget_memo does, file "
    "included, and voids its supersedes claims, so what it replaced is current again. Each fact it "
    "alone supported is withdrawn and its value erased. Name each memory in sources by the memo_id "
    "search returned it by, or by its file. A server with a shared rate limiter requires "
    "idempotency_key, as for every write."
)
#: `recall_report_stale`'s text when served beside this surface, where search names hits by memo_id.
REPORT_STALE_DESCRIPTION = (
    "Report that one memory you used is replaced by another, newer one you also saw. Use it only "
    "when two memories state different versions of the same fact, decision or status and one "
    "clearly replaces the other. Not for a memory that is merely old, partial or off topic. Name "
    "both by the memo_id search returned them by, and copy at least 20 characters exactly from "
    "each: the outdated statement and the one that replaces it. The report is queued for a person "
    "to review; nothing in memory changes until they accept it. A server with a shared rate "
    "limiter requires idempotency_key, as for every write."
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


def assistant_memos_from_env(env: Mapping[str, str]) -> AssistantMemos | None:
    """The surface `RECALL_ASSISTANT_MEMOS` asks for, or None when it is unset.

    Refuses rather than serving half of it: the folder must be inside RECALL_INDEX_ROOT, which stored
    names are keyed to (and which a lite server's automatic refresh follows). On PostgreSQL the
    names are the generation's, so RECALL_INDEX_ROOT must be the folder the corpus manifest is
    built from, or a settle finds no assistant memo to name and declines.
    """
    raw = env.get(ASSISTANT_ENV, "").strip()
    if not raw:
        return None
    root = env.get("RECALL_INDEX_ROOT", "").strip()
    if not root:
        raise AssistantSurfaceError(f"{ASSISTANT_ENV} needs RECALL_INDEX_ROOT, the folder it lies in")
    folder, index_root = Path(raw).resolve(), Path(root).resolve()
    if within(index_root, folder) is None:
        raise AssistantSurfaceError(
            f"{ASSISTANT_ENV}={str(folder)!r} is outside RECALL_INDEX_ROOT={str(index_root)!r}; the "
            "server's refresh would not follow it"
        )
    return AssistantMemos(folder, index_root=index_root,
                          settle_successors=strict_bool(env.get(SETTLE_ENV), name=SETTLE_ENV))


def _key_only(fact: object, reason: str = "reading what else is in force needs the read scope") -> dict[str, Any]:
    """`fact_key` naming the fact's own key and nothing about others: for a writer who may not read
    current facts, and for a memo the store does not serve yet."""
    from recall.types import AtomicFact

    stated = AtomicFact.from_payload(fact) if isinstance(fact, dict) else None
    key = None if stated is None else {"namespace": stated.namespace, "subject": stated.subject,
                                       "predicate": stated.predicate, "context": stated.context}
    return {"key": key, "state": "unchecked", "reason": reason}


#: Why a PostgreSQL receipt reports the memo `pending` and its key `unchecked`.
_BUILT_LATER = "the memo is indexed by the next build of the corpus"


class AssistantMemos(MemoFolder):
    """The memo folder (`recall.lite.assistant_memos`) with its lane-held writes and the view that
    names hits by memo id."""

    def __init__(
        self, folder: str | Path, *, index_root: str | Path | None = None, settle_successors: bool = False
    ) -> None:
        super().__init__(folder, index_root=index_root)
        #: Whether this server lets `remember(settle=True)` write the rule's edge (SETTLE_ENV).
        self.settle_successors = settle_successors

    def remember(
        self,
        store: LiteStore | PgVectorStore,
        embedder: Embedder,
        text: str,
        kind: str = "note",
        valid_from: str | None = None,
        valid_until: str | None = None,
        supersedes: str | None = None,
        fact: object = None,
        *,
        author: str = ASSISTANT_AUTHOR,
        request_id: str | None = None,
        expected_revision: str | None = None,
        include_receipt: bool = False,
        access_context: Any = None,
        read_facts: bool = True,
        settle: bool = False,
        **refresh_kwargs: Any,
    ) -> dict[str, Any]:
        """`write` then `refresh`, both under the store's write lane.

        A receipt for a memo that states a `fact` also says, under `fact_key`, which key the fact
        went under, what is in force for that key now, and any edge rule R(a') proposes
        (`recall.declared_keys.key_receipt`), read with the caller's `access_context`. With
        `read_facts` false (a caller without the read scope) it names the key only, and an
        `expected_revision` is refused (`revision_requires_read`): comparing it with the target
        would let a writer confirm guesses of a memo it may not read.

        With `settle` (refused unless the server's `settle_successors` is on) the edge rule R(a')
        gives the new memo is written as its `supersedes`, read inside the same lane so no write can
        move the ledger in between, and the answer's `settle` says `{"settled": <memo id>}`,
        `{"settled": None}` (nothing to settle) or `{"declined": <reason>}` (`_settle`).

        On PostgreSQL the memo is written and left to the next build of the corpus, which is the
        only way production indexes (`recall.runtime_route`): its receipt says `pending`, and a
        settle reads what the served generation holds, so it cannot see a memo written since that
        build and answers `{"settled": None}` for it. The folder scan in `write` still sees such
        a memo as a successor, so no target is superseded twice.
        """
        if expected_revision is not None and not read_facts:
            return {"refused": "expected_revision needs the read scope", "conflict": "revision_requires_read"}
        if settle and not self.settle_successors:
            return {"refused": f"settle is off on this server; the operator turns it on with {SETTLE_ENV}"}
        with self._lane(store, "remember"):
            note: dict[str, Any] | None = None
            replayed = False
            if settle:
                supersedes, note, replayed = self._settle(
                    store, fact=fact, valid_from=valid_from, supersedes=supersedes, request_id=request_id,
                    access_context=access_context, read_facts=read_facts,
                    security_policy=refresh_kwargs.get("security_policy"),
                )
            def write_with(target: str | None, note: dict[str, Any] | None) -> dict[str, Any]:
                return self.write(
                    text, kind, valid_from, valid_until, target, fact, author=author,
                    request_id=request_id, expected_revision=expected_revision, include_receipt=include_receipt,
                    has_successor=lambda target, own: self._indexed_successor(store, target, own),
                    settled_by=STATED_KEY_SUCCESSOR_RULE if note is not None and note.get("settled") else None,
                    settle_note=note,
                )

            result = write_with(supersedes, note)
            if note is not None and note.get("settled") and result.get("conflict") in _UNSETTLED and not replayed:
                # `write_memo`'s own check of the target, the only one, found the rule's older memo
                # already succeeded or gone since it was indexed: the memo is written without the edge.
                # Not on a replay: the journal holds the request with its edge, so a rewrite without
                # it would be refused as another request (CCA fix audit FA-2-5).
                note = {"declined": _UNSETTLED[result["conflict"]]}
                result = write_with(None, note)
            if note is not None and "memo_id" in result:
                result["settle"] = note
            if "memo_id" in result:
                receipt = result.get("receipt")
                if not isinstance(store, LiteStore):
                    if receipt is not None:
                        self._pending_receipt(store, receipt, fact)
                elif receipt is None:
                    self.refresh(store, embedder, **refresh_kwargs)
                else:
                    self._complete_receipt(
                        store, embedder, result["memo_id"], receipt, fact, access_context, read_facts,
                        refresh_kwargs,
                    )
            return result

    def _lane(self, store: object, purpose: str) -> AbstractContextManager[Any]:
        """The lane a write of this folder holds: the lite store's write lane, else the folder's own.

        On PostgreSQL nothing is indexed under it, so the store needs no lock here; what two servers
        writing one folder would race on is the folder itself (the journal, the next memo id, the
        successor scan), and a lite lane keyed to the journal file serialises them.
        """
        if isinstance(store, LiteStore):
            return write_session(store, purpose)
        from recall.lite.writer_lock import lite_lane

        return lite_lane(self.ledger_path, purpose=purpose)

    @staticmethod
    def _pending_receipt(store: object, receipt: dict[str, Any], fact: object) -> None:
        """Fill a PostgreSQL receipt: the memo waits for the next build, so it is not ready yet."""
        resolver = getattr(store, "resolve_calibration", None)
        try:
            status = resolver().status.value if callable(resolver) else "unavailable"
        except Exception as exc:  # BROAD-CATCH: fail-closed readiness, the memo is written
            status = "unavailable"
            receipt["error_type"] = type(exc).__name__
        receipt.update(index_state="pending", calibration_status=status, retrieval_ready=False)
        if fact is not None:
            receipt["fact_key"] = _key_only(fact, _BUILT_LATER)

    def _settle(
        self, store: LiteStore | PgVectorStore, *, fact: object, valid_from: str | None, supersedes: str | None,
        request_id: str | None, access_context: Any, read_facts: bool, security_policy: Any,
    ) -> tuple[str | None, dict[str, Any], bool]:
        """The `supersedes` a settling write carries, the answer's `settle` note, and whether both
        were replayed from the journal.

        A `supersedes` the writer named wins. A retry of a journalled request replays the target its
        first call wrote, and the note it journalled: the journal hashes `supersedes` into the
        request's identity, and on a retry the superseded memo is no longer in force, so computing
        the pair again would find none and the retry would read as another request. A request
        journalled without a note (its first call did not settle) replays its own `supersedes` as the
        writer's. Otherwise the target is `settle_target`'s over the current facts the caller may
        read, and must be a memo the assistant wrote. Whether it has a successor yet is `write`'s
        own check, and `remember` declines the edge when that check refuses it.
        """
        from recall.frontmatter import validity_bounds
        from recall.lite.memo_requests import load_ledger
        from recall.types import AtomicFact
        from recall_mcp.provenance import current_facts_memory

        if supersedes is not None:
            return supersedes, {"declined": "writer_named_supersedes"}, False
        if not isinstance(fact, dict):
            return None, {"declined": "no_fact"}, False
        if not read_facts:
            return None, {"declined": "unchecked"}, False
        own = load_ledger(self.ledger_path)
        entry = own.get("requests", {}).get(request_id) if request_id is not None else None
        if isinstance(entry, dict):
            replayed, journalled = entry.get("supersedes"), entry.get("settle")
            if isinstance(journalled, dict):
                return replayed, dict(journalled), True
            return replayed, {"declined": "writer_named_supersedes"} if replayed is not None else {"settled": None}, True
        if security_policy is not None:
            # The listing a policy filters can miss a holder of the key, and counting over the whole
            # store would let the answer reveal a memo the caller may not read (CCA STAKES-001).
            return None, {"declined": "unchecked"}, False
        try:
            stated = AtomicFact.from_payload(fact)
            start = validity_bounds({"valid_from": valid_from})[0]
        except (TypeError, ValueError, KeyError):
            return None, {"declined": "invalid"}, False  # `write` refuses it, with its own reason
        early = settle_precondition(stated, start)
        if early is not None:
            return None, {"declined": early}, False
        listing = current_facts_memory(cast(Any, store), security_policy=security_policy, access_context=access_context)
        source, reason = settle_target(listing, stated, start)
        if reason is not None:
            return None, {"declined": reason}, False
        if source is None:
            return None, {"settled": None}, False
        ref = self.ref_of_source(source)
        located = None if ref is None else self.assistant_target(ref, ledger=own)
        if ref is None or not isinstance(located, tuple):
            return None, {"declined": "not_assistant_memo"}, False
        return ref, {"settled": located[1] if located[0] == self.folder else ref}, False

    def _complete_receipt(
        self, store: LiteStore, embedder: Embedder, memo_id: str, receipt: dict[str, Any],
        fact: object, access_context: Any, read_facts: bool, refresh_kwargs: dict[str, Any],
    ) -> None:
        """Index the memo `remember` just wrote and fill its receipt, failing closed at each step."""
        from recall.calibration_v2 import CalibrationStatus

        try:
            self.refresh(store, embedder, **refresh_kwargs)
            path = self.path_of(memo_id)
            indexed = path is not None and store.source_raw_hash(str(path)) == receipt["revision"]
            receipt["index_state"] = "indexed" if indexed else "failed"
        except Exception as exc:  # BROAD-CATCH: fail-closed receipt; retry can finish indexing
            receipt.update(index_state="failed", calibration_status="unavailable",
                           retrieval_ready=False, error_type=type(exc).__name__)
            return
        try:
            resolution = store.resolve_calibration()
            receipt["calibration_status"] = resolution.status.value
            receipt["retrieval_ready"] = indexed and resolution.status is CalibrationStatus.CERTIFIED
        except Exception as exc:  # BROAD-CATCH: fail-closed readiness, keep verified indexing state
            receipt.update(calibration_status="unavailable", retrieval_ready=False,
                           error_type=type(exc).__name__)
        if fact is not None and indexed and not read_facts:
            receipt["fact_key"] = _key_only(fact)
        elif fact is not None and indexed:
            receipt["fact_key"] = self._fact_key(
                store, memo_id, fact, access_context, refresh_kwargs.get("security_policy"),
            )

    def _fact_key(
        self, store: LiteStore, memo_id: str, fact: object, access_context: Any, security_policy: Any,
    ) -> dict[str, Any]:
        """The receipt's `fact_key`: `key_receipt` over the current facts as the caller may read them.

        It reads the whole current-facts projection, inside the write lane, so its cost grows with
        the ledger; only a receipt asked for, of a memo stating a fact, pays it. A settling write has
        read it once already (`_settle`), before the memo existed, so a settling write with a
        receipt pays it twice: the receipt must show the key after the write.
        """
        from recall.declared_keys import key_receipt
        from recall.types import AtomicFact
        from recall_mcp.provenance import current_facts_memory

        path = self.path_of(memo_id)
        if path is None or not isinstance(fact, dict):  # `write` refuses both before a memo exists
            return {"state": "unchecked", "error_type": "ValueError"}
        folder = self.folder.relative_to(self.index_root).as_posix()
        folder = "" if folder == "." else folder

        def name_of(name: str) -> str:
            # A memo of this folder by its id, as the surface names it everywhere; any other by its name.
            parent, _, file = name.rpartition("/")
            stem = file.removesuffix(".md")
            return stem if parent == folder and MEMO_ID.fullmatch(stem) else name

        try:
            # Typed for PgVectorStore, it reads a lite store as the held `recall_current_facts` does.
            listing = current_facts_memory(cast(Any, store), security_policy=security_policy,
                                           access_context=access_context)
            return key_receipt(listing, path.relative_to(self.index_root).as_posix(), AtomicFact.from_payload(fact),
                               name_of=name_of)
        except Exception as exc:  # BROAD-CATCH: fail-open; the memo is written and indexed, only the key report is missing
            return {"state": "unchecked", "error_type": type(exc).__name__}

    def forget(
        self,
        store: LiteStore | PgVectorStore,
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

        A forget whose fact retraction or fact value erasure failed raises rather than delete the
        file, and so does one that found no rows by the memo's path while rows name it another
        way (`_indexed_under_another_name`): the value would stay served, or stay in the fact
        ledger, while the caller reads the memo as gone. A memo no row names was published and
        never indexed (a crash after publication, no refresh since): it is deleted, and its request
        cancelled, like any other. The answer's fields are the pinned
        contract's; a completed forget has erased the values too (`forget_memory`). A forget whose
        erasure elsewhere (the embedding cache, the atomic rescue) was incomplete adds `warning`,
        `forget_memory`'s message, which says what is left.
        """
        result = self._erase_memo(
            store, embedder, memo_id, withdraw=False, security_policy=security_policy,
            security_context=security_context, refresh_context=refresh_context, **refresh_kwargs,
        )
        if result is None:
            return {"deleted": 0}
        assert isinstance(result, ForgetResult)
        answer: dict[str, Any] = {"deleted": 1, "memo_id": memo_id, "facts_retracted": result.facts_retracted}
        if " WARNING: " in result.message or " NOTE: " in result.message:
            # The rows and the file are gone, but something else the forget erases (the embedding
            # cache, the atomic rescue) may still hold the text; the message is the only place
            # that says what is left and where, and a re-run finds no rows (CCA STAKES-001).
            answer["warning"] = result.message
        return answer

    def forget_result(
        self,
        store: LiteStore | PgVectorStore,
        embedder: Embedder,
        memo_id: str,
        *,
        security_policy: Any = None,
        security_context: Any = None,
        refresh_context: Any = None,
        **refresh_kwargs: Any,
    ) -> ForgetResult | None:
        """Forget one memo of the folder as `forget` does, its file included, answering in
        `recall_forget`'s form. None for an id that names no memo file here, which the caller
        forgets by the name it had."""
        result = self._erase_memo(
            store, embedder, memo_id, withdraw=False, security_policy=security_policy,
            security_context=security_context, refresh_context=refresh_context, **refresh_kwargs,
        )
        assert result is None or isinstance(result, ForgetResult)
        return result

    def withdraw(
        self,
        store: LiteStore | PgVectorStore,
        embedder: Embedder,
        memo_id: str,
        *,
        security_policy: Any = None,
        security_context: Any = None,
        refresh_context: Any = None,
        **refresh_kwargs: Any,
    ) -> WithdrawResult | None:
        """Withdraw one memo of the folder (`withdraw_memory`) as `forget` forgets one, its file
        included, so the next refresh does not index it again. None for an id that names no memo
        file here, which the caller withdraws by the name it had."""
        result = self._erase_memo(
            store, embedder, memo_id, withdraw=True, security_policy=security_policy,
            security_context=security_context, refresh_context=refresh_context, **refresh_kwargs,
        )
        assert result is None or isinstance(result, WithdrawResult)
        return result

    def _erase_memo(
        self,
        store: LiteStore | PgVectorStore,
        embedder: Embedder,
        memo_id: str,
        *,
        withdraw: bool,
        security_policy: Any,
        security_context: Any,
        refresh_context: Any,
        **refresh_kwargs: Any,
    ) -> ForgetResult | WithdrawResult | None:
        """The body `forget` and `withdraw` share: the journal, the rows, the file, the refresh."""
        from recall_mcp.lifecycle import forget_memory, withdraw_memory

        path = self.path_of(memo_id)
        if path is None:
            return None
        verb = "withdrawal" if withdraw else "forget"
        # A lite row's source is the file's path; a generation's is its manifest URI, which the
        # file's name under the index root (`metadata.file`) resolves to.
        name = str(path) if isinstance(store, LiteStore) else path.relative_to(self.index_root).as_posix()
        with self._lane(store, "forget"):
            # Under the lane, so a concurrent forget of the same memo cannot pass the check too.
            if not path.exists():
                return None
            # The journal first: a ledger that cannot be read or saved stops the forget before
            # any row is erased, instead of after, with the file left for the next refresh.
            previous = self.cancel_requests(memo_id)
            try:
                result = (withdraw_memory if withdraw else forget_memory)(
                    store,  # type: ignore[arg-type]  # the shared lifecycle types the Postgres store
                    [name],
                    security_policy=security_policy,
                    security_context=security_context,
                    erase_memo_requests=False,  # cancelled above, and restored if this fails
                )
                facts = result.facts_withdrawn if isinstance(result, WithdrawResult) else result.facts_retracted
                if (facts == -1 or result.facts_redacted == -1
                        or (result.sources_not_found and self._indexed_under_another_name(store, path))):
                    raise RuntimeError(f"{verb} of {memo_id} incomplete: {result.message}")
            except BaseException:
                self.restore_requests(previous)
                raise
            self.unlink_cancelled(path, previous)
            if not isinstance(store, LiteStore):
                return result  # the next build of the corpus leaves the deleted file out
            self.refresh(
                store,
                embedder,
                security_policy=security_policy,
                security_context=refresh_context if refresh_context is not None else security_context,
                **refresh_kwargs,
            )
        return result

    def folder_memo(self, value: object) -> str | None:
        """The id of the memo of this folder that `value` names, by its id (`note-0002`, `.md` or
        not) or by its file's path (absolute, or under the index root); None for anything else."""
        ref = memo_ref(value)
        if not ref:
            return None
        path = Path(ref)
        qualified = len(path.parts) > 1
        memo = path.name if qualified else ref
        if self.path_of(memo) is None:
            return None
        if qualified:
            if not path.is_absolute():
                path = self.index_root / path
            if path.parent.resolve() != self.folder.resolve():
                return None
        return memo

    def _indexed_successor(self, store: LiteStore | PgVectorStore, target: str, own: str | None = None) -> bool:
        """Whether an indexed document other than memo `own` claims to supersede memo `target` (a
        memo id here, or another folder's qualified id, `target_of`), as the store resolves
        `supersedes` (folded case, any markdown extension, anywhere under the index root, and
        forgotten claimants kept as ghosts)."""
        from recall.supersession import store_supersession_graph

        graph = store_supersession_graph(store)

        def key(path: Path | None) -> str | None:
            if path is None:
                return None
            # Computes a memo key under the index root; no file is read.
            # ast-grep-ignore: containment-via-confined
            rel = path.relative_to(self.index_root) if path.is_relative_to(self.index_root) else None
            return graph.key(rel.as_posix() if rel is not None else path.name, str(path))

        located = self.target_of(target)
        target_key = key(None if located is None else located[0] / f"{located[1]}.md")
        if target_key is None:
            return False
        own_key = key(self.path_of(own)) if own is not None else None
        return any(claimant != own_key for claimant in graph.live_claimants(target_key))

    def _indexed_under_another_name(self, store: LiteStore | PgVectorStore, path: Path) -> bool:
        """Whether rows hold this memo under a name other than its path.

        Two ways: by its stored file name, the one `refresh` gives it (under the index root when
        the folder is inside it, else under the folder); or under any other source of the same file
        name and the same bytes (the folder indexed from another absolute path, or keyed to another
        root). Rows of the same name and other bytes are not this memo's text, and nothing else in
        a row ties it to this memo.
        """
        from recall.lite.memo_requests import revision

        # Computes a memo key under the index root; no file is read.
        # ast-grep-ignore: containment-via-confined
        key = path.relative_to(self.index_root).as_posix() if path.is_relative_to(self.index_root) else path.name
        if store.sources_for_identifiers([key]):
            return True
        text = revision(path)
        return any(
            digest == text and PurePath(source.replace("\\", "/")).name == path.name
            for source, digest in store.source_raw_hashes().items()
        )

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
        # Computes a memo id under the index root; no file is read.
        # ast-grep-ignore: containment-via-confined
        if path.is_absolute() and path.is_relative_to(self.index_root):
            return path.relative_to(self.index_root).as_posix().removesuffix(".md")
        return path.stem

    def files_named(self, store: LiteStore, names: list[object]) -> dict[str, tuple[str, str]]:
        """`{name: (file, source)}` for each name that `memo_id` gives exactly one stored source:
        its file under the index root, as the stale-report queue names it, and the source, as the
        ledger does. Read as `memo_id` writes it (own folder or under the index root, `.md`
        dropped or not); a name naming no row, or several, is absent and left to the tool to judge
        as written."""
        keys: dict[str, list[str]] = {}
        for raw in names:
            name = str(raw).strip()
            ref = memo_ref(name)
            if not ref:
                continue
            path = within(self.index_root, (self.index_root if "/" in ref else self.folder) / ref)
            if path is None:
                continue
            # Computes a memo key under the index root; no file is read.
            # ast-grep-ignore: containment-via-confined
            rel = path.relative_to(self.index_root).as_posix()
            keys[name] = [f"{rel}.md", rel]
        found = store.sources_for_identifiers([key for pair in keys.values() for key in pair])
        named: dict[str, tuple[str, str]] = {}
        for name, pair in keys.items():
            file = next((key for key in pair if key in found), None)
            if file is not None and len(found[file]) == 1:
                named[name] = (file, found[file][0])
        return named

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
            **({"similar_subjects": payload["similar_subjects"]} if "similar_subjects" in payload else {}),
        }
