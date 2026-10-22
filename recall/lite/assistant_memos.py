"""A folder of one-fact memos and its authorship ledger: the file half of the assistant surface.

`recall_mcp.assistant` serves this over MCP. The file writes live here, outside `recall_mcp`, which
makes no file write call of its own (`tests/test_mcp_rewrite_plan.py`); the server reaches the
folder only through these methods, and only the folder RECALL_ASSISTANT_MEMOS names.

What it writes, and nothing else:

- one memo file per `write`, created exclusively (never over an existing file), with frontmatter
  `valid_from`, `valid_until`, `supersedes` and `recall_graph` written by this module, never taken
  from the caller's text;
- the authorship ledger `.recall-authors.json` in the folder, replaced atomically; no index reads it;
- the deletion of one memo file of the folder by `delete`, after its rows are forgotten.

Memo ids are `<kind>-<number>`, the number at least four digits and never reused: the ledger keeps
every id ever written, forgotten ones included, and a new id is one past the largest in the ledger
or the folder.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from recall.frontmatter import validity_bounds
from recall.lite.memo_requests import MEMO_ID, cancel_requests, restore_requests, write_memo

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.lite import LiteStore
    from recall.lite.refresh import RefreshReport

__all__ = ["LEDGER_NAME", "MEMO_ID", "MemoFolder"]

LEDGER_NAME = ".recall-authors.json"


def _opens_a_fence(text: str) -> bool:
    """Whether the memo body written for `text` would open with a frontmatter fence.

    It is the opening test of `recall.frontmatter.frontmatter_span` (any number of leading BOMs
    removed, then whitespace, and the line must be exactly `---`), run on the first line exactly as
    `write` puts it in the file. A lighter check such as `text.lstrip().startswith("---")` misses a
    BOM before the fence, which the parser forgives and `str.strip` keeps.
    """
    first = text.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n", 1)[0]
    return first.lstrip("\ufeff").strip() == "---"


class MemoFolder:
    """One folder of one-fact memos and the ledger of who wrote each."""

    def __init__(self, folder: str | Path, *, index_root: str | Path | None = None) -> None:
        self.folder = Path(folder).resolve()
        self.index_root = Path(index_root).resolve() if index_root is not None else self.folder
        self.ledger_path = self.folder / LEDGER_NAME

    # -- authorship -------------------------------------------------------------------------------
    def authors(self) -> dict[str, str]:
        """Memo id -> who wrote it, for every memo this surface ever wrote (forgotten ones too)."""
        if not self.ledger_path.exists():
            return {}
        data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.get("authors", {}).items()}

    # -- writes -----------------------------------------------------------------------------------
    def write(
        self,
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
    ) -> dict[str, Any]:
        """Write one memo file without indexing it. Returns {memo_id} or {refused: reason}.

        The checks run in this order, and each refusal writes nothing: text that opens with a
        frontmatter delimiter (it would be read as the memo's own frontmatter), the dates, the
        structured fact, then whether `supersedes` names a memo the assistant wrote. Hold the store's write
        lane around this and the refresh after it (`remember` does), so two writers cannot take
        one id.
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if _opens_a_fence(text):
            # A memo written without frontmatter of its own would otherwise begin with the
            # caller's block, read as frontmatter: a `supersedes` the author check never saw, or a
            # `recall_graph` fact that was never validated.
            return {"refused": "text may not begin with a frontmatter delimiter (---)"}
        kind = re.sub(r"[^a-z]", "", kind.lower()) or "note"
        for label, value in (("valid_from", valid_from), ("valid_until", valid_until)):
            if value is not None:
                try:
                    date.fromisoformat(value)
                except ValueError:
                    return {"refused": f"{label} must be a date YYYY-MM-DD, got {value!r}"}
        try:
            # The reader's own rule, so the refresh never meets a memo it refuses: an empty window
            # (`valid_from` after `valid_until`) fails the indexer like a malformed date.
            validity_bounds({"valid_from": valid_from, "valid_until": valid_until})
        except ValueError as exc:
            return {"refused": str(exc)}
        graph = None
        if fact is not None:
            from recall.types import AtomicFact

            if not isinstance(fact, dict):
                return {"refused": "fact must be an object with subject, predicate and object"}
            try:
                graph = {"facts": [AtomicFact.from_payload(fact).to_payload()]}
            except (TypeError, ValueError, KeyError) as exc:
                return {"refused": f"fact is not a valid structured fact: {exc}"}
        authors = self.authors()
        if supersedes is not None:
            if supersedes not in authors:
                return {"refused": f"no memo {supersedes!r}"}
            if authors[supersedes] != "assistant":
                return {"refused": f"{supersedes} was not written by the assistant; it cannot be superseded"}
        front = []
        if valid_from:
            front.append(f"valid_from: {valid_from}")
        if valid_until:
            front.append(f"valid_until: {valid_until}")
        if supersedes:
            front.append(f"supersedes: {supersedes}.md")
        if graph:
            front.append("recall_graph: " + json.dumps(graph, ensure_ascii=False))
        body = ("---\n" + "\n".join(front) + "\n---\n" if front else "") + text.strip() + "\n"
        return write_memo(
            self.folder, self.ledger_path, body, kind=kind, author=author, supersedes=supersedes,
            request_id=request_id, expected_revision=expected_revision, include_receipt=include_receipt,
        )

    def refresh(self, store: LiteStore, embedder: Embedder, **kwargs: Any) -> RefreshReport:
        """Index the folder now, through the lite refresh the server itself runs, keyed to the index root.

        Raises when the refresh does not end REFRESHED or UNCHANGED: a memo that was written and
        silently left unindexed would read, to the model, as a fact it never stored.
        """
        from recall.lite.refresh import Outcome, refresh

        report = refresh(store, [self.folder], embedder, force=True, key_root=self.index_root, **kwargs)
        if report.outcome not in (Outcome.REFRESHED, Outcome.UNCHANGED):
            raise RuntimeError(f"memo refresh {report.outcome.value}: {report.reason}")
        return report

    def path_of(self, memo_id: str) -> Path | None:
        """The file of a memo id of this folder, or None for a string that is not a memo id."""
        return self.folder / f"{memo_id}.md" if MEMO_ID.match(memo_id) else None

    def delete(self, memo_id: str) -> None:
        """Delete one memo file of this folder. Forget its rows first (`recall_mcp.assistant`)."""
        path = self.path_of(memo_id)
        if path is None:
            raise ValueError(f"{memo_id!r} is not a memo id")
        previous = cancel_requests(self.ledger_path, memo_id)
        try:
            path.unlink()
        except OSError:
            # The file is still there (a Windows editor holding it, say): the request it came from
            # must stay finishable rather than read as forgotten.
            if path.exists():
                restore_requests(self.ledger_path, memo_id, previous)
            raise
