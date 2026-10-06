"""Triage for `closure-marker-unlinked`: show the sentence, offer the edge, or record "not one".

The warning says a memo's prose names another memo next to "replaces" or "supersedes" while its
frontmatter declares nothing. Only a person can say whether that sentence is a supersession, so
this module never writes an edge. It finds, per warned memo, the sentence the linter matched, the
memos that sentence names (resolved to files, exactly one match each), and which way an edge would
point by the marker's voice, the rule `recall.fix` uses: "X replaces Y" puts `supersedes: Y` on X,
"X is replaced by Y" puts `supersedes: X` on Y. Declaring goes through the memo editor, with its
preview, digest check and undo.

"Not a supersession" is recorded in ``<root>/.recall/lint-dismissals.sqlite3`` against a hash of
the sentence. Editing the sentence changes the hash, so the warning comes back for the new text
instead of staying silenced for prose nobody reviewed.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from recall.dashboard.graph import _memo_files
from recall.document import parse_document
from recall.errors import RecallError
from recall.frontmatter import supersedes_key
from recall.lint import MEMO_REFERENCE, LintIssue, _sentence_around, closure_marker_naming_a_memo, lint_corpus, prose_only

CODE = "closure-marker-unlinked"
#: A memo linking at least this many others is an index page: its lines describe the memos they
#: link to, so an edge never belongs on the index itself.
INDEX_LINKS = 40
_PASSIVE = re.compile(r"(?:superseded|replaced)\s+by", re.IGNORECASE)
MAX_NOTE = 1000


class TriageRefused(ValueError, RecallError):
    """Nothing was recorded; the message says why."""


@dataclass(frozen=True)
class Finding:
    file: str
    marker: str
    sentence: str
    sentence_sha: str
    #: Memos the sentence names, resolved to files in this store.
    candidates: tuple[str, ...]
    #: True when the marker is passive ("replaced by X"): the edge belongs on the candidate.
    passive: bool
    #: The warned memo is an index page; a line there describes the memo it links to.
    index_page: bool
    dismissed: bool


def default_dismissal_path(root: Path) -> Path:
    return root / ".recall" / "lint-dismissals.sqlite3"


def sentence_digest(sentence: str) -> str:
    return hashlib.sha256(" ".join(sentence.split()).encode("utf-8")).hexdigest()[:16]


class DismissalLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self, write: bool) -> sqlite3.Connection | None:
        if not write and not self.path.exists():
            return None
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=10.0)
            connection.execute(
                "CREATE TABLE IF NOT EXISTS dismissals (file TEXT NOT NULL, code TEXT NOT NULL, "
                "sentence_sha TEXT NOT NULL, sentence TEXT NOT NULL, reviewer TEXT NOT NULL, note TEXT NOT NULL, "
                "dismissed_at TEXT NOT NULL, PRIMARY KEY (file, code, sentence_sha))"
            )
            return connection
        return sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, timeout=10.0)

    def dismissed(self) -> set[tuple[str, str]]:
        connection = self._connect(write=False)
        if connection is None:
            return set()
        try:
            return {(f, s) for f, s in connection.execute("SELECT file, sentence_sha FROM dismissals WHERE code = ?", (CODE,))}
        except sqlite3.Error:
            return set()
        finally:
            connection.close()

    def history(self) -> list[tuple[str, str, str, str, str]]:
        connection = self._connect(write=False)
        if connection is None:
            return []
        try:
            return list(connection.execute(
                "SELECT file, sentence, reviewer, note, dismissed_at FROM dismissals WHERE code = ? ORDER BY dismissed_at DESC", (CODE,)
            ))
        except sqlite3.Error:
            return []
        finally:
            connection.close()

    def record(self, finding: Finding, *, reviewer: str, note: str, at: datetime) -> None:
        connection = self._connect(write=True)
        assert connection is not None
        try:
            with connection:
                connection.execute(
                    "INSERT OR REPLACE INTO dismissals VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (finding.file, CODE, finding.sentence_sha, finding.sentence, reviewer, note, at.isoformat()),
                )
        finally:
            connection.close()

    def remove(self, file: str, sentence_sha: str) -> bool:
        connection = self._connect(write=True)
        assert connection is not None
        try:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM dismissals WHERE file = ? AND code = ? AND sentence_sha = ?", (file, CODE, sentence_sha)
                )
            return cursor.rowcount > 0
        finally:
            connection.close()


def closure_findings(root: Path, issues: list[LintIssue] | None = None) -> list[Finding]:
    """One finding per memo `recall lint` warns about, dismissed or not.

    `issues` is `lint_corpus(root)` when the caller already ran it; linting a store is the cost here.
    """
    root = root.resolve()
    names = [path.relative_to(root).as_posix() for path in _memo_files(root)]
    by_key: dict[str, list[str]] = {}
    for name in names:
        by_key.setdefault(supersedes_key(Path(name).name), []).append(name)

    def resolve(reference: str) -> str | None:
        text = reference.strip()
        if text.startswith("[[") and text.endswith("]]"):
            text = text[2:-2].split("|", 1)[0].split("#", 1)[0]
        elif text.startswith("]("):
            text = text[2:-1]
        matches = by_key.get(supersedes_key(Path(text.strip()).name), [])
        return matches[0] if len(matches) == 1 else None

    dismissed = DismissalLedger(default_dismissal_path(root)).dismissed()
    findings: list[Finding] = []
    for issue in lint_corpus(root) if issues is None else issues:
        if issue.code != CODE:
            continue
        try:
            prose = prose_only(parse_document((root / issue.file).read_text(encoding="utf-8-sig")).human_body)
        except (OSError, UnicodeDecodeError):
            continue
        marker = closure_marker_naming_a_memo(prose)
        if marker is None:
            continue
        sentence = " ".join(_sentence_around(prose, marker.start(), marker.end()).split())
        candidates: list[str] = []
        for reference in MEMO_REFERENCE.finditer(sentence):
            found = resolve(reference.group(0))
            if found and found != issue.file and found not in candidates:
                candidates.append(found)
        digest = sentence_digest(sentence)
        findings.append(Finding(
            file=issue.file, marker=marker.group(0), sentence=sentence, sentence_sha=digest,
            candidates=tuple(candidates), passive=bool(_PASSIVE.fullmatch(marker.group(0))),
            index_page=len(MEMO_REFERENCE.findall(prose)) >= INDEX_LINKS,
            dismissed=(issue.file, digest) in dismissed,
        ))
    return findings


def dismiss(root: Path, file: str, sentence_sha: str, *, reviewer: str, note: str, now: datetime) -> Finding:
    """Record that the sentence a person was shown is not a supersession."""
    reviewer = reviewer.strip()
    if not reviewer:
        raise TriageRefused("Say who decided: the reviewer field is required.")
    current = next((f for f in closure_findings(root) if f.file == file), None)
    if current is None:
        raise TriageRefused(f"{file} no longer carries this warning; nothing to dismiss.")
    if current.sentence_sha != sentence_sha:
        raise TriageRefused(f"The sentence in {file} changed since the page was shown; reload and look again.")
    DismissalLedger(default_dismissal_path(root.resolve())).record(current, reviewer=reviewer, note=note.strip()[:MAX_NOTE], at=now)
    return current


def restore(root: Path, file: str, sentence_sha: str) -> bool:
    """Take a dismissal back, so the warning shows again."""
    return DismissalLedger(default_dismissal_path(root.resolve())).remove(file, sentence_sha)
