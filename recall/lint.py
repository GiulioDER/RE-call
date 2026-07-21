"""Write-time completeness lint for the supersession graph.

Supersession is a RELATION between two memories, captured at write time (`supersedes:`
frontmatter). Its residual failure mode is authorial: a new memo that replaces an old one but
never declares the edge leaves an orphan that looks valid forever. Read-time code cannot catch
that — both memos look fine in isolation — but the omission IS lintable. Run `recall lint`
before indexing (or in CI over a memory corpus) to catch:

errors (break the trust layer's correctness):
- ``dangling-supersedes``  — the edge names a file that does not exist in the corpus
- ``self-supersedes``      — a document claims to supersede itself
- ``supersession-cycle``   — following the chain revisits a document
- ``invalid-date``         — malformed ``valid_from``/``valid_until`` (the Indexer would refuse it)
- ``ambiguous-supersedes-target`` — the edge names a basename carried by MORE than one document,
  so which document is superseded cannot be resolved
- ``ambiguous-supersedes-source`` — the declaring document's own basename is carried by more than
  one document, so which document is the successor cannot be resolved

  Both ambiguity codes are errors because read-time acts on them: `recall.trust` returns the
  ``ambiguous_supersession`` verdict and abstains rather than guess an endpoint. A corpus that
  lints clean must be one the engine will actually answer from.

warnings (smells that usually mean a missing or ambiguous edge):
- ``version-sibling-unlinked``     — ``x_v1.md`` / ``x_v2.md`` naming with no edge into the older one
- ``closure-marker-unlinked``      — body prose says superseded/replaced/deprecated but the
  frontmatter declares no relation and no validity window (the relation lives only in prose,
  where retrieval cannot act on it)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from recall.frontmatter import parse_frontmatter, validity_bounds

#: Prose that usually accompanies a closure/replacement decision. Deliberately short and
#: high-precision: a chatty list would drown real omissions in noise.
CLOSURE_MARKERS = re.compile(
    r"\b(superseded by|supersedes|replaced by|replaces|deprecated|obsolete)\b", re.IGNORECASE
)
_VERSION_STEM = re.compile(r"^(?P<stem>.+)_v(?P<num>\d+)$")

Level = Literal["error", "warning"]


@dataclass(frozen=True)
class LintIssue:
    file: str
    level: Level
    code: str
    message: str


def _find_cycles(graph: dict[str, list[str]]) -> list[frozenset[str]]:
    """Distinct cycles in a directed multigraph (iterative colored DFS, O(V+E))."""
    color: dict[str, int] = {}  # absent=white, 1=grey (on path), 2=black (done)
    cycles: list[frozenset[str]] = []
    seen_cycles: set[frozenset[str]] = set()
    for start in graph:
        if color.get(start):
            continue
        color[start] = 1
        stack = [(start, iter(graph.get(start, ())))]
        path = [start]
        while stack:
            node, edges = stack[-1]
            advanced = False
            for nxt in edges:
                c = color.get(nxt, 0)
                if c == 0:
                    color[nxt] = 1
                    stack.append((nxt, iter(graph.get(nxt, ()))))
                    path.append(nxt)
                    advanced = True
                    break
                if c == 1:  # back edge onto the current path: a cycle
                    members = frozenset(path[path.index(nxt):])
                    if members not in seen_cycles:
                        seen_cycles.add(members)
                        cycles.append(members)
            if not advanced:
                stack.pop()
                path.pop()
                color[node] = 2
    return cycles


#: Default file pattern — shared with the CLI so the two defaults cannot drift.
DEFAULT_GLOB = "**/*.md"


def lint_corpus(path: str | Path, glob: str = DEFAULT_GLOB) -> list[LintIssue]:
    """Lint a markdown corpus; returns issues sorted by (level, file). Empty list = clean.

    Raises FileNotFoundError on a nonexistent root. An individual unreadable file (bad UTF-8,
    permissions) becomes an ``unreadable-file`` error and the REST of the corpus is still
    linted — one broken file must not abort a CI run with zero issues reported.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"corpus path does not exist: {root}")
    files = sorted(root.glob(glob)) if root.is_dir() else [root]
    # keys are root-relative paths so same-named files in different directories cannot shadow
    # each other; `supersedes:` targets stay basenames (the frontmatter convention)
    rel = {f: (f.relative_to(root).as_posix() if root.is_dir() else f.name) for f in files}
    name_count: dict[str, int] = {}
    for f in files:
        name_count[f.name] = name_count.get(f.name, 0) + 1
    names = set(name_count)
    # superseded basename -> ALL files claiming to supersede it (a single-valued map would
    # drop edges on fan-in and could hide a declared cycle behind a third superseder)
    superseders: dict[str, list[str]] = {}
    issues: list[LintIssue] = []

    metas: dict[str, dict[str, str]] = {}
    bodies: dict[str, str] = {}
    readable: list[Path] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8-sig")
        except (UnicodeDecodeError, OSError) as exc:
            issues.append(LintIssue(rel[f], "error", "unreadable-file", str(exc)))
            continue
        readable.append(f)
        meta, body = parse_frontmatter(text)
        metas[rel[f]], bodies[rel[f]] = meta, body
        try:
            validity_bounds(meta)
        except ValueError as exc:
            issues.append(LintIssue(rel[f], "error", "invalid-date", str(exc)))
        target = meta.get("supersedes")
        if not target:
            continue
        if target == f.name:
            issues.append(
                LintIssue(rel[f], "error", "self-supersedes",
                          "a document cannot supersede itself")
            )
        elif target not in names:
            issues.append(
                LintIssue(rel[f], "error", "dangling-supersedes",
                          f"supersedes {target!r}, which does not exist in the corpus — "
                          f"the chain breaks here")
            )
        else:
            # Both endpoints matter: the trust layer withdraws the edge if EITHER basename is
            # carried by several documents (it cannot tell which one is superseded, or which
            # one is the successor). These are errors, not smells — read-time now refuses to
            # answer from the affected memory, so a clean lint would be a lie.
            if name_count[target] > 1:
                issues.append(
                    LintIssue(rel[f], "error", "ambiguous-supersedes-target",
                              f"supersedes {target!r}, but {name_count[target]} files share "
                              f"that basename — the reference cannot be resolved unambiguously, "
                              f"so retrieval will refuse to trust it (ambiguous_supersession)")
                )
            if name_count[f.name] > 1:
                issues.append(
                    LintIssue(rel[f], "error", "ambiguous-supersedes-source",
                              f"declares `supersedes: {target}` but {name_count[f.name]} files "
                              f"share this file's basename {f.name!r} — which document is the "
                              f"successor cannot be resolved, so retrieval will refuse to "
                              f"trust {target!r} (ambiguous_supersession)")
                )
            superseders.setdefault(target, []).append(f.name)

    for members in _find_cycles(superseders):
        issues.append(
            LintIssue(min(members), "error", "supersession-cycle",
                      "supersession chain forms a cycle: " + " -> ".join(sorted(members)))
        )

    superseded_targets = set(superseders)  # basenames some file claims to supersede
    # group version siblings by (stem, parsed int) — reconstructing "stem_v{n+1}" as a string
    # missed zero-padded series (x_v01/x_v02) and non-contiguous ones (x_v1/x_v3) entirely
    by_stem: dict[tuple[str, str], list[tuple[int, Path]]] = {}
    for f in readable:
        m = _VERSION_STEM.match(f.stem)
        if m:
            by_stem.setdefault((m["stem"], f.suffix), []).append((int(m["num"]), f))
    for versions in by_stem.values():
        versions.sort(key=lambda t: t[0])
        for (_, older), (_, newer) in zip(versions, versions[1:]):
            if older.name not in superseded_targets:
                issues.append(
                    LintIssue(newer.name, "warning", "version-sibling-unlinked",
                              f"{newer.name} looks like the successor of {older.name} but no "
                              f"file declares `supersedes: {older.name}` — retrieval will "
                              f"keep serving both as valid")
                )

    for f in readable:
        meta = metas[rel[f]]
        declares_relation = (
            "supersedes" in meta or "valid_until" in meta or f.name in superseded_targets
        )
        if declares_relation:
            continue
        hit = CLOSURE_MARKERS.search(bodies[rel[f]])
        if hit:
            issues.append(
                LintIssue(rel[f], "warning", "closure-marker-unlinked",
                          f"body says {hit.group(0)!r} but the frontmatter declares no "
                          f"supersession edge or validity window — the relation exists only "
                          f"in prose, where retrieval cannot act on it")
            )

    order = {"error": 0, "warning": 1}
    return sorted(issues, key=lambda i: (order[i.level], i.file, i.code))
