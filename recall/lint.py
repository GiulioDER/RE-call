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

warnings (smells that usually mean a missing edge):
- ``version-sibling-unlinked`` — ``x_v1.md`` / ``x_v2.md`` naming with no edge into the older one
- ``closure-marker-unlinked``  — body prose says superseded/replaced/deprecated but the
  frontmatter declares no relation and no validity window (the relation lives only in prose,
  where retrieval cannot act on it)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from recall.frontmatter import parse_frontmatter, validity_bounds

#: Prose that usually accompanies a closure/replacement decision. Deliberately short and
#: high-precision: a chatty list would drown real omissions in noise.
CLOSURE_MARKERS = re.compile(
    r"\b(superseded by|supersedes|replaced by|replaces|deprecated|obsolete)\b", re.IGNORECASE
)
_VERSION_STEM = re.compile(r"^(?P<stem>.+)_v(?P<num>\d+)$")


@dataclass(frozen=True)
class LintIssue:
    file: str
    level: str  # "error" | "warning"
    code: str
    message: str


def lint_corpus(path: str | Path, glob: str = "**/*.md") -> list[LintIssue]:
    """Lint a markdown corpus; returns issues sorted by (level, file). Empty list = clean."""
    root = Path(path)
    files = sorted(root.glob(glob)) if root.is_dir() else [root]
    names = {f.name for f in files}
    supersedes: dict[str, str] = {}
    issues: list[LintIssue] = []

    metas: dict[str, dict[str, str]] = {}
    bodies: dict[str, str] = {}
    for f in files:
        meta, body = parse_frontmatter(f.read_text(encoding="utf-8-sig"))
        metas[f.name], bodies[f.name] = meta, body
        try:
            validity_bounds(meta)
        except ValueError as exc:
            issues.append(LintIssue(f.name, "error", "invalid-date", str(exc)))
        target = meta.get("supersedes")
        if not target:
            continue
        if target == f.name:
            issues.append(
                LintIssue(f.name, "error", "self-supersedes",
                          "a document cannot supersede itself")
            )
        elif target not in names:
            issues.append(
                LintIssue(f.name, "error", "dangling-supersedes",
                          f"supersedes {target!r}, which does not exist in the corpus — "
                          f"the chain breaks here")
            )
        else:
            supersedes[target] = f.name

    # cycles: walk each chain once; a revisit is a cycle (report at the first file seen)
    reported_cycles: set[frozenset[str]] = set()
    for start in supersedes:
        seen: list[str] = [start]
        cur = start
        while cur in supersedes:
            cur = supersedes[cur]
            if cur in seen:
                members = frozenset(seen[seen.index(cur):])
                if members not in reported_cycles:
                    reported_cycles.add(members)
                    issues.append(
                        LintIssue(min(members), "error", "supersession-cycle",
                                  "supersession chain forms a cycle: "
                                  + " -> ".join(sorted(members)))
                    )
                break
            seen.append(cur)

    superseded_targets = set(supersedes)  # files something points at
    for f in files:
        m = _VERSION_STEM.match(f.stem)
        if not m:
            continue
        newer = f"{m['stem']}_v{int(m['num']) + 1}{f.suffix}"
        if newer in names and f.name not in superseded_targets:
            issues.append(
                LintIssue(newer, "warning", "version-sibling-unlinked",
                          f"{newer} looks like the successor of {f.name} but declares no "
                          f"`supersedes: {f.name}` — retrieval will keep serving both as valid")
            )

    for f in files:
        meta = metas[f.name]
        declares_relation = (
            "supersedes" in meta or "valid_until" in meta or f.name in superseded_targets
        )
        if declares_relation:
            continue
        hit = CLOSURE_MARKERS.search(bodies[f.name])
        if hit:
            issues.append(
                LintIssue(f.name, "warning", "closure-marker-unlinked",
                          f"body says {hit.group(0)!r} but the frontmatter declares no "
                          f"supersession edge or validity window — the relation exists only "
                          f"in prose, where retrieval cannot act on it")
            )

    order = {"error": 0, "warning": 1}
    return sorted(issues, key=lambda i: (order[i.level], i.file, i.code))
