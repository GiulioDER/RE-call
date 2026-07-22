"""Write-time gate: ask for the supersession edge while the author still knows the answer.

`recall lint` finds 60 memos in a real corpus whose prose says *"superseded by X"* while the
frontmatter declares nothing, and `recall lint --fix` proved that **none of them** can be
resolved mechanically after the fact: the distinctions that matter — narrating versus declaring,
part versus whole, augmenting versus replacing — are invisible to a pattern and obvious to the
person writing. By the time a linter sees the memo, the one who knew has moved on.

So this runs at the other end: on the file being written, before it lands.

**The trade-off inverts.** `--fix` refuses everything it cannot prove, because it writes
unattended. This SURFACES every candidate it can find, because a human is right there to pick
one — a false candidate costs a glance, while a missing one costs the edge. Same extraction,
opposite disposition, and the reason is who is in the room.

Meant for a pre-commit hook or an editor save action; `--strict` makes it block.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from recall.fix import _REF, _is_index
from recall.frontmatter import parse_frontmatter, supersedes_key
from recall.lint import CLOSURE_MARKERS, DEFAULT_GLOB

#: Any document reference anywhere in the body — no marker proximity required. At write time the
#: author picks, so a wide net beats a precise one.
_ANY_REF = re.compile(_REF)


@dataclass(frozen=True)
class CheckResult:
    file: str
    marker: str            # the closure phrase that triggered the prompt
    candidates: list[str]  # document names named in the body, best guesses first
    declared: bool         # already has supersedes: or valid_until:

    @property
    def needs_attention(self) -> bool:
        return bool(self.marker) and not self.declared


def _marker_in(body: str) -> str:
    m = CLOSURE_MARKERS.search(body)
    return m.group(0) if m else ""


def check_file(path: str | Path, corpus_names: set[str] | None = None) -> CheckResult:
    """Inspect one memo. `corpus_names` (stems) filters candidates to real documents."""
    p = Path(path)
    meta, body = parse_frontmatter(p.read_text(encoding="utf-8-sig", errors="replace"))
    declared = bool(meta.get("supersedes") or meta.get("valid_until"))
    marker = "" if _is_index(p.name) else _marker_in(body)

    seen: list[str] = []
    for m in _ANY_REF.finditer(body):
        ref = next((g for g in m.groups() if g), "").strip()
        key = supersedes_key(ref)
        if not key or key == supersedes_key(p.name) or key in seen:
            continue
        if corpus_names is not None and key not in corpus_names:
            continue
        seen.append(key)
    return CheckResult(p.name, marker, seen, declared)


def corpus_names(corpus_dir: str | Path, glob: str = DEFAULT_GLOB) -> set[str]:
    root = Path(corpus_dir)
    return {supersedes_key(f.name) for f in root.glob(glob)}


def format_prompt(r: CheckResult) -> str:
    """The message an author sees at commit time — a question with the answer pre-filled."""
    lines = [
        f"{r.file}: says {r.marker!r} but declares no supersession edge or validity window.",
        "    The relation will exist only in prose, where retrieval cannot act on it.",
    ]
    if r.candidates:
        lines.append("    Documents this memo mentions — add whichever it replaces:")
        lines.extend(f"        supersedes: {c}" for c in r.candidates[:5])
    else:
        lines.append("    Add `supersedes: <name>`, or `valid_until: YYYY-MM-DD` if it expires.")
    lines.append("    If it merely relates to them, leave it — an augmenting memo is not a"
                 " successor.")
    return "\n".join(lines)
