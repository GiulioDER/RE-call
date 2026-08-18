"""Fail when a `path:line` citation in the docs no longer points at what it pointed at.

Filed as follow up work in `docs/preregistrations/2026-08-18-uncalibrated-first-run.md`, which
states the problem: **line numbers in this repository drift faster than a document can be written.**
Measured there, one merge moved 13 of 13 checked citations while a draft was in progress, and two
further merges moved one each. Measured again on 2026-08-18, a single commit (`79a0d6ed`) shifted
`recall/index.py` by 26 lines and broke 23 citations across 5 documents at once. A citation that
silently stops resolving is worse than none, because it reads as evidence while pointing at
unrelated code.

## Why this asks git rather than reading the line

The obvious design is the one that document proposes: match each citation against a quoted anchor
nearby. **I built that first and measured it, and it does not work on this corpus.** It produced 33
findings against a tree whose citations had just been repaired by hand, and the great majority were
correct citations flagged wrongly, because a documentation line routinely carries several backticked
symbols and several citations and no reliable pairing between them. A check with thirty false alarms
is not a strict check, it is a check somebody switches off, and then the coverage is gone. The
numbers are in `tests/test_doc_citations.py`, which pins the reasoning so it is not rediscovered.

So this asks the only thing that knows the answer exactly. For each document it takes the last
commit that touched that document, and asks git how the cited file changed between that commit and
HEAD. Git reports which lines moved and by how much, so drift becomes arithmetic rather than
inference:

* the line moved  -> FAIL, naming the line it moved to, so the repair is copying one number;
* the line's own content was edited or deleted -> FAIL, because the citation may now be pointing at
  something that no longer says what was cited;
* the file did not change since the document was written -> nothing to report.

That last clause is what makes the check quiet. Editing a document re baselines its citations,
which is correct: an author who touches a document has had the chance to look at them.

## What it does not do

It cannot see uncommitted edits, because it compares committed trees. It needs real history, so CI
must check out with `fetch-depth: 0`; with a shallow clone it says so and fails rather than passing
vacuously. And it says nothing about whether a citation was ever right, only whether it still means
what it meant, which is the failure this repository actually keeps hitting.

Run it exactly as CI does:

    python scripts/check_doc_citations.py
"""
from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Same directory (`scripts/`), so this resolves when run as CI runs it:
# `python scripts/check_doc_citations.py` puts `scripts/` on `sys.path[0]`. Importing rather than
# restating the list is deliberate; the reasoning is at FROZEN_PREFIXES' use below.
from check_citation_anchors import FROZEN_PREFIXES

REPO = Path(__file__).resolve().parent.parent
#: Directories whose citations are never asked to be current, shared with the anchor checker.
#:
#: Imported rather than restated. `scripts/check_citation_anchors.py` defines this and says why the
#: list must be single-sourced: "which documents are exempt from being asked at all is one fact
#: about a document, not two. Two lists would drift, and the drift would be silent in the direction
#: that matters." That warning described a hazard; until now the two checkers ACTUALLY disagreed
#: about `docs/preregistrations/`, and this closes it.
#:
#: The consequence of the gap was concrete, and #396 documented it while working around it: #395
#: had to repoint five citations inside a pre-registration, four of them above `## Result` and one
#: inside "What I predict", purely to keep THIS checker green. The anchor checker never asked for
#: those edits, because its own comments hold that "a gate must never be able to force an edit to
#: an immutable record". #396 named teaching this checker the same prefix as the better fix and
#: left it undone because it is a change to a checker rather than to a list. This is that change.
#:
#: A pre-registration's `path:line` is evidence of what was true when it was written, so a line
#: that has moved makes the citation HISTORICAL, not wrong. `frozen_above` entries in the policy
#: stay meaningful for documents that want a live zone below a marker; this prefix decides whether
#: the document is asked at all.
POLICY = REPO / "docs" / "citation-policy.toml"

#: A citation is a repo relative path (it must carry a directory, so a bare `cli.py:487` shorthand
#: is left alone) plus a line or line range, in backticks.
CITATION = re.compile(
    r"`(?P<path>[A-Za-z0-9_][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_.-]+)+"
    r"\.(?:py|sql|toml|yml|yaml|sh)):(?P<start>\d+)(?:-(?P<end>\d+))?`"
)
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class Shallow(SystemExit):
    """A shallow clone cannot answer the question, and must not answer it with silence."""


@dataclass(frozen=True)
class Finding:
    doc: str
    doc_line: int
    citation: str
    detail: str

    def render(self) -> str:
        return f"{self.doc}:{self.doc_line}  `{self.citation}`\n    {self.detail}"


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, check=False
    )
    if result.returncode != 0:
        return ""
    return result.stdout.decode("utf-8", errors="replace")


def load_policy() -> tuple[list[dict], list[dict]]:
    if not POLICY.exists():
        return [], []
    data = tomllib.loads(POLICY.read_text(encoding="utf-8"))
    return data.get("exempt", []), data.get("frozen_above", [])


def frozen_line(doc: Path, rel: str, zones: list[dict]) -> int | None:
    """The document line below which citations are live, or None when there is no frozen zone.

    Refuses loudly when a declared marker has vanished. A policy naming a marker that no longer
    exists would silently stop protecting a frozen zone, or silently exempt a live one, and both
    failures look exactly like a passing check.
    """
    for zone in zones:
        if zone["path"] != rel:
            continue
        for i, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
            if line.startswith(zone["marker"]):
                return i
        raise SystemExit(
            f"citation policy names a frozen-zone marker that is not in {rel}: "
            f"{zone['marker']!r}. Update docs/citation-policy.toml, or remove the entry. Refusing "
            f"to check this document while the policy disagrees with it."
        )
    return None


def line_map(path: str, base: str) -> dict[int, int | None] | None:
    """How lines of `path` moved between `base` and the working tree, as {old: new or None}.

    None as a value means the line itself was changed or removed, which is a stronger finding than
    a move: whatever was cited may no longer be there at all. A None RETURN means the file did not
    change, so every citation into it is still good.
    """
    diff = git("diff", "--unified=0", "--no-color", f"{base}..HEAD", "--", path)
    if not diff.strip():
        return None

    hunks: list[tuple[int, int, int, int]] = []
    for line in diff.splitlines():
        m = HUNK.match(line)
        if m:
            old_start, old_count, new_start, new_count = (
                int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1)
            )
            hunks.append((old_start, old_count, new_start, new_count))
    return {"hunks": hunks}  # type: ignore[return-value]


def resolve(old_line: int, hunks: list[tuple[int, int, int, int]]) -> int | None:
    """Where `old_line` ended up, or None when the line itself was edited or deleted."""
    offset = 0
    for old_start, old_count, _new_start, new_count in hunks:
        if old_line < old_start:
            break
        if old_start <= old_line < old_start + old_count:
            return None  # inside a changed hunk: the cited content itself moved or went away
        offset += new_count - old_count
    return old_line + offset


def check() -> tuple[list[Finding], list[str]]:
    exempt, zones = load_policy()
    docs = sorted({p for p in (REPO / "docs").rglob("*.md")} | set(REPO.glob("*.md")))
    findings: list[Finding] = []
    skipped: list[str] = []
    cache: dict[tuple[str, str], list[tuple[int, int, int, int]] | None] = {}

    for doc in docs:
        rel = doc.relative_to(REPO).as_posix()
        if any(fnmatch.fnmatch(rel, rule["path"]) for rule in exempt):
            continue

        # An explicit `[[frozen_above]]` zone BEATS the blanket prefix skip, and the ORDER of these
        # two statements is the whole reason they are written out rather than folded together.
        #
        # The prefix says "this whole document is a record". A zone says something stronger and more
        # specific: the head is a record, and the tail below the marker is live and must keep failing
        # the build. Skipping on the prefix first therefore discards the tail's coverage, and discards
        # it SILENTLY. The citations simply stop being read;
        # `test_the_policy_declares_the_frozen_zone_and_its_marker_still_exists` keeps passing because
        # it calls `frozen_line` directly; and the policy file goes on describing a live zone that
        # nothing checks.
        #
        # Measured on the tree before this change, in a throwaway repository: a stale citation in a
        # zone the policy declared LIVE produced **zero** findings. Both `[[frozen_above]]` entries
        # live under `docs/preregistrations/`, so the entire mechanism was unreachable.
        # `test_a_declared_live_zone_is_still_checked_under_a_frozen_prefix` pins it.
        boundary = frozen_line(doc, rel, zones)
        if boundary is None and rel.startswith(FROZEN_PREFIXES):
            continue

        # ⚠️ A document with uncommitted edits cannot be checked, and this is not a shortcut.
        # The baseline is the last commit that TOUCHED the document, so citations sitting in the
        # working tree are newer than that baseline, and mapping them forward from it reports
        # drift for citations that were just repaired against the current tree. Measured while
        # building this: six citations corrected by hand were all reported stale, and all six
        # findings vanished on commit. CI always has a clean tree, so nothing is lost there.
        if git("status", "--porcelain", "--", rel).strip():
            skipped.append(rel)
            continue

        base = git("log", "-1", "--format=%H", "--", rel).strip()
        if not base:
            continue  # never committed, or no history for it; nothing to compare against

        for doc_line, text in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
            if boundary is not None and doc_line < boundary:
                continue  # registered predictions: their citations are history, not pointers
            for m in CITATION.finditer(text):
                findings.extend(check_one(rel, doc_line, m, base, cache))
    return findings, skipped


def check_one(
    rel: str, doc_line: int, m: re.Match, base: str,
    cache: dict[tuple[str, str], list[tuple[int, int, int, int]] | None],
) -> list[Finding]:
    path, cited = m.group("path"), m.group(0).strip("`")
    source = REPO / path
    if not source.exists():
        return [Finding(rel, doc_line, cited, f"{path} does not exist")]

    body = source.read_text(encoding="utf-8", errors="replace").splitlines()
    start, end = int(m.group("start")), int(m.group("end") or m.group("start"))
    if start > len(body):
        return [Finding(rel, doc_line, cited, f"{path} has only {len(body)} lines")]

    key = (path, base)
    if key not in cache:
        raw = line_map(path, base)
        cache[key] = None if raw is None else raw["hunks"]  # type: ignore[index]
    hunks = cache[key]
    if hunks is None:
        return []  # file unchanged since this document was last touched

    moved = resolve(start, hunks)
    if moved is None:
        return [Finding(rel, doc_line, cited,
                        f"line {start} of {path} was edited or removed after this document was "
                        f"last committed ({base[:8]}), so the citation may no longer name what it "
                        f"named. Re read it and write the current line.")]
    if moved != start:
        if moved > len(body):
            # The arithmetic ran off the end of the file, which means the diff was a rewrite
            # rather than a shift. Saying "cite line 1937" of a 1247 line file would be worse than
            # saying nothing, so this reports the drift and declines to invent a destination.
            return [Finding(rel, doc_line, cited,
                            f"{path} was rewritten rather than shifted since {base[:8]} (it now "
                            f"has {len(body)} lines), so where line {start} went cannot be "
                            f"computed. Re read the passage and write the current line.")]
        span = f"{moved}-{moved + (end - start)}" if end != start else str(moved)
        return [Finding(rel, doc_line, cited,
                        f"the content at line {start} moved {moved - start:+d} since {base[:8]}, "
                        f"so it is now at `{path}:{span}`. Confirm before copying it in: that is "
                        f"the right answer only if this citation was accurate when the document "
                        f"was last committed, and this repository contains citations that were "
                        f"already wrong before they drifted.")]
    return []


def main() -> int:
    if (REPO / ".git" / "shallow").exists():
        raise Shallow(
            "this is a shallow clone, so git cannot say how the cited files moved. Check out with "
            "fetch-depth: 0 (CI) or run `git fetch --unshallow` locally. Refusing to report a pass "
            "that was never measured."
        )
    findings, skipped = check()
    for rel in skipped:
        print(f"skipped {rel}: uncommitted changes, so its baseline is unknowable",
              file=sys.stderr)
    for finding in findings:
        print(finding.render(), file=sys.stderr)
    if findings:
        print(f"\n{len(findings)} stale citation(s). Each names the line it moved to, so the "
              f"repair is copying one number. If a citation is deliberately frozen, declare it in "
              f"docs/citation-policy.toml rather than updating it.", file=sys.stderr)
        return 1
    print("every citation still points where it pointed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
