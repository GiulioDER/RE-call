"""Seed a project's corpus at install time, so the first session finds something.

Memory is worthless on day one and valuable on day thirty, which is the wrong shape for a trial.
A user who installs recall, opens Claude, asks it something and gets an abstention has learned that
recall returns nothing. The `SessionStart` digest is honest about an empty corpus and stays silent
on one, so without this step the whole integration is invisible on the first day.

**Transcripts are deliberately NOT seeded, and that reverses the original plan for this step.**
Indexing the user's existing Claude Code sessions looked like the strongest possible onboarding:
nobody else can do it, and it would fill the corpus with real project history. Measured on this
machine on 2026-08-19 before writing any of it:

    122 project directories, 4,011 transcripts, 1.8 GB total

and for a single worktree, 11 MB of transcript against 756 KB of durable documents in the same
project, a ratio of about 15 to 1. Re-measure with:

```bash
find ~/.claude/projects -name '*.jsonl' | wc -l && du -sh ~/.claude/projects
```

Three things follow, and together they retire the idea in this form. Embedding 1.8 GB locally is
hours of work at install time. The content is mostly tool output and conversation scaffolding, so
the nearest match to "what did we decide about X" is as likely to be a directory listing as a
decision. And a transcript holds every secret the user has ever pasted into a session, which is not
something to ingest as a default on the strength of a good intention.

What survives is the goal rather than the method: the valuable part of a transcript is the
decisions, not the turns, and extracting those is what `recall.truth_extraction` is for. That is a
separate feature with an LLM in the loop and a cost model, not a file walk, and it should be
measured on its own rather than smuggled in here.

So this seeds durable prose the project already maintains: `CLAUDE.md`, `memory/`, `docs/`, and the
top-level markdown. Small, high signal, already curated by a human, and free of anything the user
did not deliberately write down.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

#: Files worth seeding, relative to the project root. Order is priority order: when the byte
#: budget runs out, what is left out is whatever came last.
SEED_FILES: tuple[str, ...] = ("CLAUDE.md", "CLAUDE.local.md", "README.md")

#: Directories walked recursively, in priority order. `memory/` first because it is the corpus
#: this product is actually about; a user who has written memos wants those searchable before
#: they want their changelog searchable.
SEED_DIRS: tuple[str, ...] = ("memory", "docs")

#: Extensions seeded. Prose only. `recall.extraction` handles PDF, DOCX and the rest, and a
#: project's binaries are a deliberate choice for the user to make later rather than a default to
#: impose at install time.
SEED_SUFFIXES: frozenset[str] = frozenset({".md", ".txt", ".rst"})

#: Never descended into. `.claude` earns its place for a specific reason: a worktree lives at
#: `.claude/worktrees/<name>`, so descending would seed every OTHER checkout of this repository
#: through the current one, multiplying the corpus by the number of worktrees and attributing
#: every copy to this project.
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".claude",
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "site-packages",
        "venv",
    }
)

#: Budget, applied to the plan rather than to the walk, so what was dropped can be reported.
#: A first install must not spend ten minutes embedding a documentation monorepo before the user
#: has seen the product do anything.
MAX_SEED_BYTES = 8 * 1024 * 1024
MAX_SEED_FILES = 1000


@dataclass(frozen=True)
class SeedPlan:
    """What seeding WOULD index, computed before anything is read or embedded.

    Separating the plan from the act is what lets the wizard tell the user what it is about to
    ingest and get an answer first. It also feeds `Indexer.index_path(files=...)` the same set that
    was measured, rather than re-walking and indexing a set nobody counted.
    """

    root: Path
    files: tuple[Path, ...]
    total_bytes: int
    #: Files that matched but did not fit the budget. Reported, never silently dropped: a cap that
    #: says nothing reads as "everything was covered" when it was not.
    dropped: tuple[Path, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.files

    def describe(self) -> str:
        kb = self.total_bytes / 1024
        text = f"{len(self.files)} files, {kb:.0f} KB"
        if self.dropped:
            text += f" ({len(self.dropped)} more left out to stay inside the install budget)"
        return text


def _readable_size(path: Path) -> int | None:
    """The file's size, or None when it cannot be stated. Never raises.

    A file that cannot be stat'd is not a file that can be indexed, and an install must not abort
    because one path in a documentation tree is unreadable.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size if stat.st_size > 0 else None


def _walk(directory: Path) -> list[Path]:
    """Every seedable file under `directory`, skipping the directories that are never worth it."""
    found: list[Path] = []
    if not directory.is_dir():
        return found
    stack = [directory]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name not in SKIP_DIRS and not entry.is_symlink():
                    stack.append(entry)
            elif entry.suffix.lower() in SEED_SUFFIXES:
                found.append(entry)
    return found


def plan_seed(
    root: Path,
    *,
    max_bytes: int = MAX_SEED_BYTES,
    max_files: int = MAX_SEED_FILES,
) -> SeedPlan:
    """Decide what to seed from `root`, in priority order, inside a budget.

    Deterministic: the same tree yields the same plan, so a user who is shown a plan and accepts it
    gets the thing they were shown.
    """
    root = root.resolve()
    candidates: list[Path] = []
    for name in SEED_FILES:
        candidate = root / name
        if candidate.is_file():
            candidates.append(candidate)
    for name in SEED_DIRS:
        candidates.extend(_walk(root / name))
    for entry in sorted(root.glob("*.md")):
        if entry.is_file():
            candidates.append(entry)

    chosen: list[Path] = []
    dropped: list[Path] = []
    seen: set[Path] = set()
    total = 0
    for candidate in candidates:
        # Top-level `*.md` re-offers `CLAUDE.md` and `README.md`, which are already in by name.
        # De-duplicating here rather than in the sources keeps each source's rule simple and makes
        # the priority order the single thing that decides what survives the budget.
        if candidate in seen:
            continue
        seen.add(candidate)
        size = _readable_size(candidate)
        if size is None:
            continue
        if len(chosen) >= max_files or total + size > max_bytes:
            dropped.append(candidate)
            continue
        chosen.append(candidate)
        total += size
    return SeedPlan(root=root, files=tuple(chosen), total_bytes=total, dropped=tuple(dropped))


def seed_corpus(
    *,
    dsn: str,
    embedder_name: str,
    plan: SeedPlan,
    env: dict[str, str] | None = None,
    tenant: str | None = None,
    table: str | None = None,
    print_fn: Callable[..., None] = print,
) -> int:
    """Index a plan. Returns the number of chunks written, 0 on any failure.

    Best effort, like the rest of the wizard's post-setup work: the answers the user just gave are
    already persisted in `.env`, and a seeding failure must cost them a printed line rather than
    the install.

    `index_path(root, files=...)` rather than a call per source. One call means one walk and one
    prune pass, and passing the measured set is what the `files` parameter exists for. The prune is
    safe with a partial list because it asks the DISK whether a source is gone rather than
    inferring it from absence from `files`.
    """
    if plan.is_empty:
        print_fn(
            "Nothing to seed: no CLAUDE.md, memory/, docs/ or top-level markdown under "
            f"{plan.root}. The first session will have an empty corpus until something is indexed."
        )
        return 0
    try:
        # Imported here rather than at module scope, matching `index_memory_directory`. Resolving
        # an embedder can download a model, and this module is imported when the wizard starts,
        # long before anyone has agreed to seed anything. A caller that only wants `plan_seed`, to
        # show the user what WOULD happen, should pay nothing for the machinery that does it.
        from recall.embeddings import resolve_embedder
        from recall.store import redacted_dsn
        from recall.index import Indexer, chunk_text
        from recall.store import DEFAULT_TABLE, DEFAULT_TENANT, PgVectorStore

        embedder = resolve_embedder(embedder_name, env=env)
        with PgVectorStore(
            dsn,
            dim=embedder.dim,
            table=table or DEFAULT_TABLE,
            tenant=tenant or DEFAULT_TENANT,
        ) as store:
            store.check_schema()
            indexer = Indexer(store, embedder, chunker=chunk_text)
            stats = indexer.index_path(plan.root, files=list(plan.files))
    except Exception as exc:
        # ⚠️ The exception text can carry the DSN verbatim, password included. A MALFORMED dsn
        # makes psycopg echo the whole connection string back: `missing "=" after
        # "postgresql://user:PASSWORD@host" in connection info string`. The three WELL-FORMED
        # failures (unreachable port, bad host, wrong password) are all clean, which is exactly
        # what makes it easy to miss. Found by the wizard session in its own preflight.
        #
        # Replacing the known DSN handles the echo, which is the observed leak. It is not a
        # general scrubber: `recall.store.scrub_dsn_secrets` is that, and it lands with #434, at
        # which point this should call it instead.
        print_fn(
            f"Could not seed the corpus: {str(exc).replace(dsn, redacted_dsn(dsn))} — run "
            f"'python -m recall.cli index {plan.root}' once the schema is applied for this "
            "embedder's dimension."
        )
        return 0
    print_fn(f"Seeded {stats.chunks} chunks from {stats.files} files under {plan.root}")
    if plan.dropped:
        print_fn(
            f"{len(plan.dropped)} further files were left out to keep the install quick. "
            f"Index them with 'python -m recall.cli index {plan.root}'."
        )
    return int(stats.chunks)


__all__ = [
    "MAX_SEED_BYTES",
    "MAX_SEED_FILES",
    "SEED_DIRS",
    "SEED_FILES",
    "SEED_SUFFIXES",
    "SeedPlan",
    "plan_seed",
    "seed_corpus",
]
