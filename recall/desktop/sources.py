"""Source classification and safe file selection for drag and drop."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from recall.extraction import DOCUMENT_EXTENSIONS as EXTRACTABLE_DOCUMENT_EXTENSIONS
from recall.desktop.models import SourceCategory

DOCUMENT_EXTENSIONS = EXTRACTABLE_DOCUMENT_EXTENSIONS
CODE_EXTENSIONS = frozenset({".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".rb", ".c", ".h", ".cpp", ".hpp", ".cs", ".sh", ".sql"})

SCAN_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "site-packages",
        "venv",
    }
)
CLAUDE_MEMORY_FILENAMES = frozenset({"claude.md", "claude.local.md", "memory.md"})
#: ⚠️ `.jsonl` is deliberately absent. Claude Code's session transcripts are `.jsonl`, they are the
#: bulk of that directory by an order of magnitude (measured 2026-08-19: 4,011 files, 1.8 GB, about
#: 15:1 against the same project's durable documents), and they hold every secret the user has ever
#: pasted into a session. Nothing here is currently reachable through it, because `.jsonl` is in
#: neither `DOCUMENT_EXTENSIONS` nor `CODE_EXTENSIONS` and the suffix filter runs first, but the
#: entry stated an intention that would have taken effect the moment the extractable set grew.
CLAUDE_MEMORY_EXTENSIONS = frozenset({".md", ".txt", ".json", ".yaml", ".yml"})

#: Where Claude keeps its OWN configuration, relative to the user's home directory. Written once
#: and read by both the scan roots and the restriction below, because two copies of this list
#: drifting apart is a restriction that silently stops covering a directory.
CLAUDE_CONFIG_SUBPATHS = (
    (".claude",),
    (".config", "claude"),
    ("AppData", "Roaming", "Claude"),
    ("AppData", "Local", "Claude"),
)


def category_extensions(category: SourceCategory) -> frozenset[str]:
    if category is SourceCategory.CODE:
        return CODE_EXTENSIONS
    return frozenset(DOCUMENT_EXTENSIONS - CODE_EXTENSIONS)


def collect_files(paths: list[str | Path], category: SourceCategory | None = None) -> tuple[Path, ...]:
    allowed = (
        category_extensions(category)
        if category is not None
        else DOCUMENT_EXTENSIONS | CODE_EXTENSIONS
    )
    result: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            candidates = sorted(item for item in path.rglob("*") if item.is_file())
        else:
            candidates = [path]
        for candidate in candidates:
            if candidate.suffix.lower() in allowed and candidate not in result:
                result.append(candidate)
    return tuple(result)


def default_scan_roots(home: str | Path | None = None) -> tuple[Path, ...]:
    """Return the small set of user folders that a local scan is allowed to inspect."""
    base = Path(home).expanduser() if home is not None else Path.home()
    candidates = (
        base / "Documents",
        base / "Desktop",
        base / "Downloads",
        *(base.joinpath(*parts) for parts in CLAUDE_CONFIG_SUBPATHS),
    )
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        key = str(candidate.resolve()).casefold()
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return tuple(result)


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def claude_config_directories(home: str | Path | None = None) -> tuple[Path, ...]:
    """The directories holding Claude's own configuration, resolved for prefix comparison."""
    base = Path(home).expanduser() if home is not None else Path.home()
    result: list[Path] = []
    seen: set[str] = set()
    for parts in CLAUDE_CONFIG_SUBPATHS:
        resolved = _resolved(base.joinpath(*parts))
        key = str(resolved).casefold()
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return tuple(result)


def _claude_config_base(resolved: Path, config_directories: tuple[Path, ...]) -> Path | None:
    """The Claude configuration directory this file lies inside, or None.

    ⚠️ **A prefix test against the REAL configuration directories, not a substring test.** This
    used to ask whether any component of the path was named `.claude`, which is true of any project
    that happens to live under one — including every checkout made by this repository's own
    documented worktree workflow, `<repo>/.claude/worktrees/<name>`. Such a checkout was classified
    as a Claude config folder and restricted to memory files, so a scan of an ordinary project
    returned almost nothing and looked like a project with no documents rather than like a filter
    that had fired. Measured on this worktree before the fix: scanning `docs/` found **0 of 86**
    markdown files.

    Applied per FILE rather than per root, which is strictly wider than the test it replaces: a
    scan rooted at the user's home directory now restricts what is under `~/.claude` instead of
    treating the whole home as unrestricted because its own name is not `claude`.
    """
    for base in config_directories:
        if resolved == base or base in resolved.parents:
            return base
    return None


def _is_claude_memory_file(path: Path, root: Path) -> bool:
    relative_parts = {part.casefold() for part in path.relative_to(root).parts[:-1]}
    filename = path.name.casefold()
    return (
        filename in CLAUDE_MEMORY_FILENAMES
        or bool(relative_parts & {"memory", "memories"})
    ) and path.suffix.casefold() in CLAUDE_MEMORY_EXTENSIONS


def scan_files(
    roots: Iterable[str | Path],
    category: SourceCategory | None = None,
    *,
    max_files: int = 5000,
    home: str | Path | None = None,
) -> tuple[Path, ...]:
    """Find supported local sources while pruning generated and private dependency folders.

    Files inside Claude's own configuration directories are restricted to named memory files and
    memory directories, so a scan does not ingest settings, credentials, caches, or conversation
    transcripts by accident. That restriction is a prefix test against the real configuration
    locations: see `_claude_config_base` for what it replaced and why.

    `home` exists so the restriction can be pointed at a temporary directory in tests. Left unset
    it reads the real home, which is what the UI wants.
    """
    allowed = category_extensions(category) if category is not None else DOCUMENT_EXTENSIONS | CODE_EXTENSIONS
    config_directories = claude_config_directories(home)
    result: list[Path] = []
    seen: set[str] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if not root.is_dir():
            continue
        for current, directories, filenames in os.walk(root, topdown=True, onerror=lambda _error: None):
            directories[:] = sorted(
                directory
                for directory in directories
                if directory.casefold() not in SCAN_EXCLUDED_DIRECTORIES
            )
            for filename in sorted(filenames):
                path = Path(current) / filename
                if path.suffix.casefold() not in allowed:
                    continue
                resolved = _resolved(path)
                config_base = _claude_config_base(resolved, config_directories)
                if config_base is not None and not _is_claude_memory_file(resolved, config_base):
                    continue
                key = str(resolved).casefold()
                if key in seen:
                    continue
                seen.add(key)
                result.append(path)
                if len(result) >= max_files:
                    return tuple(result)
    return tuple(result)


def classify(path: str | Path) -> SourceCategory | None:
    suffix = Path(path).suffix.lower()
    if suffix in CODE_EXTENSIONS:
        return SourceCategory.CODE
    if suffix in DOCUMENT_EXTENSIONS:
        return SourceCategory.DOCUMENTS
    return None


def display_type(path: str | Path, category: SourceCategory) -> str:
    """Use a useful format label in the desktop table while retaining broad categories."""
    if category is SourceCategory.CODE:
        return "Code"
    suffix = Path(path).suffix.lower().removeprefix(".")
    return suffix.upper() if suffix else "Document"
