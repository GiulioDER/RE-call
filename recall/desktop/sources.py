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
CLAUDE_MEMORY_EXTENSIONS = frozenset({".md", ".txt", ".json", ".jsonl", ".yaml", ".yml"})


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
        base / ".claude",
        base / ".config" / "claude",
        base / "AppData" / "Roaming" / "Claude",
        base / "AppData" / "Local" / "Claude",
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


def _is_claude_root(root: Path) -> bool:
    return root.name.casefold() == "claude" or ".claude" in {part.casefold() for part in root.parts}


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
) -> tuple[Path, ...]:
    """Find supported local sources while pruning generated and private dependency folders.

    Claude folders are restricted to named memory files and memory directories so a scan does
    not ingest settings, credentials, caches, or conversation databases by accident.
    """
    allowed = category_extensions(category) if category is not None else DOCUMENT_EXTENSIONS | CODE_EXTENSIONS
    result: list[Path] = []
    seen: set[str] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if not root.is_dir():
            continue
        claude_root = _is_claude_root(root)
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
                if claude_root and not _is_claude_memory_file(path, root):
                    continue
                try:
                    key = str(path.resolve()).casefold()
                except OSError:
                    key = str(path.absolute()).casefold()
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
