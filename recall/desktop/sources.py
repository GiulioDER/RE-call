"""Source classification and safe file selection for drag and drop."""

from __future__ import annotations

from pathlib import Path

from recall.extraction import DOCUMENT_EXTENSIONS as EXTRACTABLE_DOCUMENT_EXTENSIONS
from recall.desktop.models import SourceCategory

DOCUMENT_EXTENSIONS = EXTRACTABLE_DOCUMENT_EXTENSIONS
CODE_EXTENSIONS = frozenset({".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".rb", ".c", ".h", ".cpp", ".hpp", ".cs", ".sh", ".sql"})


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
