"""Print the stable `file:ord` label + a text preview for every chunk of the eval corpus.

Run offline (no DB) to author queries.json gold labels:
    python -m recall.eval._show_ids
"""
from __future__ import annotations

from pathlib import Path

from recall.index import chunk_text

CORPUS = Path(__file__).parent / "corpus"


def main() -> None:
    for f in sorted(CORPUS.glob("*.md")):
        # Deliberately chunks raw file bytes, frontmatter and any derived block included: this
        # is eval-label authoring, not a corpus evidence path, so it is outside the
        # parse_document seam (docs/DERIVED_BLOCK_DESIGN.md) by design.
        chunks = chunk_text(f.read_text(encoding="utf-8"))
        for i, c in enumerate(chunks):
            preview = c.replace("\n", " ")[:80]
            print(f"{f.name}:{i}  {preview!r}")


if __name__ == "__main__":
    main()
