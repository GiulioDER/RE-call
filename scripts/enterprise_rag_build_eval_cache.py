"""Build a minimal EnterpriseRAG evaluator document cache.

The official EnterpriseRAG evaluator expects generated JSON documents under
``generated_data/sources``. The public release also ships an exported text ZIP.
For score verification, the evaluator only needs documents referenced by the
gold questions and by the answer file. This script converts just those exported
text files into the JSON shape required by the evaluator.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def _load_needed_ids(questions: Path, answers: Path) -> set[str]:
    needed: set[str] = set()
    with questions.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            needed.update(row.get("expected_doc_ids") or [])
    with answers.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            needed.update(row.get("document_ids") or [])
    return needed


def _doc_id_from_export_name(name: str) -> str | None:
    base = Path(name).name
    if not base.startswith("dsid_") or not base.endswith(".txt"):
        return None
    parts = base.split("_", 2)
    if len(parts) < 2:
        return None
    return f"dsid_{parts[1]}"


def build_cache(
    *,
    questions: Path,
    answers: Path,
    documents_zip: Path,
    sources_out: Path,
    index_out: Path,
) -> None:
    needed = _load_needed_ids(questions, answers)
    missing = set(needed)
    index: dict[str, str] = {}
    sources_out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(documents_zip) as archive:
        for name in archive.namelist():
            dsid = _doc_id_from_export_name(name)
            if dsid is None or dsid not in missing:
                continue
            text = archive.read(name).decode("utf-8", errors="replace")
            lines = text.splitlines()
            title = lines[0].strip() if lines else dsid
            content = "\n".join(lines[1:]).strip() if len(lines) > 1 else text
            rel = Path(name).with_suffix(".json")
            target = sources_out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "dataset_doc_uuid": dsid,
                "title_field_name": "title",
                "content_field_names": ["content"],
                "title": title,
                "content": content,
                "exported_text_path": name,
            }
            target.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            index[dsid] = str(rel)
            missing.remove(dsid)
            if not missing:
                break

    index_out.parent.mkdir(parents=True, exist_ok=True)
    index_out.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"needed_ids={len(needed)} written={len(index)} missing={len(missing)}")
    if missing:
        preview = ", ".join(sorted(missing)[:20])
        raise SystemExit(f"missing referenced document ids: {preview}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--documents-zip", required=True, type=Path)
    parser.add_argument(
        "--sources-out",
        default=Path("generated_data/sources"),
        type=Path,
    )
    parser.add_argument(
        "--index-out",
        default=Path("generated_data/uuid_index.json"),
        type=Path,
    )
    args = parser.parse_args()
    build_cache(
        questions=args.questions,
        answers=args.answers,
        documents_zip=args.documents_zip,
        sources_out=args.sources_out,
        index_out=args.index_out,
    )


if __name__ == "__main__":
    main()
