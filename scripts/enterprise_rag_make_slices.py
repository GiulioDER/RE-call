"""Create deterministic question-id slices from the official EnterpriseRAG release."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def make_slices(
    questions_path: Path,
    out_dir: Path,
    *,
    seed: str = "366",
    dev_fraction: float = 0.5,
    question_types: set[str] | None = None,
) -> dict[str, object]:
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1")
    selected_types = {value.lower() for value in (question_types or set())}
    rows: list[tuple[str, str]] = []
    with questions_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row["question_id"])
            question_type = str(row.get("question_type", "unknown")).lower()
            if selected_types and question_type not in selected_types:
                continue
            rows.append((question_id, question_type))

    dev: list[str] = []
    confirmation: list[str] = []
    for question_id, _ in rows:
        digest = hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        (dev if bucket < dev_fraction else confirmation).append(question_id)
    dev.sort()
    confirmation.sort()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dev.ids").write_text("\n".join(dev) + "\n", encoding="utf-8")
    (out_dir / "confirmation.ids").write_text(
        "\n".join(confirmation) + "\n", encoding="utf-8"
    )
    manifest = {
        "questions": str(questions_path),
        "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
        "seed": seed,
        "dev_fraction": dev_fraction,
        "question_types": sorted(selected_types),
        "total": len(rows),
        "dev": len(dev),
        "confirmation": len(confirmation),
        "counts_by_type": dict(sorted(Counter(question_type for _, question_type in rows).items())),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", default="366")
    parser.add_argument("--dev-fraction", type=float, default=0.5)
    parser.add_argument("--question-types")
    args = parser.parse_args()
    types = (
        {value.strip().lower() for value in args.question_types.split(",") if value.strip()}
        if args.question_types
        else None
    )
    print(json.dumps(make_slices(args.questions, args.out_dir, seed=args.seed, dev_fraction=args.dev_fraction, question_types=types), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
