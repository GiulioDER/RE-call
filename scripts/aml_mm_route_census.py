"""Stage 0 of the MM-1/MM-3 pre-registration: which public questions reach C9's image route.

Applies the served ``route_query`` to each question string of MemEye (8 MCQ scenarios), MemLens
and MobileMem-Omni, and reports the share routed to ``multimodal``. No model, no network beyond
the files already downloaded, no service. Pre-registration:
``docs/preregistrations/2026-09-25-aml-c9-multimodal-scope-and-dates.md``.

Usage::

    python scripts/aml_mm_route_census.py --data-dir DIR --output census.json

``DIR`` holds ``memeye/<scenario>.json``, ``memlens_32k.parquet``,
``mobilemem_filtered_questions.jsonl`` and ``mobilemem_questions.jsonl``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from recall_aml.specialists import route_query

MEMEYE_SCENARIOS = (
    "Brand_Memory_Test",
    "Card_Playlog_Test",
    "Cartoon_Entertainment_Companion",
    "Home_Renovation_Interior_Design",
    "Multi-Scene_Visual_Case_Archive_Assistant",
    "Outdoor_Navigation_Route_Memory_Assistant",
    "Personal_Health_Dashboard_Assistant",
    "Social_Chat_Memory_Test",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def share(routes: Iterable[str]) -> dict[str, Any]:
    """Route counts and the multimodal share over every question given."""
    counts = Counter(routes)
    total = sum(counts.values())
    return {
        "n": total,
        "routes": dict(sorted(counts.items())),
        "multimodal_share": counts["multimodal"] / total if total else None,
    }


def memeye(data_dir: Path) -> dict[str, Any]:
    routes: list[str] = []
    by_scenario: dict[str, list[str]] = defaultdict(list)
    by_axis: dict[str, list[str]] = defaultdict(list)
    files: dict[str, str] = {}
    for scenario in MEMEYE_SCENARIOS:
        path = data_dir / "memeye" / f"{scenario}.json"
        files[scenario] = _sha256(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for qa in payload["human-annotated QAs"]:
            route = route_query(str(qa["question"]))
            routes.append(route)
            by_scenario[scenario].append(route)
            for group in qa.get("point", []):
                for axis in group if isinstance(group, list) else [group]:
                    by_axis[str(axis)].append(route)
    return {
        "files_sha256": files,
        "all": share(routes),
        "by_scenario": {key: share(value) for key, value in by_scenario.items()},
        "by_axis": {key: share(value) for key, value in sorted(by_axis.items())},
    }


def memlens(data_dir: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    path = data_dir / "memlens_32k.parquet"
    table = pq.read_table(path, columns=["question", "question_type"]).to_pylist()
    routes = [route_query(str(row["question"])) for row in table]
    by_type: dict[str, list[str]] = defaultdict(list)
    for row, route in zip(table, routes):
        by_type[str(row["question_type"])].append(route)
    return {
        "file_sha256": _sha256(path),
        "all": share(routes),
        "by_question_type": {key: share(value) for key, value in sorted(by_type.items())},
    }


def mobilemem(path: Path) -> dict[str, Any]:
    by_language: dict[str, list[str]] = defaultdict(list)
    by_type: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            user = json.loads(line)
            language = str(user["language"])
            for question in user["questions"]:
                route = route_query(str(question["question"]))
                by_language[language].append(route)
                by_type[f"{language}:{question.get('question_type')}"].append(route)
    return {
        "file_sha256": _sha256(path),
        "all": share(route for routes in by_language.values() for route in routes),
        "by_language": {key: share(value) for key, value in sorted(by_language.items())},
        "by_language_and_type": {key: share(value) for key, value in sorted(by_type.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "memeye": memeye(args.data_dir),
        "memlens_32k": memlens(args.data_dir),
        "mobilemem_filtered": mobilemem(args.data_dir / "mobilemem_filtered_questions.jsonl"),
        "mobilemem_all": mobilemem(args.data_dir / "mobilemem_questions.jsonl"),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for name in ("memeye", "memlens_32k", "mobilemem_filtered", "mobilemem_all"):
        print(name, json.dumps(report[name]["all"]))
    for name in ("mobilemem_filtered", "mobilemem_all"):
        for language, value in report[name]["by_language"].items():
            print(f"{name}[{language}]", json.dumps(value))


if __name__ == "__main__":
    main()
