"""Stage 1 metrics and apparatus checks for the MM-1/MM-3 pre-registration.

Pre-registration: ``docs/preregistrations/2026-09-25-aml-c9-multimodal-scope-and-dates.md``.
Reads the JSONL written by ``scripts/aml_mm_scope_stage1.py`` and the MemEye cache, and reports:

* the stratum each question falls in, by arm B's route (non-multimodal is what MM-1 is about);
* "clue image delivered": a clue round's item carrying an image among the items the frozen
  MemEye answer packer would admit under the 117,760-token budget;
* any-clue Recall@10 by session, and the median number of admitted items;
* apparatus checks 2 to 5 of the record (check 1 is a unit test, 6 belongs to Stage 2, 7 to the
  run summary).

MM-3 (arms P+t and D+t) is built here by applying the production
``recall_aml.window_format.dated_multimodal_items`` to the stored P and D responses.

Usage::

    python scripts/aml_mm_scope_report.py --results FILE.jsonl --cache-dir DIR --out REPORT.json
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

from recall_aml.models import SearchItem
from recall_aml.window_format import dated_multimodal_items

ANSWER_TOKEN_BUDGET = 117_760
IMAGE_TOKEN_ESTIMATE = 1_000
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def load_questions(cache_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    identity = json.loads((cache_dir / "identity.json").read_text(encoding="utf-8"))
    questions: dict[tuple[str, str], dict[str, Any]] = {}
    for scenario in identity["scenarios"]:
        dataset = json.loads((cache_dir / f"{scenario}.json").read_text(encoding="utf-8"))
        for qa in dataset["human-annotated QAs"]:
            axes = sorted(
                {
                    str(value)
                    for group in qa.get("point", [])
                    for value in (group if isinstance(group, list) else [group])
                }
            )
            questions[(scenario, str(qa["question_id"]))] = {
                "question": qa["question"],
                "clues": [str(clue) for clue in qa["clue"]],
                "axes": axes,
            }
    return questions


def image_index(cache_dir: Path) -> dict[str, Path]:
    import hashlib

    index: dict[str, Path] = {}
    for path in (cache_dir / "image").rglob("*"):
        if path.is_file():
            index[hashlib.sha256(path.read_bytes()).hexdigest()] = path
    return index


def admitted_prefix(items: list[dict[str, Any]], question: str) -> int:
    """How many ranked items the frozen MemEye packer (``pack_answer_content``) would admit."""
    estimated = math.ceil(len(question) / 4) + 128
    admitted = 0
    for rank, item in enumerate(items, start=1):
        cost = math.ceil(len(f"\nMemory {rank}, session {item.get('session_id', '')}:\n") / 4)
        content = item["content"]
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        for part in parts:
            cost += IMAGE_TOKEN_ESTIMATE if part["type"] == "image" else math.ceil(len(part["text"]) / 4)
        if estimated + cost > ANSWER_TOKEN_BUDGET:
            break
        estimated += cost
        admitted += 1
    return admitted


def has_image(item: dict[str, Any]) -> bool:
    return isinstance(item["content"], list) and any(part["type"] == "image" for part in item["content"])


def row_metrics(items: list[dict[str, Any]], question: dict[str, Any]) -> dict[str, float]:
    clues = set(question["clues"])
    admitted = admitted_prefix(items, question["question"])
    sessions = [str(item.get("session_id", "")) for item in items]
    return {
        "clue_image_delivered": float(
            any(str(item.get("session_id")) in clues and has_image(item) for item in items[:admitted])
        ),
        "any_clue_recall_at_10": float(bool(clues.intersection(sessions[:10]))),
        "any_clue_recall_at_100": float(bool(clues.intersection(sessions[:100]))),
        "admitted_items": float(admitted),
        "image_items_admitted": float(sum(has_image(item) for item in items[:admitted])),
    }


def rebuild(items: list[dict[str, Any]], images: dict[str, Path]) -> list[SearchItem]:
    """The exact SearchItems the service returned, with each image restored from the cache."""
    rebuilt: list[SearchItem] = []
    for item in items:
        content: Any = item["content"]
        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for part in content:
                if part["type"] == "image":
                    path = images[part["sha256"]]
                    url = f"data:{MIME[path.suffix.lower()]};base64," + base64.b64encode(path.read_bytes()).decode()
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                else:
                    parts.append({"type": "text", "text": part["text"]})
            content = parts
        rebuilt.append(
            SearchItem.model_validate(
                {
                    "id": item["id"],
                    "content": content,
                    "created_at": item["created_at"],
                    "source": "",
                    "session_id": item["session_id"] or "",
                    "kind": item["kind"] or "",
                    "score": item["score"],
                }
            )
        )
    return rebuilt


def check_dates_render_only(items: list[dict[str, Any]], images: dict[str, Path]) -> bool:
    """Apparatus check 4 on one response: same ids, order and scores, one leading date part."""
    before = rebuild(items, images)
    after = dated_multimodal_items(before)
    if [item.id for item in after] != [item.id for item in before]:
        return False
    for old, new in zip(before, after):
        if new.score != old.score:
            return False
        if isinstance(old.content, list) and old.created_at is not None:
            if not isinstance(new.content, list) or new.content[1:] != old.content:
                return False
            if len(new.content) != len(old.content) + 1 or new.content[0].type != "text":
                return False
        elif new != old:
            return False
    return True


def report(results: Path, cache_dir: Path) -> dict[str, Any]:
    questions = load_questions(cache_dir)
    images = image_index(cache_dir)
    rows: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    with results.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            rows[(row["scenario"], row["question_id"], row["rotation"], row["arm"])] = row
    by_question: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for (scenario, question_id, _rotation, arm), row in sorted(rows.items()):
        by_question[(scenario, question_id)][arm].append(row)

    missing = sorted(set(questions) - set(by_question))
    checks: dict[str, Any] = {"questions_missing": len(missing)}
    per_question: list[dict[str, Any]] = []
    b_leg_matches_route = d_leg_always = True
    dates_render_only = True
    identical_b_rotations = total_b_rotations = 0
    for key, arms in sorted(by_question.items()):
        question = questions[key]
        routes = {row["route"] for row in arms["B"]}
        stratum = "multimodal" if routes == {"multimodal"} else "other" if "multimodal" not in routes else "mixed"
        record: dict[str, Any] = {"scenario": key[0], "question_id": key[1], "stratum": stratum, "axes": question["axes"]}
        for arm in ("B", "B2", "P", "D"):
            metrics = [row_metrics(row["items"], question) for row in arms[arm]]
            record[arm] = {name: mean(m[name] for m in metrics) for name in metrics[0]}
        for row in arms["B"]:
            b_leg_matches_route &= (row["visual_leg"] == "1") == (row["route"] == "multimodal")
        for row in arms["D"]:
            d_leg_always &= row["visual_leg"] == "1"
        for row in arms["P"] + arms["D"]:
            dates_render_only &= check_dates_render_only(row["items"], images)
        for b_row, b2_row in zip(arms["B"], arms["B2"]):
            total_b_rotations += 1
            identical_b_rotations += [i["id"] for i in b_row["items"]] == [i["id"] for i in b2_row["items"]]
        per_question.append(record)

    def stratum_summary(stratum: str) -> dict[str, Any]:
        chosen = [record for record in per_question if record["stratum"] == stratum]
        if not chosen:
            return {"n": 0}
        summary: dict[str, Any] = {"n": len(chosen)}
        for arm in ("B", "B2", "P", "D"):
            summary[arm] = {
                "clue_image_delivered": mean(r[arm]["clue_image_delivered"] for r in chosen),
                "any_clue_recall_at_10": mean(r[arm]["any_clue_recall_at_10"] for r in chosen),
                "any_clue_recall_at_100": mean(r[arm]["any_clue_recall_at_100"] for r in chosen),
                "median_admitted_items": median(r[arm]["admitted_items"] for r in chosen),
                "mean_image_items_admitted": mean(r[arm]["image_items_admitted"] for r in chosen),
            }
        return summary

    strata = {name: stratum_summary(name) for name in ("other", "multimodal", "mixed")}
    other, visual = strata["other"], strata["multimodal"]
    checks.update(
        {
            "2_b_delivers_no_image_off_route": other.get("n", 0) > 0 and other["B"]["clue_image_delivered"] == 0.0,
            "2_b_delivers_images_on_route": visual.get("n", 0) > 0 and visual["B"]["clue_image_delivered"] > 0.0,
            "3_b_visual_leg_only_on_route": b_leg_matches_route,
            "3_d_visual_leg_every_query": d_leg_always,
            "4_dates_render_only": dates_render_only,
            "5_b_b2_identical_ranked_lists": identical_b_rotations / total_b_rotations if total_b_rotations else None,
        }
    )
    return {"strata": strata, "checks": checks, "questions": per_question}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.results, args.cache_dir)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"strata": result["strata"], "checks": result["checks"]}, indent=2))


if __name__ == "__main__":
    main()
