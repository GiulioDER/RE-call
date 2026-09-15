"""Measure a span grounded reader over the original dense top 20."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.token_f1 import normalize as squad_normalize  # noqa: E402
from benchmarks.token_f1 import token_f1  # noqa: E402
from scripts.run_anchor_boosted_query_dev import _cutoff_counts  # noqa: E402
from scripts.run_colbert_dense_selector_dev import validate_collection  # noqa: E402


EXPECTED_MODEL = "deepset/roberta-base-squad2"
EXPECTED_REVISION = "adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
EXPECTED_LICENCE = "cc-by-4.0"
EXPECTED_CONFIG = {
    "max_length": 512,
    "doc_stride": 128,
    "max_answer_tokens": 96,
    "null_margin_threshold": 0.0,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def prepare_input(collection: Mapping[str, Any], output: Path) -> dict[str, int]:
    rows = list(collection["rows"])
    validate_collection(rows)
    public_rows: list[dict[str, object]] = []
    pair_count = 0
    for row in rows:
        candidates = list(row["candidates"])
        if len(candidates) != 20:
            raise RuntimeError("reader input requires exactly 20 dense candidates per row")
        pair_count += len(candidates)
        public_rows.append(
            {
                "qid": str(row["query_index"]),
                "question": str(row["query"]),
                "candidates": [
                    {
                        "doc_id": str(candidate["chunk_id"]),
                        "text": str(candidate["text"]),
                        "dense_rank": rank,
                    }
                    for rank, candidate in enumerate(candidates, 1)
                ],
            }
        )
    _jsonl(output, public_rows)
    return {"queries": len(public_rows), "pairs": pair_count}


def load_reader_output(
    path: Path,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    header: dict[str, object] | None = None
    results: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("_header"):
                if header is not None:
                    raise RuntimeError("reader output has more than one identity header")
                header = dict(row)
                continue
            qid = str(row["qid"])
            if qid in results:
                raise RuntimeError(f"reader output repeats qid {qid}")
            results[qid] = dict(row)
    if header is None:
        raise RuntimeError("reader output has no identity header")
    if header.get("model") != EXPECTED_MODEL or header.get("revision") != EXPECTED_REVISION:
        raise RuntimeError("reader output has the wrong model identity")
    if header.get("licence") != EXPECTED_LICENCE:
        raise RuntimeError("reader output has the wrong model licence")
    if header.get("config") != EXPECTED_CONFIG:
        raise RuntimeError("reader output has the wrong frozen reader configuration")
    if not header.get("transformers_version") or not header.get("torch_version"):
        raise RuntimeError("reader output does not record library versions")
    return header, results


def _validated_selection(
    row: Mapping[str, Any], result: Mapping[str, object]
) -> dict[str, object] | None:
    candidates = list(row["candidates"])
    candidates_scored = result.get("candidates_scored")
    if not isinstance(candidates_scored, int) or isinstance(candidates_scored, bool):
        raise RuntimeError("reader output has an invalid candidate count")
    if candidates_scored != len(candidates):
        raise RuntimeError("reader did not score the complete candidate pool")
    selected_id = result.get("selected_doc_id")
    margin_value = result.get("margin")
    if not isinstance(margin_value, (int, float)) or isinstance(margin_value, bool):
        raise RuntimeError("reader output has an invalid margin")
    margin = float(margin_value)
    if selected_id is None:
        if margin > 0.0 or result.get("answer") not in {"", None}:
            raise RuntimeError("reader null output is inconsistent with its margin or answer")
        return None
    if margin <= 0.0:
        raise RuntimeError("reader selected a span without a positive margin")
    by_id = {str(candidate["chunk_id"]): candidate for candidate in candidates}
    if str(selected_id) not in by_id:
        raise RuntimeError("reader selected a chunk outside the original dense top 20")
    candidate = dict(by_id[str(selected_id)])
    start, end = result.get("start"), result.get("end")
    if not isinstance(start, int) or isinstance(start, bool):
        raise RuntimeError("reader selection has an invalid start offset")
    if not isinstance(end, int) or isinstance(end, bool) or end <= start:
        raise RuntimeError("reader selection has an invalid end offset")
    text = str(candidate["text"])
    answer = str(result.get("answer", ""))
    if text[start:end] != answer or not answer:
        raise RuntimeError("reader answer is not the exact selected chunk substring")
    candidate.update(
        {
            "answer": answer,
            "start": start,
            "end": end,
            "margin": margin,
        }
    )
    return candidate


def apply_reader(
    rows: list[dict[str, Any]], results: Mapping[str, Mapping[str, object]]
) -> list[dict[str, Any]]:
    expected_qids = {str(row["query_index"]) for row in rows}
    if set(results) != expected_qids:
        raise RuntimeError("reader output qids do not match the frozen collection")
    measured: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        qid = str(row["query_index"])
        raw = _validated_selection(row, results[qid])
        gated = raw if bool(row["eligible"]) else None
        row["raw_selection"] = raw
        row["gated_selection"] = gated
        row["membership_preserved"] = len(row["candidates"]) == 20
        measured.append(row)
    return measured


def _arm_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    answerable = [row for row in rows if row["expected_answerability"] == "answerable"]
    controls = [row for row in rows if row["expected_answerability"] == "unanswerable"]
    selections = [row[field] for row in answerable if row[field] is not None]
    exact_quotes = [
        squad_normalize(str(selected["answer"]))
        == squad_normalize(str(row["answer_span"]))
        for row in answerable
        if (selected := row[field]) is not None
    ]
    f1_values = [
        token_f1(str(selected["answer"]), str(row["answer_span"]))
        for row in answerable
        if (selected := row[field]) is not None
    ]
    return {
        "answerable_selections": len(selections),
        "control_activations": sum(row[field] is not None for row in controls),
        "selected_gold_source_rows": sum(bool(selected["gold_source"]) for selected in selections),
        "selected_exact_bearing_rows": sum(bool(selected["exact_span"]) for selected in selections),
        "literal_substring_rows": len(selections),
        "exact_quote_match_rows": sum(exact_quotes),
        "mean_selected_quote_token_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    validate_collection(rows)
    answerable = [row for row in rows if row["expected_answerability"] == "answerable"]
    candidate = _arm_summary(rows, "gated_selection")
    raw = _arm_summary(rows, "raw_selection")
    selected_changes = [
        row
        for row in answerable
        if row["gated_selection"] is not None
        and row["gated_selection"]["chunk_id"] != row["candidates"][0]["chunk_id"]
    ]
    exact_gains = sum(
        bool(row["gated_selection"]["exact_span"]) and not bool(row["candidates"][0]["exact_span"])
        for row in answerable
        if row["gated_selection"] is not None
    )
    exact_losses = sum(
        bool(row["candidates"][0]["exact_span"])
        and (row["gated_selection"] is None or not bool(row["gated_selection"]["exact_span"]))
        for row in answerable
    )
    gold_gains = sum(
        bool(row["gated_selection"]["gold_source"])
        and not bool(row["candidates"][0]["gold_source"])
        for row in answerable
        if row["gated_selection"] is not None
    )
    gold_losses = sum(
        bool(row["candidates"][0]["gold_source"])
        and (row["gated_selection"] is None or not bool(row["gated_selection"]["gold_source"]))
        for row in answerable
    )
    summary: dict[str, Any] = {
        "empty_base_rows": len(rows),
        "answerable_rows": len(answerable),
        "control_rows": len(rows) - len(answerable),
        "eligible_answerable": sum(bool(row["eligible"]) for row in answerable),
        "eligible_controls": sum(
            bool(row["eligible"])
            for row in rows
            if row["expected_answerability"] == "unanswerable"
        ),
        "original_dense_gold_source_by_cutoff": _cutoff_counts(
            answerable, "original_min_gold_rank"
        ),
        "original_dense_exact_span_by_cutoff": _cutoff_counts(
            answerable, "original_min_exact_rank"
        ),
        "raw_reader": raw,
        "anchor_gated_reader": candidate,
        "selected_chunk_changed_from_dense_rank1": len(selected_changes),
        "selected_gold_gains": gold_gains,
        "selected_gold_losses": gold_losses,
        "selected_exact_gains": exact_gains,
        "selected_exact_losses": exact_losses,
        "membership_preserved_rows": sum(bool(row["membership_preserved"]) for row in rows),
    }
    proceed = (
        candidate["selected_gold_source_rows"] >= 7
        and candidate["selected_exact_bearing_rows"] >= 5
        and candidate["control_activations"] == 0
        and candidate["literal_substring_rows"] == candidate["answerable_selections"]
        and summary["membership_preserved_rows"] == len(rows)
        and exact_gains >= 2
        and exact_losses == 0
    )
    summary["decision"] = (
        "PROCEED_FRESH_SPAN_GROUNDED_READER_VALIDATION"
        if proceed
        else "STOP_OFF_THE_SHELF_READER_ON_THIS_COHORT"
    )
    return summary


def prepare(args: argparse.Namespace) -> None:
    collection = json.loads(args.collection.read_text(encoding="utf-8"))
    if _sha256(args.collection) != args.collection_sha256:
        raise RuntimeError("collection SHA256 does not match the preregistration")
    counts = prepare_input(collection, args.output)
    print(json.dumps({"input_sha256": _sha256(args.output), **counts}))


def report(args: argparse.Namespace) -> None:
    collection = json.loads(args.collection.read_text(encoding="utf-8"))
    if _sha256(args.collection) != args.collection_sha256:
        raise RuntimeError("collection SHA256 does not match the preregistration")
    header, results = load_reader_output(args.reader_output)
    rows = apply_reader(list(collection["rows"]), results)
    summary = summarize(rows)
    artifact = {
        "schema_version": 1,
        "protocol": "2026-09-15-span-grounded-reader-dense-top20-development",
        "measured_at": datetime.now(UTC).isoformat(),
        "preregistration_commit": args.preregistration_commit,
        "collection_sha256": args.collection_sha256,
        "reader_output_sha256": _sha256(args.reader_output),
        "reader_identity": header,
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({key: value for key, value in artifact.items() if key != "rows"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--collection", type=Path, required=True)
    prepare_parser.add_argument("--collection-sha256", required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--collection", type=Path, required=True)
    report_parser.add_argument("--collection-sha256", required=True)
    report_parser.add_argument("--reader-output", type=Path, required=True)
    report_parser.add_argument("--preregistration-commit", required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
