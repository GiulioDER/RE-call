"""Run the preregistered extractive reader. This script is intended for VPS2 only."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any, Sequence


MODEL = "deepset/roberta-base-squad2"
REVISION = "adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
LICENCE = "cc-by-4.0"
MAX_LENGTH = 512
DOC_STRIDE = 128
MAX_ANSWER_TOKENS = 96
NULL_MARGIN_THRESHOLD = 0.0


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _best_span(
    start_logits: Sequence[float],
    end_logits: Sequence[float],
    offsets: Sequence[Sequence[int]],
    sequence_ids: Sequence[int | None],
    cls_index: int,
) -> tuple[float, int, int]:
    null_score = float(start_logits[cls_index]) + float(end_logits[cls_index])
    best: tuple[float, int, int] | None = None
    context_indexes = [index for index, value in enumerate(sequence_ids) if value == 1]
    for start_index in context_indexes:
        for end_index in context_indexes:
            if end_index < start_index or end_index - start_index + 1 > MAX_ANSWER_TOKENS:
                continue
            start = int(offsets[start_index][0])
            end = int(offsets[end_index][1])
            if end <= start:
                continue
            margin = (
                float(start_logits[start_index]) + float(end_logits[end_index]) - null_score
            )
            candidate = (margin, start, end)
            if best is None or candidate[0] > best[0] or (
                candidate[0] == best[0] and (candidate[1], candidate[2]) < (best[1], best[2])
            ):
                best = candidate
    if best is None:
        return float("-inf"), 0, 0
    return best


def score_row(model: Any, tokenizer: Any, torch: Any, row: dict[str, Any], batch_size: int) -> dict[str, object]:
    question = str(row["question"])
    candidates = list(row["candidates"])
    contexts = [str(candidate["text"]) for candidate in candidates]
    encoded = tokenizer(
        [question] * len(contexts),
        contexts,
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )
    sample_mapping = encoded.pop("overflow_to_sample_mapping")
    offsets = encoded.pop("offset_mapping")
    feature_count = int(encoded["input_ids"].shape[0])
    all_start: list[list[float]] = []
    all_end: list[list[float]] = []
    with torch.inference_mode():
        for begin in range(0, feature_count, batch_size):
            batch = {key: value[begin : begin + batch_size] for key, value in encoded.items()}
            output = model(**batch)
            all_start.extend(output.start_logits.cpu().tolist())
            all_end.extend(output.end_logits.cpu().tolist())

    best_by_candidate: dict[int, tuple[float, int, int]] = {}
    cls_token_id = int(tokenizer.cls_token_id)
    for feature_index in range(feature_count):
        candidate_index = int(sample_mapping[feature_index])
        ids = encoded["input_ids"][feature_index].tolist()
        cls_index = ids.index(cls_token_id)
        span = _best_span(
            all_start[feature_index],
            all_end[feature_index],
            offsets[feature_index].tolist(),
            encoded.sequence_ids(feature_index),
            cls_index,
        )
        previous = best_by_candidate.get(candidate_index)
        if previous is None or span[0] > previous[0] or (
            span[0] == previous[0] and (span[1], span[2]) < (previous[1], previous[2])
        ):
            best_by_candidate[candidate_index] = span

    ranked = sorted(
        (
            (span[0], int(candidates[index]["dense_rank"]), span[1], span[2], index)
            for index, span in best_by_candidate.items()
        ),
        key=lambda value: (-value[0], value[1], value[2], value[3]),
    )
    margin, _dense_rank, start, end, candidate_index = ranked[0]
    if margin <= NULL_MARGIN_THRESHOLD:
        return {
            "qid": str(row["qid"]),
            "selected_doc_id": None,
            "answer": "",
            "start": None,
            "end": None,
            "margin": margin,
            "candidates_scored": len(candidates),
        }
    selected = candidates[candidate_index]
    context = str(selected["text"])
    return {
        "qid": str(row["qid"]),
        "selected_doc_id": str(selected["doc_id"]),
        "answer": context[start:end],
        "start": start,
        "end": end,
        "margin": margin,
        "candidates_scored": len(candidates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    torch.set_num_threads(args.torch_threads)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL, revision=REVISION, use_safetensors=True
    )
    model.eval()
    rows = _load_rows(args.input)
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as fh:
        header = {
            "_header": True,
            "protocol": "2026-09-15-span-grounded-reader-v1",
            "scored_at": datetime.now(UTC).isoformat(),
            "model": MODEL,
            "revision": REVISION,
            "licence": LICENCE,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "config": {
                "max_length": MAX_LENGTH,
                "doc_stride": DOC_STRIDE,
                "max_answer_tokens": MAX_ANSWER_TOKENS,
                "null_margin_threshold": NULL_MARGIN_THRESHOLD,
            },
        }
        fh.write(json.dumps(header) + "\n")
        for completed, row in enumerate(rows, 1):
            result = score_row(model, tokenizer, torch, row, args.batch_size)
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"span reader {completed}/{len(rows)}", flush=True)
    print(json.dumps({"rows": len(rows), "elapsed_seconds": time.perf_counter() - started}))


if __name__ == "__main__":
    main()
