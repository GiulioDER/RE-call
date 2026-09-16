"""Train and score the preregistered task specific dense pool selector."""

from __future__ import annotations

import argparse
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence


MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
SEED = 20260916
EPOCHS = 2.0
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
MAX_LENGTH = 512
NEGATIVES_PER_QUERY = 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def flatten_training_pairs(
    payload: Mapping[str, Any], *, enforce_row_count: bool = True
) -> tuple[list[str], list[str], list[float]]:
    rows = list(payload.get("train_rows", []))
    if enforce_row_count and len(rows) != 186:
        raise RuntimeError(f"expected 186 train rows, got {len(rows)}")
    queries: list[str] = []
    responses: list[str] = []
    labels: list[float] = []
    for row in rows:
        negatives = [str(value) for value in row.get("negative_texts", [])]
        if len(negatives) != NEGATIVES_PER_QUERY:
            raise RuntimeError(f"row {row.get('id')} does not have four negatives")
        query = str(row["query"])
        positive = str(row["positive_text"])
        if positive in negatives:
            raise RuntimeError(f"row {row.get('id')} labels its positive as a negative")
        queries.extend([query] * (1 + len(negatives)))
        responses.extend([positive, *negatives])
        labels.extend([1.0, *([0.0] * len(negatives))])
    return queries, responses, labels


def order_from_scores(
    candidates: Sequence[Mapping[str, Any]], scores: Mapping[str, float]
) -> list[str]:
    ids = [str(candidate["chunk_id"]) for candidate in candidates]
    if set(ids) != set(scores) or len(ids) != len(scores):
        raise RuntimeError("score membership differs from the candidate pool")
    rank = {str(candidate["chunk_id"]): int(candidate["dense_rank"]) for candidate in candidates}
    return sorted(ids, key=lambda chunk_id: (-float(scores[chunk_id]), rank[chunk_id]))


def _score_rows(model: Any, rows: Sequence[Mapping[str, Any]], *, batch_size: int) -> list[dict[str, Any]]:
    pairs: list[tuple[str, str]] = []
    spans: list[tuple[int, int]] = []
    for row in rows:
        start = len(pairs)
        query = str(row["query"])
        candidates = list(row["candidates"])
        pairs.extend((query, str(candidate["text"])) for candidate in candidates)
        spans.append((start, len(pairs)))
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    output = []
    for row, (start, stop) in zip(rows, spans, strict=True):
        candidates = list(row["candidates"])
        values = {
            str(candidate["chunk_id"]): float(score)
            for candidate, score in zip(candidates, scores[start:stop], strict=True)
        }
        output.append({"id": str(row["id"]), "scores": values})
    return output


def _model_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(value for value in path.rglob("*") if value.is_file()):
        relative = file_path.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = file_path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    return {
        name: version(name)
        for name in ("torch", "sentence-transformers", "transformers", "datasets")
    }


def _load_base() -> Any:
    from sentence_transformers.cross_encoder import CrossEncoder

    return CrossEncoder(MODEL, revision=REVISION, max_length=MAX_LENGTH)


def train_validate(args: argparse.Namespace) -> None:
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    score_rows = list(payload.get("score_rows", []))
    if len(score_rows) != 33:
        raise RuntimeError(f"expected 33 validation rows, got {len(score_rows)}")
    queries, responses, labels = flatten_training_pairs(payload)

    import torch

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    base = _load_base()
    first_parameter_name, first_parameter = next(iter(base.model.named_parameters()))
    base_fingerprint = first_parameter.detach().cpu().clone()
    started_base = time.perf_counter()
    base_scores = _score_rows(base, score_rows, batch_size=BATCH_SIZE)
    base_score_seconds = time.perf_counter() - started_base
    del base
    gc.collect()

    from datasets import Dataset
    from sentence_transformers.cross_encoder import (
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    model = _load_base()
    dataset = Dataset.from_dict(
        {"query": queries, "response": responses, "label": labels}
    ).shuffle(seed=SEED)
    loss = BinaryCrossEntropyLoss(model)
    training_args = CrossEncoderTrainingArguments(
        output_dir=str(args.model_output / "_run"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        fp16=False,
        bf16=False,
        logging_steps=25,
        save_strategy="no",
        report_to=[],
        seed=SEED,
        dataloader_num_workers=0,
    )
    started_train = time.perf_counter()
    CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
    ).train()
    train_seconds = time.perf_counter() - started_train
    trained_parameter = dict(model.model.named_parameters())[first_parameter_name]
    weight_drift = float((trained_parameter.detach().cpu() - base_fingerprint).abs().max())
    if weight_drift == 0.0:
        raise RuntimeError("training left the sampled model weights unchanged")
    started_score = time.perf_counter()
    trained_scores = _score_rows(model, score_rows, batch_size=BATCH_SIZE)
    trained_score_seconds = time.perf_counter() - started_score
    args.model_output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.model_output))
    model_digest = _model_digest(args.model_output)
    _write(
        args.output,
        {
            "schema_version": 1,
            "protocol": "2026-09-16-task-specific-dense-hard-negative-selector-validation-scores",
            "input_sha256": _sha256(args.input),
            "model": MODEL,
            "revision": REVISION,
            "versions": _versions(),
            "hyperparameters": {
                "seed": SEED,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "warmup_ratio": 0.1,
                "max_length": MAX_LENGTH,
                "negatives_per_query": NEGATIVES_PER_QUERY,
            },
            "training_pairs": {
                "positive": int(sum(labels)),
                "negative": len(labels) - int(sum(labels)),
            },
            "weight_parameter": first_parameter_name,
            "weight_drift_max_abs": weight_drift,
            "model_digest": model_digest,
            "train_seconds": train_seconds,
            "base_score_seconds": base_score_seconds,
            "trained_score_seconds": trained_score_seconds,
            "base_scores": base_scores,
            "trained_scores": trained_scores,
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model_output": str(args.model_output),
                "model_digest": model_digest,
                "weight_drift_max_abs": weight_drift,
            }
        ),
        flush=True,
    )


def score_internal_test(args: argparse.Namespace) -> None:
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = list(payload.get("score_rows", []))
    if len(rows) != 29:
        raise RuntimeError(f"expected 29 internal test rows, got {len(rows)}")
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder

    torch.set_num_threads(4)
    base = _load_base()
    base_scores = _score_rows(base, rows, batch_size=BATCH_SIZE)
    del base
    gc.collect()
    trained = CrossEncoder(str(args.model), max_length=MAX_LENGTH)
    trained_scores = _score_rows(trained, rows, batch_size=BATCH_SIZE)
    _write(
        args.output,
        {
            "schema_version": 1,
            "protocol": "2026-09-16-task-specific-dense-hard-negative-selector-test-scores",
            "input_sha256": _sha256(args.input),
            "model_digest": _model_digest(args.model),
            "versions": _versions(),
            "base_scores": base_scores,
            "trained_scores": trained_scores,
        },
    )
    print(json.dumps({"output": str(args.output), "rows": len(rows)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train-validate")
    train_parser.add_argument("--input", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--model-output", type=Path, required=True)
    score_parser = sub.add_parser("score-internal-test")
    score_parser.add_argument("--input", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train-validate":
        train_validate(args)
    else:
        score_internal_test(args)


if __name__ == "__main__":
    main()
