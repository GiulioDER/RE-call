#!/usr/bin/env python
"""Encode passages to pruned SPLADE vectors. Runs anywhere with a GPU; touches no database.

    python scripts/encode_sparse.py --input passages.jsonl --output vectors.jsonl \
        --model prithivida/Splade_PP_en_v1 --batch-size 64

Input is JSONL with `id` and `text` per line. Output is JSONL with `id`, `weights`
({term_id: weight}, already pruned) and `nnz`, plus a header line carrying the encoder identity.

NOTE: this script takes NO DSN and reads NO credentials, deliberately. The encode is the only part
of the pipeline that wants rented hardware, and rented hardware is the last place a database
credential should be. `scripts/load_sparse.py` does the writing, from a host you trust.

(This docstring is printed by argparse and is kept ASCII-only on purpose: a non-ASCII character
here crashes `--help` outright under a cp1252 console, which is the default on Windows. Found by
running it, not by reading it.)

The split also bounds the blast radius of a reclaimed instance: `--resume` skips ids already in the
output, so a spot instance dying at 80% costs the remaining 20% and not the run.

`--limit` exists for the throughput measurement that must precede a full run. Encode 1000, read the
reported rate, and decide from a measured number rather than an estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def read_passages(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def already_done(path: Path) -> set[str]:
    """Ids present in an existing output file, so `--resume` can skip them.

    A malformed trailing line (the shape an interrupted write leaves behind) is ignored rather
    than fatal — that line's id simply is not counted as done, so it gets encoded again. Re-doing
    one passage is the cheap error here; treating a truncated file as complete is the expensive
    one.
    """
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "id" in row:
                done.add(str(row["id"]))
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL with id + text")
    parser.add_argument("--output", type=Path, required=True, help="JSONL of sparse vectors")
    parser.add_argument("--model", default=None, help="defaults to the apache-2.0 encoder")
    parser.add_argument("--revision", default=None, help="pin the model revision (reproducibility)")
    parser.add_argument("--top-k", type=int, default=1000, help="prune budget; pgvector caps at 1000")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="encode only the first N passages")
    parser.add_argument("--resume", action="store_true", help="skip ids already in --output")
    parser.add_argument(
        "--accept-noncommercial-license", action="store_true",
        help="required for naver/splade-v3 (cc-by-nc-sa-4.0); RE-call itself is MIT",
    )
    args = parser.parse_args(argv)

    from recall.sparse import DEFAULT_MODEL, SpladeEncoder

    encoder = SpladeEncoder.from_pretrained(
        args.model or DEFAULT_MODEL,
        top_k=args.top_k,
        revision=args.revision,
        accept_noncommercial_license=args.accept_noncommercial_license,
        max_length=args.max_length,
    )

    passages = read_passages(args.input, args.limit)
    done = already_done(args.output) if args.resume else set()
    pending = [p for p in passages if str(p["id"]) not in done]
    print(
        json.dumps({
            "event": "start", "passages": len(passages), "skipped": len(passages) - len(pending),
            "profile": encoder.profile.profile_id, "fingerprint": encoder.profile.fingerprint(),
        }),
        flush=True,
    )

    mode = "a" if (args.resume and args.output.exists()) else "w"
    written = 0
    empty = 0
    started = time.perf_counter()
    with args.output.open(mode, encoding="utf-8") as out:
        if mode == "w":
            # The identity travels WITH the vectors. A bare file of term weights cannot say which
            # model produced it, and vectors from two encoders are silently mixable.
            out.write(json.dumps({
                "_header": True,
                "model_name": encoder.profile.model_name,
                "profile_id": encoder.profile.profile_id,
                "artifact_digest": encoder.profile.artifact_digest,
                "top_k": encoder.profile.top_k,
                "dimension": encoder.profile.dimension,
                "fingerprint": encoder.profile.fingerprint(),
            }) + "\n")
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            vectors = encoder.encode([str(p["text"]) for p in batch])
            for passage, weights in zip(batch, vectors, strict=True):
                if not weights:
                    # Recorded, not silently dropped: an all-empty run means a broken encoder, and
                    # the count is what makes that visible instead of a mysteriously small index.
                    empty += 1
                    continue
                out.write(json.dumps({
                    "id": str(passage["id"]),
                    "weights": {str(term): round(float(w), 5) for term, w in weights.items()},
                    "nnz": len(weights),
                }) + "\n")
                written += 1
            elapsed = time.perf_counter() - started
            print(
                json.dumps({
                    "event": "progress", "written": written, "empty": empty,
                    "elapsed_s": round(elapsed, 1),
                    "rate_per_s": round(written / elapsed, 2) if elapsed > 0 else None,
                }),
                flush=True,
            )

    elapsed = time.perf_counter() - started
    rate = written / elapsed if elapsed > 0 else 0.0
    print(
        json.dumps({
            "event": "done", "written": written, "empty": empty,
            "elapsed_s": round(elapsed, 1), "rate_per_s": round(rate, 2),
            # The whole reason --limit exists: turn a measured rate into a defensible projection
            # for the full corpus, instead of quoting an estimate as if it were a plan.
            "projected_hours_for_366479": round(366479 / rate / 3600, 2) if rate else None,
        }),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
